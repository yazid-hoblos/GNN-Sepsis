#!/usr/bin/env python3
"""
Extract consensus biomarker proteins across all 4 ML models on ComplEx embeddings.

This script:
1. Loads feature importance from SVM, RF, MLP, XGBoost on ComplEx
2. Maps each to protein biomarkers
3. Identifies proteins appearing in top-K across multiple models
4. Creates consensus ranking
5. Outputs results for downstream analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Set

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


def load_embeddings_and_mapping(version: str = 'v2.11', embedding_type: str = 'ComplEx'):
    """Load entity embeddings and entity mapping."""
    embeddings_path = Path(f"models/executions/GSE54514_enriched_ontology_degfilter{version}") / embedding_type / f"outputmodel_{embedding_type}_entity_embeddings.npy"
    entity_mapping_path = Path(f"models/executions/GSE54514_enriched_ontology_degfilter{version}") / embedding_type / f"outputmodel_{embedding_type}_entity_mapping.csv"
    
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")
    if not entity_mapping_path.exists():
        raise FileNotFoundError(f"Entity mapping not found: {entity_mapping_path}")
    
    embeddings = np.load(embeddings_path)
    entity_df = pd.read_csv(entity_mapping_path)
    
    logger.info(f"Loaded embeddings: {embeddings.shape}")
    logger.info(f"Loaded entity mapping: {len(entity_df)} entities")
    
    return embeddings, entity_df


def map_features_to_proteins(feature_importance_csv: str, version: str = 'v2.11', 
                             embedding_type: str = 'ComplEx', top_k: int = 3) -> pd.DataFrame:
    """Map Feature_i to top protein entities in each embedding dimension."""
    
    logger.info(f"Loading features from: {feature_importance_csv}")
    features_df = pd.read_csv(feature_importance_csv)
    
    embeddings, entity_df = load_embeddings_and_mapping(version, embedding_type)
    
    results = []
    
    for _, row in features_df.iterrows():
        feature_name = row['feature']
        importance = row['importance']
        
        # Extract dimension from feature name (e.g., "Feature_73" -> 73)
        try:
            dim_idx = int(feature_name.split('_')[1])
        except (IndexError, ValueError):
            logger.warning(f"Could not parse feature: {feature_name}")
            continue
        
        if dim_idx >= embeddings.shape[1]:
            logger.warning(f"Feature dimension {dim_idx} out of range (max {embeddings.shape[1]-1})")
            continue
        
        # Get absolute values for this dimension
        dim_values = np.abs(embeddings[:, dim_idx])
        
        # Get top K entities
        top_indices = np.argsort(dim_values)[-top_k:][::-1]
        top_entities = entity_df.iloc[top_indices]['label'].tolist()
        top_scores = dim_values[top_indices].tolist()
        
        # Filter to only proteins
        proteins = []
        protein_scores = []
        for entity, score in zip(top_entities, top_scores):
            if entity.startswith('Protein_'):
                proteins.append(entity)
                protein_scores.append(score)
        
        # Only keep if has at least one protein
        if proteins:
            results.append({
                'feature': feature_name,
                'dimension': dim_idx,
                'importance': importance,
                'proteins': ';'.join(proteins),
                'top_entity': proteins[0] if proteins else None,
                'top_entity_score': protein_scores[0] if protein_scores else 0,
            })
    
    return pd.DataFrame(results)


def extract_consensus_proteins(model_dirs: Dict[str, str], top_k_per_model: int = 20) -> Dict[str, List]:
    """Extract consensus proteins across all models."""
    
    all_proteins = {}  # protein -> list of (model, score, feature)
    model_results = {}
    
    for model_name, feature_csv in model_dirs.items():
        logger.info(f"\nProcessing {model_name}...")
        
        if not Path(feature_csv).exists():
            logger.warning(f"Feature file not found: {feature_csv}")
            continue
        
        # Map features to proteins
        mapped_df = map_features_to_proteins(feature_csv)
        
        if len(mapped_df) == 0:
            logger.warning(f"No protein mappings for {model_name}")
            continue
        
        # Get top K proteins by score
        top_proteins = mapped_df.nlargest(top_k_per_model, 'top_entity_score')
        
        model_results[model_name] = {
            'total_features': len(mapped_df),
            'top_proteins_df': top_proteins,
            'unique_proteins': top_proteins['top_entity'].unique().tolist()
        }
        
        logger.info(f"  {model_name}: {len(mapped_df)} features, {len(top_proteins)} top features mapped")
        logger.info(f"  Unique proteins in top {top_k_per_model}: {len(model_results[model_name]['unique_proteins'])}")
        
        # Track proteins
        for _, row in top_proteins.iterrows():
            protein = row['top_entity']
            score = row['top_entity_score']
            feature = row['feature']
            
            if protein not in all_proteins:
                all_proteins[protein] = []
            all_proteins[protein].append({
                'model': model_name,
                'score': score,
                'feature': feature
            })
    
    logger.info(f"\n" + "="*80)
    logger.info(f"Total unique proteins across models: {len(all_proteins)}")
    
    # Calculate consensus metrics
    consensus_data = []
    for protein, appearances in all_proteins.items():
        num_models = len(set(a['model'] for a in appearances))
        avg_score = np.mean([a['score'] for a in appearances])
        max_score = max([a['score'] for a in appearances])
        
        consensus_data.append({
            'protein': protein,
            'protein_name': protein.replace('Protein_', ''),
            'num_models': num_models,
            'avg_score': avg_score,
            'max_score': max_score,
            'models': ';'.join(sorted(set(a['model'] for a in appearances))),
            'features': ';'.join([a['feature'] for a in appearances])
        })
    
    consensus_df = pd.DataFrame(consensus_data)
    consensus_df = consensus_df.sort_values('avg_score', ascending=False)
    
    return consensus_df, model_results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract consensus biomarker proteins across ComplEx+4ML models'
    )
    parser.add_argument('--results-dir', type=str,
                       default='results/interpretability/v2.11_minmax/run_20260114_032006',
                       help='Path to results directory with all models')
    parser.add_argument('--top-k-per-model', type=int, default=20,
                       help='Top K features per model to consider')
    parser.add_argument('--output-dir', type=str,
                       default='results/interpretability/complex_svm_mapped',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return
    
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Define model paths
    model_dirs = {
        'SVM': str(results_dir / 'svm_Complex_protein_embeddings' / 'feature_importance.csv'),
        'RandomForest': str(results_dir / 'random_forest_Complex_protein_embeddings' / 'feature_importance.csv'),
        'MLP': str(results_dir / 'sklearn_mlp_Complex_protein_embeddings' / 'feature_importance.csv'),
        'XGBoost': str(results_dir / 'xgboost_Complex_protein_embeddings' / 'feature_importance.csv'),
    }
    
    logger.info("\n" + "="*80)
    logger.info("EXTRACTING CONSENSUS PROTEINS")
    logger.info("="*80)
    
    # Extract consensus
    consensus_df, model_results = extract_consensus_proteins(model_dirs, args.top_k_per_model)
    
    # Save consensus
    consensus_csv = output_dir / 'complex_consensus_proteins.csv'
    consensus_df.to_csv(consensus_csv, index=False)
    logger.info(f"\nSaved consensus proteins: {consensus_csv}")
    
    # Save per-model details
    for model_name, results in model_results.items():
        model_csv = output_dir / f'complex_{model_name.lower()}_top_proteins.csv'
        results['top_proteins_df'][['feature', 'top_entity', 'top_entity_score', 'importance']].to_csv(model_csv, index=False)
        logger.info(f"Saved {model_name} proteins: {model_csv}")
    
    # Summary statistics
    logger.info("\n" + "="*80)
    logger.info("CONSENSUS SUMMARY")
    logger.info("="*80)
    logger.info(f"\nTotal unique proteins: {len(consensus_df)}")
    
    # Group by number of models
    model_counts = consensus_df['num_models'].value_counts().sort_index(ascending=False)
    logger.info(f"\nProteins appearing in N models:")
    for n_models, count in model_counts.items():
        logger.info(f"  {n_models} models: {count} proteins")
    
    logger.info(f"\nTop 15 consensus biomarkers (by avg score):")
    for idx, (_, row) in enumerate(consensus_df.head(15).iterrows(), 1):
        logger.info(f"  {idx:2}. {row['protein_name']:20} | Score: {row['avg_score']:.4f} | Models: {row['num_models']}/4 ({row['models']})")
    
    # Proteins in all 4 models
    all_models_df = consensus_df[consensus_df['num_models'] == 4]
    logger.info(f"\nProteins appearing in ALL 4 models: {len(all_models_df)}")
    if len(all_models_df) > 0:
        logger.info("  " + ", ".join(all_models_df['protein_name'].head(10).tolist()))
        if len(all_models_df) > 10:
            logger.info(f"  ... and {len(all_models_df) - 10} more")
    
    logger.info(f"\n✓ Consensus analysis complete!")


if __name__ == '__main__':
    main()
