#!/usr/bin/env python3
"""
Map ComplEx+SVM features to biological entities and prepare for downstream analysis.

This script:
1. Loads ComplEx+SVM feature importance
2. Maps Feature_i to top entities using embeddings
3. Creates interpretable biomarker tables
4. Outputs results for downstream biological analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
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


def map_features_to_entities(feature_importance_csv: str, version: str = 'v2.11', 
                             embedding_type: str = 'ComplEx', top_k: int = 3) -> pd.DataFrame:
    """Map Feature_i to top entities in each embedding dimension."""
    
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
        
        mapped_feature = "; ".join(top_entities)
        
        results.append({
            'feature': feature_name,
            'dimension': dim_idx,
            'importance': importance,
            'mapped_feature': mapped_feature,
            'top_entities': top_entities,
            'entity_scores': top_scores,
        })
    
    return pd.DataFrame(results)


def filter_to_proteins(mapped_df: pd.DataFrame) -> pd.DataFrame:
    """Filter biomarkers to only Protein entities."""
    filtered_rows = []
    
    for _, row in mapped_df.iterrows():
        # Filter top_entities and entity_scores to only proteins
        proteins = []
        protein_scores = []
        
        for entity, score in zip(row['top_entities'], row['entity_scores']):
            if entity.startswith('Protein_'):
                proteins.append(entity)
                protein_scores.append(score)
        
        # Only keep features that have at least one protein
        if proteins:
            row['top_entities'] = proteins
            row['entity_scores'] = protein_scores
            mapped_feature = "; ".join(proteins)
            row['mapped_feature'] = mapped_feature
            filtered_rows.append(row)
    
    if not filtered_rows:
        logger.warning("No proteins found in mapped features!")
        return pd.DataFrame()
    
    return pd.DataFrame(filtered_rows)


def categorize_entities(mapped_df: pd.DataFrame) -> Dict[str, List[str]]:
    """Categorize entities by type (e.g., Protein, Pathway, GeneOntology)."""
    categorized = {}
    
    for _, row in mapped_df.iterrows():
        entities = row['top_entities']
        for entity in entities:
            # Extract entity type from prefix
            if '_' in entity:
                entity_type = entity.split('_')[0]
            else:
                entity_type = 'Unknown'
            
            if entity_type not in categorized:
                categorized[entity_type] = []
            if entity not in categorized[entity_type]:
                categorized[entity_type].append(entity)
    
    return categorized


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Map ComplEx+SVM features to biological entities'
    )
    parser.add_argument('--feature-csv', type=str,
                       default='results/interpretability/v2.11_minmax/run_20260114_032006/svm_Complex_protein_embeddings/feature_importance.csv',
                       help='Path to feature importance CSV')
    parser.add_argument('--version', type=str, default='v2.11',
                       help='Model version')
    parser.add_argument('--embedding-type', type=str, default='ComplEx',
                       help='Embedding type')
    parser.add_argument('--top-k', type=int, default=3,
                       help='Number of top entities per feature')
    parser.add_argument('--output-dir', type=str, default='results/interpretability/complex_svm_mapped',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    
    # Map features to entities
    logger.info("Mapping features to entities...")
    mapped_df = map_features_to_entities(
        args.feature_csv,
        version=args.version,
        embedding_type=args.embedding_type,
        top_k=args.top_k
    )
    
    # Filter to proteins only
    logger.info("Filtering to protein entities only...")
    mapped_df = filter_to_proteins(mapped_df)
    logger.info(f"After protein filtering: {len(mapped_df)} features with protein mappings")
    
    # Save mapped features
    mapped_csv = output_dir / 'complex_svm_features_mapped.csv'
    mapped_df.to_csv(mapped_csv, index=False)
    logger.info(f"Saved mapped features: {mapped_csv}")
    
    # Save only feature + mapped_feature for downstream use
    feature_map_csv = output_dir / 'feature_mapping.csv'
    feature_map = mapped_df[['feature', 'mapped_feature']]
    feature_map.to_csv(feature_map_csv, index=False)
    logger.info(f"Saved feature mapping: {feature_map_csv}")
    
    # Categorize entities
    logger.info("Categorizing entities...")
    categorized = categorize_entities(mapped_df)
    
    logger.info(f"\nEntity categories found:")
    for entity_type, entities in sorted(categorized.items()):
        logger.info(f"  {entity_type}: {len(entities)} unique entities")
    
    # Save entity categories
    categories_csv = output_dir / 'entity_categories.csv'
    category_rows = []
    for entity_type, entities in categorized.items():
        for entity in entities:
            category_rows.append({'entity_type': entity_type, 'entity': entity})
    pd.DataFrame(category_rows).to_csv(categories_csv, index=False)
    logger.info(f"Saved entity categories: {categories_csv}")
    
    # Create biomarker summary (top entities by importance)
    logger.info("Creating biomarker summary...")
    biomarker_rows = []
    for _, row in mapped_df.iterrows():
        for entity, score in zip(row['top_entities'], row['entity_scores']):
            biomarker_rows.append({
                'feature': row['feature'],
                'feature_importance': row['importance'],
                'entity': entity,
                'entity_score': score,
                'entity_type': entity.split('_')[0] if '_' in entity else 'Unknown',
            })
    
    biomarker_df = pd.DataFrame(biomarker_rows)
    # Sort by entity score (descending)
    biomarker_df = biomarker_df.sort_values('entity_score', ascending=False)
    
    biomarker_csv = output_dir / 'complex_svm_biomarkers.csv'
    biomarker_df.to_csv(biomarker_csv, index=False)
    logger.info(f"Saved biomarkers: {biomarker_csv}")
    
    # Summary statistics
    logger.info("\n" + "="*60)
    logger.info("SUMMARY: ComplEx+SVM Feature Mapping (PROTEINS ONLY)")
    logger.info("="*60)
    logger.info(f"Total features with protein mappings: {len(mapped_df)}")
    logger.info(f"Unique proteins: {len(biomarker_df['entity'].unique())}")
    logger.info(f"Unique entity types: {len(categorized)}")
    
    logger.info("\nTop 10 biomarkers by entity score:")
    for _, row in biomarker_df.head(10).iterrows():
        logger.info(f"  {row['entity']:30} (score: {row['entity_score']:.4f}, type: {row['entity_type']})")
    
    logger.info(f"\nAll results saved to: {output_dir}")
    logger.info("Files generated:")
    logger.info(f"  - complex_svm_features_mapped.csv (full details)")
    logger.info(f"  - feature_mapping.csv (for downstream use)")
    logger.info(f"  - complex_svm_biomarkers.csv (biomarkers sorted by score)")
    logger.info(f"  - entity_categories.csv (entity classification)")


if __name__ == '__main__':
    main()
