"""
Map features per dataset BEFORE consolidation.

This script processes each model-dataset combo and maps features to biological labels
based on the dataset type:
- gene_expression: probe ID → gene symbol (from GPL6947)
- *_embeddings: embedding dim → top entity labels (from auto_feature_mapping)

Usage:
    python -m src.interpretability.map_features_per_dataset \
        --run-dir results/interpretability/v2.11_none/run_* \
        --gpl-file data/GPL6947-13512.txt
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_gpl_mapping(gpl_file: Path) -> dict:
    """Load GPL6947 probe ID to gene symbol mapping."""
    mapping = {}
    try:
        with open(gpl_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    probe_id = parts[0]
                    gene_symbol = parts[1] if parts[1] else f"probe_{probe_id}"
                    mapping[probe_id] = gene_symbol
    except Exception as e:
        print(f"Warning: Could not load GPL mapping from {gpl_file}: {e}")
    return mapping


def map_gene_expression_features(features_df: pd.DataFrame, gpl_mapping: dict) -> pd.DataFrame:
    """Map gene expression probe IDs to gene symbols."""
    features_df['mapped_feature'] = features_df['feature'].apply(
        lambda feat: gpl_mapping.get(feat.replace('Feature_', ''), feat)
    )
    features_df['dataset'] = 'gene_expression'
    return features_df


def map_embedding_features(features_df: pd.DataFrame, embedding_type: str) -> pd.DataFrame:
    """Map embedding dimensions to labels (use dimension-to-entity mapping if available)."""
    # For embeddings, we can use auto_feature_mapping logic or just label as-is
    features_df['mapped_feature'] = features_df['feature']  # Keep as-is for now
    features_df['dataset'] = embedding_type.replace('_embeddings', '')
    return features_df


def process_model_dataset_combo(model_dir: Path, gpl_mapping: dict) -> pd.DataFrame | None:
    """Process a single model-dataset combo directory and map features."""
    feature_importance_path = model_dir / 'feature_importance.csv'
    if not feature_importance_path.exists():
        return None
    
    # Read feature importance
    features_df = pd.read_csv(feature_importance_path)
    if 'feature' not in features_df.columns:
        return None
    
    # Extract dataset type from directory name (e.g., random_forest_gene_expression)
    dir_name = model_dir.name
    model_name = None
    dataset_type = None
    
    # Parse: {model}_{dataset}
    parts = dir_name.rsplit('_', 2)  # Split from right to handle underscore in dataset names
    if len(parts) >= 2:
        # Handle multi-word datasets like "protein_embeddings"
        for possible_dataset in [
            'gene_expression',
            'Complex_protein_embeddings',
            'RGCN_protein_embeddings',
            'concatenated_protein_embeddings',
            'Complex_sample_embeddings',
            'RGCN_sample_embeddings',
            'concatenated_sample_embeddings'
        ]:
            if dir_name.endswith(possible_dataset):
                model_name = dir_name[:-len(possible_dataset)-1]
                dataset_type = possible_dataset
                break
    
    if not dataset_type:
        print(f"Warning: Could not determine dataset type for {dir_name}")
        dataset_type = 'unknown'
    
    # Map based on dataset type
    if 'gene_expression' in dataset_type:
        features_df = map_gene_expression_features(features_df, gpl_mapping)
    elif 'embeddings' in dataset_type:
        features_df = map_embedding_features(features_df, dataset_type)
    else:
        features_df['mapped_feature'] = features_df['feature']
        features_df['dataset'] = dataset_type
    
    features_df['model'] = model_name or 'unknown'
    features_df['model_dataset'] = dir_name
    
    return features_df


def main():
    parser = argparse.ArgumentParser(
        description='Map features per dataset before consolidation'
    )
    parser.add_argument(
        '--run-dir',
        required=True,
        help='Glob pattern for run directories (e.g., results/interpretability/v2.11_none/run_*)'
    )
    parser.add_argument(
        '--gpl-file',
        default=str(project_root / 'data' / 'GPL6947-13512.txt'),
        help='Path to GPL6947 annotation file for probe mapping'
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Output directory for mapped files (default: same as run-dir)'
    )
    args = parser.parse_args()
    
    # Load GPL mapping
    gpl_mapping = load_gpl_mapping(Path(args.gpl_file))
    print(f"Loaded {len(gpl_mapping)} probe mappings from {args.gpl_file}")
    
    # Find run directories
    run_dirs = sorted(glob(args.run_dir))
    if not run_dirs:
        print(f"No run directories found matching {args.run_dir}")
        return
    
    print(f"Found {len(run_dirs)} run directory(ies)")
    
    all_features = []
    for run_dir in run_dirs:
        run_path = Path(run_dir)
        print(f"\nProcessing {run_path.name}...")
        
        # Find all model-dataset subdirectories
        for model_dir in sorted(run_path.glob('*_*')):
            if not model_dir.is_dir():
                continue
            
            mapped_df = process_model_dataset_combo(model_dir, gpl_mapping)
            if mapped_df is not None:
                all_features.append(mapped_df)
                print(f"  ✓ {model_dir.name}: {len(mapped_df)} features")
            else:
                print(f"  ✗ {model_dir.name}: no feature_importance.csv")
    
    if not all_features:
        print("No features found to map!")
        return
    
    # Combine and save
    combined_df = pd.concat(all_features, ignore_index=True)
    
    out_dir = Path(args.output_dir) if args.output_dir else run_dirs[0] if len(run_dirs) == 1 else project_root / 'results' / 'feature_mapping'
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = out_dir / 'feature_mapping_per_dataset.csv'
    combined_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Saved {len(combined_df)} mapped features to {output_path}")
    print(f"\nSample mappings:")
    print(combined_df[['model_dataset', 'feature', 'mapped_feature', 'dataset']].head(10))


if __name__ == '__main__':
    main()
