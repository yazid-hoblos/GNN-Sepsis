#!/usr/bin/env python
"""
Train downstream ML models on embeddings from full KG vs optimized KG.

Replicates the exact training approach from src/ml/train_all.py but on:
1. Full KG embeddings (old embeddings from models/executions/v2.11/)
2. Optimized KG embeddings (new embeddings from retrain_embeddings.py output)

Usage:
    python src/gnn/train_downstream_models.py \
        --old-emb-dir models/executions/v2.11/ComplEx/ \
        --new-emb-dir lean_ComplEx/ \
        --output-dir results/ComplEx_comparison/ \
        --model-types random_forest sklearn_mlp
"""

import argparse
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, roc_curve
)
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ml.model_trainer import MLModel


def load_embeddings(emb_path, map_path):
    """Load embeddings and entity mapping."""
    emb = np.load(str(emb_path))
    mapping = pd.read_csv(str(map_path))
    
    # Handle complex embeddings by taking absolute value
    if np.iscomplexobj(emb):
        emb = np.abs(emb)
    
    return emb, mapping


def load_sample_ids_and_labels(mapping_df, nodes_csv_path=None):
    """Extract sample IDs and disease labels from entity mapping.
    
    If nodes_csv_path provided, uses hasDiseaseStatus column (more reliable).
    Otherwise falls back to pattern matching on sample names.
    
    The nodes_csv_path should point to node_features.csv which is in long format:
    node_id, name_feature, value_feature
    """
    sample_rows = mapping_df[mapping_df['label'].str.startswith('Sample_', na=False)]
    sample_ids = sample_rows['label'].tolist()
    
    if not sample_ids:
        return [], np.array([], dtype=int)
    
    # Try to load from nodes CSV for accurate disease status
    labels = []
    if nodes_csv_path and Path(nodes_csv_path).exists():
        try:
            df_nodes = pd.read_csv(nodes_csv_path)
            
            # Filter for hasDiseaseStatus rows (long format)
            disease_rows = df_nodes[df_nodes['name_feature'] == 'hasDiseaseStatus']
            
            # Create mapping from sample name to disease status
            sample_to_status = {}
            for _, row in disease_rows.iterrows():
                node_id = str(row['node_id'])
                status = str(row['value_feature']).lower()
                # Convert to binary: healthy=0, anything else=1
                sample_to_status[node_id] = 0 if status == 'healthy' else 1
            
            # Map labels using the disease status
            for sample_id in sample_ids:
                if sample_id in sample_to_status:
                    labels.append(sample_to_status[sample_id])
                else:
                    # Fallback to pattern matching
                    labels.append(0 if 'control' in sample_id.lower() else 1)
            
            print(f"      Loaded disease status for {len([l for l in labels if l == 1])} diseased samples")
        except Exception as e:
            print(f"      Warning: Could not load from nodes CSV ({e}). Falling back to pattern matching.")
            # Fallback to pattern matching
            for sample_id in sample_ids:
                labels.append(0 if 'control' in sample_id.lower() else 1)
    else:
        # No CSV provided, use pattern matching
        for sample_id in sample_ids:
            labels.append(0 if 'control' in sample_id.lower() else 1)
    
    return sample_ids, np.array(labels, dtype=int)


def prepare_embedding_dataframes(old_emb, old_map, new_emb, new_map, nodes_csv_path=None):
    """
    Prepare DataFrames compatible with MLModel training.
    
    For old KG: use sample embeddings directly
    For new KG: use average protein embeddings (repeated for all samples)
    
    Parameters
    ----------
    nodes_csv_path : str, optional
        Path to nodes.csv file for accurate disease status extraction
    
    Returns:
        tuple: (df_old, df_new, y) where y contains disease labels from old KG
    """
    # Load sample information from old KG
    old_sample_ids, y = load_sample_ids_and_labels(old_map, nodes_csv_path)
    
    if len(old_sample_ids) == 0:
        raise ValueError("No samples found in old KG mapping")
    
    # Map sample IDs to embedding indices (old KG)
    old_sample_map = {label: idx for idx, label in enumerate(old_map['label'])}
    
    # Extract old sample embeddings
    old_sample_emb = []
    y_valid = []
    
    for sample_id, label in zip(old_sample_ids, y):
        if sample_id in old_sample_map:
            idx = old_sample_map[sample_id]
            old_sample_emb.append(old_emb[idx])
            y_valid.append(label)
    
    old_sample_emb = np.array(old_sample_emb)
    y_valid = np.array(y_valid)
    
    # For new KG: compute average protein embedding
    new_protein_indices = [
        idx for idx, label in enumerate(new_map['label']) 
        if label.startswith('Protein_')
    ]
    
    if len(new_protein_indices) > 0:
        avg_protein_emb = np.mean(new_emb[new_protein_indices], axis=0)
        # Repeat for all samples
        new_sample_emb = np.tile(avg_protein_emb, (len(y_valid), 1))
    else:
        raise ValueError("No proteins found in new KG")
    
    # Create DataFrames compatible with MLModel
    # MLModel expects a 'disease_status' column for labels
    n_samples = len(y_valid)
    
    df_old = pd.DataFrame(old_sample_emb)
    df_old.columns = [f'feature_{i}' for i in range(old_sample_emb.shape[1])]
    df_old['disease_status'] = y_valid
    
    df_new = pd.DataFrame(new_sample_emb)
    df_new.columns = [f'feature_{i}' for i in range(new_sample_emb.shape[1])]
    df_new['disease_status'] = y_valid
    
    return df_old, df_new, y_valid


def train_and_get_metrics(df, model_type, dataset_name, cache_dir, random_state=42, split_ratio=0.2):
    """
    Train model exactly like train_all() does and return metrics.
    """
    # Create and train model using MLModel - exactly like src/ml/utils.py does
    ml_model = MLModel(
        model_type=model_type,
        df=df,
        dataset_name=dataset_name,
        save_model=False,
        version='embedding_comparison',
        normalization='none',
        split_ratio=split_ratio,
        random_state=random_state
    )
    
    # Train and evaluate using the built-in method
    ml_model.train_evaluate()
    
    # Extract metrics
    metrics = {
        'accuracy': accuracy_score(ml_model.y_test, ml_model.y_pred),
        'precision': precision_score(ml_model.y_test, ml_model.y_pred, zero_division=0),
        'recall': recall_score(ml_model.y_test, ml_model.y_pred, zero_division=0),
        'f1': f1_score(ml_model.y_test, ml_model.y_pred, zero_division=0),
        'auc': roc_auc_score(ml_model.y_test, ml_model.y_proba) if len(np.unique(ml_model.y_test)) > 1 else 0.0,
    }
    
    # ROC curve
    if len(np.unique(ml_model.y_test)) > 1:
        fpr, tpr, _ = roc_curve(ml_model.y_test, ml_model.y_proba)
    else:
        fpr, tpr = np.array([0, 1]), np.array([0, 1])
    
    metrics['fpr'] = fpr
    metrics['tpr'] = tpr
    
    return metrics


def plot_comparison(results, output_dir):
    """Create comparison plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not results:
        print("  No successful model trainings. Skipping plot generation.")
        return
    
    model_types = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    # Bar plot comparing metrics
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        old_vals = [results[mt]['old'][metric] for mt in model_types]
        new_vals = [results[mt]['new'][metric] for mt in model_types]
        
        x = np.arange(len(model_types))
        width = 0.35
        
        ax.bar(x - width/2, old_vals, width, label='Full KG', alpha=0.8)
        ax.bar(x + width/2, new_vals, width, label='Optimized KG', alpha=0.8)
        
        ax.set_ylabel('Score')
        ax.set_title(f'{metric.upper()}')
        ax.set_xticks(x)
        ax.set_xticklabels(model_types, rotation=45)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    # Remove extra subplot
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ROC curves
    fig, axes = plt.subplots(1, len(model_types), figsize=(12, 4))
    if len(model_types) == 1:
        axes = [axes]
    
    for idx, model_type in enumerate(model_types):
        ax = axes[idx]
        
        # Old KG
        old_results = results[model_type]['old']
        ax.plot(old_results['fpr'], old_results['tpr'], 
               label=f"Full KG (AUC={old_results['auc']:.3f})", linewidth=2)
        
        # New KG
        new_results = results[model_type]['new']
        ax.plot(new_results['fpr'], new_results['tpr'], 
               label=f"Optimized KG (AUC={new_results['auc']:.3f})", linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{model_type}')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-emb-dir', required=True, 
                       help='Directory with old (full KG) embeddings')
    parser.add_argument('--new-emb-dir', required=True,
                       help='Directory with new (optimized KG) embeddings')
    parser.add_argument('--output-dir', default='results/embedding_comparison',
                       help='Output directory for results')
    parser.add_argument('--model-types', nargs='+', 
                       default=['random_forest', 'sklearn_mlp'],
                       choices=['svm', 'random_forest', 'xgboost', 'sklearn_mlp', 'pytorch_mlp'],
                       help='ML models to train')
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--split-ratio', type=float, default=0.2)
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("DOWNSTREAM PERFORMANCE COMPARISON")
    print("="*60)
    
    # Load embeddings
    print(f"\nLoading embeddings...")
    old_emb_files = list(Path(args.old_emb_dir).glob('outputmodel_*_entity_embeddings.npy'))
    if not old_emb_files:
        old_emb_files = list(Path(args.old_emb_dir).glob('*_entity_embeddings.npy'))
    
    new_emb_files = list(Path(args.new_emb_dir).glob('*_entity_embeddings.npy'))
    
    if not old_emb_files:
        raise FileNotFoundError(f"No embeddings found in {args.old_emb_dir}")
    if not new_emb_files:
        raise FileNotFoundError(f"No embeddings found in {args.new_emb_dir}")
    
    old_emb_path = old_emb_files[0]
    old_map_path = old_emb_path.parent / old_emb_path.name.replace('_embeddings.npy', '_mapping.csv')
    
    new_emb_path = new_emb_files[0]
    new_map_path = new_emb_path.parent / new_emb_path.name.replace('_embeddings.npy', '_mapping.csv')
    
    old_emb, old_map = load_embeddings(old_emb_path, old_map_path)
    new_emb, new_map = load_embeddings(new_emb_path, new_map_path)
    
    print(f"  Old KG embeddings: {old_emb.shape}")
    print(f"  New KG embeddings: {new_emb.shape}")
    print(f"  Old KG mapping: {len(old_map)} entities")
    print(f"  New KG mapping: {len(new_map)} entities")
    
    # Prepare DataFrames
    print(f"\nPreparing data for training...")
    
    # Find nodes.csv for disease status extraction
    nodes_csv_path = None
    old_emb_dir_path = Path(args.old_emb_dir)
    # Look for nodes.csv in the execution directory or parent
    for search_path in [old_emb_dir_path, old_emb_dir_path.parent]:
        potential_nodes = search_path / 'node_features.csv'
        if potential_nodes.exists():
            nodes_csv_path = str(potential_nodes)
            print(f"  Found node features: {nodes_csv_path}")
            break
    
    if not nodes_csv_path:
        print(f"  Warning: Could not find node_features.csv. Using pattern matching for disease status.")
    
    df_old, df_new, y = prepare_embedding_dataframes(old_emb, old_map, new_emb, new_map, nodes_csv_path)
    
    print(f"  Samples: {len(y)}")
    print(f"  Class distribution: {np.bincount(y)}")
    print(f"  Old KG features: {df_old.shape[1]-1}")
    print(f"  New KG features: {df_new.shape[1]-1}")
    
    # Set MLModel global variables
    MLModel.set_global_variable("CACHE_DIR", str(output_dir))
    MLModel.set_global_variable("DEFAULT_RANDOM_STATE", args.random_state)
    MLModel.set_global_variable("DEFAULT_SPLIT_RATIO", args.split_ratio)
    
    # Train and evaluate models
    results = {}
    for model_type in args.model_types:
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()}")
        print(f"{'='*60}")
        
        # Old embeddings
        print(f"\n  Full KG embeddings...")
        try:
            old_metrics = train_and_get_metrics(
                df_old, model_type, f'{model_type}_old_kg', 
                str(output_dir), args.random_state, args.split_ratio
            )
            print(f"    Accuracy: {old_metrics['accuracy']:.4f}")
            print(f"    Precision: {old_metrics['precision']:.4f}")
            print(f"    Recall: {old_metrics['recall']:.4f}")
            print(f"    F1: {old_metrics['f1']:.4f}")
            print(f"    AUC: {old_metrics['auc']:.4f}")
        except Exception as e:
            print(f"    Failed: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # New embeddings
        print(f"\n  Optimized KG embeddings...")
        try:
            new_metrics = train_and_get_metrics(
                df_new, model_type, f'{model_type}_new_kg', 
                str(output_dir), args.random_state, args.split_ratio
            )
            print(f"    Accuracy: {new_metrics['accuracy']:.4f}")
            print(f"    Precision: {new_metrics['precision']:.4f}")
            print(f"    Recall: {new_metrics['recall']:.4f}")
            print(f"    F1: {new_metrics['f1']:.4f}")
            print(f"    AUC: {new_metrics['auc']:.4f}")
        except Exception as e:
            print(f"    Failed: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Difference
        print(f"\n  Difference (Optimized - Full):")
        print(f"    Accuracy: {new_metrics['accuracy'] - old_metrics['accuracy']:+.4f}")
        print(f"    F1: {new_metrics['f1'] - old_metrics['f1']:+.4f}")
        print(f"    AUC: {new_metrics['auc'] - old_metrics['auc']:+.4f}")
        
        results[model_type] = {
            'old': old_metrics,
            'new': new_metrics
        }
    
    # Save results
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")
    
    results_summary = {}
    for model_type, metrics_dict in results.items():
        results_summary[model_type] = {
            'old': {k: float(v) if not isinstance(v, (list, np.ndarray)) else None 
                   for k, v in metrics_dict['old'].items()},
            'new': {k: float(v) if not isinstance(v, (list, np.ndarray)) else None 
                   for k, v in metrics_dict['new'].items()}
        }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"  Results saved to {output_dir}/results.json")
    
    # Create plots
    plot_comparison(results, output_dir)
    
    print(f"\nComparison complete!")


if __name__ == '__main__':
    main()
