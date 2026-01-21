"""
Consolidate interpretability results across models and datasets.

This script aggregates feature importance rankings from all models/datasets
to identify consensus biomarkers, validate separation, and produce summary reports.

Usage:
    python -m src.ml.consolidate_interpretability --run-dir results/interpretability/v2.11_none/run_20260114_XXXXXX
    python -m src.ml.consolidate_interpretability --run-dir results/interpretability/v2.11_none/run_* --all-runs
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from collections import Counter, defaultdict

# Setup
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# ==================================================================================== #
# CONSOLIDATION LOGIC
# ==================================================================================== #

class InterpretabilityConsolidator:
    """
    Consolidate interpretability results across multiple model-dataset combinations.
    """
    
    def __init__(self, run_dirs, output_dir=None, importance_norm: str = 'minmax', feature_map: dict | None = None):
        """
        Parameters
        ----------
        run_dirs : list or str
            Path(s) to interpretability run directories
        output_dir : str, optional
            Where to save consolidated outputs (default: run_dir/../consolidated/)
        importance_norm : str
            How to normalize per-model importances before aggregation: 'none', 'minmax', or 'zscore'
        feature_map : dict, optional
            Optional mapping from feature name (e.g., Feature_73) to biological ID/label
        """
        if isinstance(run_dirs, str):
            run_dirs = [run_dirs]
        
        self.run_dirs = [Path(d) for d in run_dirs if Path(d).exists()]
        if not self.run_dirs:
            raise ValueError("No valid run directories provided")
        
        self.output_dir = Path(output_dir) if output_dir else self.run_dirs[0].parent / "consolidated"
        self.output_dir.mkdir(exist_ok=True)
        
        self.results = []
        self.feature_importance_all = []
        self.importance_norm = importance_norm
        self.feature_map = feature_map or {}
        
    def load_results(self):
        """Load all analysis_summary.csv files from run directories."""
        print("Loading results from run directories...")
        
        for run_dir in self.run_dirs:
            summary_path = run_dir / "analysis_summary.csv"
            if summary_path.exists():
                df = pd.read_csv(summary_path)
                self.results.append(df)
                print(f"  ✓ Loaded: {summary_path.parent.name}")
        
        if self.results:
            self.all_results_df = pd.concat(self.results, ignore_index=True)
            print(f"Total analyses: {len(self.all_results_df)}")
        else:
            print("⚠ No analysis_summary.csv files found")
    
    def load_feature_importance(self):
        """Load feature_importance.csv from each model-dataset combo."""
        print("\nLoading feature importance rankings...")
        
        feature_data = []
        for run_dir in self.run_dirs:
            for model_dir in run_dir.glob("*_*"):
                if model_dir.is_dir():
                    importance_path = model_dir / "feature_importance.csv"
                    if importance_path.exists():
                        df = pd.read_csv(importance_path)
                        # Parse model_type and dataset from directory name
                        parts = model_dir.name.rsplit('_', 1)
                        if len(parts) == 2:
                            # Handle compound names like sklearn_mlp_Complex_protein_embeddings
                            # The file naming format: {model_type}_{dataset}
                            model_type_dataset = model_dir.name
                            # Try to identify model type
                            for prefix in ['sklearn_mlp_', 'pytorch_mlp_', 'random_forest_', 'xgboost_', 'svm_']:
                                if model_type_dataset.startswith(prefix):
                                    model_type = prefix.rstrip('_')
                                    dataset = model_type_dataset[len(prefix):]
                                    break
                            else:
                                # Fallback
                                model_type = parts[0]
                                dataset = parts[1]
                            
                            df['model_type'] = model_type
                            df['dataset'] = dataset
                            df['data_repr'] = dataset  # Track representation type for disambiguation
                            df['rank'] = range(1, len(df) + 1)
                            # Normalize per model if requested
                            imp = df['importance'].astype(float)
                            if self.importance_norm == 'minmax':
                                max_abs = np.abs(imp).max()
                                if max_abs > 0:
                                    norm_imp = np.abs(imp) / max_abs
                                else:
                                    norm_imp = np.zeros_like(imp)
                            elif self.importance_norm == 'zscore':
                                mu = imp.mean()
                                sigma = imp.std() if imp.std() > 0 else 1.0
                                norm_imp = (imp - mu) / sigma
                                norm_imp = np.abs(norm_imp)
                            else:  # none
                                norm_imp = np.abs(imp)
                            df['normalized_importance'] = norm_imp
                            feature_data.append(df)
        
        if feature_data:
            self.feature_importance_df = pd.concat(feature_data, ignore_index=True)
            print(f"✓ Loaded {len(self.feature_importance_df)} feature rankings")
        else:
            print("⚠ No feature_importance.csv files found")
    
    def compute_consensus_ranking(self, top_n=50, min_accuracy=0.5):
        """
        Compute weighted consensus ranking of features.
        
        Parameters
        ----------
        top_n : int
            Number of top features to include from each model
        min_accuracy : float
            Minimum test accuracy to include model in consensus
        """
        print(f"\nComputing consensus ranking (min_accuracy={min_accuracy})...")
        
        # Filter by accuracy
        valid_models = self.all_results_df[
            (self.all_results_df['test_accuracy'] >= min_accuracy) &
            (self.all_results_df['status'] == 'success')
        ]
        print(f"  Valid models: {len(valid_models)} / {len(self.all_results_df)}")
        
        # Build feature scores
        feature_scores = defaultdict(lambda: {'count': 0, 'total_weight': 0, 'total_importance': 0})
        
        for _, row in valid_models.iterrows():
            model_key = f"{row['model_type']}_{row['dataset']}"
            model_data = self.feature_importance_df[
                (self.feature_importance_df['model_type'] == row['model_type']) &
                (self.feature_importance_df['dataset'] == row['dataset'])
            ]
            
            if model_data.empty:
                continue
            
            # Weight by accuracy and inverse rank
            model_weight = row['test_accuracy']  # Higher accuracy = higher weight
            
            for idx, feat_row in model_data.head(top_n).iterrows():
                feature = feat_row['feature']
                rank = feat_row['rank']
                importance = feat_row.get('normalized_importance', feat_row['importance'])
                
                # Inverse rank weighting: rank 1 gets weight 1, rank N gets weight 1/N
                rank_weight = 1.0 / rank
                
                feature_scores[feature]['count'] += 1
                feature_scores[feature]['total_weight'] += model_weight * rank_weight
                feature_scores[feature]['total_importance'] += importance
        
        # Convert to dataframe
        consensus_list = []
        dataset_counter = defaultdict(lambda: defaultdict(int))  # Track features per dataset
        
        for feature, scores in sorted(feature_scores.items(), key=lambda x: x[1]['total_weight'], reverse=True):
            mapped = self.feature_map.get(feature, feature)
            # Find which datasets this feature appeared in
            feat_datasets = self.feature_importance_df[
                self.feature_importance_df['feature'] == feature
            ]['dataset'].unique().tolist()
            
            consensus_list.append({
                'feature': feature,
                'mapped_feature': mapped,
                'datasets': '|'.join(feat_datasets),  # Track which datasets contributed
                'n_models': scores['count'],
                'consensus_score': scores['total_weight'],
                'mean_importance': scores['total_importance'] / scores['count'],
                'pct_models': 100 * scores['count'] / len(valid_models)
            })
        
        self.consensus_df = pd.DataFrame(consensus_list)
        
        # Save
        consensus_path = self.output_dir / "consensus_biomarkers.csv"
        self.consensus_df.to_csv(consensus_path, index=False)
        print(f"✓ Saved consensus ranking: {consensus_path}")
        
        # Print top 20
        print("\nTop 20 Consensus Biomarkers:")
        print(self.consensus_df.head(20).to_string(index=False))
    
    def create_feature_heatmap(self, top_n=30):
        """
        Create heatmap of top features × models showing importance scores.
        """
        print(f"\nCreating feature × model heatmap...")
        
        # Get top features
        top_features = self.consensus_df.head(top_n)['feature'].tolist()
        
        # Build matrix: features × model-dataset
        pivot_data = []
        for _, row in self.all_results_df.iterrows():
            if row['status'] != 'success':
                continue
            
            model_key = f"{row['model_type']}\n{row['dataset']}"
            model_importance = self.feature_importance_df[
                (self.feature_importance_df['model_type'] == row['model_type']) &
                (self.feature_importance_df['dataset'] == row['dataset'])
            ].set_index('feature')['normalized_importance'].to_dict()
            
            for feat in top_features:
                importance = model_importance.get(feat, 0)
                pivot_data.append({
                    'feature': feat,
                    'model': model_key,
                    'importance': importance
                })
        
        pivot_df = pd.DataFrame(pivot_data).pivot(index='feature', columns='model', values='importance')
        pivot_df = pivot_df.fillna(0)
        
        # Plot
        fig, ax = plt.subplots(figsize=(16, 12))
        sns.heatmap(pivot_df, cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Feature Importance'})
        ax.set_title(f'Top {top_n} Consensus Biomarkers across Models', fontsize=14, fontweight='bold')
        ax.set_xlabel('Model × Dataset', fontsize=12, fontweight='bold')
        ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        heatmap_path = self.output_dir / "feature_heatmap.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved heatmap: {heatmap_path}")
    
    def create_summary_report(self, top_k=20):
        """
        Create a human-readable summary report.
        """
        print(f"\nGenerating summary report...")
        
        report_path = self.output_dir / "CONSOLIDATION_REPORT.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("INTERPRETABILITY CONSOLIDATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Overall stats
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total analyses: {len(self.all_results_df)}\n")
            f.write(f"Successful: {(self.all_results_df['status'] == 'success').sum()}\n")
            f.write(f"Failed: {(self.all_results_df['status'] == 'failed').sum()}\n")
            mean_acc = self.all_results_df[self.all_results_df['status'] == 'success']['test_accuracy'].mean()
            f.write(f"Mean test accuracy (successful): {mean_acc:.3f}\n")
            f.write(f"Unique features across all analyses: {self.feature_importance_df['feature'].nunique()}\n\n")
            
            # Per-dataset stats
            f.write("PER-DATASET PERFORMANCE\n")
            f.write("-" * 80 + "\n")
            dataset_stats = self.all_results_df.groupby('dataset').agg({
                'test_accuracy': ['mean', 'std', 'count'],
                'status': lambda x: (x == 'success').sum()
            }).round(3)
            f.write(str(dataset_stats) + "\n\n")
            
            # Per-model stats
            f.write("PER-MODEL PERFORMANCE\n")
            f.write("-" * 80 + "\n")
            model_stats = self.all_results_df.groupby('model_type').agg({
                'test_accuracy': ['mean', 'std', 'count'],
                'status': lambda x: (x == 'success').sum()
            }).round(3)
            f.write(str(model_stats) + "\n\n")
            
            # Top consensus biomarkers
            f.write(f"TOP {top_k} CONSENSUS BIOMARKERS\n")
            f.write("-" * 80 + "\n")
            for idx, row in self.consensus_df.head(top_k).iterrows():
                f.write(f"{idx+1:2d}. {row['feature']}\n")
                f.write(f"    Consensus Score: {row['consensus_score']:.4f}\n")
                f.write(f"    Appears in {row['n_models']} models ({row['pct_models']:.1f}% coverage)\n")
                f.write(f"    Mean Importance: {row['mean_importance']:.4f}\n\n")
            
            # Confidence tiers
            f.write("BIOMARKER CONFIDENCE TIERS\n")
            f.write("-" * 80 + "\n")
            n_valid_models = len(self.all_results_df[self.all_results_df['status'] == 'success'])
            f.write(f"Based on {n_valid_models} successful analyses:\n\n")
            
            tier_high = self.consensus_df[self.consensus_df['pct_models'] >= 75]
            f.write(f"HIGH CONFIDENCE (≥75% coverage): {len(tier_high)} features\n")
            for feat in tier_high['feature'].head(10).tolist():
                f.write(f"  • {feat}\n")
            
            tier_med = self.consensus_df[(self.consensus_df['pct_models'] >= 50) & (self.consensus_df['pct_models'] < 75)]
            f.write(f"\nMEDIUM CONFIDENCE (50-75% coverage): {len(tier_med)} features\n")
            for feat in tier_med['feature'].head(10).tolist():
                f.write(f"  • {feat}\n")
            
            tier_low = self.consensus_df[(self.consensus_df['pct_models'] >= 25) & (self.consensus_df['pct_models'] < 50)]
            f.write(f"\nLOW CONFIDENCE (25-50% coverage): {len(tier_low)} features\n")
            for feat in tier_low['feature'].head(10).tolist():
                f.write(f"  • {feat}\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        print(f"✓ Saved report: {report_path}")
    
    def create_biomarker_ranking_table(self, top_k=100):
        """
        Export top biomarkers with detailed metadata.
        """
        print(f"\nExporting top {top_k} biomarker details...")
        
        # Merge with per-model details
        top_features = self.consensus_df.head(top_k)['feature'].tolist()
        
        details = []
        for feat in top_features:
            feat_data = self.feature_importance_df[self.feature_importance_df['feature'] == feat]
            consensus_row = self.consensus_df[self.consensus_df['feature'] == feat].iloc[0]
            
            details.append({
                'rank': len(details) + 1,
                'feature': feat,
                'consensus_score': consensus_row['consensus_score'],
                'n_models': int(consensus_row['n_models']),
                'pct_models': consensus_row['pct_models'],
                'mean_importance': consensus_row['mean_importance'],
                'min_importance': feat_data['normalized_importance'].min(),
                'max_importance': feat_data['normalized_importance'].max(),
                'std_importance': feat_data['normalized_importance'].std()
            })
        
        details_df = pd.DataFrame(details)
        details_path = self.output_dir / "top_biomarkers_detailed.csv"
        details_df.to_csv(details_path, index=False)
        print(f"✓ Saved detailed biomarker table: {details_path}")
    
    def run_full_consolidation(self, top_n=50, min_accuracy=0.5):
        """Run all consolidation steps."""
        self.load_results()
        self.load_feature_importance()
        self.compute_consensus_ranking(top_n=top_n, min_accuracy=min_accuracy)
        self.create_feature_heatmap(top_n=30)
        self.create_summary_report(top_k=20)
        self.create_biomarker_ranking_table(top_k=top_n)
        
        print(f"\n{'='*80}")
        print(f"✓ Consolidation complete!")
        print(f"  Output directory: {self.output_dir}")
        print(f"{'='*80}\n")


# ==================================================================================== #
# CLI
# ==================================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description='Consolidate interpretability results across models and datasets'
    )
    parser.add_argument(
        '--run-dir',
        required=True,
        help='Path to interpretability run directory (or glob pattern for multiple)'
    )
    parser.add_argument(
        '--output-dir',
        help='Output directory for consolidated results (default: run_dir/../consolidated)'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=50,
        help='Top N features to include in consensus ranking'
    )
    parser.add_argument(
        '--min-accuracy',
        type=float,
        default=0.5,
        help='Minimum test accuracy to include model in consensus'
    )
    parser.add_argument(
        '--importance-norm',
        default='minmax',
        choices=['none', 'minmax', 'zscore'],
        help='Per-model importance normalization before aggregation'
    )
    parser.add_argument(
        '--feature-map',
        help='Optional CSV mapping features to biological labels (cols: feature,mapped_feature or feature,label)'
    )
    
    args = parser.parse_args()
    
    # Handle glob patterns
    run_dirs = sorted(glob(args.run_dir))
    if not run_dirs:
        print(f"✗ No run directories found matching: {args.run_dir}")
        sys.exit(1)
    
    print(f"Found {len(run_dirs)} run directory(ies)")
    
    feature_map = {}
    if args.feature_map:
        map_path = Path(args.feature_map)
        if map_path.exists():
            fmap_df = pd.read_csv(map_path)
            key_col = 'feature' if 'feature' in fmap_df.columns else fmap_df.columns[0]
            val_col = None
            for cand in ['mapped_feature', 'label', 'name', 'gene', 'protein']:
                if cand in fmap_df.columns:
                    val_col = cand
                    break
            if val_col is None and len(fmap_df.columns) > 1:
                val_col = fmap_df.columns[1]
            if val_col:
                feature_map = dict(zip(fmap_df[key_col], fmap_df[val_col]))
                print(f"Loaded feature map with {len(feature_map)} entries from {map_path}")
            else:
                print(f"⚠ Could not determine value column in feature map; skipping map load")
        else:
            print(f"⚠ Feature map file not found: {map_path}")
    
    # Run consolidation
    consolidator = InterpretabilityConsolidator(run_dirs, output_dir=args.output_dir,
                                                importance_norm=args.importance_norm,
                                                feature_map=feature_map)
    consolidator.run_full_consolidation(top_n=args.top_n, min_accuracy=args.min_accuracy)


if __name__ == '__main__':
    main()
