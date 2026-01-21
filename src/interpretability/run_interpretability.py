"""
Run interpretability analysis on trained models
------------------------------------------------

This script runs SHAP analysis and feature importance on all trained models
and generates comprehensive reports and visualizations.

Usage:
    # Analyze single model
    python -m src.ml.run_interpretability --dataset Complex_protein_embeddings --model random_forest

    # Analyze all models for a dataset
    python -m src.ml.run_interpretability --dataset Complex_protein_embeddings --all-models

    # Analyze all datasets for a model
    python -m src.ml.run_interpretability --model random_forest --all-datasets

    # Full analysis (all models + all datasets)
    python -m src.ml.run_interpretability --full-analysis

    # Analyze multiple datasets/models at once
    python -m src.ml.run_interpretability --datasets Complex_protein_embeddings concatenated_sample_embeddings \
                                          --models random_forest xgboost --version v2.11 --normalization none
"""

import sys
import argparse
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ml.interpretability import ModelInterpreter
from src.ml.load_matrix import load_df
from src.ml.model_trainer import MLModel


def load_model_with_data(model_path):
    """Load a model and attach test data by recreating the train/test split."""
    # Load model
    try:
        ml_model = joblib.load(model_path)
    except Exception as e:
        print(f"  ⚠ joblib.load failed: {type(e).__name__}: {str(e)[:80]}")
        print(f"  → Rebuilding model from scratch for interpretability.")
        stem = model_path.stem.replace('_gridsearch_model', '')
        # Determine model_type and dataset
        if stem.startswith('sklearn_mlp_'):
            model_type = 'sklearn_mlp'
            dataset = stem.replace('sklearn_mlp_', '')
        elif stem.startswith('pytorch_mlp_'):
            model_type = 'pytorch_mlp'
            dataset = stem.replace('pytorch_mlp_', '')
        elif stem.startswith('random_forest_'):
            model_type = 'random_forest'
            dataset = stem.replace('random_forest_', '')
        elif stem.startswith('xgboost_'):
            model_type = 'xgboost'
            dataset = stem.replace('xgboost_', '')
        elif stem.startswith('svm_'):
            model_type = 'svm'
            dataset = stem.replace('svm_', '')
        else:
            parts = stem.split('_', 1)
            model_type = parts[0]
            dataset = parts[1] if len(parts) > 1 else stem

        # Parse version and normalization from parent directory
        parent = model_path.parent.name
        version = parent.split('_')[0] if '_' in parent else 'unknown'
        normalization = parent.split('_')[1] if '_' in parent and len(parent.split('_')) > 1 else 'none'

        df = load_df(dataset)
        ml_model = MLModel(model_type=model_type, df=df, dataset_name=dataset, save_model=False, version=version, normalization=normalization)
        print(f"  Training {model_type} on {dataset}...")
        ml_model.train_evaluate()
        print(f"  ✓ Model trained successfully")
    
    # Load data and recreate test split
    df = load_df(ml_model.dataset_name)
    X = df.drop('disease_status', axis=1).values
    y = df['disease_status'].values
    
    # Recreate the same split used during training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=ml_model.split_ratio,
        random_state=ml_model.random_state,
        stratify=y
    )
    
    # Set test data on the model
    ml_model.X_test = X_test
    ml_model.y_test = y_test
    ml_model.y_pred = ml_model.best_model.predict(X_test)
    
    return ml_model


def analyze_model(model_path, output_dir, sample_indices=[0, 1, 5, 10]):
    """Run complete interpretability analysis on a single model"""
    
    print(f"\nAnalyzing: {model_path.name}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    try:
        ml_model = load_model_with_data(model_path)
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        return None
    
    # Initialize interpreter
    try:
        interpreter = ModelInterpreter(ml_model)
    except Exception as e:
        print(f"  ✗ Failed to initialize interpreter: {e}")
        return None
    
    # Compute test accuracy if not already available
    test_accuracy = None
    if hasattr(ml_model, 'test_score'):
        test_accuracy = ml_model.test_score
    elif hasattr(ml_model, 'best_model') and ml_model.X_test is not None:
        from sklearn.metrics import accuracy_score
        y_pred = ml_model.best_model.predict(ml_model.X_test)
        test_accuracy = accuracy_score(ml_model.y_test, y_pred)
    
    results = {
        'model_path': str(model_path),
        'model_type': ml_model.model_type,
        'dataset': ml_model.dataset_name,
        'version': ml_model.version if hasattr(ml_model, 'version') else 'unknown',
        'test_accuracy': test_accuracy,
        'status': 'success'
    }
    
    try:
        # 1. SHAP Analysis
        print("  Computing SHAP values...")
        shap_values = interpreter.compute_shap_values()
        
        if shap_values is not None:
            results['mean_abs_shap'] = abs(shap_values).mean()
            
            print("  Generating SHAP plots...")
            # Summary barplot
            fig = interpreter.plot_shap_summary(show=False)
            plt.savefig(output_dir / "shap_summary.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            # Beeswarm plot (distribution)
            fig = interpreter.plot_shap_beeswarm(max_features=15, show=False)
            plt.savefig(output_dir / "shap_beeswarm.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            # Dependence plots (top 3 features)
            fig = interpreter.plot_shap_dependence(top_n=3, show=False)
            plt.savefig(output_dir / "shap_dependence_top3.png", dpi=150, bbox_inches='tight')
            plt.close()
        else:
            print("  ⚠ SHAP values unavailable (likely memory constraints); skipping SHAP plots.")
            results['mean_abs_shap'] = None
        
        # 2. Feature Statistics by Class
        print("  Generating feature distribution plots...")
        try:
            fig = interpreter.plot_feature_stats_by_class(top_n=10, show=False)
            plt.savefig(output_dir / "feature_stats_by_class.png", dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"  ⚠ Feature stats plot failed: {str(e)[:80]}...")
        
        # 3. Feature Importance (consolidated)
        print("  Computing feature importance...")
        try:
            # Try SHAP first, will fall back to permutation if SHAP is unavailable
            importance_df = interpreter.get_feature_importance(top_k=50, method='shap')
            importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
            results['top_feature'] = importance_df.iloc[0]['feature']
            results['top_importance'] = importance_df.iloc[0]['importance']
            
            fig = interpreter.plot_feature_importance(top_k=20, method='shap', show=False)
            plt.savefig(output_dir / "feature_importance.png", dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"  ⚠ SHAP feature importance failed ({str(e)[:60]}...); trying permutation...")
            try:
                importance_df = interpreter.get_feature_importance(top_k=50, method='permutation')
                importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
                results['top_feature'] = importance_df.iloc[0]['feature']
                results['top_importance'] = importance_df.iloc[0]['importance']
                
                fig = interpreter.plot_feature_importance(top_k=20, method='permutation', show=False)
                plt.savefig(output_dir / "feature_importance.png", dpi=150, bbox_inches='tight')
                plt.close()
            except Exception as e2:
                print(f"  ⚠ Feature importance (both methods) failed: {str(e2)[:60]}...")
                results['top_feature'] = None
                results['top_importance'] = None
        
        # 4. Permutation Importance
        print("  Computing permutation importance...")
        try:
            fig = interpreter.plot_permutation_importance(max_features=15, show=False)
            plt.savefig(output_dir / "permutation_importance.png", dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"  ⚠ Permutation importance failed: {str(e)[:80]}...")
        
        # 5. Individual Predictions
        print(f"  Explaining {len(sample_indices)} sample predictions...")
        predictions_dir = output_dir / "predictions"
        predictions_dir.mkdir(exist_ok=True)
        
        explanations = []
        for idx in sample_indices:
            if idx < len(ml_model.X_test):
                try:
                    explanation = interpreter.explain_prediction(idx, top_k=10)
                    explanations.append(explanation)
                    
                    fig = interpreter.plot_single_prediction(idx, show=False)
                    plt.savefig(predictions_dir / f"sample_{idx}.png", dpi=150, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    print(f"  ⚠ Sample {idx} explanation failed: {str(e)[:60]}...")
        
        # Save explanations
        if explanations:
            explanations_df = pd.DataFrame([
                {
                    'sample_idx': e['sample_idx'],
                    'true_label': e['true_label'],
                    'predicted_label': e['predicted_label'],
                    'prediction_proba': e['prediction_proba'],
                    'top_feature': e['top_features'][0]['feature'] if e['top_features'] else None,
                    'top_shap': e['top_features'][0]['shap_value'] if e['top_features'] else None
                }
                for e in explanations
            ])
            explanations_df.to_csv(predictions_dir / "explanations.csv", index=False)
        
        print(f"  ✓ Analysis complete")
        
    except Exception as e:
        print(f"  ✗ Analysis failed: {e}")
        results['status'] = 'failed'
        results['error'] = str(e)
        import traceback
        traceback.print_exc()
    
    return results


def run_analysis(
    datasets=None,
    model_types=None,
    version='v2.11',
    normalization='none',
    output_base='results/interpretability',
    sample_indices=None,
    timestamp=False,
):
    """Run interpretability analysis on specified models"""
    
    dump_dir = Path(project_root) / "dump" / f"{version}_{normalization}"
    base_dir = Path(project_root) / output_base / f"{version}_{normalization}"
    if timestamp:
        run_tag = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        output_base = base_dir / run_tag
    else:
        output_base = base_dir
    
    if not dump_dir.exists():
        print(f"✗ Dump directory not found: {dump_dir}")
        return []
    
    # Find all models matching criteria
    models_to_analyze = []
    for model_path in sorted(dump_dir.glob("*_gridsearch_model.joblib")):
        # Parse model name: {model_type}_{dataset}_gridsearch_model.joblib
        stem = model_path.stem.replace('_gridsearch_model', '')
        
        # Handle compound model names (sklearn_mlp, pytorch_mlp)
        if stem.startswith('sklearn_mlp_'):
            model_type = 'sklearn_mlp'
            dataset = stem.replace('sklearn_mlp_', '')
        elif stem.startswith('pytorch_mlp_'):
            model_type = 'pytorch_mlp'
            dataset = stem.replace('pytorch_mlp_', '')
        elif stem.startswith('random_forest_'):
            model_type = 'random_forest'
            dataset = stem.replace('random_forest_', '')
        elif stem.startswith('xgboost_'):
            model_type = 'xgboost'
            dataset = stem.replace('xgboost_', '')
        elif stem.startswith('svm_'):
            model_type = 'svm'
            dataset = stem.replace('svm_', '')
        else:
            # Fallback: assume first part is model type
            parts = stem.split('_', 1)
            if len(parts) == 2:
                model_type = parts[0]
                dataset = parts[1]
            else:
                continue
        
        # Filter by criteria
        if datasets and dataset not in datasets:
            continue
        if model_types and model_type not in model_types:
            continue
        
        models_to_analyze.append((model_path, model_type, dataset))
    
    if not models_to_analyze:
        print(f"✗ No models found matching criteria")
        return []
    
    print(f"\n{'='*80}")
    print(f"INTERPRETABILITY ANALYSIS")
    print(f"{'='*80}")
    print(f"Version: {version}")
    print(f"Normalization: {normalization}")
    print(f"Models to analyze: {len(models_to_analyze)}")
    print(f"Output directory: {output_base}")
    print(f"{'='*80}\n")
    
    # Run analysis on each model
    all_results = []
    for i, (model_path, model_type, dataset) in enumerate(models_to_analyze, 1):
        print(f"\n[{i}/{len(models_to_analyze)}] {model_type} on {dataset}")
        print(f"{'-'*80}")
        
        output_dir = output_base / f"{model_type}_{dataset}"
        results = analyze_model(model_path, output_dir, sample_indices=sample_indices or [0, 1, 5, 10])
        
        if results:
            all_results.append(results)
    
    # Save summary
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_path = output_base / "analysis_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\n{'='*80}")
        print(f"ANALYSIS SUMMARY")
        print(f"{'='*80}")
        print(f"Total models analyzed: {len(all_results)}")
        print(f"Successful: {sum(1 for r in all_results if r['status'] == 'success')}")
        print(f"Failed: {sum(1 for r in all_results if r['status'] == 'failed')}")
        print(f"\nSummary saved to: {summary_path}")
        print(f"{'='*80}\n")
        
        # Generate model comparison table (top features across models)
        print("Generating model comparison table...")
        comparison_data = []
        for result in all_results:
            if result['status'] == 'success':
                model_output_dir = output_base / f"{result['model_type']}_{result['dataset']}"
                importance_path = model_output_dir / "feature_importance.csv"
                
                if importance_path.exists():
                    importance_df = pd.read_csv(importance_path)
                    top_5_features = importance_df.head(5)['feature'].tolist()
                    top_5_importance = importance_df.head(5)['importance'].tolist()
                    
                    comparison_data.append({
                        'model_type': result['model_type'],
                        'dataset': result['dataset'],
                        'test_accuracy': result.get('test_accuracy'),
                        'top_1': top_5_features[0] if len(top_5_features) > 0 else None,
                        'top_1_importance': top_5_importance[0] if len(top_5_importance) > 0 else None,
                        'top_2': top_5_features[1] if len(top_5_features) > 1 else None,
                        'top_3': top_5_features[2] if len(top_5_features) > 2 else None,
                        'top_4': top_5_features[3] if len(top_5_features) > 3 else None,
                        'top_5': top_5_features[4] if len(top_5_features) > 4 else None,
                    })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_path = output_base / "model_comparison_top_features.csv"
            comparison_df.to_csv(comparison_path, index=False)
            print(f"✓ Model comparison table saved to: {comparison_path}\n")
        
        # Print top models by feature importance
        if 'top_importance' in summary_df.columns:
            print("\nTop 5 models by feature importance:")
            top_models = summary_df.nlargest(5, 'top_importance')[
                ['model_type', 'dataset', 'test_accuracy', 'top_feature', 'top_importance']
            ]
            print(top_models.to_string(index=False))
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Run interpretability analysis on trained ML models'
    )
    parser.add_argument(
        '--dataset',
        help='Specific dataset to analyze'
    )
    parser.add_argument(
        '--model',
        help='Specific model type to analyze'
    )
    parser.add_argument(
        '--datasets',
        nargs='*',
        help='Multiple datasets to analyze'
    )
    parser.add_argument(
        '--models',
        nargs='*',
        help='Multiple model types to analyze'
    )
    parser.add_argument(
        '--all-datasets',
        action='store_true',
        help='Analyze all datasets'
    )
    parser.add_argument(
        '--all-models',
        action='store_true',
        help='Analyze all model types'
    )
    parser.add_argument(
        '--full-analysis',
        action='store_true',
        help='Analyze all models on all datasets'
    )
    parser.add_argument(
        '--version',
        default='v2.11',
        help='Model version to analyze'
    )
    parser.add_argument(
        '--normalization',
        default='none',
        choices=['none', 'standard', 'robust', 'minmax'],
        help='Normalization type'
    )
    parser.add_argument(
        '--output-dir',
        default='results/interpretability',
        help='Base output directory'
    )
    parser.add_argument(
        '--timestamp',
        action='store_true',
        help='Append timestamp to output run folder'
    )
    parser.add_argument(
        '--sample-indices',
        nargs='*',
        type=int,
        default=[0, 1, 2],
        help='Test indices to explain with local plots'
    )
    
    args = parser.parse_args()
    
    # Determine what to analyze
    datasets = None
    model_types = None
    
    if args.full_analysis:
        # Analyze everything
        pass
    elif args.all_datasets:
        # All datasets, specific model
        model_types = [args.model] if args.model else None
    elif args.all_models:
        # All models, specific dataset
        datasets = [args.dataset] if args.dataset else None
    else:
        # Specific model and dataset
        if args.dataset:
            datasets = [args.dataset]
        if args.model:
            model_types = [args.model]

    # Override with explicit multi-selects if provided
    if args.datasets:
        datasets = args.datasets
    if args.models:
        model_types = args.models
    
    # Run analysis
    results = run_analysis(
        datasets=datasets,
        model_types=model_types,
        version=args.version,
        normalization=args.normalization,
        output_base=args.output_dir,
        sample_indices=args.sample_indices,
        timestamp=args.timestamp,
    )
    
    sys.exit(0 if results else 1)


if __name__ == '__main__':
    main()
