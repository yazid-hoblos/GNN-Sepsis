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


def load_model_with_data(model_path):
    """Load a model and attach test data by recreating the train/test split."""
    # Load model
    ml_model = joblib.load(model_path)
    
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
        results['mean_abs_shap'] = abs(shap_values).mean()
        
        print("  Generating SHAP summary plot...")
        fig = interpreter.plot_shap_summary(show=False)
        plt.savefig(output_dir / "shap_summary.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Feature Importance
        print("  Computing feature importance...")
        importance_df = interpreter.get_feature_importance(top_k=50)
        importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
        results['top_feature'] = importance_df.iloc[0]['feature']
        results['top_importance'] = importance_df.iloc[0]['importance']
        
        print("  Generating feature importance plot...")
        fig = interpreter.plot_feature_importance(top_k=20, show=False)
        plt.savefig(output_dir / "feature_importance.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Individual Predictions
        print(f"  Explaining {len(sample_indices)} sample predictions...")
        predictions_dir = output_dir / "predictions"
        predictions_dir.mkdir(exist_ok=True)
        
        explanations = []
        for idx in sample_indices:
            if idx < len(ml_model.X_test):
                explanation = interpreter.explain_prediction(idx, top_k=10)
                explanations.append(explanation)
                
                fig = interpreter.plot_single_prediction(idx, show=False)
                plt.savefig(predictions_dir / f"sample_{idx}.png", dpi=150, bbox_inches='tight')
                plt.close()
        
        # Save explanations
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
    output_base='results/interpretability'
):
    """Run interpretability analysis on specified models"""
    
    dump_dir = Path(project_root) / "dump" / f"{version}_{normalization}"
    output_base = Path(project_root) / output_base / f"{version}_{normalization}"
    
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
        results = analyze_model(model_path, output_dir)
        
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
        choices=['none', 'standard', 'robust'],
        help='Normalization type'
    )
    parser.add_argument(
        '--output-dir',
        default='results/interpretability',
        help='Base output directory'
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
    
    # Run analysis
    results = run_analysis(
        datasets=datasets,
        model_types=model_types,
        version=args.version,
        normalization=args.normalization,
        output_base=args.output_dir
    )
    
    sys.exit(0 if results else 1)


if __name__ == '__main__':
    main()
