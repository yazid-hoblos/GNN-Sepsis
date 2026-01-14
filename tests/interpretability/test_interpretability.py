"""
Test script for interpretability module
----------------------------------------

Tests SHAP analysis, feature importance, and prediction explanations
for trained ML models on different datasets.

Usage:
    python -m tests.ml.test_interpretability
    python -m tests.ml.test_interpretability --dataset Complex_protein_embeddings --model random_forest
    python -m tests.ml.test_interpretability --version v2.11 --quick
"""

import sys
import argparse
from pathlib import Path
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ml.model_trainer import MLModel
from src.ml.interpretability import ModelInterpreter
from src.ml.load_matrix import load_df
import joblib
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split


def check_sklearn_version():
    """Check scikit-learn version and warn about compatibility"""
    try:
        from packaging import version
        sklearn_version = version.parse(sklearn.__version__)
        min_version = version.parse("1.3.0")
        
        if sklearn_version < min_version:
            print(f"\n⚠️  WARNING: scikit-learn version {sklearn.__version__} detected")
            print(f"   Models were likely trained with scikit-learn >= 1.3.0")
            print(f"   You may encounter compatibility issues.\n")
            print(f"   To fix, run: pip install --upgrade 'scikit-learn>=1.3.0'\n")
            return False
    except ImportError:
        # packaging not available, just check version string
        if sklearn.__version__ < "1.3.0":
            print(f"\n⚠️  WARNING: scikit-learn {sklearn.__version__} may be incompatible")
            print(f"   Upgrade recommended: pip install --upgrade 'scikit-learn>=1.3.0'\n")
            return False
    return True


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


def test_shap_analysis(model_path, output_dir="tests/ml/output"):
    """Test SHAP value computation"""
    print(f"\n{'='*60}")
    print(f"Testing SHAP Analysis")
    print(f"{'='*60}")
    
    # Load model
    print(f"Loading model from: {model_path}")
    try:
        ml_model = load_model_with_data(model_path)
        print(f"Dataset: {ml_model.dataset_name}")
        print(f"Test set: {len(ml_model.X_test)} samples, {ml_model.X_test.shape[1]} features")
    except ValueError as e:
        if "node array" in str(e) and "incompatible dtype" in str(e):
            print(f"\n✗ Scikit-learn version incompatibility detected!")
            print(f"\nThe model was trained with a newer version of scikit-learn.")
            print(f"Current version: {sklearn.__version__}")
            print(f"Required version: >= 1.3.0\n")
            print(f"Fix: pip install --upgrade 'scikit-learn>=1.3.0'\n")
            raise RuntimeError("Scikit-learn version mismatch") from e
        raise
    
    # Initialize interpreter
    interpreter = ModelInterpreter(ml_model)
    
    # Compute SHAP values
    print("Computing SHAP values...")
    shap_values = interpreter.compute_shap_values()
    
    print(f"✓ SHAP values computed successfully")
    print(f"  Shape: {shap_values.shape}")
    print(f"  Mean absolute SHAP: {abs(shap_values).mean():.4f}")
    
    # Generate plots
    print("\nGenerating SHAP summary plot...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig = interpreter.plot_shap_summary(show=False)
    plt.savefig(output_path / "test_shap_summary.png", dpi=150, bbox_inches='tight')
    print(f"✓ Saved to: {output_path / 'test_shap_summary.png'}")
    plt.close()
    
    return True


def test_feature_importance(model_path, output_dir="tests/ml/output"):
    """Test feature importance ranking"""
    print(f"\n{'='*60}")
    print(f"Testing Feature Importance")
    print(f"{'='*60}")
    
    # Load model
    ml_model = load_model_with_data(model_path)
    interpreter = ModelInterpreter(ml_model)
    
    # Get feature importance
    print("Computing feature importance...")
    importance_df = interpreter.get_feature_importance(top_k=20)
    
    print(f"✓ Feature importance computed")
    print(f"\nTop 10 features:")
    print(importance_df.head(10).to_string(index=False))
    
    # Plot
    print("\nGenerating feature importance plot...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig = interpreter.plot_feature_importance(top_k=15, show=False)
    plt.savefig(output_path / "test_feature_importance.png", dpi=150, bbox_inches='tight')
    print(f"✓ Saved to: {output_path / 'test_feature_importance.png'}")
    plt.close()
    
    return True


def test_single_prediction(model_path, sample_idx=0, output_dir="tests/ml/output"):
    """Test single prediction explanation"""
    print(f"\n{'='*60}")
    print(f"Testing Single Prediction Explanation")
    print(f"{'='*60}")
    
    # Load model
    ml_model = load_model_with_data(model_path)
    interpreter = ModelInterpreter(ml_model)
    
    # Explain prediction
    print(f"Explaining prediction for sample {sample_idx}...")
    explanation = interpreter.explain_prediction(sample_idx, top_k=10)
    
    print(f"✓ Prediction explained")
    print(f"\nTrue label: {explanation['true_label']}")
    print(f"Predicted label: {explanation['predicted_label']}")
    print(f"Prediction probability: {explanation['prediction_proba']:.4f}")
    
    print(f"\nTop contributing features:")
    for feat in explanation['top_features'][:5]:
        print(f"  {feat['feature']:20s}: {feat['shap_value']:+.4f}")
    
    # Plot
    print("\nGenerating force plot...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig = interpreter.plot_single_prediction(sample_idx, show=False)
    plt.savefig(output_path / f"test_prediction_{sample_idx}.png", dpi=150, bbox_inches='tight')
    print(f"✓ Saved to: {output_path / f'test_prediction_{sample_idx}.png'}")
    plt.close()
    
    return True


def test_model_comparison(models, output_dir="tests/ml/output"):
    """Test cross-model feature importance comparison"""
    print(f"\n{'='*60}")
    print(f"Testing Model Comparison")
    print(f"{'='*60}")
    
    interpreters = []
    labels = []
    
    for model_path, label in models:
        print(f"Loading {label}...")
        ml_model = load_model_with_data(model_path)
        interpreters.append(ModelInterpreter(ml_model))
        labels.append(label)
    
    # Compare feature importance
    print("Comparing feature importance across models...")
    comparison_df = ModelInterpreter.compare_feature_importance(
        interpreters, labels, top_k=15
    )
    
    print(f"✓ Comparison complete")
    print(f"\nFeature importance correlation:")
    print(comparison_df.corr())
    
    # Plot
    print("\nGenerating comparison plot...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig = ModelInterpreter.plot_feature_comparison(
        interpreters, labels, top_k=10, show=False
    )
    plt.savefig(output_path / "test_model_comparison.png", dpi=150, bbox_inches='tight')
    print(f"✓ Saved to: {output_path / 'test_model_comparison.png'}")
    plt.close()
    
    return True


def run_quick_test(dataset='Complex_protein_embeddings', model_type='random_forest', 
                   version='v2.11', normalization='none'):
    """Run a quick test on a single model"""
    print(f"\n{'#'*60}")
    print(f"# QUICK TEST: {model_type} on {dataset}")
    print(f"# Version: {version}, Normalization: {normalization}")
    print(f"{'#'*60}")
    
    # Construct model path
    model_path = (
        Path(project_root) / "dump" / f"{version}_{normalization}" / 
        f"{model_type}_{dataset}_gridsearch_model.joblib"
    )
    
    if not model_path.exists():
        print(f"✗ Model not found: {model_path}")
        print(f"  Available models in dump/:")
        dump_dir = Path(project_root) / "dump"
        if dump_dir.exists():
            for version_dir in sorted(dump_dir.iterdir()):
                if version_dir.is_dir():
                    print(f"    {version_dir.name}/")
        return False
    
    try:
        # Run tests
        test_shap_analysis(model_path)
        test_feature_importance(model_path)
        test_single_prediction(model_path, sample_idx=0)
        test_single_prediction(model_path, sample_idx=5)
        
        print(f"\n{'='*60}")
        print(f"✓ All tests passed!")
        print(f"{'='*60}")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_comprehensive_test(version='v2.11', normalization='none'):
    """Run comprehensive tests across multiple models"""
    print(f"\n{'#'*60}")
    print(f"# COMPREHENSIVE TEST")
    print(f"# Version: {version}, Normalization: {normalization}")
    print(f"{'#'*60}")
    
    # Test configurations
    configs = [
        ('random_forest', 'Complex_protein_embeddings'),
        ('random_forest', 'RGCN_protein_embeddings'),
        ('xgboost', 'Complex_protein_embeddings'),
    ]
    
    results = {}
    
    for model_type, dataset in configs:
        model_path = (
            Path(project_root) / "dump" / f"{version}_{normalization}" / 
            f"{model_type}_{dataset}_gridsearch_model.joblib"
        )
        
        if not model_path.exists():
            print(f"\n✗ Skipping {model_type} on {dataset} (not found)")
            continue
        
        print(f"\n{'-'*60}")
        print(f"Testing: {model_type} on {dataset}")
        print(f"{'-'*60}")
        
        try:
            test_shap_analysis(model_path, output_dir=f"tests/ml/output/{model_type}_{dataset}")
            test_feature_importance(model_path, output_dir=f"tests/ml/output/{model_type}_{dataset}")
            results[f"{model_type}_{dataset}"] = "PASS"
            print(f"✓ Tests passed for {model_type} on {dataset}")
        except Exception as e:
            results[f"{model_type}_{dataset}"] = f"FAIL: {e}"
            print(f"✗ Tests failed for {model_type} on {dataset}")
    
    # Model comparison test
    if len(results) >= 2:
        print(f"\n{'-'*60}")
        print(f"Testing Model Comparison")
        print(f"{'-'*60}")
        
        models = []
        for model_type, dataset in configs[:2]:  # Compare first 2 models
            model_path = (
                Path(project_root) / "dump" / f"{version}_{normalization}" / 
                f"{model_type}_{dataset}_gridsearch_model.joblib"
            )
            if model_path.exists():
                models.append((model_path, f"{model_type}_{dataset}"))
        
        if len(models) >= 2:
            test_model_comparison(models, output_dir="tests/ml/output")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    for config, result in results.items():
        status = "✓" if result == "PASS" else "✗"
        print(f"{status} {config:50s} {result}")
    
    return all(r == "PASS" for r in results.values())


def main():
    # Check scikit-learn version first
    check_sklearn_version()
    
    parser = argparse.ArgumentParser(
        description='Test interpretability module for ML models'
    )
    parser.add_argument(
        '--dataset', 
        default='Complex_protein_embeddings',
        choices=[
            'gene_expression',
            'RGCN_sample_embeddings',
            'Complex_sample_embeddings',
            'concatenated_sample_embeddings',
            'RGCN_protein_embeddings',
            'Complex_protein_embeddings',
            'concatenated_protein_embeddings'
        ],
        help='Dataset to test'
    )
    parser.add_argument(
        '--model',
        default='random_forest',
        choices=['svm', 'random_forest', 'xgboost', 'sklearn_mlp', 'pytorch_mlp'],
        help='Model type to test'
    )
    parser.add_argument(
        '--version',
        default='v2.11',
        help='Model version to test'
    )
    parser.add_argument(
        '--normalization',
        default='none',
        choices=['none', 'standard', 'robust'],
        help='Normalization type'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick test on single model'
    )
    parser.add_argument(
        '--comprehensive',
        action='store_true',
        help='Run comprehensive test on multiple models'
    )
    
    args = parser.parse_args()
    
    if args.comprehensive:
        success = run_comprehensive_test(args.version, args.normalization)
    elif args.quick:
        success = run_quick_test(args.dataset, args.model, args.version, args.normalization)
    else:
        # Default: quick test
        success = run_quick_test(args.dataset, args.model, args.version, args.normalization)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
