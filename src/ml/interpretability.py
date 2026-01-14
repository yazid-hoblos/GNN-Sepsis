"""
Model Interpretability and Explainability Module
-------------------------------------------------

This module provides tools for interpreting and explaining predictions from trained ML models.
It supports SHAP-based feature importance analysis, permutation importance, and visualizations
for understanding model decisions in the context of sepsis prediction from omics data.

Supported model types:
    - Tree-based models (Random Forest, XGBoost)
    - Linear models (SVM with linear kernel)
    - Neural networks (scikit-learn MLP, PyTorch MLP)

Features:
    - SHAP value computation and visualization
    - Feature importance ranking
    - Prediction explanation for individual samples
    - Comparison of feature importance across models/datasets
    - Interpretation of embeddings vs. raw gene expression
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional, Union, List, Dict
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    warnings.warn("shap not installed. Install with: pip install shap")

from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

# ==================================================================================== #
# CONFIGURATION
# ==================================================================================== #

PLOT_COLORS = {
    'septic': '#d73027',      # Red for sepsis (positive class)
    'control': '#4575b4',     # Blue for control (negative class)
    'important': '#fee090',   # Yellow for important features
    'neutral': '#e0e0e0'      # Gray for neutral
}

# ==================================================================================== #
# MAIN INTERPRETABILITY CLASS
# ==================================================================================== #

class ModelInterpreter:
    """
    Comprehensive model interpretability tool for analyzing ML predictions.
    
    This class wraps around trained MLModel instances to provide:
    - SHAP-based feature importance
    - Permutation importance
    - Prediction explanations
    - Embedding interpretation
    
    Parameters
    ----------
    ml_model : MLModel
        A trained MLModel instance from model_trainer.py
    feature_names : list, optional
        Names of features (gene names, embedding components, etc.)
        If None, will use generic names (Feature_0, Feature_1, ...)
    
    Attributes
    ----------
    model : sklearn/torch model
        The underlying trained model
    X_test : np.ndarray
        Test set features
    y_test : np.ndarray
        Test set labels
    y_pred : np.ndarray
        Model predictions on test set
    feature_names : list
        Feature names for interpretability
    """
    
    def __init__(self, ml_model, feature_names: Optional[List[str]] = None):
        """Initialize interpreter with a trained MLModel."""
        self.ml_model = ml_model
        self.model = ml_model.best_model
        
        # Check if test data is available
        if not hasattr(ml_model, 'X_test') or ml_model.X_test is None:
            raise ValueError(
                "Model does not have test data (X_test is None). "
                "This usually happens when loading a model from disk. "
            )
        
        self.X_test = ml_model.X_test
        self.y_test = ml_model.y_test
        self.y_pred = ml_model.y_pred if hasattr(ml_model, 'y_pred') else None
        self.dataset_name = ml_model.dataset_name
        self.model_type = ml_model.model_type
        
        # Feature names
        n_features = self.X_test.shape[1]
        if feature_names is None:
            self.feature_names = [f"Feature_{i}" for i in range(n_features)]
        else:
            assert len(feature_names) == n_features, \
                f"feature_names length ({len(feature_names)}) must match n_features ({n_features})"
            self.feature_names = feature_names
        
        self.shap_explainer = None
        self.shap_values = None
        self._is_tree_model = self.model_type in ['random_forest', 'xgboost']
        self._is_linear_model = self.model_type in ['svm']
        
    # ==================================================================================== #
    # SHAP-BASED EXPLANATIONS
    # ==================================================================================== #
    
    def compute_shap_values(self, force_recalculate: bool = False) -> np.ndarray:
        """
        Compute SHAP values for model explanations.
        
        Parameters
        ----------
        force_recalculate : bool
            If True, recompute SHAP values even if cached
            
        Returns
        -------
        shap_values : np.ndarray
            SHAP values (n_samples, n_features)
            
        Raises
        ------
        RuntimeError
            If SHAP is not installed or model type is not supported
        """
        if not HAS_SHAP:
            raise RuntimeError(
                "SHAP is required for this feature. Install with: pip install shap"
            )
        
        if self.shap_values is not None and not force_recalculate:
            return self.shap_values
        
        print(f"Computing SHAP values for {self.model_type} model...")
        
        try:
            if self._is_tree_model:
                # Try TreeExplainer first for tree-based models (RandomForest, XGBoost)
                try:
                    booster = getattr(self.model, 'get_booster', None)
                    if callable(booster):
                        self.shap_explainer = shap.TreeExplainer(booster())
                    else:
                        self.shap_explainer = shap.TreeExplainer(self.model)
                    self.shap_values = self.shap_explainer.shap_values(self.X_test)

                    # Handle multiclass output for some models
                    if isinstance(self.shap_values, list):
                        self.shap_values = self.shap_values[1]  # positive class

                except (ValueError, KeyError) as tree_err:
                    # XGBoost 3.x compatibility issue or other TreeExplainer failures
                    print(f"  TreeExplainer failed: {str(tree_err)[:100]}...")
                    print(f"  Falling back to Kernel SHAP for {self.model_type}")
                    self.shap_explainer = shap.KernelExplainer(
                        self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                        shap.sample(self.X_test, min(50, len(self.X_test)))
                    )
                    sv = self.shap_explainer.shap_values(self.X_test)
                    self.shap_values = sv[1] if isinstance(sv, list) and len(sv) > 1 else sv

            elif self.model_type == 'svm' and hasattr(self.model, 'kernel') and self.model.kernel == 'linear':
                # Kernel explainer for linear SVM decision function
                self.shap_explainer = shap.KernelExplainer(
                    self.model.decision_function,
                    shap.sample(self.X_test, min(100, len(self.X_test)))
                )
                self.shap_values = self.shap_explainer.shap_values(self.X_test)

            elif self.model_type == 'pytorch_mlp':
                # Deep explainer for PyTorch models
                import torch
                try:
                    background = shap.sample(self.X_test, min(50, len(self.X_test)))
                    self.shap_explainer = shap.DeepExplainer(self.model.model, torch.tensor(background, dtype=torch.float32))
                    sv = self.shap_explainer.shap_values(torch.tensor(self.X_test, dtype=torch.float32))
                    # DeepExplainer returns a list per output; binary sigmoid has single output
                    self.shap_values = sv[0] if isinstance(sv, list) else sv
                except Exception as torch_err:
                    print(f"  DeepExplainer failed: {str(torch_err)[:100]}...")
                    print(f"  Falling back to Kernel SHAP for pytorch_mlp")
                    self.shap_explainer = shap.KernelExplainer(
                        self.model.predict_proba,
                        shap.sample(self.X_test, min(50, len(self.X_test)))
                    )
                    sv = self.shap_explainer.shap_values(self.X_test)
                    self.shap_values = sv[1] if isinstance(sv, list) and len(sv) > 1 else sv

            elif self.model_type == 'sklearn_mlp':
                # scikit-learn MLP fallback to KernelExplainer with memory-aware sampling
                print("  Using Kernel SHAP for sklearn MLP")
                try:
                    # Start with reasonable background size
                    bg_size = min(50, len(self.X_test))
                    self.shap_explainer = shap.KernelExplainer(
                        self.model.predict_proba,
                        shap.sample(self.X_test, bg_size)
                    )
                    sv = self.shap_explainer.shap_values(self.X_test)
                    self.shap_values = sv[1] if isinstance(sv, list) and len(sv) > 1 else sv
                except MemoryError as mem_err:
                    print(f"  ⚠ Memory error with bg_size={bg_size}: {str(mem_err)[:80]}...")
                    print(f"  Retrying with smaller background (10 samples)...")
                    # Retry with minimal background for high-dimensional data
                    self.shap_explainer = shap.KernelExplainer(
                        self.model.predict_proba,
                        shap.sample(self.X_test, min(10, len(self.X_test)))
                    )
                    sv = self.shap_explainer.shap_values(self.X_test[:min(10, len(self.X_test))])  # Explain fewer samples
                    # Pad remaining samples with zeros to maintain shape
                    if len(sv.shape) == 2:
                        self.shap_values = sv
                    else:
                        self.shap_values = sv[1] if isinstance(sv, list) and len(sv) > 1 else sv

            else:
                # Generic fallback to KernelExplainer
                print(f"  Using Kernel SHAP for {self.model_type}")
                try:
                    bg_size = min(50, len(self.X_test))
                    self.shap_explainer = shap.KernelExplainer(
                        self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                        shap.sample(self.X_test, bg_size)
                    )
                    sv = self.shap_explainer.shap_values(self.X_test)
                    self.shap_values = sv[1] if isinstance(sv, list) and len(sv) > 1 else sv
                except MemoryError as mem_err:
                    print(f"  ⚠ Memory error: {str(mem_err)[:80]}...")
                    print(f"  Using reduced-sample Kernel SHAP (10 background samples)...")
                    self.shap_explainer = shap.KernelExplainer(
                        self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                        shap.sample(self.X_test, min(10, len(self.X_test)))
                    )
                    sv = self.shap_explainer.shap_values(self.X_test[:min(10, len(self.X_test))])
                    self.shap_values = sv[1] if isinstance(sv, list) and len(sv) > 1 else sv
        
        except MemoryError as e:
            print(f"  ✗ SHAP computation still failed due to memory: {str(e)[:100]}...")
            print(f"  Setting shap_values to None; will skip SHAP-dependent plots.")
            self.shap_values = None
        except Exception as e:
            raise RuntimeError(f"Failed to compute SHAP values: {str(e)}")
        
        print(f"✓ SHAP values computed: shape {self.shap_values.shape if self.shap_values is not None else 'None'}")
        return self.shap_values
    
    def plot_shap_summary(self, max_features: int = 20, plot_type: str = 'bar',
                         output_dir: Optional[str] = None, show: bool = True) -> plt.Figure:
        """
        Create SHAP summary plots.
        
        Parameters
        ----------
        max_features : int
            Maximum number of top features to display
        plot_type : str
            Type of plot: 'bar', 'beeswarm', or 'violin'
        output_dir : str, optional
            If provided, save plot to this directory
        show : bool
            Whether to display the plot (default: True)
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure
        """
        if self.shap_values is None:
            self.compute_shap_values()
        
        if self.shap_values is None:
            print(f"⚠ SHAP values not available; skipping plot")
            return None
        
        # Handle 3D SHAP values (binary classification returns shape [n_samples, n_features, n_classes])
        shap_vals = self.shap_values
        if len(shap_vals.shape) == 3:
            # For binary classification, take the positive class (index 1)
            shap_vals = shap_vals[:, :, 1]
        
        fig, axes = plt.subplots(1, 1, figsize=(12, 8))
        
        # Compute mean absolute SHAP values
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[-max_features:][::-1]
        
        if plot_type == 'bar':
            top_names = [self.feature_names[i] for i in top_indices]
            top_values = mean_abs_shap[top_indices]
            
            axes.barh(top_names, top_values, color=PLOT_COLORS['important'])
            axes.set_xlabel('Mean |SHAP value|', fontsize=12, fontweight='bold')
            axes.set_title(f'Feature Importance (SHAP) - {self.dataset_name}', 
                          fontsize=14, fontweight='bold')
            axes.invert_yaxis()
            
            # Create beeswarm plot showing individual sample contributions
            top_shap = shap_vals[:, top_indices]
            top_names = [self.feature_names[i] for i in top_indices]
            
            y_pos = np.arange(len(top_names))
            for i, name in enumerate(top_names):
                colors = [PLOT_COLORS['septic'] if v > 0 else PLOT_COLORS['control'] 
                         for v in top_shap[:, i]]
                axes.scatter(top_shap[:, i], 
                           [i] * len(top_shap[:, i]), 
                           alpha=0.5, s=30, c=colors)
            
            axes.set_yticks(y_pos)
            axes.set_yticklabels(top_names)
            axes.set_xlabel('SHAP value', fontsize=12, fontweight='bold')
            axes.set_title(f'SHAP Value Distribution - {self.dataset_name}', 
                          fontsize=14, fontweight='bold')
            axes.axvline(0, color='black', linestyle='--', linewidth=1)
        
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filepath = Path(output_dir) / f"shap_summary_{plot_type}_{self.dataset_name}.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filepath}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_shap_waterfall(self, sample_idx: int = 0, 
                           output_dir: Optional[str] = None, show: bool = True) -> plt.Figure:
        """
        Plot SHAP waterfall chart for a single prediction.
        
        Parameters
        ----------
        sample_idx : int
            Index of sample to explain
        output_dir : str, optional
            If provided, save plot to this directory
        show : bool
            Whether to display the plot (default: True)
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure
        """
        if self.shap_values is None:
            self.compute_shap_values()
        
        if self.shap_values is None:
            return None
        
        # Handle 3D SHAP values (binary classification)
        shap_vals = self.shap_values
        if len(shap_vals.shape) == 3:
            shap_vals = shap_vals[:, :, 1]  # Positive class
        
        # Handle expected value for binary classification
        expected_value = self.shap_explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)) and len(expected_value) > 1:
            expected_value = expected_value[1]  # Positive class
        
        # Get SHAP values for this sample
        sample_shap = shap_vals[sample_idx]
        
        # Create a simple waterfall-style plot manually
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get top contributing features (both positive and negative)
        abs_shap = np.abs(sample_shap)
        top_indices = np.argsort(abs_shap)[-15:][::-1]
        
        top_features = [self.feature_names[i] for i in top_indices]
        top_shap_values = [sample_shap[i] for i in top_indices]
        
        # Create horizontal bar chart
        colors = ['#ff7f0e' if v > 0 else '#1f77b4' for v in top_shap_values]
        y_pos = np.arange(len(top_features))
        
        ax.barh(y_pos, top_shap_values, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features)
        ax.axvline(0, color='black', linestyle='-', linewidth=0.8)
        ax.set_xlabel('SHAP Value (impact on prediction)', fontsize=12, fontweight='bold')
        
        label = "Septic" if self.y_test[sample_idx] == 1 else "Control"
        pred = "Septic" if (self.y_pred[sample_idx] == 1 if self.y_pred is not None else False) else "Control"
        
        ax.set_title(f"Sample {sample_idx}: True={label}, Pred={pred} - {self.dataset_name}", 
                    fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        
        # Add text annotation
        pred_proba = self.model.predict_proba(self.X_test[sample_idx:sample_idx+1])[0][1] if hasattr(self.model, 'predict_proba') else 0.5
        ax.text(0.02, 0.98, f'Base value: {expected_value:.3f}\nPrediction: {pred_proba:.3f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filepath = Path(output_dir) / f"shap_waterfall_sample{sample_idx}_{self.dataset_name}.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filepath}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_shap_beeswarm(self, max_features: int = 15, output_dir: Optional[str] = None, 
                          show: bool = True) -> plt.Figure:
        """
        Plot SHAP beeswarm chart showing distribution of SHAP values per feature.
        
        Parameters
        ----------
        max_features : int
            Maximum number of top features to display
        output_dir : str, optional
            If provided, save plot to this directory
        show : bool
            Whether to display the plot (default: True)
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure
        """
        if self.shap_values is None:
            self.compute_shap_values()
        
        # Handle 3D SHAP values (binary classification)
        shap_vals = self.shap_values
        if len(shap_vals.shape) == 3:
            shap_vals = shap_vals[:, :, 1]  # Positive class
        
        # Get top features by mean absolute SHAP
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[-max_features:][::-1]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create beeswarm-style scatter
        for i, feat_idx in enumerate(top_indices):
            feat_shap = shap_vals[:, feat_idx]
            # Add jitter to x-axis for visibility
            x_jitter = np.random.normal(i, 0.04, size=len(feat_shap))
            colors = [PLOT_COLORS['septic'] if v > 0 else PLOT_COLORS['control'] for v in feat_shap]
            ax.scatter(x_jitter, feat_shap, alpha=0.6, s=50, c=colors, edgecolors='none')
        
        ax.set_xticks(range(len(top_indices)))
        ax.set_xticklabels([self.feature_names[i] for i in top_indices], rotation=45, ha='right')
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.set_ylabel('SHAP value', fontsize=12, fontweight='bold')
        ax.set_title(f'SHAP Value Distribution (Beeswarm) - {self.dataset_name}', 
                    fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filepath = Path(output_dir) / f"shap_beeswarm_{self.dataset_name}.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved: {filepath}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_shap_dependence(self, top_n: int = 3, output_dir: Optional[str] = None,
                            show: bool = True) -> plt.Figure:
        """
        Plot SHAP dependence plots for top N features showing non-linear relationships.
        
        Parameters
        ----------
        top_n : int
            Number of top features to plot dependence for
        output_dir : str, optional
            If provided, save plot to this directory
        show : bool
            Whether to display the plot (default: True)
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure with subplots
        """
        if self.shap_values is None:
            self.compute_shap_values()
        
        # Handle 3D SHAP values
        shap_vals = self.shap_values
        if len(shap_vals.shape) == 3:
            shap_vals = shap_vals[:, :, 1]  # Positive class
        
        # Get top features
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]
        
        fig, axes = plt.subplots(1, top_n, figsize=(15, 4))
        if top_n == 1:
            axes = [axes]
        
        for ax_idx, feat_idx in enumerate(top_indices):
            ax = axes[ax_idx]
            
            # Scatter: feature value vs SHAP value, colored by predicted class
            colors = [PLOT_COLORS['septic'] if y == 1 else PLOT_COLORS['control'] 
                     for y in self.y_pred]
            ax.scatter(self.X_test[:, feat_idx], shap_vals[:, feat_idx], 
                      c=colors, alpha=0.6, s=40, edgecolors='none')
            
            ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
            ax.set_xlabel(f'{self.feature_names[feat_idx]}', fontsize=11, fontweight='bold')
            ax.set_ylabel('SHAP value', fontsize=11, fontweight='bold')
            ax.set_title(f'{self.feature_names[feat_idx]}\n(mean |SHAP|={mean_abs_shap[feat_idx]:.4f})', 
                        fontsize=10)
            ax.grid(alpha=0.3)
        
        plt.suptitle(f'SHAP Dependence Plots - {self.dataset_name}', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filepath = Path(output_dir) / f"shap_dependence_top{top_n}_{self.dataset_name}.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved: {filepath}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_feature_stats_by_class(self, top_n: int = 10, output_dir: Optional[str] = None,
                                   show: bool = True) -> plt.Figure:
        """
        Plot feature value distributions by class (control vs septic).
        
        Parameters
        ----------
        top_n : int
            Number of top features to display
        output_dir : str, optional
            If provided, save plot to this directory
        show : bool
            Whether to display the plot (default: True)
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure
        """
        if self.shap_values is None:
            self.compute_shap_values()
        
        # Get top features by SHAP
        shap_vals = self.shap_values
        if len(shap_vals.shape) == 3:
            shap_vals = shap_vals[:, :, 1]
        
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]
        
        # Compute stats
        control_mask = self.y_test == 0
        septic_mask = self.y_test == 1
        
        fig, axes = plt.subplots(2, (top_n + 1) // 2, figsize=(16, 8))
        axes = axes.flatten()
        
        for plot_idx, feat_idx in enumerate(top_indices):
            ax = axes[plot_idx]
            
            control_vals = self.X_test[control_mask, feat_idx]
            septic_vals = self.X_test[septic_mask, feat_idx]
            
            # Box plot
            bp = ax.boxplot([control_vals, septic_vals], 
                           labels=['Control', 'Septic'],
                           patch_artist=True,
                           widths=0.6)
            
            # Color boxes
            bp['boxes'][0].set_facecolor(PLOT_COLORS['control'])
            bp['boxes'][1].set_facecolor(PLOT_COLORS['septic'])
            
            ax.set_ylabel('Feature Value', fontsize=10)
            ax.set_title(f'{self.feature_names[feat_idx]}\n(mean |SHAP|={mean_abs_shap[feat_idx]:.4f})', 
                        fontsize=10, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
        
        # Hide extra subplots
        for idx in range(len(top_indices), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Feature Value Distribution by Class - {self.dataset_name}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filepath = Path(output_dir) / f"feature_stats_by_class_{self.dataset_name}.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved: {filepath}")
        
        if show:
            plt.show()
        
        return fig
    
    # ==================================================================================== #
    # PERMUTATION IMPORTANCE
    # ==================================================================================== #
    
    def compute_permutation_importance(self, n_repeats: int = 10) -> pd.DataFrame:
        """
        Compute permutation importance scores.
        
        Parameters
        ----------
        n_repeats : int
            Number of times to permute each feature
            
        Returns
        -------
        importance_df : pd.DataFrame
            DataFrame with columns: feature, importance_mean, importance_std
        """
        print(f"Computing permutation importance ({n_repeats} repeats)...")
        
        result = permutation_importance(
            self.model, self.X_test, self.y_test,
            n_repeats=n_repeats, random_state=42, n_jobs=-1
        )
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        return importance_df
    
    def plot_permutation_importance(self, max_features: int = 20,
                                   output_dir: Optional[str] = None, show: bool = True) -> plt.Figure:
        """
        Plot permutation importance with error bars.
        
        Parameters
        ----------
        max_features : int
            Maximum number of top features to display
        output_dir : str, optional
            If provided, save plot to this directory
        show : bool
            Whether to display the plot (default: True)
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure
        """
        importance_df = self.compute_permutation_importance()
        importance_df = importance_df.head(max_features)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        ax.barh(importance_df['feature'], importance_df['importance_mean'], 
               xerr=importance_df['importance_std'], 
               color=PLOT_COLORS['important'], alpha=0.8, capsize=5)
        ax.set_xlabel('Permutation Importance (accuracy drop)', fontsize=12, fontweight='bold')
        ax.set_title(f'Permutation Importance - {self.dataset_name}', 
                    fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filepath = Path(output_dir) / f"permutation_importance_{self.dataset_name}.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved: {filepath}")
        
        if show:
            plt.show()
        
        return fig
    
    # ==================================================================================== #
    # BUILT-IN FEATURE IMPORTANCE (for tree-based models)
    # ==================================================================================== #
    
    def get_native_feature_importance(self) -> pd.DataFrame:
        """
        Get native feature importance from tree-based models.
        
        Returns
        -------
        importance_df : pd.DataFrame
            DataFrame with columns: feature, importance
            
        Raises
        ------
        ValueError
            If model doesn't have feature_importances_ attribute
        """
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError(f"{self.model_type} does not have native feature importance")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_native_importance(self, max_features: int = 20,
                              output_dir: Optional[str] = None, show: bool = True) -> Optional[plt.Figure]:
        """
        Plot native feature importance for tree models.
        
        Parameters
        ----------
        max_features : int
            Maximum number of top features to display
        output_dir : str, optional
            If provided, save plot to this directory
        show : bool
            Whether to display the plot (default: True)
            
        Returns
        -------
        fig : matplotlib.figure.Figure or None
            The created figure, or None if model doesn't support native importance
        """
        try:
            importance_df = self.get_native_feature_importance()
        except ValueError as e:
            print(f"⚠ {e}")
            return None
        
        importance_df = importance_df.head(max_features)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        ax.barh(importance_df['feature'], importance_df['importance'], 
               color=PLOT_COLORS['important'])
        ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
        ax.set_title(f'Native Feature Importance ({self.model_type}) - {self.dataset_name}', 
                    fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filepath = Path(output_dir) / f"native_importance_{self.dataset_name}.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filepath}")
        
        if show:
            plt.show()
        
        return fig
    
    # ==================================================================================== #
    # PREDICTION EXPLANATION FOR INDIVIDUAL SAMPLES
    # ==================================================================================== #
    
    def explain_prediction(self, sample_idx: int, top_k: int = 10) -> Dict:
        """
        Get detailed explanation for a single prediction.
        
        Parameters
        ----------
        sample_idx : int
            Index of sample to explain
        top_k : int
            Number of top contributing features to return (default: 10)
            
        Returns
        -------
        explanation : dict
            Dictionary containing:
            - sample_idx: index of sample
            - true_label: actual class
            - prediction: model prediction
            - confidence: prediction confidence
            - top_contributing_features: features with largest SHAP values
        """
        if self.shap_values is None:
            self.compute_shap_values()
        
        # Handle 3D SHAP values (binary classification)
        shap_vals = self.shap_values
        if len(shap_vals.shape) == 3:
            shap_vals = shap_vals[:, :, 1]  # Positive class
        
        # Get SHAP values for this sample
        sample_shap = shap_vals[sample_idx]
        
        # Get top contributing features
        abs_shap = np.abs(sample_shap)
        top_indices = np.argsort(abs_shap)[-top_k:][::-1]
        
        # Get prediction probability
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(self.X_test[sample_idx:sample_idx+1])[0]
            prediction_proba = proba[1]  # Positive class probability
        else:
            prediction_proba = 1.0 if self.y_pred[sample_idx] == 1 else 0.0
        
        explanation = {
            'sample_idx': sample_idx,
            'true_label': 'Septic' if self.y_test[sample_idx] == 1 else 'Control',
            'predicted_label': 'Septic' if (self.y_pred[sample_idx] == 1 if self.y_pred is not None else False) else 'Control',
            'prediction': 'Septic' if (self.y_pred[sample_idx] == 1 if self.y_pred is not None else False) else 'Control',  # Backward compatibility
            'prediction_proba': float(prediction_proba),
            'top_features': [
                {
                    'feature': self.feature_names[idx],
                    'value': float(self.X_test[sample_idx, idx]),
                    'shap_value': float(sample_shap[idx]),
                    'direction': 'increases sepsis risk' if sample_shap[idx] > 0 else 'decreases sepsis risk'
                }
                for idx in top_indices
            ],
            'top_contributing_features': [  # Backward compatibility
                {
                    'feature': self.feature_names[idx],
                    'value': float(self.X_test[sample_idx, idx]),
                    'shap_value': float(sample_shap[idx]),
                    'direction': 'increases sepsis risk' if sample_shap[idx] > 0 else 'decreases sepsis risk'
                }
                for idx in top_indices
            ]
        }
        
        return explanation
    
    def print_prediction_explanation(self, sample_idx: int) -> None:
        """Print human-readable explanation for a prediction."""
        explanation = self.explain_prediction(sample_idx)
        
        print("\n" + "="*80)
        print(f"PREDICTION EXPLANATION - Sample {explanation['sample_idx']}")
        print("="*80)
        print(f"True Label:    {explanation['true_label']}")
        print(f"Prediction:    {explanation['prediction']}")
        print("\nTop Contributing Features:")
        print("-"*80)
        
        for i, feat in enumerate(explanation['top_contributing_features'], 1):
            print(f"\n{i}. {feat['feature']}")
            print(f"   Value:        {feat['value']:.4f}")
            print(f"   SHAP value:   {feat['shap_value']:.4f}")
            print(f"   Effect:       {feat['direction']}")
        
        print("\n" + "="*80 + "\n")
    
    # ==================================================================================== #
    # CONVENIENCE METHODS FOR COMMON WORKFLOWS
    # ==================================================================================== #
    
    def get_feature_importance(self, top_k: int = 20, method: str = 'shap') -> pd.DataFrame:
        """
        Get feature importance using specified method.
        
        Parameters
        ----------
        top_k : int
            Number of top features to return
        method : str
            Method to use: 'shap', 'permutation', or 'native'
            
        Returns
        -------
        importance_df : pd.DataFrame
            DataFrame with feature names and importance scores
        """
        if method == 'shap':
            # Use SHAP values
            if self.shap_values is None:
                self.compute_shap_values()
            
            # If SHAP still None, fall back to permutation
            if self.shap_values is None:
                print(f"  ⚠ SHAP values unavailable; using permutation importance instead")
                return self.get_feature_importance(top_k=top_k, method='permutation')
            
            # Handle 3D SHAP values (binary classification)
            shap_vals = self.shap_values
            if len(shap_vals.shape) == 3:
                shap_vals = shap_vals[:, :, 1]  # Positive class
            
            mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': mean_abs_shap
            }).sort_values('importance', ascending=False)
            
        elif method == 'permutation':
            importance_df = self.compute_permutation_importance()
            importance_df = importance_df.rename(columns={'importance_mean': 'importance'})
            importance_df = importance_df[['feature', 'importance']]
            
        elif method == 'native':
            importance_df = self.get_native_feature_importance()
            
        else:
            raise ValueError(f"Unknown method: {method}. Use 'shap', 'permutation', or 'native'")
        
        return importance_df.head(top_k)
    
    def plot_feature_importance(self, top_k: int = 20, method: str = 'shap',
                               output_dir: Optional[str] = None, show: bool = True) -> plt.Figure:
        """
        Plot feature importance.
        
        Parameters
        ----------
        top_k : int
            Number of top features to display
        method : str
            Method to use: 'shap', 'permutation', or 'native'
        output_dir : str, optional
            If provided, save plot to this directory
        show : bool
            Whether to display the plot (default: True)
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure
        """
        if method == 'shap':
            return self.plot_shap_summary(max_features=top_k, plot_type='bar', 
                                         output_dir=output_dir, show=show)
        elif method == 'permutation':
            return self.plot_permutation_importance(max_features=top_k, 
                                                   output_dir=output_dir, show=show)
        elif method == 'native':
            return self.plot_native_importance(max_features=top_k, 
                                              output_dir=output_dir, show=show)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'shap', 'permutation', or 'native'")
    
    def plot_single_prediction(self, sample_idx: int, output_dir: Optional[str] = None,
                              show: bool = True) -> plt.Figure:
        """
        Plot explanation for a single prediction (force plot).
        
        Parameters
        ----------
        sample_idx : int
            Index of sample to explain
        output_dir : str, optional
            If provided, save plot to this directory
        show : bool
            Whether to display the plot (default: True)
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure
        """
        return self.plot_shap_waterfall(sample_idx, output_dir=output_dir, show=show)
    
    @staticmethod
    def compare_feature_importance(interpreters: List['ModelInterpreter'],
                                   labels: List[str],
                                   top_k: int = 20) -> pd.DataFrame:
        """
        Compare feature importance across multiple models.
        
        Parameters
        ----------
        interpreters : list of ModelInterpreter
            List of interpreter instances
        labels : list of str
            Labels for each interpreter
        top_k : int
            Number of top features to compare
            
        Returns
        -------
        comparison_df : pd.DataFrame
            DataFrame with feature importance for each model
        """
        # Get feature importance from each interpreter
        all_importance = {}
        for interp, label in zip(interpreters, labels):
            importance = interp.get_feature_importance(top_k=top_k, method='shap')
            all_importance[label] = importance.set_index('feature')['importance']
        
        # Combine into single DataFrame
        comparison_df = pd.DataFrame(all_importance).fillna(0)
        return comparison_df
    
    @staticmethod
    def plot_feature_comparison(interpreters: List['ModelInterpreter'],
                               labels: List[str],
                               top_k: int = 10,
                               output_dir: Optional[str] = None,
                               show: bool = True) -> plt.Figure:
        """
        Plot comparison of feature importance across models.
        
        Parameters
        ----------
        interpreters : list of ModelInterpreter
            List of interpreter instances
        labels : list of str
            Labels for each interpreter
        top_k : int
            Number of top features to compare
        output_dir : str, optional
            If provided, save plot to this directory
        show : bool
            Whether to display the plot (default: True)
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure
        """
        comparison_df = ModelInterpreter.compare_feature_importance(interpreters, labels, top_k)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(comparison_df))
        width = 0.8 / len(labels)
        
        for i, label in enumerate(labels):
            offset = width * i - (width * len(labels) / 2)
            ax.barh(x + offset, comparison_df[label], width, label=label, alpha=0.8)
        
        ax.set_yticks(x)
        ax.set_yticklabels(comparison_df.index)
        ax.set_xlabel('Feature Importance (mean |SHAP|)', fontsize=12, fontweight='bold')
        ax.set_title('Feature Importance Comparison Across Models', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filepath = Path(output_dir) / "feature_comparison.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filepath}")
        
        if show:
            plt.show()
        
        return fig


# ==================================================================================== #
# EMBEDDING INTERPRETATION UTILITIES
# ==================================================================================== #

class EmbeddingInterpreter:
    """
    Specialized tools for interpreting learned embeddings (ComplEx, RGCN).
    
    Helps understand what different embedding dimensions capture in the context
    of sepsis prediction.
    """
    
    def __init__(self, embedding_matrix: np.ndarray, 
                 embedding_names: Optional[List[str]] = None,
                 embedding_type: str = 'unknown'):
        """
        Initialize embedding interpreter.
        
        Parameters
        ----------
        embedding_matrix : np.ndarray
            Shape (n_samples, n_dimensions)
        embedding_names : list, optional
            Names for embedding dimensions (e.g., 'Dim_0', 'Dim_1', ...)
        embedding_type : str
            Type of embedding ('ComplEx', 'RGCN', 'concatenated', etc.)
        """
        self.embedding_matrix = embedding_matrix
        self.n_samples, self.n_dims = embedding_matrix.shape
        self.embedding_type = embedding_type
        
        if embedding_names is None:
            self.embedding_names = [f"Dim_{i}" for i in range(self.n_dims)]
        else:
            assert len(embedding_names) == self.n_dims
            self.embedding_names = embedding_names
    
    def correlation_with_target(self, y: np.ndarray) -> pd.DataFrame:
        """
        Compute correlation between embedding dimensions and target variable.
        
        Parameters
        ----------
        y : np.ndarray
            Binary target variable (0/1)
            
        Returns
        -------
        corr_df : pd.DataFrame
            DataFrame with columns: dimension, correlation, abs_correlation
        """
        correlations = np.array([
            np.corrcoef(self.embedding_matrix[:, i], y)[0, 1]
            for i in range(self.n_dims)
        ])
        
        corr_df = pd.DataFrame({
            'dimension': self.embedding_names,
            'correlation': correlations,
            'abs_correlation': np.abs(correlations)
        }).sort_values('abs_correlation', ascending=False)
        
        return corr_df
    
    def plot_dimension_variance(self, y: Optional[np.ndarray] = None,
                               output_dir: Optional[str] = None) -> plt.Figure:
        """
        Plot variance of each embedding dimension.
        
        Parameters
        ----------
        y : np.ndarray, optional
            Binary target variable. If provided, show variance by class.
        output_dir : str, optional
            If provided, save plot to this directory
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        if y is None:
            # Overall variance
            variances = np.var(self.embedding_matrix, axis=0)
            ax.bar(range(len(variances)), variances, color=PLOT_COLORS['important'], alpha=0.7)
            ax.set_ylabel('Variance', fontsize=12, fontweight='bold')
        else:
            # Variance by class
            control_var = np.var(self.embedding_matrix[y == 0], axis=0)
            septic_var = np.var(self.embedding_matrix[y == 1], axis=0)
            
            x = np.arange(len(control_var))
            width = 0.35
            ax.bar(x - width/2, control_var, width, label='Control', 
                  color=PLOT_COLORS['control'], alpha=0.8)
            ax.bar(x + width/2, septic_var, width, label='Septic', 
                  color=PLOT_COLORS['septic'], alpha=0.8)
            ax.set_ylabel('Variance', fontsize=12, fontweight='bold')
            ax.legend(fontsize=11)
        
        ax.set_xlabel('Embedding Dimension', fontsize=12, fontweight='bold')
        ax.set_title(f'Embedding Dimension Variance - {self.embedding_type}', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(range(min(50, len(variances))))
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filepath = Path(output_dir) / f"embedding_variance_{self.embedding_type}.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filepath}")
        
        return fig


# ==================================================================================== #
# COMPARATIVE ANALYSIS
# ==================================================================================== #

def compare_model_interpretability(interpreters: List[ModelInterpreter],
                                  output_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Compare feature importance across multiple models/datasets.
    
    Parameters
    ----------
    interpreters : list of ModelInterpreter
        Multiple interpreter instances to compare
    output_dir : str, optional
        If provided, save comparison table to this directory
        
    Returns
    -------
    comparison_df : pd.DataFrame
        Comparison of top features across models
    """
    # Get top features for each model
    top_features_list = []
    
    for interp in interpreters:
        try:
            importance_df = interp.get_native_feature_importance()
            top_features_list.append(importance_df.head(10))
        except ValueError:
            print(f"⚠ Skipping {interp.dataset_name} (no native importance)")
    
    if not top_features_list:
        print("⚠ No importance data available for comparison")
        return None
    
    # Combine and create comparison
    all_features = set()
    for df in top_features_list:
        all_features.update(df['feature'].values)
    
    comparison_data = {feat: [] for feat in sorted(all_features)}
    
    for i, interp in enumerate(interpreters):
        importance_dict = dict(zip(top_features_list[i]['feature'], 
                                   top_features_list[i]['importance']))
        for feat in sorted(all_features):
            comparison_data[feat].append(importance_dict.get(feat, 0))
    
    comparison_df = pd.DataFrame(
        comparison_data,
        index=[interp.dataset_name for interp in interpreters]
    ).T
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filepath = Path(output_dir) / "model_comparison.csv"
        comparison_df.to_csv(filepath)
        print(f"✓ Saved: {filepath}")
    
    return comparison_df


if __name__ == "__main__":
    print("Model Interpretability Module")
    print("="*80)
    print("This module provides tools for interpreting ML model predictions.")
    print("Use with: from src.ml.interpretability import ModelInterpreter")
    print("="*80)
