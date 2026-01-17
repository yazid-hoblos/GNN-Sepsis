# Model Interpretability Module

## Overview

The interpretability module provides comprehensive tools for explaining and understanding the ML model predictions. It aims to answer key questions:

- **What features matter most?** → Feature importance analysis
- **Why did the model predict sepsis/control?** → SHAP explanations
- **Are embeddings meaningful?** → Embedding interpretation
- **Which model is most interpretable?** → Cross-model comparison

---

## Features

### 1. **SHAP-Based Feature Importance**
Uses SHapley Additive exPlanations to compute the contribution of each feature to predictions.

**Supported Models:**
- ✅ Random Forest
- ✅ XGBoost
- ✅ SVM (linear kernel)
- ✅ Neural Networks (scikit-learn MLP, PyTorch MLP)

---

### 2. **Permutation Importance**
Measures how much a model's performance drops when feature values are randomly shuffled.

```python
# Compute permutation importance
importance_df = interpreter.compute_permutation_importance(n_repeats=10)
# Visualize
fig = interpreter.plot_permutation_importance(max_features=20)
```

**Advantages:**
- Model-agnostic (works with any model)
- Captures feature interactions
- Shows with confidence intervals

---

### 3. **Native Feature Importance**
For tree-based models, we use their built-in importance scores.

```python
# For Random Forest / XGBoost
importance_df = interpreter.get_native_feature_importance()
fig = interpreter.plot_native_importance(max_features=20)
```

---

### 4. **Individual Prediction Explanations**
Explain specific predictions using SHAP waterfall plots.

```python
# Explain prediction for sample 0
interpreter.print_prediction_explanation(sample_idx=0)

# Output:
# ============================================================================
# PREDICTION EXPLANATION - Sample 0
# ============================================================================
# True Label:    Septic
# Prediction:    Septic
#
# Top Contributing Features:
# 1. Feature_45
#    Value:        0.5234
#    SHAP value:   0.3421
#    Effect:       increases sepsis risk
#
# ... (more features)
```

**Visual Explanation:**
```python
fig = interpreter.plot_shap_waterfall(sample_idx=0)
```

Creates a waterfall chart showing how each feature pushes the prediction from the base value to the final decision.

---

### 5. **Embedding Interpretation**
For ComplEx, RGCN, and other embedding-based datasets, analyze what dimensions capture.

---

### 6. **Cross-Model Comparison**
Compare feature importance across multiple models and datasets (`compare_model_interpretability`).

---

## Key Classes

### `ModelInterpreter`

Main class for model interpretability analysis.

**Methods:**
- `compute_shap_values(force_recalculate=False)` → SHAP values
- `plot_shap_summary(max_features=20, plot_type='bar', output_dir=None)` → SHAP bar plot
- `plot_shap_waterfall(sample_idx=0, output_dir=None)` → SHAP waterfall plot
- `compute_permutation_importance(n_repeats=10)` → Permutation importance DataFrame
- `plot_permutation_importance(max_features=20, output_dir=None)` → Permutation plot
- `get_native_feature_importance()` → DataFrame (tree models only)
- `plot_native_importance(max_features=20, output_dir=None)` → Native importance plot
- `explain_prediction(sample_idx)` → Dictionary with explanation
- `print_prediction_explanation(sample_idx)` → Human-readable explanation

---

### `EmbeddingInterpreter`

Specialized class for embedding analysis.

**Methods:**
- `correlation_with_target(y)` → DataFrame of dimension-target correlations
- `plot_dimension_variance(y=None, output_dir=None)` → Variance plot

---

### Utility Functions

- `compare_model_interpretability(interpreters, output_dir=None)` → Comparison DataFrame

---

## Output Directory Structure

```
results/
├── interpretability/
│   ├── shap_summary_bar_gene_expression.png
│   ├── shap_summary_beeswarm_Complex_protein_embeddings.png
│   ├── permutation_importance_RGCN_sample_embeddings.png
│   ├── native_importance_random_forest_gene_expression.png
│   ├── shap_waterfall_sample0_Complex_protein_embeddings.png
│   ├── shap_waterfall_sample1_Complex_protein_embeddings.png
│   ├── embedding_variance_ComplEx.png
│   └── model_comparison.csv
```

---

## Interpreting Results

**SHAP Values:**
- Positive SHAP value → Feature increases prediction toward sepsis
- Negative SHAP value → Feature decreases prediction toward sepsis
- Larger magnitude → More important for that prediction

**Permutation Importance:**
- Higher score → Feature is more important
- Error bars show variance across permutations
- 0 means feature has no impact

**Native Importance (Trees):**
- Relative to tree structure
- Captures feature usage in splits
- May not reflect conditional importance

---

## Examples

### Example 1: Find Most Important Gene in Gene Expression

```python
from src.ml.interpretability import ModelInterpreter
import joblib

# Load gene expression model
model = joblib.load("dump/v2.10_standard/random_forest_gene_expression_gridsearch_model.joblib")
interpreter = ModelInterpreter(model)

# Get top features
importance = interpreter.get_native_feature_importance()
top_gene = importance.iloc[0]['feature']
top_importance = importance.iloc[0]['importance']

print(f"Most important gene: {top_gene} (importance: {top_importance:.4f})")
```

### Example 2: Compare Embedding vs. Raw Gene Expression

```python
from src.ml.interpretability import compare_model_interpretability

# Load 3 models
models = [
    "dump/v2.10_standard/random_forest_gene_expression_gridsearch_model.joblib",
    "dump/v2.10_standard/random_forest_ComplEx_protein_embeddings_gridsearch_model.joblib",
    "dump/v2.10_standard/random_forest_RGCN_protein_embeddings_gridsearch_model.joblib",
]

interpreters = [ModelInterpreter(joblib.load(m)) for m in models]

# Compare
comparison = compare_model_interpretability(interpreters)
print(comparison.head(15))
```

### Example 3: Understand Why Model Misclassified a Sample

```python
from src.ml.interpretability import ModelInterpreter
import joblib

model = joblib.load("path/to/model.joblib")
interpreter = ModelInterpreter(model)

# Find misclassified sample
misclassified_idx = 0  # Example index
for i in range(len(interpreter.y_test)):
    if interpreter.y_test[i] != interpreter.y_pred[i]:
        misclassified_idx = i
        break

# Explain the wrong prediction
print(f"Sample {misclassified_idx} was misclassified:")
interpreter.print_prediction_explanation(misclassified_idx)

# Visualize
fig = interpreter.plot_shap_waterfall(sample_idx=misclassified_idx)
```
---

## References

- [SHAP Documentation](https://shap.readthedocs.io/)
- [A Unified Approach to Interpreting Model Predictions (SHAP paper)](https://arxiv.org/abs/1705.07874)
- [Permutation Importance (Scikit-learn)](https://scikit-learn.org/stable/modules/permutation_importance.html)
- ["Why Should I Trust You?" (LIME paper)](https://arxiv.org/abs/1602.04938)

---