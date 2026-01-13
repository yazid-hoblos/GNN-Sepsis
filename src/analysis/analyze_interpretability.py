"""
Quick interpretability analysis script
Compare embeddings vs gene expression and cross-model consistency
"""

import pandas as pd
from pathlib import Path
import numpy as np

results_dir = Path('results/interpretability/v2.11_none')

print("="*80)
print("INTERPRETABILITY ANALYSIS SUMMARY")
print("="*80)

# 1. Compare embedding types (ComplEx vs RGCN)
print("\n1. EMBEDDING TYPE COMPARISON")
print("-"*80)

complex_path = results_dir / 'random_forest_Complex_protein_embeddings' / 'feature_importance.csv'
rgcn_path = results_dir / 'random_forest_RGCN_protein_embeddings' / 'feature_importance.csv'

if complex_path.exists() and rgcn_path.exists():
    complex_df = pd.read_csv(complex_path)
    rgcn_df = pd.read_csv(rgcn_path)
    
    print(f"ComplEx top importance:  {complex_df.iloc[0]['importance']:.4f}")
    print(f"RGCN top importance:     {rgcn_df.iloc[0]['importance']:.4f}")
    print(f"RGCN advantage:          {rgcn_df.iloc[0]['importance']/complex_df.iloc[0]['importance']:.2f}x")
    print(f"\nComplEx mean importance: {complex_df['importance'].mean():.4f}")
    print(f"RGCN mean importance:    {rgcn_df['importance'].mean():.4f}")
else:
    print("⚠ Run: python -m src.ml.run_interpretability --model random_forest --dataset RGCN_protein_embeddings")

# 2. Compare with gene expression
print("\n2. EMBEDDINGS VS GENE EXPRESSION")
print("-"*80)

gene_path = results_dir / 'random_forest_gene_expression' / 'feature_importance.csv'

if gene_path.exists():
    gene_df = pd.read_csv(gene_path)
    
    print(f"Gene expression top:     {gene_df.iloc[0]['importance']:.4f}")
    if rgcn_path.exists():
        print(f"RGCN top:                {rgcn_df.iloc[0]['importance']:.4f}")
        print(f"\nEmbedding advantage:     {rgcn_df.iloc[0]['importance']/gene_df.iloc[0]['importance']:.2f}x")
    
    print(f"\nGene expr mean:          {gene_df['importance'].mean():.4f}")
    if rgcn_path.exists():
        print(f"RGCN mean:               {rgcn_df['importance'].mean():.4f}")
else:
    print("⚠ Run: python -m src.ml.run_interpretability --model random_forest --dataset gene_expression")

# 3. Cross-model consistency
print("\n3. CROSS-MODEL CONSISTENCY (ComplEx embeddings)")
print("-"*80)

models = ['random_forest', 'xgboost', 'sklearn_mlp', 'svm']
available_models = []

for model in models:
    path = results_dir / f"{model}_Complex_protein_embeddings" / "feature_importance.csv"
    if path.exists():
        available_models.append(model)
        df = pd.read_csv(path)
        print(f"\n{model.upper()}")
        print(f"  Top feature:  {df.iloc[0]['feature']} (importance: {df.iloc[0]['importance']:.4f})")
        print(f"  Top 3: {', '.join(df.head(3)['feature'].tolist())}")

if len(available_models) >= 2:
    print(f"\n✓ Found {len(available_models)} models")
    
    # Check overlap in top 10 features
    top_features = {}
    for model in available_models:
        path = results_dir / f"{model}_Complex_protein_embeddings" / "feature_importance.csv"
        df = pd.read_csv(path)
        top_features[model] = set(df.head(10)['feature'].tolist())
    
    # Compute pairwise overlap
    if len(available_models) >= 2:
        model1, model2 = available_models[0], available_models[1]
        overlap = len(top_features[model1] & top_features[model2])
        print(f"\nTop-10 overlap ({model1} vs {model2}): {overlap}/10 features")
else:
    print("\n⚠ Run more models: python -m src.ml.run_interpretability --dataset Complex_protein_embeddings --all-models")

# 4. Load summary statistics
print("\n4. ANALYSIS SUMMARY")
print("-"*80)

summary_path = results_dir / 'analysis_summary.csv'
if summary_path.exists():
    summary_df = pd.read_csv(summary_path)
    print(f"\nTotal analyses: {len(summary_df)}")
    print(f"Successful: {(summary_df['status'] == 'success').sum()}")
    
    if 'test_accuracy' in summary_df.columns:
        print("\nTest Accuracies:")
        for _, row in summary_df.iterrows():
            if pd.notna(row.get('test_accuracy')):
                print(f"  {row['model_type']:15s} on {row['dataset']:35s}: {row['test_accuracy']:.3f}")
else:
    print("⚠ No summary file found yet")

# 5. Key recommendations
print("\n" + "="*80)
print("KEY FINDINGS & RECOMMENDATIONS")
print("="*80)

if rgcn_path.exists() and complex_path.exists():
    rgcn_importance = rgcn_df.iloc[0]['importance']
    complex_importance = complex_df.iloc[0]['importance']
    
    if rgcn_importance > complex_importance * 1.5:
        print("✓ RGCN embeddings are significantly more predictive than ComplEx")
        print("  → Focus on RGCN for GNN interpretability")
    elif complex_importance > rgcn_importance * 1.5:
        print("✓ ComplEx embeddings are significantly more predictive than RGCN")
        print("  → Focus on ComplEx for GNN interpretability")
    else:
        print("✓ ComplEx and RGCN have similar predictive power")
        print("  → Investigate both architectures")

if gene_path.exists() and rgcn_path.exists():
    if rgcn_df['importance'].mean() > gene_df['importance'].mean():
        print("✓ Graph embeddings capture more predictive information than raw expression")
        print("  → Graph structure adds value")
    else:
        print("⚠ Raw gene expression may be as good as embeddings")
        print("  → Consider if graph complexity is justified")

print("\nNext steps:")
print("  1. Map important dimensions back to biological entities")
print("  2. Implement HAN architecture with attention mechanisms")
print("  3. Use GNNExplainer to identify important subgraphs")
print("  4. Analyze pathway-level importance")

print("\n" + "="*80)
