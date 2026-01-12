"""
Exploratory Data Analysis (EDA) for GSE54514 Sepsis Dataset
Analyzes embeddings, patient distribution, and data characteristics
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore")

from load_embeddings import load_patient_data, load_all_entities


def analyze_embeddings(dataset='GSE54514', model='ComplEx'):
    """
    Comprehensive analysis of graph embeddings and patient data.
    """
    print("="*80)
    print(f"EXPLORATORY DATA ANALYSIS: {dataset} - {model} Embeddings")
    print("="*80)
    
    # Set control label
    control_label = "healthy"
    
    # File paths
    emb_path = f"./models/executions/{dataset}_enriched_ontology_degfilterv2.9_outputmodel_{model}_entity_embeddings.npy"
    node_features_csv = f"./models/executions/{dataset}_enriched_ontology_degfilterv2.9_node_features.csv"
    map_csv = f"./models/executions/{dataset}_enriched_ontology_degfilterv2.9_outputmodel_{model}_entity_mapping.csv"
    edge_csv = f"./models/executions/{dataset}_enriched_ontology_degfilterv2.9_edge_attributes.csv"
    
    # ==========================================
    # 1. Load and analyze patient data
    # ==========================================
    print("\n" + "="*80)
    print("1. PATIENT DATA ANALYSIS")
    print("="*80)
    
    patient_ids, X_patients, y_patients, df_patients = load_patient_data(
        emb_path, map_csv, node_features_csv, control_label
    )
    
    print(f"\n📊 Dataset Overview:")
    print(f"  Total patients: {len(patient_ids)}")
    print(f"  Embedding dimension: {X_patients.shape[1]}")
    print(f"  Feature space: {X_patients.shape}")
    
    # Class distribution
    unique, counts = np.unique(y_patients, return_counts=True)
    print(f"\n📈 Class Distribution:")
    print(f"  Healthy (0): {counts[0]} samples ({counts[0]/len(y_patients)*100:.1f}%)")
    print(f"  Disease (1): {counts[1]} samples ({counts[1]/len(y_patients)*100:.1f}%)")
    print(f"  Class ratio (Disease/Healthy): {counts[1]/counts[0]:.2f}")
    
    # Check for imbalance
    if counts[1]/counts[0] > 2 or counts[1]/counts[0] < 0.5:
        print(f"  ⚠️  Class imbalance detected! Consider stratified sampling.")
    else:
        print(f"  ✅ Classes relatively balanced")
    
    # Patient features
    print(f"\n🔍 Patient Features Available:")
    if not df_patients.empty:
        feature_cols = [col for col in df_patients.columns if col not in ['entity_id', 'label', 'node_id', 'embedding']]
        print(f"  Available features: {', '.join(feature_cols[:10])}")
        if len(feature_cols) > 10:
            print(f"  ... and {len(feature_cols) - 10} more")
    
    # ==========================================
    # 2. Embedding statistics
    # ==========================================
    print("\n" + "="*80)
    print("2. EMBEDDING STATISTICS")
    print("="*80)
    
    print(f"\n📐 Embedding Properties:")
    print(f"  Min value: {X_patients.min():.4f}")
    print(f"  Max value: {X_patients.max():.4f}")
    print(f"  Mean: {X_patients.mean():.4f}")
    print(f"  Std: {X_patients.std():.4f}")
    print(f"  Median: {np.median(X_patients):.4f}")
    
    # Check for NaN or infinite values
    nan_count = np.isnan(X_patients).sum()
    inf_count = np.isinf(X_patients).sum()
    print(f"\n🔍 Data Quality:")
    print(f"  NaN values: {nan_count}")
    print(f"  Infinite values: {inf_count}")
    if nan_count > 0 or inf_count > 0:
        print(f"  ⚠️  Data quality issues detected!")
    else:
        print(f"  ✅ No missing or infinite values")
    
    # Dimension-wise variance
    dim_var = X_patients.var(axis=0)
    print(f"\n📊 Dimension-wise Variance:")
    print(f"  Min variance: {dim_var.min():.6f}")
    print(f"  Max variance: {dim_var.max():.6f}")
    print(f"  Mean variance: {dim_var.mean():.6f}")
    print(f"  Dimensions with near-zero variance (<0.001): {(dim_var < 0.001).sum()}")
    
    # ==========================================
    # 3. Load ALL entities (genes, pathways, etc.)
    # ==========================================
    print("\n" + "="*80)
    print("3. COMPLETE KNOWLEDGE GRAPH ANALYSIS")
    print("="*80)
    
    entity_ids, X_all, entity_labels, df_all = load_all_entities(
        emb_path, map_csv, node_features_csv
    )
    
    print(f"\n🌐 Knowledge Graph Entities:")
    print(f"  Total entities: {len(entity_ids)}")
    print(f"  Patients: {len(patient_ids)}")
    print(f"  Other entities (genes/pathways): {len(entity_ids) - len(patient_ids)}")
    
    # Entity type distribution (if available)
    if 'node_id' in df_all.columns:
        entity_types = {}
        for label in entity_labels[:100]:  # Sample first 100
            if label.startswith('Sample'):
                entity_types['Patient'] = entity_types.get('Patient', 0) + 1
            elif label.startswith('GO:'):
                entity_types['GO Term'] = entity_types.get('GO Term', 0) + 1
            elif label.startswith('REACT:'):
                entity_types['Pathway'] = entity_types.get('Pathway', 0) + 1
            elif label.isdigit():
                entity_types['Gene'] = entity_types.get('Gene', 0) + 1
            else:
                entity_types['Other'] = entity_types.get('Other', 0) + 1
        
        print(f"\n📋 Entity Types (from sample):")
        for etype, count in sorted(entity_types.items(), key=lambda x: -x[1]):
            print(f"  {etype}: ~{count}")
    
    # ==========================================
    # 4. Edge/Relation analysis
    # ==========================================
    print("\n" + "="*80)
    print("4. GRAPH STRUCTURE ANALYSIS")
    print("="*80)
    
    if os.path.exists(edge_csv):
        df_edges = pd.read_csv(edge_csv)
        print(f"\n🔗 Graph Edges:")
        print(f"  Total edges: {len(df_edges)}")
        
        if 'predicate' in df_edges.columns or 'relation' in df_edges.columns:
            rel_col = 'predicate' if 'predicate' in df_edges.columns else 'relation'
            relation_counts = df_edges[rel_col].value_counts()
            print(f"\n  Top relation types:")
            for rel, count in relation_counts.head(10).items():
                print(f"    {rel}: {count} edges")
        
        if 'subject' in df_edges.columns and 'object' in df_edges.columns:
            print(f"\n  Unique source nodes: {df_edges['subject'].nunique()}")
            print(f"  Unique target nodes: {df_edges['object'].nunique()}")
            
            # Average degree
            avg_degree = len(df_edges) / df_edges['subject'].nunique()
            print(f"  Average node degree: {avg_degree:.2f}")
    else:
        print(f"\n⚠️  Edge file not found: {edge_csv}")
    
    # ==========================================
    # 5. Return data for visualization
    # ==========================================
    return {
        'X_patients': X_patients,
        'y_patients': y_patients,
        'patient_ids': patient_ids,
        'df_patients': df_patients,
        'X_all': X_all,
        'entity_labels': entity_labels,
        'df_all': df_all,
        'embedding_dim': X_patients.shape[1]
    }


def visualize_embeddings(data, dataset='GSE54514', model='ComplEx', output_dir='eda'):
    """
    Create visualizations of the embeddings.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    X_patients = data['X_patients']
    y_patients = data['y_patients']
    
    print("\n" + "="*80)
    print("5. VISUALIZATION")
    print("="*80)
    
    # Set up plotting style
    sns.set_style('whitegrid')
    
    # ==========================================
    # Plot 1: Class distribution
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar plot
    unique, counts = np.unique(y_patients, return_counts=True)
    axes[0].bar(['Healthy', 'Disease'], counts, color=['steelblue', 'coral'], alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Number of Samples', fontsize=12)
    axes[0].set_title('Class Distribution', fontsize=14, fontweight='bold')
    for i, count in enumerate(counts):
        axes[0].text(i, count + 1, str(count), ha='center', fontweight='bold')
    
    # Pie chart
    axes[1].pie(counts, labels=['Healthy', 'Disease'], autopct='%1.1f%%', 
                colors=['steelblue', 'coral'], startangle=90)
    axes[1].set_title('Class Proportion', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset}_{model}_class_distribution.png'), dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {dataset}_{model}_class_distribution.png")
    plt.close()
    
    # ==========================================
    # Plot 2: Embedding value distribution
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Histogram of all values
    axes[0, 0].hist(X_patients.flatten(), bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0, 0].set_xlabel('Embedding Value', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Distribution of All Embedding Values', fontsize=12, fontweight='bold')
    axes[0, 0].axvline(X_patients.mean(), color='red', linestyle='--', label=f'Mean: {X_patients.mean():.3f}')
    axes[0, 0].legend()
    
    # Dimension-wise variance
    dim_var = X_patients.var(axis=0)
    axes[0, 1].plot(dim_var, color='darkorange', linewidth=2)
    axes[0, 1].set_xlabel('Dimension', fontsize=11)
    axes[0, 1].set_ylabel('Variance', fontsize=11)
    axes[0, 1].set_title('Variance per Embedding Dimension', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Mean per dimension for each class
    mean_healthy = X_patients[y_patients == 0].mean(axis=0)
    mean_disease = X_patients[y_patients == 1].mean(axis=0)
    
    axes[1, 0].plot(mean_healthy, label='Healthy', alpha=0.7, linewidth=2)
    axes[1, 0].plot(mean_disease, label='Disease', alpha=0.7, linewidth=2)
    axes[1, 0].set_xlabel('Dimension', fontsize=11)
    axes[1, 0].set_ylabel('Mean Value', fontsize=11)
    axes[1, 0].set_title('Mean Embedding per Class', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Difference between classes
    diff = np.abs(mean_healthy - mean_disease)
    top_dims = np.argsort(diff)[-20:]  # Top 20 discriminative dimensions
    
    axes[1, 1].barh(range(20), diff[top_dims], color='purple', alpha=0.7)
    axes[1, 1].set_xlabel('Absolute Difference', fontsize=11)
    axes[1, 1].set_ylabel('Dimension', fontsize=11)
    axes[1, 1].set_title('Top 20 Discriminative Dimensions', fontsize=12, fontweight='bold')
    axes[1, 1].set_yticks(range(20))
    axes[1, 1].set_yticklabels([f'Dim {d}' for d in top_dims])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset}_{model}_embedding_distributions.png'), dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {dataset}_{model}_embedding_distributions.png")
    plt.close()
    
    # ==========================================
    # Plot 3: PCA visualization
    # ==========================================
    print("\n  Running PCA (2D projection)...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_patients)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[y_patients == 0, 0], X_pca[y_patients == 0, 1], 
                         c='steelblue', alpha=0.6, s=50, label='Healthy', edgecolors='black', linewidth=0.5)
    scatter = plt.scatter(X_pca[y_patients == 1, 0], X_pca[y_patients == 1, 1], 
                         c='coral', alpha=0.6, s=50, label='Disease', edgecolors='black', linewidth=0.5)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
    plt.title(f'PCA: Patient Embeddings ({model})', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset}_{model}_pca.png'), dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {dataset}_{model}_pca.png")
    plt.close()
    
    # ==========================================
    # Plot 4: t-SNE visualization
    # ==========================================
    print("  Running t-SNE (2D projection)... (this may take a minute)")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_patients)-1))
    X_tsne = tsne.fit_transform(X_patients)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(X_tsne[y_patients == 0, 0], X_tsne[y_patients == 0, 1], 
               c='steelblue', alpha=0.6, s=50, label='Healthy', edgecolors='black', linewidth=0.5)
    plt.scatter(X_tsne[y_patients == 1, 0], X_tsne[y_patients == 1, 1], 
               c='coral', alpha=0.6, s=50, label='Disease', edgecolors='black', linewidth=0.5)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.title(f't-SNE: Patient Embeddings ({model})', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset}_{model}_tsne.png'), dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {dataset}_{model}_tsne.png")
    plt.close()
    
    # ==========================================
    # Plot 5: Explained variance (PCA)
    # ==========================================
    pca_full = PCA(random_state=42)
    pca_full.fit(X_patients)
    
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(pca_full.explained_variance_ratio_[:50], marker='o', linewidth=2, markersize=4)
    axes[0].set_xlabel('Principal Component', fontsize=11)
    axes[0].set_ylabel('Explained Variance Ratio', fontsize=11)
    axes[0].set_title('Scree Plot (First 50 PCs)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(cumvar, linewidth=2, color='darkorange')
    axes[1].axhline(y=0.95, color='red', linestyle='--', label='95% variance')
    axes[1].set_xlabel('Number of Components', fontsize=11)
    axes[1].set_ylabel('Cumulative Explained Variance', fontsize=11)
    axes[1].set_title('Cumulative Explained Variance', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Find number of components for 95% variance
    n_95 = np.argmax(cumvar >= 0.95) + 1
    axes[1].axvline(x=n_95, color='red', linestyle='--', alpha=0.5)
    axes[1].text(n_95+2, 0.5, f'{n_95} components\nfor 95% variance', fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset}_{model}_explained_variance.png'), dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {dataset}_{model}_explained_variance.png")
    plt.close()
    
    print(f"\n✅ All visualizations saved to: {output_dir}/")


def main():
    """
    Main function to run complete EDA.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Explore embeddings and patient data")
    parser.add_argument('--dataset', type=str, default='GSE54514', help='Dataset name')
    parser.add_argument('--model', type=str, default='ComplEx', choices=['ComplEx', 'RGCN'],
                        help='Embedding model to analyze')
    parser.add_argument('--output_dir', type=str, default='eda',
                        help='Output directory for visualizations')
    args = parser.parse_args()
    
    # Analyze embeddings
    data = analyze_embeddings(args.dataset, args.model)
    
    # Create visualizations
    visualize_embeddings(data, args.dataset, args.model, args.output_dir)
    
    print("\n" + "="*80)
    print("✅ EXPLORATORY DATA ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved in: {args.output_dir}/")
    print("\nNext steps:")
    print("  1. Review the visualizations to understand data structure")
    print("  2. Run: python train_classifier.py --dataset GSE54514 --model ComplEx")
    print("  3. Compare ComplEx vs RGCN embeddings")
    print("="*80)


if __name__ == "__main__":
    main()
