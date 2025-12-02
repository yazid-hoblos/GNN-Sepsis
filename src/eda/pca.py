#!/usr/bin/env python

# python -m src.eda.pca

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from ..ml.load_matrix import load_df

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, default='v2.10', help='Data version to load')
parser.add_argument('--save_dir', type=str, default='figures/projections/pca', help='Directory to save PCA plots')
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)

gene_exp_df = load_df('gene_expression', folder_version=args.version)
complex_protein_df = load_df('Complex_protein_embeddings', folder_version=args.version)
rgcn_protein_df = load_df('RGCN_protein_embeddings', folder_version=args.version)
complex_sample_df = load_df('Complex_sample_embeddings', folder_version=args.version)
rgcn_sample_df = load_df('RGCN_sample_embeddings', folder_version=args.version)
concatenated_protein_df = load_df('concatenated_protein_embeddings', folder_version=args.version)
concatenated_sample_df = load_df('concatenated_sample_embeddings', folder_version=args.version)

def plot_pca(df, title):
    X = df.drop(columns=['disease_status'])
    y = df['disease_status']
    X_scaled = StandardScaler().fit_transform(X)
    X_pca = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'], index=df.index)
    pca_df['disease_status'] = y.values

    plt.figure(figsize=(10, 8))
    sns.set(style='whitegrid', font_scale=1.2)
    palette = sns.color_palette("Set2")
    scatter = sns.scatterplot(
        data=pca_df,
        x='PC1',
        y='PC2',
        hue='disease_status',
        palette=palette,
        s=120,
        alpha=0.9,
        edgecolor='k',
        linewidth=0.7
    )

    plt.title(title + ' PCA', fontsize=18, weight='bold')
    plt.xlabel('PC1', fontsize=14)
    plt.ylabel('PC2', fontsize=14)
    plt.legend(title='Disease Status', fontsize=12, title_fontsize=13, loc='best', frameon=True, shadow=True)
    sns.despine(trim=True)
    plt.tight_layout()
    save_path = f'{args.save_dir}/{title.replace(" ", "_").lower()}.png'
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"-- saved PCA plot for {title} at {save_path} --")

plot_pca(gene_exp_df, 'Gene Expression')
plot_pca(complex_protein_df, 'Complex Protein Embeddings')
plot_pca(rgcn_protein_df, 'RGCN Protein Embeddings')
plot_pca(complex_sample_df, 'Complex Sample Embeddings')
plot_pca(rgcn_sample_df, 'RGCN Sample Embeddings')
plot_pca(concatenated_protein_df, 'Concatenated Protein Embeddings')
plot_pca(concatenated_sample_df, 'Concatenated Sample Embeddings')
print(f"-- PCA plots saved in directory: {args.save_dir}")