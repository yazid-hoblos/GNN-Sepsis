#!/usr/bin/env python

# python -m src.eda.umap

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from ..ml.load_matrix import load_df

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, default='v2.10', help='Data version to load')
parser.add_argument('--save_dir', type=str, default='figures/projections/umap', help='Directory to save UMAP plots')
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)

gene_exp_df = load_df('gene_expression', folder_version=args.version)
complex_protein_df = load_df('Complex_protein_embeddings', folder_version=args.version)
rgcn_protein_df = load_df('RGCN_protein_embeddings', folder_version=args.version)
complex_sample_df = load_df('Complex_sample_embeddings', folder_version=args.version)
rgcn_sample_df = load_df('RGCN_sample_embeddings', folder_version=args.version)
concatenated_protein_df = load_df('concatenated_protein_embeddings', folder_version=args.version)
concatenated_sample_df = load_df('concatenated_sample_embeddings', folder_version=args.version)

def plot_umap(df, title, save_dir):
    y = df['disease_status']
    X = df.drop(columns=['disease_status'])
    X_scaled = StandardScaler().fit_transform(X)
    X_umap = UMAP(n_components=2, random_state=42).fit_transform(X_scaled)
    plot_df = df.copy()
    plot_df['UMAP1'] = X_umap[:, 0]
    plot_df['UMAP2'] = X_umap[:, 1]

    plt.figure(figsize=(10, 8))
    sns.set(style='whitegrid', font_scale=1.2)
    palette = sns.color_palette("Set2")
    scatter = sns.scatterplot(
        data=plot_df,
        x='UMAP1',
        y='UMAP2',
        hue='disease_status',
        palette=palette,
        s=120,
        alpha=0.9,
        edgecolor='k',
        linewidth=0.7
    )

    plt.title(title + ' UMAP', fontsize=18, weight='bold')
    plt.xlabel('UMAP1', fontsize=14)
    plt.ylabel('UMAP2', fontsize=14)
    plt.legend(title='Disease Status', fontsize=12, title_fontsize=13, loc='best', frameon=True, shadow=True)
    sns.despine(trim=True)
    plt.tight_layout()
    save_path = f'{save_dir}/{title.replace(" ", "_").lower()}.png'
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"-- saved UMAP plot for {title} at {save_path} --")
    

plot_umap(gene_exp_df, 'Gene Expression', args.save_dir)
plot_umap(complex_protein_df, 'Complex Protein Embeddings', args.save_dir)
plot_umap(rgcn_protein_df, 'RGCN Protein Embeddings', args.save_dir)
plot_umap(complex_sample_df, 'Complex Sample Embeddings', args.save_dir)
plot_umap(rgcn_sample_df, 'RGCN Sample Embeddings', args.save_dir)
plot_umap(concatenated_protein_df, 'Concatenated Protein Embeddings', args.save_dir)
plot_umap(concatenated_sample_df, 'Concatenated Sample Embeddings', args.save_dir)
