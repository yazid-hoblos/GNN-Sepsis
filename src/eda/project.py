#!/usr/bin/env python

'''
this uses load_df with normalization when computing the protein embeddings
If wanna plot pca/tsne/umap without normalization, use the scripts pca, tsne, umap in src/eda 
'''


import argparse
from ..ml.load_matrix import load_df
from .projection_plotting import plot_pca, plot_tsne, plot_umap

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, default='v2.10')
parser.add_argument('--normalization', type=str, default='robust')
parser.add_argument('--method', type=str, choices=['pca', 'tsne', 'umap'], default='pca')
parser.add_argument('--dataset_name', type=str, nargs='*', default=[
    'gene_expression',
    'Complex_protein_embeddings',
    'RGCN_protein_embeddings',
    'Complex_sample_embeddings',
    'RGCN_sample_embeddings',
    'concatenated_protein_embeddings',
    'concatenated_sample_embeddings'
])
parser.add_argument('--output_dir', type=str, default='results/figures/projections')
args = parser.parse_args()

method_map = {
    'pca': plot_pca,
    'tsne': plot_tsne,
    'umap': plot_umap
}

plot_func = method_map[args.method]

for name in args.dataset_name:  
    df = load_df(name, folder_version=args.version, normalization=args.normalization)
    title = name.replace('_', ' ').title()
    plot_func(df, title, f"{args.output_dir}/{args.method}_{args.version}_{args.normalization}")
    print(f"-- saved {args.method} plot for {name} in {args.output_dir}/{args.method}_{args.normalization}/title")
