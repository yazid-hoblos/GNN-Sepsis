#!/usr/bin/env python

import argparse
import os
from ..ml.load_matrix import load_df
from .projection_plotting import plot_pca

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, default='v2.10')
parser.add_argument('--dataset_name', type=str, nargs='*', default=[
    'gene_expression',
    'Complex_protein_embeddings',
    'RGCN_protein_embeddings',
    'Complex_sample_embeddings',
    'RGCN_sample_embeddings',
    'concatenated_protein_embeddings',
    'concatenated_sample_embeddings'
])
parser.add_argument('--output_dir', type=str, default='figures/projections/pca')
args = parser.parse_args()

for name in args.dataset_name:
    os.makedirs(args.output_dir, exist_ok=True)
    df = load_df(name, folder_version=args.version)
    title = name.replace('_', ' ').title()
    plot_pca(df, title, args.output_dir)
    print(f"-- saved pca plot for {name} in {args.output_dir}")