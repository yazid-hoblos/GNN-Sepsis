#!/usr/bin/env python

'''
-- Training combinations of ML models on combinations of datasets for all specified versions --

by default, trains all available models on all available datasets for versions v2.10 and v2.11

AVAILABLE OPTIONS:  
    model-types: svm, random_forest, xgboost, pytorch_mlp, sklearn_mlp
    datasets: gene_expression, RGCN_sample_embeddings, Complex_sample_embeddings, concatenated_sample_embeddings, RGCN_protein_embeddings, Complex_protein_embeddings, concatenated_protein_embeddings
    versions: v2.10, v2.11, v2.9
    normalization: robust, standard, minmax, log1p, none

run this script to train all ML models on all datasets for all specified versions
saves trained models in joblib files in dump/{version}/ folder

> [!TIP]
> run it from the project root as:

    python -m src.ml.train_all -h  
    python -m src.ml.train_all      # -- which is equivalent to:  
    python -m src.ml.train_all --versions v2.10 v2.11 --dump-dir ./dump/ --threads 12  
    
    python -m src.ml.train_all --versions v2.10 --model-types svm
    python -m src.ml.train_all --versions v2.10 --datasets gene_expression RGCN_sample_embeddings --model-types random_forest
'''


import os
import argparse
import textwrap
# -- package relative imports are essential when running as module --

# from src.ml.load_matrix import load_df
from src.ml.model_trainer import MLModel
from src.ml.utils import train_all, set_num_threads,load_hyperparameters

import os


def get_args():
    '''reads command line arguments'''
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(__doc__),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--versions", nargs="+", default=["v2.10", "v2.11"])
    parser.add_argument("--normalization", default="robust")
    parser.add_argument("--logging", action="store_true", help="Whether to enable logging to file")
    parser.add_argument("--cache-dir", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../dump")))
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--model-types", nargs="+", default=["svm", "random_forest", "xgboost", "sklearn_mlp"])
    parser.add_argument("--datasets", nargs="+", default=["gene_expression", "RGCN_sample_embeddings", "Complex_sample_embeddings", "concatenated_sample_embeddings", "RGCN_protein_embeddings", "Complex_protein_embeddings", "concatenated_protein_embeddings"])
    parser.add_argument("--split-ratio", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--hyperparameter-file", type=str, default=None, help="Path to a JSON file containing hyperparameter grids for models")
    parser.add_argument("--default-logging", action="store_true", help="Whether to enable default logging to file")
    parser.add_argument("--kfold", type=int, default=3, help="Number of folds for k-fold cross-validation")
    parser.add_argument("--dump-dir", type=str, default="./dump/", help="Directory where trained models will be saved")

    return parser.parse_args()



def main():
    args = get_args()
    set_num_threads(args.threads)
    os.makedirs(args.dump_dir, exist_ok=True)
    MLModel.DEFAULT_LOGGING=args.logging
    MLModel.HYPERPARAMS_FILE=args.hyperparameter_file
    MLModel.SPLIT_RATIO=args.split_ratio
    MLModel.RANDOM_STATE=args.random_state
    MLModel.K_FOLD=args.kfold

    hyperparam_dict = load_hyperparameters(args.hyperparameter_file)
    MLModel.HYPERPARAMS = hyperparam_dict
    for dataset in args.datasets:
        print(f'-- datasets to be used: {args.datasets} --')
    for model in args.model_types:
        print(f'-- model types to be used: {args.model_types} --')
    for norm in args.normalization:
        print(f'-- normalization to be used: {args.normalization} --')


    for version in args.versions:
        os.makedirs(args.dump_dir, exist_ok=True)
        print(f"-- Training all models for version: {version} (with {args.normalization} normalization)--")
        train_all(version=version, cache_dir=args.cache_dir,model_types=args.model_types,datasets=args.datasets,normalization=args.normalization)

if __name__ == "__main__":
    main()
