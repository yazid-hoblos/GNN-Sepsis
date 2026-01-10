#!/bin/env/python

import os
os.chdir('../../src/ml')

from load_matrix import load_df
from model_trainer import MLModel
import argparse
from joblib import Parallel, delayed,dump

dump_dir='dump/'
MLModel.DEFAULT_SAVE=False  # -- disable default saving after training, will be doing this manually here for testing purposes


# -- parallel training
# -- also found in notebooks/train_all_parallel.ipynb

def train_one(dataset, model,version='v2.10',dump_dir=dump_dir):
    if dataset in ["RGCN_protein_embeddings", "concatenated_protein_embeddings"] and model=='svm':
        print('-- skipping SVM on RGCN protein features for now --')
        return
    pid = os.getpid()
    print(f"[PID {pid}] Training model: {model} on dataset: {dataset}")
    df = load_df(dataset,folder_version=version)
    ml_model = MLModel(model_type=model, df=df, dataset_name=f'{model}_{dataset}')
    ml_model.train_evaluate()

    joblib_path = os.path.join(dump_dir, f'{model}_{dataset}_MLmodel.joblib')
    dump(ml_model, joblib_path)
    print(f'[PID {pid}] Model saved to {joblib_path}')

    return 

def train_all(datasets:list=['gene_expression', 'RGCN_sample_embeddings', 'Complex_sample_embeddings', 'concatenated_sample_embeddings', 'RGCN_protein_embeddings', 'Complex_protein_embeddings', 'concatenated_protein_embeddings'],model_types=MLModel.MODELS,version='v2.10',dump_dir=dump_dir):
    results = Parallel(n_jobs=8)(
        delayed(train_one)(dataset, model)
        for dataset in datasets
        for model in model_types
    )
    return results



def read_arguments():
    
    parser = argparse.ArgumentParser(description="Train ML models on various datasets.")
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['gene_expression', 'RGCN_sample_embeddings', 'Complex_sample_embeddings', 'concatenated_sample_embeddings', 'RGCN_protein_embeddings', 'Complex_protein_embeddings', 'concatenated_protein_embeddings'],
        help='list of datasets to train models on'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=['svm','xgboost','mlp'],
        help='list of models to train'
    )
    parser.add_argument('--kfold', type=int, default=5, help='number of folds for cross-validation')
    parser.add_argument('--split_ratio', type=float, default=0.2, help='train-test split ratio')
    parser.add_argument('--random_state', type=int, default=42, help='random state for reproducibility')
    # -- help message for each argument
    # parser.add_argument('--help', action='help', help='''this script trains multiple ML models on specified datasets, it can customized with:
    # --datasets: list of datasets to train models on (default: all available datasets for load_df)
    # --models: list of models to train (default: svm, xgboost, mlp)
    # --kfold: number of folds for cross-validation (default: 3)
    # --split_ratio: train-test split ratio (default: 0.2)
    # --random_state: random state for reproducibility (default: 42)
    # ''')
    return parser.parse_args()

if __name__ == "__main__":
    args = read_arguments()
    if args.kfold:
        MLModel.set_global_variable('DEFAULT_KFOLD', args.kfold)
    if args.split_ratio:
        MLModel.set_global_variable('DEFAULT_SPLIT_RATIO', args.split_ratio)
    if args.random_state:
        MLModel.set_global_variable('DEFAULT_RANDOM_STATE', args.random_state)
    
    MLModel.set_global_variable('DEFAULT_KFOLD', args.kfold)
    MLModel.set_global_variable('DEFAULT_SPLIT_RATIO', args.split_ratio)
    MLModel.set_global_variable('DEFAULT_RANDOM_STATE', args.random_state)

    train_all(datasets=args.datasets, model_types=args.models)

    l1=['RGCN_sample_embeddings', 'Complex_sample_embeddings', 'RGCN_protein_embeddings', 'Complex_protein_embeddings']
    l2=['gene_expression', 'concatenated_sample_embeddings', 'concatenated_protein_embeddings']

    # -- testing examples
    # train_all(datasets=l1)
    # train_one('RGCN_protein_embeddings', 'svm')
    # train_one('concatenated_protein_embeddings', 'svm')