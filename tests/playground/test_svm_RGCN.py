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
    dump_dir=os.path.join(dump_dir, version)
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


if __name__ == "__main__":



    l1=['RGCN_sample_embeddings', 'Complex_sample_embeddings', 'RGCN_protein_embeddings', 'Complex_protein_embeddings']
    l2=['gene_expression', 'concatenated_sample_embeddings', 'concatenated_protein_embeddings']
    l=['RGCN_sample_embeddings',
        'Complex_sample_embeddings',
        'RGCN_protein_embeddings',
        'Complex_protein_embeddings',
        'gene_expression',
        'concatenated_sample_embeddings',
        'concatenated_protein_embeddings']

    # -- testing examples
    # train_all(datasets=l1)
    train_one('RGCN_protein_embeddings', 'svm', version='v2.10', dump_dir='dump/v2.10/')
    # train_one('concatenated_protein_embeddings', 'svm')