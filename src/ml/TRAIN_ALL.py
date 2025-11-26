#!/bin/env/python

from load_matrix import load_df
from training_utils import MLModel
import argparse

prepare_df = lambda df: df.set_index('label').iloc[:, 3:]

def train_all(datasets:list=['gene_expression', 'RGCN_sample_embeddings', 'Complex_sample_embeddings', 'concatenated_sample_embeddings', 'RGCN_protein_embeddings', 'Complex_protein_embeddings', 'concatenated_protein_embeddings'],model_types=MLModel.MODELS):
    trained_MLModels={}

    for dataset in datasets:
        print('dataset preping:', dataset)
        df=prepare_df( load_df(dataset))
        print('preppeddd')
        for model in models:
            ml_model=MLModel(model_type=model, df=df,dataset_name=f'{model}_{dataset}')
            ml_model.train_evaluate()
            trained_MLModels[f'{model}_{dataset}']=ml_model
    return trained_MLModels

def read_arguments():
    
    parser = argparse.ArgumentParser(description="Train ML models on various datasets.")
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['gene_expression', 'RGCN_sample_embeddings', 'Complex_sample_embeddings', 'concatenated_sample_embeddings', 'RGCN_protein_embeddings', 'Complex_protein_embeddings', 'concatenated_protein_embeddings'],
        help='list of datasets to train models on'
    )
    #         'DEFAULT_KFOLD': lambda v: setattr(cls, 'DEFAULT_KFOLD', v),
    #         'DEFAULT_SPLIT_RATIO': lambda v: setattr(cls, 'DEFAULT_SPLIT_RATIO', v),
    #         'DEFAULT_RANDOM_STATE': lambda v: setattr(cls, 'DEFAULT_RANDOM_STATE', v),
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
    ALL_MODELS = train_all(datasets=args.datasets, models=args.models)
    # -- need a way to save tehm all tho or diretly perform analysis here