'''
utils.py contains functions to perform redundant tasks such as:
* train a specific model on a specific dataset for a specific version: --> MLModel/joblib file
* train all models on all datasets for a specififc version: ->{name:MLModel}/joblib dir
* run results on a set of trained models: -> pd.DataFrame
'''

import os

from src.ml.load_matrix import load_df
from src.ml.model_trainer import MLModel
from joblib import Parallel, delayed, dump

NUM_THREADS=8

def train_one(dataset, model):

    if dataset in ["RGCN_protein_embeddings", "concatenated_protein_embeddings"] and model=='svm':
        print('-- skipping SVM on RGCN protein features for now --')
        return
    pid = os.getpid()
    print(f"[PID {pid}] Training model: {model}  on dataset: {dataset}")
    df = load_df(dataset,folder_version=MLModel.VERSION)
    ml_model = MLModel(model_type=model, df=df, dataset_name=dataset,save_model=True)
    ml_model.train_evaluate()

    return 

def train_all(datasets:list=['gene_expression', 'RGCN_sample_embeddings', 'Complex_sample_embeddings', 'concatenated_sample_embeddings', 'RGCN_protein_embeddings', 'Complex_protein_embeddings', 'concatenated_protein_embeddings'],
              model_types=MLModel.AVAILABLE_MODELS,
              version='v2.10',cache_dir='../../dump/',
              split_ratio=0.3, random_state=42):
    
    MLModel.CACHE_DIR=cache_dir
    MLModel.DEFAULT_SAVE=True
    MLModel.VERSION=version
    MLModel.SPLIT_RATIO=split_ratio
    MLModel.RANDOM_STATE=random_state
    MLModel.DEFAULT_LOGGING=True

    from datetime import datetime
    date=datetime.now().strftime("%Y%m%d_%H%M%S")
    MLModel.set_global_variable("SYSOUT_FILE",f"train_all_{version}_{date}_training_utils.log") # -- since parallelized better not have a file for each model/dataset

    results = Parallel(n_jobs=NUM_THREADS, backend='threading', verbose=10)(
        delayed(train_one)(dataset, model)
        for dataset in datasets
        for model in model_types
    )
    return results

def set_num_threads(num_threads:int):
    global NUM_THREADS
    NUM_THREADS=num_threads
    print(f'-- set NUM_THREADS to {NUM_THREADS}')