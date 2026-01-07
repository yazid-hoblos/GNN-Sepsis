'''
utils.py contains functions to perform redundant tasks such as:
traininf a model on all datasets, training all models on all datasets, setting number of threads, loading hyperparameters from JSON file

Functions:
- train_one(dataset, model,version)
- train_all(datasets, model_types, version,cache_dir, split_ratio, random_state)
- set_num_threads(num_threads)
- load_hyperparameters(hyperparam_file)
- load_models(dump_dir,version)
'''

import os

from src.ml.load_matrix import load_df
from src.ml.model_trainer import MLModel
from joblib import Parallel, delayed, dump, load
import json

NUM_THREADS=8


REQUIRED_HYPERPARAM_KEYS = {
    "SVM_HYPERPARAMS",
    "RANDOM_FOREST_HYPERPARAMS",
    "XGBOOST_HYPERPARAMS",
    "PYTORCH_MLP_HYPERPARAMS",
    "SKLEARN_MLP_HYPERPARAMS"
}

def load_hyperparameters(hyperparam_file):
    """
    validate and load hyperparameter JSON file
    """

    if hyperparam_file is None:
        return None
    if not os.path.exists(hyperparam_file):
        raise FileNotFoundError(
            f"Hyperparameter file not found: {hyperparam_file}"
        )
    if not os.path.isfile(hyperparam_file):
        raise ValueError(
            f"Provided hyperparameter path is not a file: {hyperparam_file}"
        )
    try:
        with open(hyperparam_file, "r") as f:
            hyperparams = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse JSON in hyperparameter file '{hyperparam_file}': {e}"
        )
    missing = REQUIRED_HYPERPARAM_KEYS - set(hyperparams.keys())
    if missing:
        raise ValueError(
            f"Hyperparameter file '{hyperparam_file}' is missing required keys: "
            f"{', '.join(missing)}"
        )
    return hyperparams

def train_one(dataset, model,version,normalization):
    pid = os.getpid()
    print(f"[PID {pid}] Training model: {model}  on dataset: {dataset} (version={version})")
    df = load_df(dataset,folder_version=version, normalization=normalization)
    ml_model = MLModel(model_type=model, df=df, dataset_name=dataset,save_model=True, version=version,normalization=normalization)
    ml_model.train_evaluate()

    return 

def train_all(datasets:list=['gene_expression', 'RGCN_sample_embeddings', 'Complex_sample_embeddings', 'concatenated_sample_embeddings', 'RGCN_protein_embeddings', 'Complex_protein_embeddings', 'concatenated_protein_embeddings'],
              model_types=MLModel.AVAILABLE_MODELS,
              version='v2.10',cache_dir='../../dump/',
              normalization="robust",
              split_ratio=0.3, random_state=42):
    
    MLModel.CACHE_DIR=cache_dir
    MLModel.DEFAULT_SAVE=True
    MLModel.SPLIT_RATIO=split_ratio
    MLModel.RANDOM_STATE=random_state
    MLModel.DEFAULT_LOGGING=True

    from datetime import datetime
    date=datetime.now().strftime("%Y%m%d_%H%M%S")
    MLModel.set_global_variable("SYSOUT_FILE",f"train_all_{version}_{normalization}_{date}_training_utils.log") # -- since parallelized better not have a file for each model/dataset
    print(f"-- utils.train_all called with version={version}, normalization={normalization}, cache_dir={cache_dir} --")

    results = Parallel(n_jobs=NUM_THREADS, backend='threading', verbose=10)(
        delayed(train_one)(dataset, model,version, normalization)
        for dataset in datasets
        for model in model_types
    )
    return 

def set_num_threads(num_threads:int):
    global NUM_THREADS
    NUM_THREADS=num_threads
    print(f'-- set NUM_THREADS to {NUM_THREADS}')


def load_models(dump_dir:str,version:str,normalization:str="robust"):
    '''
    loads all trained models for a specific version from dump_dir/version/
    returns a dict of {model_name: MLModel instance}
    '''
    version_norm=f"{version}_{normalization}"
    version_dir=os.path.join(dump_dir,version_norm)
    model_files=[f for f in os.listdir(version_dir) if f.endswith('.joblib')]
    models_dict={}
    for mf in model_files:
        model_path=os.path.join(version_dir,mf)
        ml_model=load(model_path)
        models_dict[mf]=ml_model
    return models_dict