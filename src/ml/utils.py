'''
utils.py contains functions to perform redundant tasks such as:
* train a specific model on a specific dataset for a specific version: --> MLModel/joblib file
* train all models on all datasets for a specififc version: ->{name:MLModel}/joblib dir
* run results on a set of trained models: -> pd.DataFrame
'''

from load_matrix import load_df
from model_trainer import MLModel
import os
from joblib import Parallel, delayed,dump


def train_one(dataset, model,version='v2.10',dump_dir=None):

    if dataset in ["RGCN_protein_embeddings", "concatenated_protein_embeddings"] and model=='svm':
        print('-- skipping SVM on RGCN protein features for now --')
        return
    pid = os.getpid()
    print(f"[PID {pid}] Training model: {model} on dataset: {dataset}")
    df = load_df(dataset,folder_version=version)
    ml_model = MLModel(model_type=model, df=df, dataset_name=f'{model}_{dataset}')
    ml_model.train_evaluate()

    joblib_path = os.path.join(dump_dir, f'{model}_{dataset}_MLmodel.joblib')
    if dump_dir is not None:
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