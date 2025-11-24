# -- short numpy docstring explaining the purpose of the model and the main 2 functions --
'''
this module provides functions to define, train, and evaluate machine learning models using:
- scikit-learn SVM
- xgboost XGBClassifier
- scikit-learn MLPClassifier

this is the 2nd step after performing data loading and preprocessing:  
INPUT: pandas DataFrame with features and 'disease_status' label
OUTPUT: trained model(s) and evaluation predictions

e.g., code snippets to train and evaluate a model:

train and evaluate at the same time (better)
>>> model, y_test, y_pred, y_proba = train_evaluate_model(df, 'svm') 
- model: trained GridSearchCV object
- y_test: true labels for test set
- y_pred: predicted labels for test set
- y_proba: predicted probabilities for positive class for test set

If you wanna train all the models at once (using one function call):
>>> models, y_test, y_pred, y_proba = train_evaluate_model(df, 'all')
This will return dictionaries of models, y_test, y_pred, y_proba for each model type, 
each 1 of the 4 returned dictionaries will be of the shape: 
- models: { 'svm':GridSearchCV, 'xgboost':GridSearchCV, 'mlp':GridSearchCV }
- y_test: { 'svm':array, 'xgboost':array, 'mlp':array }
- y_pred: { 'svm':array, 'xgboost':array, 'mlp':array }
- y_proba: { 'svm':array, 'xgboost':array, 'mlp':array }

or to train and evaluate separately (!need to be careful to provide the same train/test split and random state!)
>>> model=train_model(df,'svm')
>>> y_test, y_pred, y_proba = evaluate_model(model, df)

Note that we use defualt values for hyperparameter grids, cross-validation folds, test split ratio, random state, and scoring metric (accuracy - for gridsearch)
You can change all of these by providing optional arguments to the functions:

>>> model=train_model(df,'svm', hyperparameters={'C':[0.1,1,10]}, split_ratio=0.3, random_state=123, kfold=5)

'''

# - [x] make a pretty print function for the hyperparameters dictionary and the best model
# - [x] GLOBAL variables to be added instead of hardcoded values
# - [x] add docstrings to functions
# - [x] make some functions private if not needed to be used outside the module
# - [ ] make it take json or yaml hyperparameter inputs and parse them to proper types
# - [ ] add option to save trained model to file
# - [ ] logging

import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost
import numpy as np
import pandas as pd
import pprint
# import joblib

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# -- GLOBAL CONSTANTS --
DEFAULT_KFOLD = 3
DEFAULT_SPLIT_RATIO = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_SCORING = "accuracy"
DEFAULT_SAVE_MODEL_PATH = None

pp = pprint.PrettyPrinter(indent=4)

# -- DEFAULT HYPERPARAMETER GRIDS --
SVM_HYPERPARAMS = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}
XGBOOST_HYPERPARAMS = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 0.2]}
MLP_HYPERPARAMS = {'hidden_layer_sizes': [(50,), (100,), (50,50)], 'activation': ['relu', 'tanh'], 'solver': ['adam', 'sgd'], 'learning_rate_init': [0.001, 0.01, 0.1]}

# -- MODEL REGISTRY --
MODELS = {'svm': SVC(), 'xgboost': xgboost.XGBClassifier(), 'mlp': MLPClassifier()}

# -- PRIVATE HELPERS --
def _pretty_print_dict(title: str, d: dict):
    """
    pretty print a dictionary with a title

    title : str, title printed above the dictionary
    d : dict, dictionary to print
    """
    print(f"\n-- {title} --")
    pp.pprint(d)
    print("\n")

def _validate_hyperparameters(model_type: str, hyperparameters: dict):
    """
    validate that provided hyperparameters exist for the requested model type

    model_type : str, model identifier ('svm', 'xgboost', 'mlp')
    hyperparameters : dict, hyperparameter dictionary to validate
    """
    model_type = model_type.lower()
    if model_type not in MODELS:
        raise ValueError(f"-- model type '{model_type}' is not supported: choose from {list(MODELS.keys())} --")
    valid_params = MODELS[model_type].get_params().keys()
    for param in hyperparameters.keys():
        if param not in valid_params:
            raise ValueError(f"-- hyperparameter '{param}' is invalid for '{model_type}'. valid keys: {list(valid_params)} --")

def _json_to_hyperparameters(json_dict: dict) -> dict:
    """
    convert json-style values to proper python hyperparameter types

    json_dict : dict, dictionary where values may be string-formatted lists
    returns : dict, parsed hyperparameter dictionary
    """
    hyperparameters = {}
    for key, value in json_dict.items():
        if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
            value = value.strip("[]").split(",")
            value = [v.strip().strip("'").strip('"') for v in value]
        hyperparameters[key] = value
    return hyperparameters

# -- MODEL DEFINITIONS --
def _define_svm(grid_params=None, kfold=DEFAULT_KFOLD):
    """
    construct a gridsearchcv object for an svm classifier

    grid_params : dict, optional, hyperparameter grid for svm
    kfold : int, optional, number of cross-validation folds
    returns : GridSearchCV, configured gridsearchcv with SVC
    """
    if grid_params:
        _validate_hyperparameters("svm", grid_params)
    else:
        grid_params = SVM_HYPERPARAMS
    return GridSearchCV(estimator=SVC(probability=True), param_grid=grid_params,
                        scoring=DEFAULT_SCORING, cv=kfold, n_jobs=-1)

def _define_xgboost(grid_params=None, kfold=DEFAULT_KFOLD):
    """
    construct a gridsearchcv object for an xgboost classifier

    grid_params : dict, optional, hyperparameter grid
    kfold : int, optional, number of cross-validation folds
    returns : GridSearchCV, configured gridsearchcv with XGBClassifier
    """
    if grid_params:
        _validate_hyperparameters("xgboost", grid_params)
    else:
        grid_params = XGBOOST_HYPERPARAMS
    return GridSearchCV(estimator=xgboost.XGBClassifier(eval_metric="logloss"),
                        param_grid=grid_params, scoring=DEFAULT_SCORING, cv=kfold, n_jobs=-1)

def _define_mlp(grid_params=None, kfold=DEFAULT_KFOLD):
    """
    construct a gridsearchcv object for an mlp classifier

    grid_params : dict, optional, hyperparameter grid
    kfold : int, optional, number of folds for cross-validation
    returns : GridSearchCV, configured gridsearchcv with MLPClassifier
    """
    if grid_params:
        _validate_hyperparameters("mlp", grid_params)
    else:
        grid_params = MLP_HYPERPARAMS
    return GridSearchCV(estimator=MLPClassifier(max_iter=500), param_grid=grid_params,
                        scoring=DEFAULT_SCORING, cv=kfold, n_jobs=-1)

def define_model(model_type:str, hyperparameters:dict=None):
    """
    define and configure a model wrapped in gridsearchcv

    model_type : str, model identifier 'svm', 'xgboost', 'mlp', 'all'
    hyperparameters : dict, optional, hyperparameter grid
    returns : GridSearchCV or dict of GridSearchCV
    """
    model_type = model_type.lower()
    # if model_type == "all":
    #     return {name: define_model(name, hyperparameters) for name in MODELS.keys()}
    if model_type not in MODELS:
        raise ValueError(f"-- model type '{model_type}' is not supported: choose from {list(MODELS.keys())} --")
    if model_type == 'svm':
        return _define_svm(hyperparameters)
    if model_type == 'xgboost':
        return _define_xgboost(hyperparameters)
    if model_type == 'mlp':
        return _define_mlp(hyperparameters)

# -- TRAIN / EVALUATION --
def train_model(df: pd.DataFrame, model_type:str, hyperparameters: dict =None, split_ratio:float=DEFAULT_SPLIT_RATIO, random_state:int =DEFAULT_RANDOM_STATE,save_path:str = DEFAULT_SAVE_MODEL_PATH):
    """
    train a model using gridsearchcv and a train/test split

    df : pandas.DataFrame, input dataframe with features and 'disease_status'
    model_type : str, model identifier 'svm', 'xgboost', 'mlp', 'all'
    hyperparameters : dict, optional, hyperparameter grid
    split_ratio : float, optional, test set proportion
    random_state : int, optional, seed for reproducible split
    save_path : str, optional, file path to save trained model
    returns : GridSearchCV, trained model
    """
    y = df["disease_status"].astype(int).values
    X = df.drop(columns=["disease_status"]).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=random_state)
    print(f"-- split of {split_ratio} for test set with random state {random_state}")

    if model_type=="all":
        models={}
        for i,name in enumerate(MODELS.keys()):
            print(f"\n-- training model {name} ({i+1}/{len(MODELS)})")
            models[name] = train_model(df, name, hyperparameters, split_ratio, random_state, save_path)
        return models
         

    model = define_model(model_type, hyperparameters)
    print(f"-- model defined {model_type} with:")
    if hyperparameters:
        _pretty_print_dict("hyperparameters", hyperparameters)
    else:
        print("-- default hyperparameters for grid search")

    model.fit(X_train, y_train)
    print(f"-- {model_type.upper()} trained on training set")

    # if save_path:
    #     joblib.dump(model, save_path)
    #     print(f"-- model saved to {save_path}")

    return model

def evaluate_model(grid_search: GridSearchCV, df: pd.DataFrame,
                   split_ratio: float = DEFAULT_SPLIT_RATIO, random_state: int = DEFAULT_RANDOM_STATE):
    """
    evaluate a trained gridsearchcv model on a test split

    grid_search : GridSearchCV, trained model
    df : pandas.DataFrame, input dataframe with features and 'disease_status'
    split_ratio : float, optional, test set proportion
    random_state : int, optional, seed for reproducible split
    returns : y_test, y_pred, y_proba
    """
    y = df["disease_status"].astype(int).values
    X = df.drop(columns=["disease_status"]).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=random_state)

    if type(grid_search) is dict:
        Y_test,Y_pred,Y_proba={},{},{}
        for name, model in grid_search.items():
            print(f"\n-- Evaluating model: {name} --")
            _pretty_print_dict("best parameters", model.best_params_)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            print("-- predictions made on test set")
            Y_test[name]=y_test
            Y_pred[name]=y_pred
            Y_proba[name]=y_proba
        return Y_test, Y_pred, Y_proba

    # -- else single model 
    best_model = grid_search.best_estimator_
    print("-- best model from grid search:")
    _pretty_print_dict("best parameters", grid_search.best_params_)
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    print("-- predictions made on test set")

    return y_test, y_pred, y_proba

def train_evaluate_model(df: pd.DataFrame, model_type: str, hyperparameters: dict = None,
                         split_ratio: float = DEFAULT_SPLIT_RATIO, random_state: int = DEFAULT_RANDOM_STATE):
    """
    train a model and immediately evaluate it

    df : pandas.DataFrame, input dataframe with features and labels
    model_type : str, model identifier (so far: 'svm', 'xgboost', 'mlp', 'all')
    hyperparameters : dict, optional, hyperparameter grid
    split_ratio : float, optional, test set proportion
    random_state : int, optional, seed for reproducible split
    returns : model, y_test, y_pred, y_proba
    """
    model = train_model(df, model_type, hyperparameters, split_ratio, random_state)
    y_test, y_pred, y_proba = evaluate_model(model, df, split_ratio, random_state)
    return model, y_test, y_pred, y_proba
