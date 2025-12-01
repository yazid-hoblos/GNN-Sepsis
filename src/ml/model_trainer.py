"""
Classical models training util functions
----------------------------------------------

this module defines utilities to construct, train, and evaluate machine-learning classification models
using support vector machines (svm), xgboost classifiers, and multilayer perceptrons (mlp). it represents
the second step of the workflow, following data loading and preprocessing.

the module expects a pandas dataframe containing feature columns and a binary 'disease_status' label.
it returns trained model objects, predictions, and class probabilities.

It was redesigned in an oop fashion to encapsulate functionality within the `MLModel` class and store all
relevant attributes


example run
-------------
>>> model = MLModel(df, model_type="svm", dataset_name="my_data")
>>> model.train()
>>> y_test, y_pred, y_proba = model.evaluate()
>>> model.y_test, model.y_pred, model.y_proba  # equivalent to above

in a single step:

>>> model.train_evaluate()
>>> model.y_test, model.y_pred, model.y_proba

Notes
-----
The input DataFrame must contain a `disease_status` column, which will be used
as the target variable, and all remaining columns are treated as features (pay attention they should all be numeric)
"""

import os
import sys
import pprint
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost
from sklearn.neural_network import MLPClassifier # --> moving to pytorch
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)


class Logger:
    '''
    logger that duplicates stdout to a log file
    !issues! -- does not print when used in jupyter notebooks/ temporarily commented out in MLModel
    '''

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


class MLModel:
    '''
    Machine learning utility class for training and evaluating models with
    automatic hyperparameter tuning via GridSearchCV

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing features and a `disease_status` column.
    model_type : str
        Type of model: {'svm', 'xgboost', 'mlp', 'all'}.
    dataset_name : str
        Name of the dataset (used for saving models)
    version : str, optional
        Version identifier for the dataset of the format 'v2.x'
    hyperparameters : dict, optional
        Custom hyperparameters for the selected model.
    split_ratio : float, optional
        Fraction of data to use for testing (default is class attribute).
    random_state : int, optional
        Seed for reproducibility.
    save_model : bool, optional
        Whether to save trained GridSearchCV object to disk.

    Attributes
    ----------
    X_train, X_test : ndarray
        Training and test feature matrices.
    y_train, y_test : ndarray
        Training and test labels.
    model : GridSearchCV or None
        Fitted search object after calling `train()`.
    y_pred : ndarray or None
        Predictions after evaluation.
    y_proba : ndarray or None
        Predicted probabilities after evaluation.

    Global Class Attributes
    --------------------------
    DEFAULT_KFOLD : int
        Default number of folds for cross-validation.
    DEFAULT_SPLIT_RATIO : float
        Default test set fraction.
    DEFAULT_RANDOM_STATE : int
        Default random seed
    DEFAULT_SCORING : str
        Default scoring metric for GridSearchCV
    DEFAULT_SAVE : bool
        Default flag for saving trained models
    CACHE_DIR : str
        Directory to save cached models and logs
    SYSOUT_FILE : str
        Log file name for stdout redirection
    AVAILABLE_MODELS : set
        Supported model types
    SVM_HYPERPARAMS : dict
        Default hyperparameter grid for SVM
    XGBOOST_HYPERPARAMS : dict
        Default hyperparameter grid for XGBoost
    MLP_HYPERPARAMS : dict
        Default hyperparameter grid for MLP
    '''

    # ------------------ CLASS ATTRIBUTES ------------------
    DEFAULT_KFOLD = 3
    DEFAULT_SPLIT_RATIO = 0.2
    DEFAULT_RANDOM_STATE = 42
    DEFAULT_SCORING = "accuracy"
    DEFAULT_SAVE = True
    DEFAULT_LOGGING = False
    CACHE_DIR = '.cache/'
    SYSOUT_FILE = "training_utils.log"

    AVAILABLE_MODELS = {'svm', 'xgboost', 'mlp','random_forest'}
    SVM_HYPERPARAMS = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto'], 'class_weight': ['balanced', {0: 2, 1: 1}, {0: 3, 1: 1}]}
    XGBOOST_HYPERPARAMS = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7],
                           'learning_rate': [0.01, 0.1, 0.2], 'subsample': [0.6, 0.8, 1.0], 'scale_pos_weight': [36 / 127, 0.4, 0.2]}
    RANDOM_FOREST_HYPERPARAMS = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7],
                                'min_samples_split': [2, 5, 10], 'class_weight': ['balanced', 'balanced_subsample', {0: 2, 1: 1}, {0: 3, 1: 1}]}
    MLP_HYPERPARAMS = {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'activation': ['relu', 'tanh'],
                       'solver': ['adam', 'sgd'], 'learning_rate_init': [0.001, 0.01, 0.1], 'class_weight': ['balanced', {0: 2, 1: 1}, {0: 3, 1: 1}]}

    pp = pprint.PrettyPrinter(indent=4)

    @classmethod
    def get_global_variables(cls):
        return {
            'CACHE_DIR': cls.CACHE_DIR,
            'SYSOUT_FILE': cls.SYSOUT_FILE,
            'DEFAULT_SAVE': cls.DEFAULT_SAVE,
            'DEFAULT_LOGGING': cls.DEFAULT_LOGGING,
            'DEFAULT_KFOLD': cls.DEFAULT_KFOLD,
            'DEFAULT_SPLIT_RATIO': cls.DEFAULT_SPLIT_RATIO,
            'DEFAULT_RANDOM_STATE': cls.DEFAULT_RANDOM_STATE,
            'DEFAULT_SCORING': cls.DEFAULT_SCORING,
            'SVM_HYPERPARAMS': cls.SVM_HYPERPARAMS,
            'XGBOOST_HYPERPARAMS': cls.XGBOOST_HYPERPARAMS,
            'MLP_HYPERPARAMS': cls.MLP_HYPERPARAMS
        }

    @classmethod
    def initialize_logging(cls):
        '''
        Initialize logging by redirecting stdout to a file in CACHE_DIR.
        '''
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        logs_dir = os.path.join(cls.CACHE_DIR, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        sys.stdout = Logger(os.path.join(logs_dir, cls.SYSOUT_FILE))

    @classmethod
    def set_global_variable(cls, var_name, value):
        '''
        Set global class-level configuration variables.

        Parameters
        ----------
        var_name : str
            Name of the variable to modify.
        value : Any
            Value to assign.

        Raises
        ------
        ValueError
            If the variable name is unknown.
        '''
        setters = {
            'CACHE_DIR': lambda v: setattr(cls, 'CACHE_DIR', v),
            'SYSOUT_FILE': lambda v: setattr(cls, 'SYSOUT_FILE', v),
            'DEFAULT_SAVE': lambda v: setattr(cls, 'DEFAULT_SAVE', v),
            'DEFAULT_LOGGING': lambda v: setattr(cls, 'DEFAULT_LOGGING', v),
            'DEFAULT_KFOLD': lambda v: setattr(cls, 'DEFAULT_KFOLD', v),
            'DEFAULT_SPLIT_RATIO': lambda v: setattr(cls, 'DEFAULT_SPLIT_RATIO', v),
            'DEFAULT_RANDOM_STATE': lambda v: setattr(cls, 'DEFAULT_RANDOM_STATE', v),
            'DEFAULT_SCORING': lambda v: setattr(cls, 'DEFAULT_SCORING', v),
            'SVM_HYPERPARAMS': lambda v: cls._validate_hyperparameters('svm', v) or setattr(cls, 'SVM_HYPERPARAMS', v),
            'XGBOOST_HYPERPARAMS': lambda v: cls._validate_hyperparameters('xgboost', v) or setattr(cls, 'XGBOOST_HYPERPARAMS', v),
            'RANDOM_FOREST_HYPERPARAMS': lambda v: cls._validate_hyperparameters('random_forest', v) or setattr(cls, 'RANDOM_FOREST_HYPERPARAMS', v),
            'MLP_HYPERPARAMS': lambda v: cls._validate_hyperparameters('mlp', v) or setattr(cls, 'MLP_HYPERPARAMS', v)
        }
        if var_name in setters:
            setters[var_name](value)
        else:
            raise ValueError(f"-- global variable '{var_name}' is not recognized. --")

    def __init__(self, df, model_type, dataset_name, version='2.10', hyperparameters=None,
                 split_ratio=None, random_state=None, save_model=None):

        # self.df = df.copy()
        self.model_type = model_type.lower()
        self.dataset_name = dataset_name
        self.version = version
        self.hyperparameters = hyperparameters
        self.split_ratio = split_ratio if split_ratio is not None else self.DEFAULT_SPLIT_RATIO
        self.random_state = random_state if random_state is not None else self.DEFAULT_RANDOM_STATE
        self.save_model = save_model if save_model is not None else self.DEFAULT_SAVE


        # -- split data
        # -- unsaving large dataframes for memory efficiency
        y = df["disease_status"].astype(int).values
        X = df.drop(columns=["disease_status"]).values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.split_ratio, random_state=self.random_state, stratify=y
        )

        self.grid_search_model = None
        self.y_pred = None
        self.y_proba = None

        if self.DEFAULT_LOGGING:
            self.initialize_logging()
        else:
            os.makedirs(self.CACHE_DIR, exist_ok=True)


        print(f"-- Initialized MLModel with model_type='{self.model_type.upper()}', dataset_name='{self.dataset_name.upper()}' --")
        print(f'-- split ratio: {self.split_ratio}')
        print(f'-- random state: {self.random_state}')

        if self.save_model:
            print(f'-- trained model will be saved to: {self.CACHE_DIR} --')

        if self.model_type not in self.AVAILABLE_MODELS and self.model_type != 'all':
            raise ValueError(f"-- model type '{self.model_type}' is not supported --")

    @classmethod
    def _pretty_print_dict(cls, title, d):
        '''
        pretty-print a dictionary using pprint.
        '''
        print(f"\n-- {title} --")
        cls.pp.pprint(d)
        print("\n")

    @classmethod
    def _validate_hyperparameters(cls, model_type, hyperparameters):
        '''
        Validate hyperparameters for a given model type.

        Parameters
        ----------
        model_type : str
            One of {'svm', 'xgboost', 'mlp', 'random_forest'}.
        hyperparameters : dict
            Hyperparameters to validate.

        Raises
        ------
        ValueError
            If hyperparameters include invalid keys.
        '''
        model_type = model_type.lower()
        if model_type not in cls.AVAILABLE_MODELS:
            raise ValueError(f"-- model type '{model_type}' is not supported --")
        valid_params = cls.AVAILABLE_MODELS[model_type].get_params().keys()
        for param in hyperparameters.keys():
            if param not in valid_params:
                raise ValueError(f"-- hyperparameter '{param}' is invalid for '{model_type}' --")

    def _define_model(self):
        '''
        Define the GridSearchCV wrapper for the selected model.

        Returns
        -------
        GridSearchCV
            Configured grid search

        Raises
        ------
        ValueError
            If model_type is invalid
        '''
        if self.model_type == 'svm':
            params = self.hyperparameters or self.SVM_HYPERPARAMS
            self._pretty_print_dict("SVM Hyperparameters", params)
            return GridSearchCV(SVC(probability=True), param_grid=params,return_train_score=True,
                                scoring=self.DEFAULT_SCORING, cv=self.DEFAULT_KFOLD, n_jobs=-1)

        if self.model_type == 'xgboost':
            params = self.hyperparameters or self.XGBOOST_HYPERPARAMS
            self._pretty_print_dict("XGBoost Hyperparameters", params)
            return GridSearchCV(xgboost.XGBClassifier(eval_metric="logloss"), param_grid=params,return_train_score=True,
                                scoring=self.DEFAULT_SCORING, cv=self.DEFAULT_KFOLD, n_jobs=-1)

        if self.model_type == 'random_forest':
            params = self.hyperparameters or self.RANDOM_FOREST_HYPERPARAMS
            self._pretty_print_dict("Random Forest Hyperparameters", params)
            return GridSearchCV(RandomForestClassifier(), param_grid=params,return_train_score=True,
                                scoring=self.DEFAULT_SCORING, cv=self.DEFAULT_KFOLD, n_jobs=-1)

        if self.model_type == 'mlp':
            params = self.hyperparameters or self.MLP_HYPERPARAMS
            self._pretty_print_dict("MLP Hyperparameters", params)
            return GridSearchCV(MLPClassifier(max_iter=500), param_grid=params,return_train_score=True,
                                scoring=self.DEFAULT_SCORING, cv=self.DEFAULT_KFOLD, n_jobs=-1)

        raise ValueError(f"-- model type '{self.model_type}' is not supported --")

    def train(self):
        '''
        train the model using GridSearchCV
        saves `.joblib` file if `save_model=True`
        '''
        print('-'*80)
        print(f"-- Training {self.model_type.upper()} model on dataset '{self.dataset_name}' --")
        print('-'*80)
        self.grid_search_model = self._define_model()
        self.grid_search_model.fit(self.X_train, self.y_train)
        self.X_train = None  # free memory
        self.y_train = None  # free memory

        self._pretty_print_dict("Best Parameters", self.grid_search_model.best_params_)

        if self.save_model:
            filepath = os.path.join(self.CACHE_DIR, f"{self.dataset_name}_{self.model_type}_gridsearch_model.joblib")
            joblib.dump(self.grid_search_model, filepath)
            print(f"-- trained model saved to: {filepath}")

        self.best_model = self.grid_search_model.best_estimator_
        print(f"-- best model parameters: {self.grid_search_model.best_params_} --")


    def evaluate(self):
        '''
        evaluate the best model on the test set

        returns
        -------
        y_test : ndarray
            True labels.
        y_pred : ndarray
            Predicted class labels.
        y_proba : ndarray
            Predicted probabilities for the positive class.

        RuntimeError
            If model has not yet been trained
        '''
        if self.grid_search_model is None:
            raise RuntimeError("-- Model not trained yet. Call train() first. --")
        
        print('-'*80)
        print(f"-- predicting {self.model_type.upper()} model on dataset '{self.dataset_name}' --")
        print('-'*80)

        self.best_model = self.grid_search_model.best_estimator_
        self.y_pred = self.best_model.predict(self.X_test)
        self.y_proba = self.best_model.predict_proba(self.X_test)[:, 1]
        print("-- Predictions made on test set --")
        self.X_test = None  # free memory
        return self.y_test, self.y_pred, self.y_proba

    def train_evaluate(self):
        '''
        convenience function to train the model and immediately evaluate it

        returns
        -------
        tuple of ndarray
            (y_test, y_pred, y_proba)
        '''
        self.train()
        return self.evaluate()
    