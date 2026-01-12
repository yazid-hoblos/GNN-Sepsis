"""
Classical models training util functions
----------------------------------------------

this module defines utilities to construct, train, and evaluate machine-learning classification models
using support vector machines (svm), xgboost classifiers, random forests and multilayer perceptrons (mlp). 
it represents the second step of the workflow, following data loading and preprocessing.

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
- The input DataFrame must contain a `disease_status` column, which will be used as the target variable, and all remaining columns are treated as features (pay attention they should all be numeric)
- This module includes two main classes:
    1. `MLModel` - Handles model selection, training, evaluation, and optional saving.
    2. `PyTorchMLP` - A PyTorch-based multilayer perceptron implementing a scikit-learn-like API
       with `fit`, `predict`, and `predict_proba` methods, used by `MLModel` when `model_type=pytorch_mlp'`
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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.exceptions import ConvergenceWarning
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ----------------------------------------------------------------------------------------------- #

class Logger:
    '''
    logger that duplicates stdout to a log file
    [!CAUTION] -- does not print when used in jupyter notebooks/ temporarily commented out in MLModel
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

# ----------------------------------------------------------------------------------------------- #

class PyTorchMLP(BaseEstimator, ClassifierMixin):
    """
    PyTorch wrapper for a simple MLP classifier to mimic scikit-learn API in a way that is easily integrable in MLModel's GridSearchCV pipeline
    """

    possible_activations={'relu': nn.ReLU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'leaky_relu': nn.LeakyReLU, 'gelu': nn.GELU}

    def __init__(self, input_dim, cache_dir='.cache_pytorch/',name='', hidden_layer_sizes=(50,), activation='relu', dropout_rate=0,weight_decay=0.0, 
                 solver='adam', learning_rate_init=0.001, max_iter=20, batch_size=32, random_state=42,early_stopping=False):
        
        self.name=name
        
        self.input_dim = input_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = self.possible_activations[activation] if activation in self.possible_activations else nn.ReLU
        self.solver = solver
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.random_state = random_state
        self.early_stopping=False
        self.dropout_rate=dropout_rate
        self.weight_decay = weight_decay

        self.cache_dir=cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # print(f'-- PytorchMLP model initialization for {self.name}, caching plot to {self.cache_dir}, random seed at {self.random_state}')


        torch.manual_seed(self.random_state)
        torch.manual_seed(self.random_state)
        torch.cuda.manual_seed(self.random_state)
        torch.cuda.manual_seed_all(self.random_state)
        
        self._build_model()
        self.loss_fn = nn.BCEWithLogitsLoss() #-- more stable than BCE

        if self.solver.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate_init,
                weight_decay=self.weight_decay
            )
        elif self.solver.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate_init,
                weight_decay=self.weight_decay
            )
        elif self.solver.lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate_init,
                momentum=0.9, # maybe experiment ehre?
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unsupported solver: {self.solver}. Choose from 'adam', 'adamw', 'sgd'.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _build_model(self):
        layers = []
        input_size = self.input_dim
        act_fn = self.activation
        for hidden_size in self.hidden_layer_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(act_fn())
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))
            input_size = hidden_size
        layers.append(nn.Linear(input_size, 1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)



    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y.reshape(-1,1), dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        train_losses = []

        for epoch in range(self.max_iter):
            epoch_loss = 0.0
            for xb, yb in loader:
                self.optimizer.zero_grad()
                pred = self.model(xb)
                loss = self.loss_fn(pred, yb)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * xb.size(0)  # sum over batch
            epoch_loss /= len(dataset)  # average loss over dataset
            train_losses.append(epoch_loss)
            print(f'-- [pytorch mlp] epoch {epoch+1}/{self.max_iter}, loss: {epoch_loss:.4f} --')

        # -- plotting
        if hasattr(self, 'cache_dir') and self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
            plt.figure(figsize=(6,4))
            plt.plot(range(1, self.max_iter+1), train_losses, marker='o')
            plt.title(f"Training Loss per Epoch {self.name} ([!note the best model in gridsearch but last one trained])")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.grid(True)
            plt.tight_layout()
            save_path = os.path.join(self.cache_dir, f"{self.name}_train_loss.png")
            plt.savefig(save_path)
            plt.close()
            print(f"-- training loss plot saved to: {save_path} in {self.cache_dir}")

        return self


    # -- scikit-learn compatibility: allow cloning and parameter setting by GridSearchCV (imp for reproducibility of class's funcs)
    def get_params(self, deep=True):
        return {
            'input_dim': self.input_dim,
            'name': self.name,
            'cache_dir': self.cache_dir,
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'activation': self._activation_name if hasattr(self, '_activation_name') else 'relu',
            'solver': self.solver,
            'learning_rate_init': self.learning_rate_init,
            'max_iter': self.max_iter,
            'batch_size': self.batch_size,
            'random_state': self.random_state,
            'dropout_rate': self.dropout_rate,
            'weight_decay': self.weight_decay
        }

    def set_params(self, **params):
        for key, val in params.items():
            if key == 'activation':
                # -- accept either a string name or a callable/class
                if isinstance(val, str):
                    self._activation_name = val
                    self.activation = self.possible_activations[val] if val in self.possible_activations else nn.ReLU
                elif callable(val):
                    self._activation_name = getattr(val, '__name__', 'custom')
                    self.activation = val
                else:
                    raise ValueError("[!] activation must be a string name or a callable activation class")
            elif key in ('name', 'cache_dir'):
                setattr(self, key, val)
            else:
                if hasattr(self, key):
                    setattr(self, key, val)
                else:
                    # allow setting arbitrary attributes for compatibility
                    setattr(self, key, val)

        # rebuild model and optimizer if architecture or training params changed
        self._build_model()
        self.loss_fn = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate_init) if self.solver=='adam' else optim.SGD(self.model.parameters(), lr=self.learning_rate_init)
        return self

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            proba = self.model(X_tensor).numpy().flatten()
        return np.vstack([1-proba, proba]).T

    def predict(self, X):
        proba = self.predict_proba(X)[:,1]
        return (proba >= 0.5).astype(int)


# ----------------------------------------------------------------------------------------------- #

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
    normalization : str, optional
        nromalization type to perform on gene exp data when creating teh protein embeddings (saved for logging the steps taken)
        Options: 'robust', 'standard', 'minmax', 'log1p', 'none'
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
    PYTORCH_MLP_HYPERPARAMS : dict
        Default hyperparameter grid for MLP
    '''

    # ------------------ CLASS ATTRIBUTES ------------------
    DEFAULT_KFOLD = 3
    DEFAULT_SPLIT_RATIO = 0.2
    DEFAULT_RANDOM_STATE = 42
    DEFAULT_SCORING = "accuracy"
    DEFAULT_SAVE = False
    DEFAULT_LOGGING = False
    CACHE_DIR = '.cache/'
    SYSOUT_FILE = None

    AVAILABLE_MODELS = {'svm', 'xgboost', 'sklearn_mlp','pytorch_mlp','random_forest'}
    SVM_HYPERPARAMS = {'C': [0.1, 1], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto'], 'class_weight': ['balanced', {0: 2, 1: 1}, {0: 3, 1: 1}]}
    XGBOOST_HYPERPARAMS = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7],
                           'learning_rate': [0.01, 0.1, 0.2], 'subsample': [0.6, 0.8, 1.0], 'scale_pos_weight': [36 / 127, 0.4, 0.2]}
    RANDOM_FOREST_HYPERPARAMS = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7],
                                'min_samples_split': [2, 5, 10], 'class_weight': ['balanced', 'balanced_subsample', {0: 2, 1: 1}, {0: 3, 1: 1}]}
    # PYTORCH_MLP_HYPERPARAMS_OG = {
    #     'hidden_layer_sizes': [(50,), (100,), (100, 50)],
    #     'activation': ['relu'],
    #     'solver': ['adam', 'sgd'],
    #     'learning_rate_init': [0.001, 0.01, 0.1],
    #     'batch_size': [16]        #--might be more stable for a small dataset like ours
    #     # 'max_iter': [200, 500, 1000],
    #     # 'dropout': [0.0, 0.1, 0.2] #-- reconsidering
    # }
    PYTORCH_MLP_HYPERPARAMS = {
        'hidden_layer_sizes': [(50,), (100,)],
        'max_iter':[40],
        'activation': ['relu'],
        'solver': ['adam','adamW'],
        'learning_rate_init': [0.001, 0.0001],
        'batch_size': [16],
        'dropout_rate': [0.0, 0.1]
    }
    SKLEARN_MLP_HYPERPARAMS = {'hidden_layer_sizes': [(50,), (100,), (100, 50)], 'activation': ['relu'], # check if leaky_relu exists here
                       'solver': ['adam', 'sgd'], 'learning_rate_init': [0.001, 0.01, 0.1]}

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
            'PYTORCH_MLP_HYPERPARAMS': cls.PYTORCH_MLP_HYPERPARAMS,
            'SKLEARN_MLP_HYPERPARAMS': cls.SKLEARN_MLP_HYPERPARAMS
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
        Set global class-level configuration variables
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
            'PYTORCH_MLP_HYPERPARAMS': lambda v: cls._validate_hyperparameters('pytorch_mlp', v) or setattr(cls, 'PYTORCH_MLP_HYPERPARAMS', v),
            'SKLEARN_MLP_HYPERPARAMS': lambda v: cls._validate_hyperparameters('sklearn_mlp', v) or setattr(cls, 'SKLEARN_MLP_HYPERPARAMS', v)
        }
        if var_name in setters:
            setters[var_name](value)
        else:
            raise ValueError(f"-- global variable '{var_name}' is not recognized. --")

    def __init__(self, df, model_type, dataset_name, version='2.10', normalization="robust",
                 hyperparameters=None, split_ratio=None, random_state=None, kfold=None, save_model=None):

        # self.df = df.copy()
        self.model_type = model_type.lower()
        self.dataset_name = dataset_name
        self.version = version
        self.normalization=normalization
        self.hyperparameters = hyperparameters
        self.split_ratio = split_ratio if split_ratio is not None else self.DEFAULT_SPLIT_RATIO
        self.random_state = random_state if random_state is not None else self.DEFAULT_RANDOM_STATE
        self.kfold = kfold if kfold is not None else self.DEFAULT_KFOLD
        self.save_model = save_model if save_model is not None else self.DEFAULT_SAVE
        self.best_model = None

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

        print(f"-- [{self.model_type}_{self.dataset_name}] Initialized MLModel with model_type='{self.model_type.upper()}', dataset_name='{self.dataset_name.upper()}' --")
        print(f'-- [{self.model_type}_{self.dataset_name}] split ratio: {self.split_ratio}')
        print(f'-- [{self.model_type}_{self.dataset_name}] random state: {self.random_state}')

        print(f'-- [{self.model_type}_{self.dataset_name}] CACHE_DIR is: {self.CACHE_DIR} --')

        if self.model_type not in self.AVAILABLE_MODELS and self.model_type != 'all':
            raise ValueError(f"-- model type '{self.model_type}' is not supported --")

        if self.save_model:
            version_dir = os.path.abspath(os.path.join(self.CACHE_DIR, f"{self.version}_{self.normalization}"))
            print(f'-- [{self.model_type}_{self.dataset_name}] trained model will be saved to: {version_dir} --')  
        if self.SYSOUT_FILE is None:
            MLModel.SYSOUT_FILE=f"{model_type}_{dataset_name}_{version}_{normalization}_training_utils.log"
            print(f'-- [{self.model_type}_{self.dataset_name}] setting SYSOUT_FILE to: {self.SYSOUT_FILE} --')
        
        if self.DEFAULT_LOGGING:
            self.initialize_logging()
            logs_dir = os.path.join(self.CACHE_DIR, 'logs')
            print(f'-- [{self.model_type}_{self.dataset_name}] logging in f{logs_dir} --')

        else:
            os.makedirs(self.CACHE_DIR, exist_ok=True)

    def get_data(self,df):
        '''same code in init to split the data but implemented later for replication
        WITHOUT memoizing matrices as object attributes
        ensures same random seed and test split ration (+stratification if performed)
        [!] useful for shap value computation to get the original X

        param: df (from load_df)
        returns: X, y
        '''
        y = df["disease_status"].astype(int).values
        X = df.drop(columns=["disease_status"]).values
        return  X, y

    def get_data_split(self,df):
        '''same code in init to split the data but implemented later for replication
        WITHOUT memoizing matrices as object attributes
        ensures same random seed and test split ration (+stratification if performed)
        [!] useful for shap value computation to get the original X

        param: df (from load_df)
        returns: X_train, X_test, y_train, y_test
        '''
        y = df["disease_status"].astype(int).values
        X = df.drop(columns=["disease_status"]).values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.split_ratio, random_state=self.random_state, stratify=y
        )
        return  X_train, X_test, y_train, y_test

    @classmethod
    def _pretty_print_dict(cls, title, d, indent=0):
        print(f"\n-- {title} --")
        cls._print_nested_dict(d, indent=2)
        print("\n")

    @classmethod
    def _print_nested_dict(cls, d, indent):
        pad = " " * indent
        for key, value in d.items():
            if isinstance(value, dict):
                print(f"{pad}{key}:")
                cls._print_nested_dict(value, indent + 2)
            else:
                print(f"{pad}{key}: {value}")

    @classmethod
    def _validate_hyperparameters(cls, model_type, hyperparameters):
        '''
        validate hyperparameters for a given model type
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
        Define the GridSearchCV wrapper for the selected model type with appropriate hyperparameters
        returns GridSearchCV object
        '''

        if self.model_type == 'svm':
            params = self.hyperparameters or self.SVM_HYPERPARAMS
            self._pretty_print_dict("SVM Hyperparameters", params)
            # -- added max_iter to avoid long convergence times which is the case when training a linear kernel on RGCN sample embedding features
            return GridSearchCV(SVC(probability=True,max_iter=10000,random_state=self.random_state), param_grid=params,return_train_score=True,
                                scoring=self.DEFAULT_SCORING, cv=self.kfold, n_jobs=-1)

        if self.model_type == 'xgboost':
            params = self.hyperparameters or self.XGBOOST_HYPERPARAMS
            self._pretty_print_dict("XGBoost Hyperparameters", params)
            return GridSearchCV(xgboost.XGBClassifier(eval_metric="logloss",random_state=self.random_state), param_grid=params,return_train_score=True,
                                scoring=self.DEFAULT_SCORING, cv=self.DEFAULT_KFOLD, n_jobs=-1)

        if self.model_type == 'random_forest':
            params = self.hyperparameters or self.RANDOM_FOREST_HYPERPARAMS
            self._pretty_print_dict("Random Forest Hyperparameters", params)
            return GridSearchCV(RandomForestClassifier(random_state=self.random_state), param_grid=params,return_train_score=True,
                                scoring=self.DEFAULT_SCORING, cv=self.DEFAULT_KFOLD, n_jobs=-1)

        if self.model_type == 'pytorch_mlp':
            params = self.hyperparameters or self.PYTORCH_MLP_HYPERPARAMS
            self._pretty_print_dict("MLP Hyperparameters", params)
            return GridSearchCV(
                estimator=PyTorchMLP(input_dim=self.X_train.shape[1], name=f"{self.dataset_name}_{self.version}_{self.normalization}_{self.random_state}", cache_dir=os.path.join(self.CACHE_DIR,'pytorch_mlp_plots/'),random_state=self.random_state  ), #-- testing the grid
                param_grid=params,
                scoring=self.DEFAULT_SCORING,
                cv=self.DEFAULT_KFOLD,
                n_jobs=-1
            )
        if self.model_type == 'sklearn_mlp':
            params = self.hyperparameters or self.SKLEARN_MLP_HYPERPARAMS
            self._pretty_print_dict("MLP Hyperparameters", params)
            return GridSearchCV(MLPClassifier(max_iter=500,random_state=self.random_state), param_grid=params,return_train_score=True,
                                scoring=self.DEFAULT_SCORING, cv=self.DEFAULT_KFOLD, n_jobs=-1)

        raise ValueError(f"-- model type '{self.model_type}' is not supported --")

    def train(self):
        '''
        train the model using GridSearchCV
        saves `.joblib` file if `save_model=True`
        '''
        print('-'*80)
        print(f"-- [{self.model_type}_{self.dataset_name}] Training {self.model_type.upper()} model on dataset '{self.dataset_name}' --")
        print('-'*80)
        self.grid_search_model = self._define_model()
        self.grid_search_model.fit(self.X_train, self.y_train)
        self.X_train = None  # free memory
        self.y_train = None  # free memory

        self._pretty_print_dict("Best Parameters", self.grid_search_model.best_params_)

        if self.save_model:
            version_dir = os.path.join(MLModel.CACHE_DIR, f"{self.version}_{self.normalization}")
            os.makedirs(version_dir, exist_ok=True)
            filepath = os.path.join(version_dir, f"{self.model_type}_{self.dataset_name}_gridsearch_model.joblib")
            joblib.dump(self, filepath)
            print(f"-- [{self.model_type}_{self.dataset_name}] trained model saved to: {filepath}")

        self.best_model = self.grid_search_model.best_estimator_
        print(f"-- [{self.model_type}_{self.dataset_name}] best model parameters: {self.grid_search_model.best_params_} --")


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
        print(f"-- [{self.model_type}_{self.dataset_name}] predicting {self.model_type.upper()} model on dataset '{self.dataset_name}' --")
        print('-'*80)

        self.best_model = self.grid_search_model.best_estimator_
        self.y_pred = self.best_model.predict(self.X_test)
        self.y_proba = self.best_model.predict_proba(self.X_test)[:, 1]
        print("-- Predictions made on test set --")
        self.X_test = None  # free memory

        if self.save_model:
            version_dir = os.path.join(MLModel.CACHE_DIR, f"{self.version}_{self.normalization}")
            os.makedirs(version_dir, exist_ok=True)
            filepath = os.path.join(version_dir, f"{self.model_type}_{self.dataset_name}_gridsearch_model.joblib")
            joblib.dump(self, filepath)
            print(f"-- [{self.model_type}_{self.dataset_name}] evaluated model saved to: {filepath}")

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
    
    def __repr__(self):
        return f'''MLModel(
    model_type={self.model_type},
    dataset_name={self.dataset_name},
    version={self.version},
    normalization={self.normalization},
    split_ratio={self.split_ratio},
    kfold={self.DEFAULT_KFOLD},
    random_state={self.random_state},
    best_model={self.best_model}
    sysout_file={self.SYSOUT_FILE},
    cache_dir={self.CACHE_DIR},
    logging={self.DEFAULT_LOGGING},
    save_model={self.save_model},
)'''
    
    def _format_tree(self, obj, indent=0):
        """Recursively formats objects into a nested tree view."""
        if obj is None:
            return " " * indent + "None"
        pad = " " * indent
        if hasattr(obj, "get_params"):
            # sklearn estimator
            params = obj.get_params(deep=False)
            s = f"{pad}{obj.__class__.__name__}(\n"
            for k, v in params.items():
                s += f"{pad}  ├─ {k}: "
                if hasattr(v, "get_params"):
                    s += "\n" + self._format_tree(v, indent + 6)
                else:
                    s += f"{v}\n"
            return s + f"{pad})"
        elif isinstance(obj, dict):
            s = f"{pad}{{\n"
            for k, v in obj.items():
                s += f"{pad}  ├─ {k}: "
                if isinstance(v, (dict, list)) or hasattr(v, "get_params"):
                    s += "\n" + self._format_tree(v, indent + 6)
                else:
                    s += f"{v}\n"
            return s + f"{pad}}}"
        elif isinstance(obj, list):
            s = f"{pad}[\n"
            for v in obj:
                s += self._format_tree(v, indent + 4) + "\n"
            return s + pad + "]"
        else:
            return f"{pad}{obj}"

    def __str__(self):
        norm_str=f"├─ normalization: {self.normalization}\n" if self.normalization else "" # -- for older trained versions with no normalization param

        return (
            f"MLModel\n"
            f"{norm_str}"
            f"├─ model_type: {self.model_type}\n"
            f"├─ dataset_name: {self.dataset_name}\n"
            f"├─ version: {self.version}\n"
            f"├─ split_ratio: {self.split_ratio}\n"
            f"├─ random_state: {self.random_state}\n"
            f"├─ best_model:\n"
            f"{self._format_tree(self.best_model, indent=4)}\n"
            f"└─ save_model: {self.save_model}"
        )
