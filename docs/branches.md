# Branches

## `dev/ml-test`

> [!CAUTION]
> When you want to run a pthon script please run it in modular way from the main project dir or imports might fail, e.g.:


```bash
python -m tests.ml.test_load_matrix
```

### File structure

This is a suggestion of the intended design of the main new dirs in this project:
```
2526-m2geniomhe-GNN-sepsis/
├── src/                        # -- main source code for ML, GNN, data loading, utils, etc.
├── figures/                    
├── notebooks/                  # -- notebooks for exploration, training, etc.
├── tests/                      # -- tests for various modules (would have a dir per module)
└── dump/                       # -- saved trained models (.joblib files)                
```

expanded view of what to be put in main branch:
```
├── docs/
│   └── training_procedure.md  # -- detailed training procedure and explanation of scripts and notebooks
── figures
│   └── projections
│       ├── pca/              # -- PCA plots for all datasets and embeddings
│       └── umap/             # -- UMAP plots for all datasets and embeddings
├── notebooks/
│   ├── HOW_TO_TRAIN.ipynb    # -- instructions on how to train models using scripts or notebooks
│   ├── train.ipynb           # -- main training notebook
│   ├── results.ipynb         # -- main results notebook (still running new models)
│   └── svm_trial.ipynb       # -- SVM testing notebook as requested to check why such accuracy (fishy acc=1)
├── src/
│   ├── gnn
│   ├── eda
│   │   ├── pca.py              # -- PCA plots script
│   │   └── umap.py             # -- UMAP plots script
│   └── ml
│       ├── collector.py        # -- joelle code 1, metrics evaluation and viz
│       ├── evaluator.py        # -- joelle code 2
│       ├── load_matrix.py      # -- sylvia code, data loading utils
│       ├── model_trainer.py    # -- main MLModel class and training functions
│       ├── utils.py            # -- various utils for ML (train all models function, etc.)
│       └── visualizer.py       # -- joelle code 3
├── tests/                      # -- tests for various modules (if user can run them all anf they pass, can run safely the code)
│   └── ml
│       ├── test_dashboard.py
│       ├── test_load_matrix.py
│       └── test_train.py
├── dump/                       # -- saved trained models (.joblib files)
│   └── v2.x/                   # -- versioned folders for trained models
│     ├── logs/
│     │   ├── {model_type}_{dataset_name}_{version}_MLmodel.log
│     │   └── train_all_v2.x_YYYYMMDD_HHMMSS_training_utils.log
│     └── {model_type}_{dataset_name}_MLmodel.joblib
└── requirements.txt            # -- all dependencies
```
