# ML testing branch

## Updates

- [ ] get new 2.10 data version
- [x] added random forest model to MLModel class
- [ ] migrating from sklearn MLPClassifier to pytorch model inside MLModel class (fixing smtg, training takes time to test)
- [x] fixed non linearly separable data issue with SVM on RGCN (max_iter set in SVC to 10000)
- [x] prepare_df added as decorator to load_df()
- [x] fixed class weighting for svm, random forest and xgboost models
- [x] logging and saving the trained models in dump/ with versioning
- [x] added pca.py and umap.py scripts in src/eda/ folder 
- [x] new directory structure suggestion and expanded view of main branch content 
- [ ] avergaed protein embeddings by **normalized** gene expression (mentioned RobustScaler)


> [!CAUTION]
> When you want to run a pthon script please run it in modular way from the main project dir or imports might fail, e.g.:
```bash
python -m tests.ml.test_load_matrix
```

## File structure

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

## Trained models 

All models were trained using `src/ml/train_all.py` script (or `train_all_parallel.ipynb` notebook whilst testing), and saved as joblib files in `dump/` folder, initially with split_ratio for test 0.3 and random_state 42 (all these params/hyperparameters can be accessed in the MLModel object when you load it from joblib file).

```python
from joblib import load
model = load('dump/v2.9/svm_gene_expression_MLmodel.joblib')

type(model)  # -- <class 'training_utils.MLModel'>
model.dataset_name  # -- 'gene_expression'
model.model_type    # -- 'svm'
model.split_ratio   # -- 0.3
model.random_state  # -- 42
model.kfolds        # -- 3 (kfold cv used for gridsearch)
model.y_test, model.y_pred, model.y_proba  # -- used for evaluation
model.grid_search_model # -- GridSearchCV object after hyperparameter tuning
model.best_model  # -- best estimator after gridsearch
```

## pretrained models available


> [!CAUTION]
> Need to have a specific dir structure in `model/executions/`, can run this to unify it:
```bash
# -- from project main dir
mkdir -p models/executions/GSE54514_enriched_ontology_degfilterv2.9 && \
for f in models/executions/GSE54514_enriched_ontology_degfilterv2.9_*; do
  base=$(basename "$f")
  newname=${base#GSE54514_enriched_ontology_degfilterv2.9_}
  mv "$f" "models/executions/GSE54514_enriched_ontology_degfilterv2.9/$newname"
done

# -- generalize it for 2.10 qnd 2.11 as well
mkdir -p models/executions/GSE54514_enriched_ontology_degfilterv2.10 && \
for f in models/executions/GSE54514_enriched_ontology_degfilter_v2.10_*; do
  base=$(basename "$f")
  newname=${base#GSE54514_enriched_ontology_degfilter_v2.10_ovp0.2_ng3_}
  mv "$f" "models/executions/GSE54514_enriched_ontology_degfilterv2.10/$newname"
done
mkdir -p models/executions/GSE54514_enriched_ontology_degfilterv2.11 && \
for f in models/executions/GSE54514_enriched_ontology_degfilter_v2.11_*; do
  base=$(basename "$f")
  newname=${base#GSE54514_enriched_ontology_degfilter_v2.11_}
  mv "$f" "models/executions/GSE54514_enriched_ontology_degfilterv2.11/$newname"
done
```