# ML testing branch

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
```

This is a suggestion of the ML code
Find the code in `src/ml`:
```
src/ml
├── load_matrix.py      # -- sylvia - preprocessing module 
├── training_utils.py   # -- main module for training the ML model using class MLModel
├── training_utils.v0.py # -- old version of the module without MLModel class
├── collector.py        # -- joelle code 1
├── evaluator.py        # -- joelle code 2
├── visualizer.py       # -- joelle code 3
├── main_Joelle.py      # -- joelle code to run (parallel training unresolved)
├── dump                # -- saved trained MLModel objects
│   └── [model_type]_[dataset_name]_MLmodel.joblib
├── notebooks           # -- jupyter notebooks
│   ├── HOW_TO_TRAIN.ipynb    # -- example on how to use the ML training utilities
│   ├── __init__.py
│   ├── results.ipynb         # -- metrics and plots for the trained models
│   ├── train_all.ipynb       # -- example on how to train all models on all datasets
│   └── train_all_parallel.ipynb # -- parallel version of train_all.ipynb (the one ran successfully - 2 exceptions)
├── test                # -- test scripts
│   ├── test_Joelle.py        # -- joelle test
│   ├── test_load_matrix.py   # -- sylvia test
│   └── test_train.py         # -- train test
└── TRAIN_ALL.py     # -- script to train all models on all datasets from terminal and save to dump/
```

All models were trained using `TRAIN_ALL.py` script or `train_all_parallel.ipynb` notebook, and saved as joblib files in `dump/` folder, initially with split_ratio for test 0.3 and random_state 42 (all these params/hyperparameters can be accessed in the MLModel object when you load it from joblib file).

```python
from joblib import load
model = load('src/ml/dump/svm_gene_expression_MLmodel.joblib')

type(model)  # -- <class 'training_utils.MLModel'>
model.dataset_name  # -- 'gene_expression'
model.model_type    # -- 'svm'
model.split_ratio   # -- 0.3
model.random_state  # -- 42
model.kfolds        # -- 3 (kfold cv used for gridsearch)
model.y_test, model.y_pred, model.y_proba  # -- used for evaluation
model.grid_search_model # -- GridSearchCV object after hyperparameter tuning
```

> [!IMPORTANT]
> Only 2 models failed to be trained: SVM + RGCN_protein_embeddings and SVM + concatenated_protein_embeddings (seems to be something with the RGCN protein embeddings, need to investigate more) - all other models were trained successfully and saved in `dump/` folder
> Can retrain them using either the script or the filder and set other hyperparameters/params as needed