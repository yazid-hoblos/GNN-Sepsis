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

This is a suggestion of the intended design of the main dirs in this project:
```
.
├── src/                        # -- main source code for ML, GNN, data loading, utils, etc.
├── notebooks/                  # -- notebooks for exploration, training, etc.
├── tests/                      # -- tests for various modules (would have a dir per module)
└── models/                     # -- all saved models, organized by type/version
```

Expanded view of what to be put in main branch:
```
├── docs/
│   └── training_procedure.md  # -- detailed training procedure and explanation of scripts and notebooks
├── notebooks/
│   ├── HOW_TO_TRAIN.ipynb    #   -- instructions on how to train models using scripts or notebooks
│   ├── TRAIN_ALL_parallel.ipynb # (considering removing since will procide function and script to train all)
│   ├── projections.ipynb     # -- PCA, tSNE, UMAP visualizations (will consider scripting them into src/utils or exploration or smtg)  
│   ├── results.ipynb         # -- main results notebook
│   └── svm_trial.ipynb       # -- SVM testing notebook as requested (fishy acc=1)
├── src/
│   ├── gnn
│   └── ml
│       ├── collector.py        # -- joelle code 1, metrics evaluation and viz
│       ├── evaluator.py        # -- joelle code 2
│       ├── load_matrix.py      # -- sylvia code, data loading utils
│       ├── model_trainer.py    # -- main MLModel class and training functions
│       ├── utils.py            # -- various utils for ML (train all models function, etc.)
│       └── visualizer.py       # -- joelle code 3
├── tests/          # -- tests for various modules (if user can run them all anf they pass, can run safely the code)
│   └── ml
│       ├── test_dashboard.py
│       ├── test_load_matrix.py
│       └── test_train.py
└── requirements.txt            # -- all dependencies
```

All models were trained using `TRAIN_ALL.py` script or `train_all_parallel.ipynb` notebook, and saved as joblib files in `dump/` folder, initially with split_ratio for test 0.3 and random_state 42 (all these params/hyperparameters can be accessed in the MLModel object when you load it from joblib file).

```python
from joblib import load
model = load('src/ml/dump/v2.9/svm_gene_expression_MLmodel.joblib')

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

> [!IMPORTANT]
> Only 2 models failed to be trained: SVM + RGCN_protein_embeddings and SVM + concatenated_protein_embeddings (seems to be something with the RGCN protein embeddings, need to investigate more) - all other models were trained successfully and saved in `dump/` folder
> Can retrain them using either the script or the filder and set other hyperparameters/params as needed