# ML testing branch

## Updates

- [ ] get new 2.10 data version
- [x] added random forest model to MLModel class
- [ ] migrating from sklearn MLPClassifier to pytorch model inside MLModel class (prototype working, long training time so far hard to evaluate the output still)
- [x] fixed non linearly separable data issue with SVM on RGCN (max_iter set in SVC to 10000)
- [x] prepare_df added as decorator to load_df()
- [x] fixed class weighting for svm, random forest and xgboost models
- [x] logging and saving the trained models in dump/ with versioning
- [x] added pca.py and umap.py scripts in src/eda/ folder 
- [x] new directory structure suggestion and expanded view of main branch content 
- [ ] averaged protein embeddings by **normalized** gene expression (mentioned RobustScaler)


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

## Trained models 

To train the models, can either use the `src/ml/train_all.py` script or train it in your python code or notebook directly.

Check out the [`notebooks/HOW_TO_TRAIN.ipynb`](notebooks/HOW_TO_TRAIN.ipynb) notebook for detailed instructions on the usage of the object, attributes and methods.

### 1. Using the `train_all.py` script

All models were trained using `src/ml/train_all.py` script (`train_all_parallel.ipynb` notebook made for testing code), and saved as joblib files in `dump/` folder, initially with split_ratio for test 0.3 and random_state 42 (all these params/hyperparameters can be accessed in the MLModel object when you load it from joblib file).

```bash
python -m src.ml.train_all
```
This will train everything and save each modelxdataset combination in its version folder, youd have in the working dir:
```
dump/
├── logs/
├── v2.10/
└── v2.11/
```

Options can be passed to the script, run:
```bash
python -m src.ml.train_all -h
```
```
usage: train_all.py [-h] [--versions VERSIONS [VERSIONS ...]] [--logging] [--cache-dir CACHE_DIR] [--threads THREADS] [--model-types MODEL_TYPES [MODEL_TYPES ...]]
                    [--datasets DATASETS [DATASETS ...]] [--split-ratio SPLIT_RATIO] [--random-state RANDOM_STATE] [--hyperparameter-file HYPERPARAMETER_FILE] [--default-logging] [--kfold KFOLD]
                    [--dump-dir DUMP_DIR]

-- Training combinations of ML models on combinations of datasets for all specified versions --

by default, trains all available models on all available datasets for versions v2.10 and v2.11

AVAILABLE OPTIONS:  
    model-types: svm, random_forest, xgboost, pytorch_mlp, sklearn_mlp
    datasets: gene_expression, RGCN_sample_embeddings, Complex_sample_embeddings, concatenated_sample_embeddings, RGCN_protein_embeddings, Complex_protein_embeddings, concatenated_protein_embeddings
    versions: v2.10, v2.11, v2.9

run this script to train all ML models on all datasets for all specified versions
saves trained models in joblib files in dump/{version}/ folder

> [!TIP]
> run it from the project root as:

    python -m src.ml.train_all -h  
    python -m src.ml.train_all      # -- which is equivalent to:  
    python -m src.ml.train_all --versions v2.10 v2.11 --dump-dir ./dump/ --threads 12  

    python -m src.ml.train_all --versions v2.10 --model-types svm
    python -m src.ml.train_all --versions v2.10 --datasets gene_expression RGCN_sample_embeddings --model-types random_forest

options:
  -h, --help            show this help message and exit
  --versions VERSIONS [VERSIONS ...]
  --logging             Whether to enable logging to file
  --cache-dir CACHE_DIR
  --threads THREADS
  --model-types MODEL_TYPES [MODEL_TYPES ...]
  --datasets DATASETS [DATASETS ...]
  --split-ratio SPLIT_RATIO
  --random-state RANDOM_STATE
  --hyperparameter-file HYPERPARAMETER_FILE
                        Path to a JSON file containing hyperparameter grids for models
  --default-logging     Whether to enable default logging to file
  --kfold KFOLD         Number of folds for k-fold cross-validation
  --dump-dir DUMP_DIR   Directory where trained models will be saved
```

> [!NOTE]
> by default it trains all available models on all available datasets for versions v2.10 and v2.11, using 12 threads and saving the trained models in `./dump/` folder EXCEPT for pytorch_mlp model which is disabled by default to avoid long training times (and as it was still under testing), so if you want to include it you need to specify it with `--model-types pytorch_mlp` option and maybe better to train this one separately

### 2. Training a model in python

Using the MLModel class from `src/ml/model_trainer.py`, you can train a model directly in your code or notebook like this:

```python
# -- gievn that the working dir is src/ml/
from model_trainer import MLModel

# -- the minimal way to define a model with all options set to default values
>>> ml_model=MLModel(
  model_type='svm',
  df=load_df('Complex_protein_embeddings'),dataset_name='Complex_protein_embeddings',
  version='v2.11'
)
>>> ml_model
MLModel(
    model_type=svm,
    dataset_name=Complex_protein_embeddings,
    version=v2.11,
    split_ratio=0.2,
    random_state=42,
    save_model=False
)
```

```python
# -- train and evaluate the model
y_test, y_pred, y_proba = ml_model.train_evaluate()
```

### Loading a trained model file from dump/

```python
from joblib import load
model = load('dump/v2.10/svm_gene_expression_MLmodel.joblib')

type(model)  # -- <class 'training_utils.MLModel'>
model.dataset_name  # -- 'gene_expression'
model.model_type    # -- 'svm'
model.split_ratio   # -- 0.3
model.random_state  # -- 42
model.kfolds        # -- 3 (kfold cv used for gridsearch)
model.y_test, model.y_pred, model.y_proba  # -- used for evaluation
model.grid_search_model                    # -- GridSearchCV object after hyperparameter tuning
model.best_model    # -- best estimator after gridsearch
model.version       # -- 'v2.10'
```

### Loading all trained models 

If trained with `train_all.py` script, all models are saved in `dump/{version}/` folder, so you can load them all and compute metrics like this:

```python
from utils import load_models
from evaluator import Evaluator
from collector import Collector

all_models_2_10=load_models('dump','v2.10')

for model_name,ml_model in all_models_2_10.items():
    print(f"Evaluating model: {model_name}")
    ml_model.evaluate_model() 
        
    y_test, y_pred, y_proba = ml_model.y_test, ml_model.y_pred, ml_model.y_proba

    metrics = Evaluator(y_test, y_pred, y_proba).compute_metrics()
    metrics_list.append({
            "model": ml_model.model_type,
            "input": ml_model.dataset_name,
            **metrics
    })
    collector.add(ml_model.model_type, ml_model.dataset_name, y_test, y_pred, y_proba)
```

### Visualizing results

You can use the `Collector` and `Visualizer` classes to visualize the results of the trained models. For example:

```python
from visualizer import DashboardVisualizer

results=pd.DataFrame(metrics_list)
dashboard = DashboardVisualizer(results, collector)

dashboard.plot_metric_grid()
dashboard.plot_radar()
dashboard.plot_roc_curves()
dashboard.plot_pr_curves()
```

## pretrained models available

Need to have pretrained models embeddings in models/executions/ dir.

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