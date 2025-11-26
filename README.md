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
├── HOW_TO_TRAIN.ipynb      # -- example on how to use the ML training utilities
├── __init__.py
├── load_matrix.py          # -- preprocessing module -- sylvia
├── test_load_matrix.py     # -- preprocessinf test -- sylvia
├── test_train.py           # -- script to train from terminal
├── todo.md                 # -- small notes on code to enhance
├── training_utils.py       # -- main module for training the ML model using class MLModel
└── training_utils_V0.py    # -- old version of the module without MLModel class
```