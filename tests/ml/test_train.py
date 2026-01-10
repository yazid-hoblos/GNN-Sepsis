#!/usr/bin/env python

import os
from src.ml.load_matrix import load_df
import src.ml.model_trainer as model_trainer

# -----------------------------------------------------------------------------------
# -- IMPORTANT! to unify the output structure of model/executions files (module load_matrix.py depend on this structure)
# mkdir -p models/executions/GSE54514_enriched_ontology_degfilterv2.9 && \
# for f in models/executions/GSE54514_enriched_ontology_degfilterv2.9_*; do
#   base=$(basename "$f")
#   newname=${base#GSE54514_enriched_ontology_degfilterv2.9_}
#   mv "$f" "models/executions/GSE54514_enriched_ontology_degfilterv2.9/$newname"
# -----------------------------------------------------------------------------------

if __name__ == "__main__":
    gex_df=load_df('gene_expression')

    # 1.
    model=model_trainer.MLModel(model_type='svm', dataset_name='gene_expression', df=gex_df)
    # 2.
    model.train_evaluate()
    # done :D
    print('end of test_train.py')