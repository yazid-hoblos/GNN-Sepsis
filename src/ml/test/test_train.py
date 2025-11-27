#!/usr/bin/env python

import os
os.chdir(os.path.dirname(os.getcwd()))

from load_matrix import *
import training_utils

# -- IMPORTANT! to unify the output structure of model/executions files (module load_matrix.py depend on this structure)
# mkdir -p models/executions/GSE54514_enriched_ontology_degfilterv2.9 && \
# for f in models/executions/GSE54514_enriched_ontology_degfilterv2.9_*; do
#   base=$(basename "$f")
#   newname=${base#GSE54514_enriched_ontology_degfilterv2.9_}
#   mv "$f" "models/executions/GSE54514_enriched_ontology_degfilterv2.9/$newname"
# done


# -----------------------------------------------------------------------------------
# ------------ THIS IS FOR VERSION 0 OF training_utils.py ----------------------------
# -----------------------------------------------------------------------------------

# from training_utils_V0 import train_evaluate_model


# def train_eval_expression_svm():
#     print('loading gene expression data...')
#     df=load_df('gene_expression')
#     print('moving label from column to index...')
#     df.index=df['label']
#     print('removing non-gex columns...')
#     df=df.iloc[:,3:]

#     print('training and evaluating SVM model on gene expression data...')
#     gs_model,y_test,y_pred,y_proba = train_evaluate_model(df, model_type='svm', dataset_name='gene_expression')

#     return gs_model,y_test,y_pred,y_proba

# def train_eval_expression_all():
#     print('loading gene expression data...')
#     df=load_df('gene_expression')
#     print('moving label from column to index...')
#     df.index=df['label']
#     print('removing non-gex columns...')
#     df=df.iloc[:,3:]

#     print('training and evaluating ALL models on gene expression data...')
#     gs_model,y_test,y_pred,y_proba = train_evaluate_model(df, model_type='all', dataset_name='gene_expression')

#     return gs_model,y_test,y_pred,y_proba

# -----------------------------------------------------------------------------------

if __name__ == "__main__":
    gex_df=load_df('gene_expression')
    gex_df.index=gex_df['label']
    gex_df=gex_df.iloc[:,3:]

    # 1.
    model=training_utils.MLModel(model_type='svm', dataset_name='gene_expression', df=gex_df)
    # 2.
    model.train_evaluate()
    # done :D
    print('end of test_train.py')