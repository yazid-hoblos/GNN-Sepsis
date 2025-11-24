from training_utils import train_evaluate_model
from load_matrix import *

def train_eval_expression():
    print('loading gene expression data...')
    df=load_df('gene_expression')
    print('training and evaluating SVM model on gene expression data...')
    model, y_test, y_pred, y_proba = train_evaluate_model(df, 'svm')
    return model, y_test, y_pred, y_proba

if __name__ == "__main__":
    # -- ask about tree file structure paths in models/executions 
    train_eval_expression()
    print('end of test_train.py')