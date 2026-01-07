import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)

class Evaluator:
    """
    Computes scalar metrics for binary classification.
    Requires y_test, y_pred, and y_proba.
    """
    def __init__(self, y_test, y_pred, y_proba):
        self.y_test = np.asarray(y_test)
        self.y_pred = np.asarray(y_pred)
        self.y_proba = np.asarray(y_proba)

    def compute_metrics(self):
        metrics = {
            "balanced_accuracy": balanced_accuracy_score(self.y_test, self.y_pred),
            "precision": precision_score(self.y_test, self.y_pred),
            "recall": recall_score(self.y_test, self.y_pred),
            "f1": f1_score(self.y_test, self.y_pred),
            "mcc": matthews_corrcoef(self.y_test, self.y_pred),
            "auroc": roc_auc_score(self.y_test, self.y_proba),
            "auprc": average_precision_score(self.y_test, self.y_proba),
            "brier": brier_score_loss(self.y_test, self.y_proba)
        }
        return metrics
