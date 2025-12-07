import os
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create synthetic binary classification data
X, y = make_classification(
    n_samples=500,       # number of samples
    n_features=20,       # number of features
    n_informative=5,     # informative features
    n_redundant=2,
    n_repeated=0,
    n_classes=2,
    weights=[0.7, 0.3],  # class imbalance
    random_state=42
)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# Simple models dictionary
models = {
    "SVM": SVC(probability=True, random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
}

import numpy as np

inputs = {
    "Original": X_test,
    "Noisy": X_test + np.random.normal(0, 0.1, X_test.shape),
    "Scaled": (X_test - X_test.mean(axis=0)) / X_test.std(axis=0)
}

def train_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]  # probability for class 1
    return y_test, y_pred, y_proba


for name, model in models.items():
    model.train_evaluate = lambda X, m=model: train_evaluate_model(m, X_train, y_train, X, y_test)


import pandas as pd
from src.ml.evaluator import Evaluator
from src.ml.collector import ResultsCollector
from src.ml.visualizer import DashboardVisualizer

collector = ResultsCollector()
metrics_list = []

for model_name, model in models.items():
    for input_name, X in inputs.items():
        y_test, y_pred, y_proba = model.train_evaluate(X)
        
        metrics = Evaluator(y_test, y_pred, y_proba).compute_metrics()
        metrics_list.append({
            "model": model_name,
            "input": input_name,
            **metrics
        })

        collector.add(model_name, input_name, y_test, y_pred, y_proba)

results_df = pd.DataFrame(metrics_list)

dashboard = DashboardVisualizer(results_df, collector)
dashboard.plot_metric_grid()
dashboard.plot_radar()
dashboard.plot_roc_curves()
dashboard.plot_pr_curves()
