import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from load_matrix import load_df
from training_utils import MLModel
from evaluator import Evaluator
from collector import ResultsCollector
from visualizer import DashboardVisualizer

# Data preparation
prepare_df = lambda df: df.set_index('label').iloc[:, 3:]

# Set default split ratio globally
MLModel.set_global_variable('DEFAULT_SPLIT_RATIO', 0.3)


# -------------------------------------------------------------
# Helper function executed in each parallel process
# -------------------------------------------------------------
def train_single_model(model_type: str, dataset_name: str):
    """
    Runs in a separate process.
    Loads data, prepares it, trains model, and returns:
    model_name, y_true, y_pred, y_proba, metrics, df_input
    """
    df = prepare_df(load_df(dataset_name))
    ml_model = MLModel(model_type=model_type,
                       df=df,
                       dataset_name=f"{model_type}_{dataset_name}")
    ml_model.train_evaluate()

    y_true, y_pred, y_proba = ml_model.y_test, ml_model.y_pred, ml_model.y_proba

    metrics = Evaluator(y_true, y_pred, y_proba).compute()

    return {
        "model_name": f"{model_type}_{dataset_name}",
        "model_type": model_type,
        "dataset_name": dataset_name,
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "df": df
    }


# -------------------------------------------------------------
# MAIN parallel function
# -------------------------------------------------------------
def train_all_parallel(
    datasets: list = [
        'RGCN_sample_embeddings',
        'Complex_sample_embeddings',
        'RGCN_protein_embeddings',
        'Complex_protein_embeddings',
    ],
    model_types: list = ["svm","mlp"],
    max_workers: int = 4
):
    """
    Run all (dataset × model_type) combinations in parallel.
    
    Returns:
        results_df : dataframe with metrics
        collector  : ResultsCollector with predictions
    """
    tasks = []
    results = []
    collector = ResultsCollector()

    print(f"Launching {len(datasets) * len(model_types)} tasks with {max_workers} workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for dataset_name in datasets:
            for model_type in model_types:
                tasks.append(
                    executor.submit(train_single_model, model_type, dataset_name)
                )

        for future in as_completed(tasks):
            result = future.result()
            results.append(result)

    # Build metrics DataFrame
    metrics_list = []
    input_cache = {}

    for r in results:
        metrics_list.append({
            "model": r["model_type"],
            "input": r["dataset_name"],
            **r["metrics"]
        })

        # Store df only once per dataset
        if r["dataset_name"] not in input_cache:
            input_cache[r["dataset_name"]] = r["df"]

        # Store predictions for dashboard
        collector.add(
            r["model_type"],
            r["dataset_name"],
            r["y_true"],
            r["y_pred"],
            r["y_proba"]
        )

    results_df = pd.DataFrame(metrics_list)

    return results_df, collector, input_cache


# -------------------------------------------------------------
# RUN EVERYTHING
# -------------------------------------------------------------
if __name__ == "__main__":
    results_df, collector, inputs = train_all_parallel(max_workers=6)

    dashboard = DashboardVisualizer(results_df, collector)

    dashboard.plot_metric_grid()
    dashboard.plot_radar()
    dashboard.plot_roc_curves()
    dashboard.plot_pr_curves()
