import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve

class DashboardVisualizer:
    """
    Dashboard-style visualization for all metrics and plots.
    Requires:
    - results_df: DataFrame with columns ['model', 'input', metrics...]
    - collector: minimal collector with raw predictions (optional, for ROC/PR curves)
    """

    def __init__(self, results_df, collector=None):
        self.results_df = results_df
        self.collector = collector
        # Default metrics to display in dashboard
        self.metrics = ["balanced_accuracy", "precision", "recall", "f1",
                        "mcc", "auroc", "auprc", "brier"]

    # -----------------------------
    def plot_metric_grid(self, title=None):
        """Grid of barplots: metrics × models × inputs."""
        n_metrics = len(self.metrics)
        n_cols = 4
        n_rows = int(np.ceil(n_metrics / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4*n_rows))
        axes = axes.flatten()

        for i, metric in enumerate(self.metrics):
            ax = axes[i]
            sns.barplot(
                data=self.results_df,
                x="model",
                y=metric,
                hue="input",
                ax=ax
            )
            ax.set_title(metric.upper())
            if metric in ["balanced_accuracy", "precision", "recall", "f1", "auroc", "auprc"]:
                ax.set_ylim(0, 1)
            ax.legend(title="Input", loc="upper right")
            for label in ax.get_xticklabels():
                label.set_rotation(30)

        # Remove empty subplots if any
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle(title if title else "Dashboard: All Metrics Across Models and Inputs", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    # -----------------------------
    def plot_radar(self, title=None):
        """Radar plots for all models and inputs with metric labels and proper spacing."""
        metrics = ["balanced_acc" if m=="balanced_accuracy" else m for m in self.metrics]

        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # close loop

        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)

        for model in self.results_df["model"].unique():
            for input_name in self.results_df["input"].unique():
                subset = self.results_df[(self.results_df["model"] == model) &
                                        (self.results_df["input"] == input_name)]
                if subset.empty:
                    continue
                values = subset[self.metrics].values.flatten()
                values = np.concatenate((values, [values[0]]))  # close loop
                ax.plot(angles, values, label=f"{model} – {input_name}")

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=10)

        ax.legend(bbox_to_anchor=(1.2, 1), loc='upper left')

        plt.title(title if title else "Radar Plot: Metrics by Model and Input", fontsize=14)
        plt.subplots_adjust(top=0.9)  # leave space for title and top labels
        plt.show()


    # -----------------------------
    def plot_roc_curves(self, title=None):
        """Plot ROC curves using minimal collector."""
        if self.collector is None:
            print("Collector not provided. Cannot plot ROC curves.")
            return

        plt.figure(figsize=(6,6))

        for (model_name, input_name), (y_true, y_pred, y_proba) in self.collector.data.items():
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            plt.plot(fpr, tpr, label=f"{model_name} – {input_name}")

        plt.plot([0,1],[0,1],'--',color='black')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title if title else "ROC Curves: All Models & Inputs")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # -----------------------------
    def plot_pr_curves(self, title=None):
        """Plot Precision-Recall curves using minimal collector."""
        if self.collector is None:
            print("Collector not provided. Cannot plot PR curves.")
            return

        plt.figure(figsize=(6,6))

        for (model_name, input_name), (y_true, y_pred, y_proba) in self.collector.data.items():
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            plt.plot(recall, precision, label=f"{model_name} – {input_name}")

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(title if title else "Precision-Recall Curves: All Models & Inputs")
        plt.legend()
        plt.tight_layout()
        plt.show()
