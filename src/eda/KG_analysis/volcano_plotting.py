"""
volcano_plotting.py

Reusable utilities for creating volcano plots from
differential expression results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_volcano(
    df,
    fc_thresh=0.75,
    p_thresh=0.05,
    n_labels=10,
    title="Volcano Plot",
    output_file="volcano_plot.png"
):
    """
    Create a volcano plot from a differential expression DataFrame.

    Required columns in df:
    - log2_fc
    - p_value
    - protein

    Parameters
    ----------
    df : pandas.DataFrame
        Differential expression results.
    fc_thresh : float
        Absolute log2 fold-change threshold.
    p_thresh : float
        P-value significance threshold.
    n_labels : int
        Number of top features (by abs fold change) to label.
    title : str
        Plot title.
    output_file : str
        Path to save the figure.
    """

    # Copy to avoid side effects
    df = df.copy()

    # Compute -log10(p-value)
    df["-log10_p"] = -np.log10(df["p_value"] + 1e-300)

    # Color coding
    df["color"] = "lightgrey"
    df.loc[
        (df["log2_fc"] >= fc_thresh) & (df["p_value"] <= p_thresh),
        "color"
    ] = "#d62728"  # Upregulated

    df.loc[
        (df["log2_fc"] <= -fc_thresh) & (df["p_value"] <= p_thresh),
        "color"
    ] = "#1f77b4"  # Downregulated

    # Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(
        df["log2_fc"],
        df["-log10_p"],
        c=df["color"],
        s=25,
        alpha=0.5,
        edgecolors="none"
    )

    # Threshold lines
    plt.axhline(-np.log10(p_thresh), linestyle="--", alpha=0.3)
    plt.axvline(fc_thresh, linestyle="--", alpha=0.3)
    plt.axvline(-fc_thresh, linestyle="--", alpha=0.3)

    # Label top features
    sig_df = df[df["p_value"] <= p_thresh]
    if not sig_df.empty:
        sig_df = sig_df.assign(abs_fc=sig_df["log2_fc"].abs())
        top_features = sig_df.nlargest(n_labels, "abs_fc")

        for _, row in top_features.iterrows():
            label = str(row["protein"]).replace("Protein_", "")
            plt.text(
                row["log2_fc"],
                row["-log10_p"] + 0.1,
                label,
                fontsize=10,
                fontweight="bold",
                ha="center",
                va="bottom"
            )

    # Formatting
    plt.title(title, fontsize=15, pad=20)
    plt.xlabel(r"$\log_2$ Fold Change", fontsize=12)
    plt.ylabel(r"$-\log_{10}$(P-value)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Upregulated",
               markerfacecolor="#d62728", markersize=9),
        Line2D([0], [0], marker="o", color="w", label="Downregulated",
               markerfacecolor="#1f77b4", markersize=9),
        Line2D([0], [0], marker="o", color="w", label="Not Significant",
               markerfacecolor="lightgrey", markersize=9),
    ]
    plt.legend(handles=legend_elements, loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.show()
