import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_enrichment_results(df, term_type, top_n=10, 
                            x_label='Regulation Category', 
                            title_prefix='Comparative Enrichment',
                            fdr_threshold=0.5,
                            output_file=None):
    """
    Plot biological enrichment results as a bubble chart.

    Each bubble represents a biological term:
        - Bubble size reflects the Gene Ratio: the proportion of proteins
          in the category associated with that term.
        - Bubble color reflects significance (-log10 FDR).

    Gene Ratio definition:
        (# of proteins in the category associated with the term) / 
        (total number of proteins in that category)
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing enrichment results. Required columns:
        ['Term_type', 'Category', 'Term_name', 'FDR', 'Gene_Ratio', '-log10_FDR', 'Term']
    term_type : str
        Type of term to visualize ('GO' or 'Pathway').
    top_n : int, default=10
        Number of top significant terms to display per category.
    x_label : str, default='Regulation Category'
        Label for the X-axis.
    title_prefix : str, default='Comparative Enrichment'
        Prefix for the plot title.
    fdr_threshold : float, default=0.5
        Maximum FDR to include terms in the plot.
    output_file : str or None, default=None
        File path to save the plot (PNG, PDF, etc.). If None, the plot is shown interactively.
    """

    # Filter data for the selected term type
    type_df = df[df['Term_type'] == term_type].copy()
    categories = type_df['Category'].unique()

    # Select top N significant terms for each category
    selected_terms = []
    for cat in categories:
        top_terms = type_df[type_df['Category'] == cat].nsmallest(top_n, 'FDR')['Term']
        selected_terms.extend(top_terms)

    unique_terms = list(set(selected_terms))

    # Filter for plotting
    plot_data = type_df[type_df['Term'].isin(unique_terms) & (type_df['FDR'] <= fdr_threshold)].copy()
    plot_data = plot_data.sort_values(['Category', '-log10_FDR'], ascending=[False, True])

    # --- Plot Setup ---
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.set_style("whitegrid")

    # Use a continuous colormap for significance
    cmap = plt.get_cmap("magma")  # gradient from dark purple to orange
    norm = plt.Normalize(plot_data['-log10_FDR'].min(), plot_data['-log10_FDR'].max())

    # Scatter plot: bubble size = Gene Ratio, color = significance
    scatter = ax.scatter(
        x=plot_data['Category'],
        y=plot_data['Term_name'],
        c=plot_data['-log10_FDR'],
        s=plot_data['Gene_Ratio'] * 600,
        cmap=cmap,
        norm=norm,
        edgecolors='black',
        linewidths=0.8,
        zorder=3
    )

    # Colorbar for -log10 FDR
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, pad=0.05)
    cbar.set_label('-log10(FDR)', fontsize=12, fontweight='bold')
    cbar.ax.axhline(1.3, color='red', lw=2, ls='--', label='p=0.05')

    # Legend for bubble sizes (Gene Ratio)
    for r in [0.2, 0.5, 0.8, 1.0]:
        ax.scatter([], [], s=r*600, c='white', edgecolors='black', label=f'{int(r*100)}%')
    ax.legend(title='Gene Ratio\n(% of category)', bbox_to_anchor=(1.25, 1), loc='upper left')

    # X-axis formatting
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    ax.set_xlim(-0.5, len(categories) - 0.5)

    # Axis labels and title
    ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
    ax.set_ylabel('Biological Term', fontsize=12, fontweight='bold')
    plt.title(f'{title_prefix}\n', fontsize=14, pad=20)

    plt.tight_layout()

    # Save or show the plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
