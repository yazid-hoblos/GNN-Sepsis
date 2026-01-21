#!/usr/bin/env python3
"""
Run GO/Reactome enrichment for consensus biomarker proteins (optionally incl. hub proteins).

Inputs:
  - consensus proteins CSV (complex_consensus_proteins.csv)
  - optional hub proteins CSV (kg_subgraph_analysis_consensus/hub_proteins.csv)
Outputs (under output-dir/enrichment/):
  - enriched_<library>.csv for each library
  - top_terms_barplot.png (combined top terms across libraries)

Requirements: gseapy
  pip install gseapy
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


DEFAULT_LIBRARIES = [
    "GO_Biological_Process_2021",
    "GO_Molecular_Function_2021",
    "GO_Cellular_Component_2021",
    "Reactome_2022",
]


def load_consensus_proteins(consensus_csv: Path) -> set:
    df = pd.read_csv(consensus_csv)
    if 'protein' not in df.columns:
        raise ValueError(f"Expected column 'protein' in {consensus_csv}")
    proteins = set(df['protein'].unique())
    logger.info(f"Loaded {len(proteins)} consensus proteins")
    return proteins


def load_hub_proteins(hub_csv: Path) -> set:
    df = pd.read_csv(hub_csv)
    col = 'protein' if 'protein' in df.columns else 'protein_name'
    hubs = set(df[col].unique())
    logger.info(f"Loaded {len(hubs)} hub proteins")
    return hubs


def run_enrichment(genes: list, libraries: list, out_dir: Path):
    try:
        import gseapy as gp
    except ImportError as e:
        logger.error("gseapy not installed. Install with: pip install gseapy")
        raise e

    results = []
    for lib in libraries:
        logger.info(f"Running Enrichr for {lib} on {len(genes)} genes")
        enr = gp.enrichr(gene_list=genes, gene_sets=lib, cutoff=1.0, outdir=None)
        if enr.results is None or enr.results.empty:
            logger.warning(f"No results for {lib}")
            continue
        df = enr.results.copy()
        df.insert(0, 'library', lib)
        results.append(df)
        csv_path = out_dir / f"enriched_{lib}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved: {csv_path}")
    if not results:
        logger.warning("No enrichment results produced.")
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


def plot_top_terms(enr_df: pd.DataFrame, top_n: int, out_path: Path):
    if enr_df.empty:
        logger.warning("No data to plot top terms")
        return
    # Select top N across all libraries by combined score (or p-value)
    enr_df = enr_df.copy()
    if 'Combined Score' in enr_df.columns:
        enr_df.sort_values('Combined Score', ascending=False, inplace=True)
    elif 'P-value' in enr_df.columns:
        enr_df.sort_values('P-value', ascending=True, inplace=True)
    top = enr_df.head(top_n)
    plt.figure(figsize=(10, 8))
    sns.barplot(data=top, y='Term', x='Combined Score' if 'Combined Score' in top.columns else '-log10(pvalue)', hue='library')
    plt.title(f"Top {top_n} Enriched Terms")
    plt.xlabel('Combined Score' if 'Combined Score' in top.columns else 'Score')
    plt.ylabel('Term')
    plt.legend(title='Library')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='GO/Reactome enrichment for consensus biomarker proteins')
    parser.add_argument('--consensus-proteins', type=str,
                        default='results/interpretability/complex_svm_mapped/complex_consensus_proteins.csv')
    parser.add_argument('--hub-proteins', type=str,
                        default='results/interpretability/complex_svm_mapped/kg_subgraph_analysis_consensus/hub_proteins.csv')
    parser.add_argument('--include-hubs', action='store_true',
                        help='Include hub proteins along with consensus proteins')
    parser.add_argument('--libraries', type=str, nargs='+', default=DEFAULT_LIBRARIES,
                        help='Enrichr libraries to use')
    parser.add_argument('--output-dir', type=str,
                        default='results/interpretability/complex_svm_mapped/validation_consensus/enrichment')
    parser.add_argument('--top-n', type=int, default=15, help='Top terms to plot')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    genes = load_consensus_proteins(Path(args.consensus_proteins))
    if args.include_hubs and Path(args.hub_proteins).exists():
        hubs = load_hub_proteins(Path(args.hub_proteins))
        genes |= hubs
        logger.info(f"Total genes after adding hubs: {len(genes)}")

    gene_symbols = [g.replace('Protein_', '') for g in genes]

    enr_df = run_enrichment(gene_symbols, args.libraries, out_dir)

    # Plot top terms across libraries
    plot_top_terms(enr_df, args.top_n, out_dir / 'top_terms_barplot.png')

    # Save combined table
    if not enr_df.empty:
        combined_csv = out_dir / 'enrichment_combined.csv'
        enr_df.to_csv(combined_csv, index=False)
        logger.info(f"Saved combined enrichment: {combined_csv}")

    logger.info("Enrichment complete.")


if __name__ == '__main__':
    main()
