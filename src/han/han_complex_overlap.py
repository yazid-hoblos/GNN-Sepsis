#!/usr/bin/env python3
"""
Summarize HAN patient neighborhoods and compute overlap with ComplEx consensus proteins.

Outputs (default: results/interpretability/complex_svm_mapped/overlap_han_complex/):
- han_protein_frequency.csv : protein frequencies across selected patients
- overlap_proteins.csv      : proteins present in both HAN neighborhoods and ComplEx consensus
- overlap_summary.txt       : quick summary counts
"""

import argparse
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

from src.han.owl_data_loader import OWLHeteroGraphBuilder
from src.han.han_heterogeneous_analysis import extract_heterogeneous_neighborhood

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


def select_patients(preds: pd.DataFrame, k: int = 3) -> List[int]:
    septic = preds[preds['predicted_label'] == 1].nlargest(k, 'prob_septic')['sample_idx'].tolist()
    healthy = preds[preds['predicted_label'] == 0].nsmallest(k, 'prob_septic')['sample_idx'].tolist()
    selected = [int(x) for x in septic + healthy]
    logger.info(f"Selected patients -> septic: {septic}, healthy: {healthy}")
    return selected


def aggregate_han_proteins(
    data,
    patients: List[int],
    protein_idx_to_name: Dict[int, str],
    max_per_patient: int = 10,
) -> Counter:
    """Count proteins appearing in HAN neighborhoods, limited to top-N per patient to avoid noise."""
    counter = Counter()
    for pid in patients:
        neighbors_by_type, _ = extract_heterogeneous_neighborhood(data, pid)
        local_counter = Counter()
        for nodes in neighbors_by_type.get('Protein', {}).values():
            for prot_idx in nodes:
                name = protein_idx_to_name.get(prot_idx, str(prot_idx))
                local_counter[name.upper()] += 1
        for name, _ in local_counter.most_common(max_per_patient):
            counter[name] += 1
    return counter


def load_consensus(consensus_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(consensus_csv)
    if not {'protein', 'protein_name', 'avg_score', 'num_models'}.issubset(df.columns):
        raise ValueError("Consensus CSV missing required columns")
    df['protein_name_upper'] = df['protein_name'].str.upper()
    return df


def compute_overlap(han_counts: Counter, consensus_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, patient_count in han_counts.items():
        match = consensus_df[consensus_df['protein_name_upper'] == name]
        if not match.empty:
            row = match.iloc[0]
            rows.append({
                'protein': row['protein'],
                'protein_name': row['protein_name'],
                'han_patient_count': patient_count,
                'han_occurrences': patient_count,  # unique per patient aggregation
                'complex_num_models': row['num_models'],
                'complex_avg_score': row['avg_score'],
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=['protein','protein_name','han_patient_count','han_occurrences','complex_num_models','complex_avg_score'])
    return df.sort_values(['han_patient_count', 'complex_num_models', 'complex_avg_score'], ascending=False)


def main():
    parser = argparse.ArgumentParser(description='Overlap between HAN neighborhoods and ComplEx consensus proteins')
    parser.add_argument('--owl-path', type=Path, default=Path('output/new_outputs/GSE54514_enriched_ontology_degfilter_v2.11.owl'))
    parser.add_argument('--predictions-path', type=Path, default=Path('results/han_model_with_expression/sample_predictions.csv'))
    parser.add_argument('--consensus-csv', type=Path, default=Path('results/interpretability/complex_svm_mapped/complex_consensus_proteins.csv'))
    parser.add_argument('--output-dir', type=Path, default=Path('results/interpretability/complex_svm_mapped/overlap_han_complex'))
    parser.add_argument('--top-k-per-class', type=int, default=3, help='Top K patients per class to aggregate')
    parser.add_argument('--max-proteins-per-patient', type=int, default=10000, help='Limit HAN proteins per patient to top-N occurrences')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading graph and building protein mapping...")
    builder = OWLHeteroGraphBuilder(args.owl_path)
    node_to_type = builder._parse_individuals()
    builder._parse_edges()
    builder._build_node_mappings(node_to_type)
    data = builder.build_hetero_data()

    # Map protein index -> gene symbol using nodes.csv if available
    protein_idx_to_name: Dict[int, str] = {}
    nodes_csv = args.owl_path.parent / f"{args.owl_path.stem}_csv" / 'nodes.csv'
    nodes_df = None
    if nodes_csv.exists():
        nodes_df = pd.read_csv(nodes_csv)
        nodes_df = nodes_df.set_index('node_id')

    for (ntype, node_id), idx in builder.node_id_to_idx.items():
        if ntype != 'Protein':
            continue
        prot_name = str(node_id)
        if nodes_df is not None and node_id in nodes_df.index:
            cand = nodes_df.loc[node_id].get('hasName', prot_name)
            if isinstance(cand, str) and cand.strip():
                prot_name = cand.strip()
        # Fallback: strip prefix
        prot_name = prot_name.replace('Protein_', '')
        protein_idx_to_name[idx] = prot_name

    preds = pd.read_csv(args.predictions_path)
    patients = select_patients(preds, k=args.top_k_per_class)

    han_counts = aggregate_han_proteins(data, patients, protein_idx_to_name, max_per_patient=args.max_proteins_per_patient)
    han_df = pd.DataFrame([
        {'protein_name': name, 'han_patient_count': cnt}
        for name, cnt in han_counts.items()
    ]).sort_values('han_patient_count', ascending=False)
    han_csv = args.output_dir / 'han_protein_frequency.csv'
    han_df.to_csv(han_csv, index=False)
    logger.info(f"Saved HAN protein frequency: {han_csv}")

    consensus_df = load_consensus(args.consensus_csv)
    overlap_df = compute_overlap(han_counts, consensus_df)
    overlap_csv = args.output_dir / 'overlap_proteins.csv'
    overlap_df.to_csv(overlap_csv, index=False)
    logger.info(f"Saved overlap proteins: {overlap_csv}")

    summary_path = args.output_dir / 'overlap_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"Patients analyzed: {patients}\n")
        f.write(f"HAN proteins: {len(han_counts)}\n")
        f.write(f"Consensus proteins: {len(consensus_df)}\n")
        f.write(f"Overlap proteins: {len(overlap_df)}\n")
    logger.info(f"Saved summary: {summary_path}")


if __name__ == '__main__':
    main()
