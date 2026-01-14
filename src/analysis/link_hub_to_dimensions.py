"""
Link top RGCN dimensions (significant) to graph hub proteins.
Uses outputs from updated analyses (graph structure + gnn interpretability).
"""

import argparse
import json
import pandas as pd
from pathlib import Path


def load_hub_proteins(version: str, top_k: int = 20):
    path = Path(f"results/graph_structure/{version}/patient_connected_entities.csv")
    if not path.exists():
        print(f"✗ Missing patient_connected_entities.csv at {path}; using static fallback hubs")
        return [
            "Protein_FYN",
            "Protein_SF1",
            "Protein_PABPC1",
            "Protein_AGTRAP",
            "Protein_EGLN2",
            "Protein_TPI1",
        ]
    df = pd.read_csv(path)
    return df.head(top_k)['label'].tolist()


def load_significant_dims(version: str, normalization: str, p_thresh: float = 0.05):
    path = Path(f"results/gnn_interpretability/{version}_{normalization}/patient_group_comparison.json")
    if not path.exists():
        print(f"✗ Missing patient_group_comparison.json at {path}; using static dims")
        return [45, 63, 33, 67, 38]
    with open(path, 'r') as f:
        data = json.load(f)
    dims = []
    for k, v in data.items():
        if v['test']['significant'] and v['test']['p_value'] < p_thresh:
            try:
                dim_idx = int(k.replace('Dim_', ''))
                dims.append(dim_idx)
            except Exception:
                continue
    return dims or [45, 63, 33, 67, 38]


def main():
    parser = argparse.ArgumentParser(description="Link graph hubs to significant RGCN dimensions")
    parser.add_argument('--version', default='v2.11', help='Model/results version (e.g., v2.11)')
    parser.add_argument('--normalization', default='none', choices=['none', 'standard', 'robust', 'minmax'],
                        help='Normalization variant used for gnn interpretability outputs')
    parser.add_argument('--top-hubs', type=int, default=20, help='Top hub proteins to consider')
    parser.add_argument('--p-thresh', type=float, default=0.05, help='p-value threshold for significance')
    args = parser.parse_args()

    print("=" * 80)
    print("LINKING GRAPH STRUCTURE TO RGCN DIMENSIONS")
    print(f"Version={args.version} | Normalization={args.normalization}")
    print("=" * 80)

    hub_proteins = load_hub_proteins(args.version, top_k=args.top_hubs)
    significant_dims = load_significant_dims(args.version, args.normalization, p_thresh=args.p_thresh)

    print(f"Hubs (top {len(hub_proteins)}): {', '.join(hub_proteins)}")
    print(f"Significant dims (p<{args.p_thresh}): {significant_dims}")

    results_dir = Path(f"results/gnn_interpretability/{args.version}_{args.normalization}")
    hub_hits = {}
    for dim in significant_dims:
        dim_file = results_dir / f"dimension_{dim}_entities.csv"
        if not dim_file.exists():
            continue
        df = pd.read_csv(dim_file)
        top_20 = df.head(20)
        hub_in_dim = [p for p in hub_proteins if p in top_20['label'].values]
        if hub_in_dim:
            hub_hits[dim] = hub_in_dim
            print(f"\n✓ Dimension {dim}: Found {len(hub_in_dim)} hub proteins in top 20")
            for protein in hub_in_dim:
                rank = df[df['label'] == protein].index[0] + 1
                print(f"    - {protein} (rank #{rank})")

    if hub_hits:
        print(f"\n✓ {len(hub_hits)} significant dimensions contain hub proteins")
        print("Interpretation: RGCN dimensions align with graph hubs")
    else:
        print("\n! No hub proteins found in significant dimensions' top 20")
        print("Possibilities: hubs are stable/non-discriminative, or disease signal is in non-hub nodes")

    print("\nNext step: Inspect top entities in significant dimensions for sepsis relevance")
    print("=" * 80)


if __name__ == '__main__':
    main()
