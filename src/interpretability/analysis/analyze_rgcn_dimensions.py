"""
Analyze RGCN Important Dimensions
-----------------------------------

Maps top RGCN embedding dimensions (from consolidated interpretability)
back to biological entities to understand what they capture.
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.ml.gnn_interpretability import GNNInterpreter
from src.ml.load_matrix import load_df


def load_top_dimensions(version: str, normalization: str, top_k: int = 10):
    """Load top feature indices from consolidated consensus file."""
    cons_dir = Path(f"results/interpretability/{version}_{normalization}/consolidated")
    cons_path = cons_dir / "consensus_biomarkers.csv"
    if not cons_path.exists():
        print(f"✗ Missing consensus file: {cons_path}; falling back to static list")
        return [31, 33, 12, 32, 67, 38, 26, 56, 63, 45]

    df = pd.read_csv(cons_path)
    # Features are named Feature_<idx>
    df = df[df['feature'].str.startswith('Feature_')]
    df = df.head(top_k)
    dims = []
    for feat in df['feature']:
        try:
            dims.append(int(feat.split('_')[1]))
        except Exception:
            continue
    if not dims:
        print("✗ No parsable Feature_<idx> entries; falling back to static list")
        return [31, 33, 12, 32, 67, 38, 26, 56, 63, 45]
    return dims


def main():
    parser = argparse.ArgumentParser(description="Analyze RGCN embedding dimensions")
    parser.add_argument('--version', default='v2.11', help='Model/results version tag (e.g., v2.11)')
    parser.add_argument('--normalization', default='none', choices=['none', 'standard', 'robust', 'minmax'],
                        help='Normalization variant used for interpretability results')
    parser.add_argument('--top-dims', type=int, default=10, help='Number of top dimensions to analyze')
    args = parser.parse_args()

    print("=" * 80)
    print("RGCN DIMENSION ANALYSIS")
    print(f"Version={args.version} | Normalization={args.normalization}")
    print("=" * 80)

    model_dir = Path(f"models/executions/GSE54514_enriched_ontology_degfilter{args.version}")
    embeddings_path = model_dir / "outputmodel_RGCN_entity_embeddings.npy"
    entity_mapping = model_dir / "outputmodel_RGCN_entity_mapping.csv"
    node_features = model_dir / "node_features.csv"

    if not embeddings_path.exists():
        print(f"✗ Embeddings not found: {embeddings_path}")
        print("Available model dirs under models/executions:")
        base = Path("models/executions")
        for d in base.glob('*'):
            if d.is_dir():
                print(f"  - {d.name}")
        return

    important_dims = load_top_dimensions(args.version, args.normalization, top_k=args.top_dims)
    print(f"Using top dimensions: {important_dims}")

    interpreter = GNNInterpreter(
        embeddings_path=str(embeddings_path),
        entity_mapping_path=str(entity_mapping),
        node_features_path=str(node_features) if node_features.exists() else None,
        embedding_type='RGCN'
    )

    print("\nAnalyzing dimensions...")
    results = interpreter.analyze_important_dimensions(important_dims, top_k=10)

    output_dir = Path(f"results/gnn_interpretability/{args.version}_{args.normalization}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving detailed results to {output_dir}/")

    for dim, entities_df in results.items():
        output_file = output_dir / f"dimension_{dim}_entities.csv"
        entities_df.to_csv(output_file, index=False)
        print(f"  ✓ Dimension {dim}: {output_file}")

    summary_rows = []
    for dim, entities_df in results.items():
        # pick top positive entity if available
        pos_df = entities_df[entities_df['direction'] == 'positive']
        top_entity_row = pos_df.iloc[0] if len(pos_df) else entities_df.iloc[0]
        entity_label = top_entity_row.get('label', top_entity_row.get('node_id', f"Entity_{top_entity_row.get('entity_id', '?')}"))
        summary_rows.append({
            'dimension': dim,
            'top_entity': entity_label,
            'embedding_value': top_entity_row.get('embedding_value', np.nan),
            'n_entities_analyzed': len(entities_df)
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_file = output_dir / "dimension_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\n✓ Summary saved: {summary_file}")

    print("\nGenerating distribution plots...")
    for dim in important_dims[:5]:
        output_plot = output_dir / f"dimension_{dim}_distribution.png"
        interpreter.plot_dimension_distribution(dim, output_path=str(output_plot), show=False)

    print("\nGenerating correlation matrix...")
    corr_plot = output_dir / "dimension_correlation.png"
    interpreter.plot_dimension_correlation_matrix(important_dims,
                                                  output_path=str(corr_plot),
                                                  show=False)

    print("\nComparing patient groups across dimensions...")
    df = load_df('RGCN_protein_embeddings')
    y = df['disease_status'].values

    patient_comparison = {}
    for dim in important_dims:
        comparison = interpreter.compare_patient_groups(y, dim)
        patient_comparison[f'Dim_{dim}'] = comparison
        if comparison['test']['significant']:
            print(f"  Dimension {dim}: SIGNIFICANT (p={comparison['test']['p_value']:.4f})")
            print(f"    Control: {comparison['Control']['mean']:.4f} ± {comparison['Control']['std']:.4f}")
            print(f"    Septic:  {comparison['Septic']['mean']:.4f} ± {comparison['Septic']['std']:.4f}")

    comparison_file = output_dir / "patient_group_comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump(patient_comparison, f, indent=2)
    print(f"\n✓ Patient comparison saved: {comparison_file}")

    n_significant = sum(1 for d in important_dims if patient_comparison[f'Dim_{d}']['test']['significant'])
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Analyzed {len(important_dims)} dimensions; significant: {n_significant}")
    print(f"Results: {output_dir}/")
    print("Files: dimension_*_entities.csv, dimension_summary.csv, dimension_*_distribution.png, "
          "dimension_correlation.png, patient_group_comparison.json")
    print("=" * 80)


if __name__ == '__main__':
    main()
