"""
Analyze RGCN Important Dimensions
----------------------------------

Maps the top RGCN embedding dimensions (from ML interpretability)
back to biological entities to understand what they capture.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.ml.gnn_interpretability import GNNInterpreter
from src.ml.load_matrix import load_df

# Configuration
RGCN_MODEL_DIR = Path("models/executions/GSE54514_enriched_ontology_degfilterv2.9")
EMBEDDINGS_PATH = RGCN_MODEL_DIR / "outputmodel_RGCN_entity_embeddings.npy"
ENTITY_MAPPING = RGCN_MODEL_DIR / "outputmodel_RGCN_entity_mapping.csv"
NODE_FEATURES = RGCN_MODEL_DIR / "node_features.csv"

# Top 10 RGCN dimensions from interpretability analysis
IMPORTANT_DIMS = [31, 33, 12, 32, 67, 38, 26, 56, 63, 45]


def main():
    print("="*80)
    print("RGCN DIMENSION ANALYSIS")
    print("="*80)
    
    # Check paths
    if not EMBEDDINGS_PATH.exists():
        print(f"✗ Embeddings not found: {EMBEDDINGS_PATH}")
        print("\nAvailable model directories:")
        models_dir = Path("models/executions")
        if models_dir.exists():
            for d in models_dir.iterdir():
                if d.is_dir():
                    print(f"  {d.name}")
        return
    
    # Initialize interpreter
    interpreter = GNNInterpreter(
        embeddings_path=str(EMBEDDINGS_PATH),
        entity_mapping_path=str(ENTITY_MAPPING),
        node_features_path=str(NODE_FEATURES) if NODE_FEATURES.exists() else None,
        embedding_type='RGCN'
    )
    
    # Analyze important dimensions
    print("\n" + "="*80)
    print(f"ANALYZING TOP {len(IMPORTANT_DIMS)} IMPORTANT DIMENSIONS")
    print("="*80)
    
    results = interpreter.analyze_important_dimensions(IMPORTANT_DIMS, top_k=10)
    
    # Save results
    output_dir = Path("results/gnn_interpretability")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving detailed results to {output_dir}/...")
    
    for dim, entities_df in results.items():
        output_file = output_dir / f"dimension_{dim}_entities.csv"
        entities_df.to_csv(output_file, index=False)
        print(f"  ✓ Dimension {dim}: {output_file}")
    
    # Create summary
    summary_rows = []
    for dim, entities_df in results.items():
        top_entity = entities_df[entities_df['direction'] == 'positive'].iloc[0]
        
        entity_label = top_entity.get('label', top_entity.get('node_id', f'Entity_{top_entity.get("entity_id", "?")}'))
        
        summary_rows.append({
            'dimension': dim,
            'top_entity': entity_label,
            'embedding_value': top_entity['embedding_value'],
            'n_entities_analyzed': len(entities_df)
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_file = output_dir / "dimension_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    
    print(f"\n✓ Summary saved: {summary_file}")
    
    # Plot distributions for top 5 dimensions
    print(f"\nGenerating distribution plots...")
    
    for dim in IMPORTANT_DIMS[:5]:
        output_plot = output_dir / f"dimension_{dim}_distribution.png"
        interpreter.plot_dimension_distribution(dim, output_path=str(output_plot), show=False)
    
    # Correlation matrix
    print(f"\nGenerating correlation matrix...")
    corr_plot = output_dir / "dimension_correlation.png"
    interpreter.plot_dimension_correlation_matrix(IMPORTANT_DIMS, 
                                                  output_path=str(corr_plot), 
                                                  show=False)
    
    # Compare patient groups
    print(f"\nComparing patient groups across dimensions...")
    
    # Load patient labels
    df = load_df('RGCN_protein_embeddings')
    y = df['disease_status'].values
    
    patient_comparison = {}
    for dim in IMPORTANT_DIMS:
        comparison = interpreter.compare_patient_groups(y, dim)
        patient_comparison[f'Dim_{dim}'] = comparison
        
        if comparison['test']['significant']:
            print(f"  Dimension {dim}: SIGNIFICANT difference (p={comparison['test']['p_value']:.4f})")
            print(f"    Control: {comparison['Control']['mean']:.4f} ± {comparison['Control']['std']:.4f}")
            print(f"    Septic:  {comparison['Septic']['mean']:.4f} ± {comparison['Septic']['std']:.4f}")
    
    # Save patient comparison
    comparison_file = output_dir / "patient_group_comparison.json"
    import json
    with open(comparison_file, 'w') as f:
        json.dump(patient_comparison, f, indent=2)
    
    print(f"\n✓ Patient comparison saved: {comparison_file}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nAnalyzed {len(IMPORTANT_DIMS)} dimensions")
    print(f"Results saved to: {output_dir}/")
    print(f"\nFiles created:")
    print(f"  - dimension_*_entities.csv: Top entities per dimension")
    print(f"  - dimension_summary.csv: Overview of all dimensions")
    print(f"  - dimension_*_distribution.png: Value distributions")
    print(f"  - dimension_correlation.png: Correlation matrix")
    print(f"  - patient_group_comparison.json: Statistical tests")
    
    # Key findings
    n_significant = sum(1 for d in IMPORTANT_DIMS 
                       if patient_comparison[f'Dim_{d}']['test']['significant'])
    
    print(f"\nKey findings:")
    print(f"  - {n_significant}/{len(IMPORTANT_DIMS)} dimensions show significant differences between groups")
    print(f"  - These dimensions likely capture disease-relevant biology")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
