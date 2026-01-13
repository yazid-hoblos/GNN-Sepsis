"""
Link Top RGCN Dimensions to Graph Hub Proteins
-----------------------------------------------

Check if hub proteins from graph structure analysis 
appear in the significant disease-associated RGCN dimensions.
"""

import pandas as pd
from pathlib import Path

# Load the key results
print("="*80)
print("LINKING GRAPH STRUCTURE TO RGCN DIMENSIONS")
print("="*80)

# Hub proteins from graph analysis
hub_proteins = [
    "Protein_FYN",      # 777 connections
    "Protein_SF1",      # 545 connections
    "Protein_PABPC1",   # 496 connections
    "Protein_AGTRAP",   # 437 connections
    "Protein_EGLN2",    # 446 connections
    "Protein_TPI1",     # 412 connections
]

# Significant RGCN dimensions (from analyze_rgcn_dimensions.py)
significant_dims = [45, 63, 33, 67, 38]  # p < 0.05

print(f"\nChecking {len(hub_proteins)} hub proteins across {len(significant_dims)} significant dimensions:")
print(f"Hubs: {', '.join(hub_proteins)}")
print(f"Significant dims (p<0.05): {significant_dims}")

# Check each significant dimension
results_dir = Path("results/gnn_interpretability")

hub_hits = {}
for dim in significant_dims:
    dim_file = results_dir / f"dimension_{dim}_entities.csv"
    
    if dim_file.exists():
        df = pd.read_csv(dim_file)
        
        # Check how many hub proteins are in this dimension's top entities
        top_20 = df.head(20)
        hub_in_dim = [p for p in hub_proteins if p in top_20['label'].values]
        
        if hub_in_dim:
            hub_hits[dim] = hub_in_dim
            print(f"\n✓ Dimension {dim}: Found {len(hub_in_dim)} hub proteins in top 20")
            for protein in hub_in_dim:
                rank = df[df['label'] == protein].index[0] + 1
                print(f"    - {protein} (rank #{rank})")

if hub_hits:
    print(f"\n✓ {len(hub_hits)} significant dimensions contain hub proteins!")
    print("\nInterpretation:")
    print("  Hub proteins from graph structure appear in disease-significant dimensions")
    print("  This validates that RGCN captures the biological importance of these proteins")
else:
    print("\n! Hub proteins not in significant dimensions' top 20")
    print("This suggests either:")
    print("  1. Hub proteins may have stable expression (not disease-discriminative)")
    print("  2. Disease signal comes from less-connected proteins")
    print("  3. Need to check top proteins in significant dimensions for biological relevance")

print("\n" + "="*80)
print("Next step: Check top proteins in significant dimensions for sepsis relevance")
print("="*80)
