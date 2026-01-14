"""
Auto-map Feature_i indices to RGCN entity labels.

Usage (standalone):
    python -m src.ml.auto_feature_mapping --version v2.11 --embedding-type RGCN --features-csv results/interpretability/v2.11_none/consolidated/consensus_biomarkers.csv
    
Or integrate into consolidate_interpretability.py
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def auto_map_features_to_entities(features: list, version: str = 'v2.11', 
                                   embedding_type: str = 'RGCN', top_k: int = 3) -> dict:
    """
    Map Feature_i indices to top entities in that embedding dimension.
    
    Parameters
    ----------
    features : list
        List of feature names like ['Feature_73', 'Feature_63', ...]
    version : str
        Model version (e.g., v2.11)
    embedding_type : str
        Embedding type ('RGCN', 'Complex', etc.)
    top_k : int
        Number of top entities to list per feature
    
    Returns
    -------
    mapping : dict
        Feature_i -> "Entity1; Entity2; Entity3"
    """
    embeddings_path = Path(f"models/executions/GSE54514_enriched_ontology_degfilter{version}") / f"outputmodel_{embedding_type}_entity_embeddings.npy"
    entity_mapping_path = Path(f"models/executions/GSE54514_enriched_ontology_degfilter{version}") / f"outputmodel_{embedding_type}_entity_mapping.csv"
    
    if not embeddings_path.exists():
        print(f"✗ Embeddings not found: {embeddings_path}")
        return {}
    if not entity_mapping_path.exists():
        print(f"✗ Entity mapping not found: {entity_mapping_path}")
        return {}
    
    print(f"Loading {embedding_type} embeddings ({version})...")
    embeddings = np.load(embeddings_path)
    entities_df = pd.read_csv(entity_mapping_path)
    
    if 'label' not in entities_df.columns:
        if 'name' in entities_df.columns:
            entities_df['label'] = entities_df['name']
        else:
            entities_df['label'] = entities_df.iloc[:, 0]
    
    print(f"Embeddings shape: {embeddings.shape}; Entities: {len(entities_df)}")
    
    mapping = {}
    for feat in features:
        try:
            idx = int(feat.split('_')[1])
            if idx >= embeddings.shape[1]:
                print(f"  ⚠ Feature__{idx} out of bounds (max dim: {embeddings.shape[1]-1})")
                continue
            
            dim_vals = np.abs(embeddings[:, idx])
            top_indices = np.argsort(dim_vals)[-top_k:][::-1]
            top_labels = entities_df.iloc[top_indices]['label'].tolist()
            mapping[feat] = '; '.join(top_labels)
        except Exception as e:
            print(f"  ⚠ Could not map {feat}: {e}")
    
    return mapping


def main():
    parser = argparse.ArgumentParser(description="Auto-map Feature_i to entity labels")
    parser.add_argument('--version', default='v2.11', help='Model version (e.g., v2.11)')
    parser.add_argument('--embedding-type', default='RGCN', help='Embedding type (RGCN, Complex, etc.)')
    parser.add_argument('--features-csv', help='CSV file with consensus features (will read "feature" column)')
    parser.add_argument('--top-k', type=int, default=3, help='Top K entities per feature')
    parser.add_argument('--output', help='Optional: save mapping to CSV')
    args = parser.parse_args()
    
    if args.features_csv:
        df = pd.read_csv(args.features_csv)
        features = df['feature'].tolist()
    else:
        print("Please provide --features-csv with a consensus file")
        return
    
    mapping = auto_map_features_to_entities(features, version=args.version, 
                                            embedding_type=args.embedding_type, 
                                            top_k=args.top_k)
    
    print(f"\n✓ Mapped {len(mapping)} features")
    for feat, label in sorted(mapping.items())[:10]:
        print(f"  {feat} -> {label}")
    
    if args.output:
        out_df = pd.DataFrame(list(mapping.items()), columns=['feature', 'mapped_feature'])
        out_df.to_csv(args.output, index=False)
        print(f"✓ Saved mapping to {args.output}")


if __name__ == '__main__':
    main()
