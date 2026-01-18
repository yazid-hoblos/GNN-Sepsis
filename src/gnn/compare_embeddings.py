#!/usr/bin/env python3
"""
Compare embeddings trained on full KG vs optimized KG.

Compares:
  - Embedding statistics (shape, norms, variance)
  - Entity representation similarity
  - Downstream ML model performance
  - Entity importance changes
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys

def load_embeddings(emb_file, mapping_file):
    """Load embeddings and entity mapping."""
    embeddings = np.load(emb_file)
    mapping = pd.read_csv(mapping_file)
    return embeddings, mapping

def compare_shapes(old_emb, new_emb, old_map, new_map):
    """Compare embedding shapes and entity counts."""
    print("\n" + "="*60)
    print("EMBEDDING SHAPE COMPARISON")
    print("="*60)
    
    print(f"Old KG (full):")
    print(f"  - Entities: {old_emb.shape[0]}")
    print(f"  - Embedding dim: {old_emb.shape[1]}")
    print(f"  - Memory: {old_emb.nbytes / 1e6:.2f} MB")
    
    print(f"\nNew KG (optimized):")
    print(f"  - Entities: {new_emb.shape[0]}")
    print(f"  - Embedding dim: {new_emb.shape[1]}")
    print(f"  - Memory: {new_emb.nbytes / 1e6:.2f} MB")
    
    print(f"\nReduction:")
    print(f"  - Entity count: {old_emb.shape[0]} → {new_emb.shape[0]} ({100*new_emb.shape[0]/old_emb.shape[0]:.1f}%)")

def compare_statistics(old_emb, new_emb):
    """Compare embedding statistics."""
    print("\n" + "="*60)
    print("EMBEDDING STATISTICS")
    print("="*60)
    
    # Handle complex embeddings (e.g., from ComplEx) by taking absolute value
    old_emb_real = np.abs(old_emb) if np.iscomplexobj(old_emb) else old_emb
    new_emb_real = np.abs(new_emb) if np.iscomplexobj(new_emb) else new_emb
    
    # Norms
    old_norms = np.linalg.norm(old_emb_real, axis=1)
    new_norms = np.linalg.norm(new_emb_real, axis=1)
    
    print(f"\nEmbedding L2 Norms:")
    print(f"  Old KG - Mean: {old_norms.mean():.4f}, Std: {old_norms.std():.4f}, Min: {old_norms.min():.4f}, Max: {old_norms.max():.4f}")
    print(f"  New KG - Mean: {new_norms.mean():.4f}, Std: {new_norms.std():.4f}, Min: {new_norms.min():.4f}, Max: {new_norms.max():.4f}")
    
    # Per-dimension variance
    old_var = old_emb_real.var(axis=0)
    new_var = new_emb_real.var(axis=0)
    
    print(f"\nPer-dimension variance:")
    print(f"  Old KG - Mean: {old_var.mean():.6f}, Std: {old_var.std():.6f}")
    print(f"  New KG - Mean: {new_var.mean():.6f}, Std: {new_var.std():.6f}")
    
    return old_norms, new_norms, old_var, new_var

def find_shared_entities(old_map, new_map):
    """Find entities present in both embeddings."""
    old_entities = set(old_map['label'])
    new_entities = set(new_map['label'])
    shared = old_entities & new_entities
    
    print("\n" + "="*60)
    print("SHARED ENTITIES")
    print("="*60)
    print(f"Old KG entities: {len(old_entities)}")
    print(f"New KG entities: {len(new_entities)}")
    print(f"Shared entities: {len(shared)} ({100*len(shared)/len(old_entities):.1f}% of old, {100*len(shared)/len(new_entities):.1f}% of new)")
    print(f"New entities (not in old): {len(new_entities - old_entities)}")
    print(f"Removed entities (in old but not new): {len(old_entities - new_entities)}")
    
    return shared, old_entities, new_entities

def compare_shared_embeddings(old_emb, old_map, new_emb, new_map, shared_entities):
    """Compare embeddings of shared entities."""
    print("\n" + "="*60)
    print("SHARED ENTITY REPRESENTATION COMPARISON")
    print("="*60)
    
    # Build mappings for shared entities
    old_id_map = {label: idx for idx, label in zip(old_map['entity_id'], old_map['label'])}
    new_id_map = {label: idx for idx, label in zip(new_map['entity_id'], new_map['label'])}
    
    # Extract shared embeddings
    shared_old_emb = []
    shared_new_emb = []
    shared_labels = []
    
    for entity in sorted(shared_entities):
        if entity in old_id_map and entity in new_id_map:
            old_idx = old_id_map[entity]
            new_idx = new_id_map[entity]
            shared_old_emb.append(old_emb[old_idx])
            shared_new_emb.append(new_emb[new_idx])
            shared_labels.append(entity)
    
    shared_old_emb = np.array(shared_old_emb)
    shared_new_emb = np.array(shared_new_emb)
    
    # Handle complex embeddings (e.g., from ComplEx) by taking absolute value
    if np.iscomplexobj(shared_old_emb):
        shared_old_emb = np.abs(shared_old_emb)
    if np.iscomplexobj(shared_new_emb):
        shared_new_emb = np.abs(shared_new_emb)
    
    # Compute similarities
    # Cosine similarity between old and new embeddings for same entity
    cosine_sims = []
    for i in range(len(shared_old_emb)):
        sim = cosine_similarity([shared_old_emb[i]], [shared_new_emb[i]])[0, 0]
        cosine_sims.append(sim)
    
    cosine_sims = np.array(cosine_sims)
    
    print(f"\nEntity representation similarity (cosine):")
    print(f"  Mean: {cosine_sims.mean():.4f}")
    print(f"  Std: {cosine_sims.std():.4f}")
    print(f"  Min: {cosine_sims.min():.4f} ({shared_labels[np.argmin(cosine_sims)]})")
    print(f"  Max: {cosine_sims.max():.4f} ({shared_labels[np.argmax(cosine_sims)]})")
    
    # Find most changed entities
    changed_idx = np.argsort(cosine_sims)
    print(f"\nMost changed entities (lowest cosine sim):")
    for idx in changed_idx[:5]:
        print(f"  - {shared_labels[idx]}: {cosine_sims[idx]:.4f}")
    
    print(f"\nMost stable entities (highest cosine sim):")
    for idx in changed_idx[-5:]:
        print(f"  - {shared_labels[idx]}: {cosine_sims[idx]:.4f}")
    
    return cosine_sims, shared_old_emb, shared_new_emb, shared_labels

def plot_comparison(old_emb, new_emb, shared_labels, cosine_sims, output_dir):
    """Create comparison plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle complex embeddings (e.g., from ComplEx) by taking absolute value
    old_emb_real = np.abs(old_emb) if np.iscomplexobj(old_emb) else old_emb
    new_emb_real = np.abs(new_emb) if np.iscomplexobj(new_emb) else new_emb
    
    # 1. Embedding norms distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    old_norms = np.linalg.norm(old_emb_real, axis=1)
    new_norms = np.linalg.norm(new_emb_real, axis=1)
    
    axes[0].hist(old_norms, bins=50, alpha=0.7, label='Old KG')
    axes[0].set_xlabel('L2 Norm')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Embedding Norms Distribution (Full KG)')
    axes[0].legend()
    
    axes[1].hist(new_norms, bins=50, alpha=0.7, label='New KG', color='orange')
    axes[1].set_xlabel('L2 Norm')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Embedding Norms Distribution (Optimized KG)')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'embedding_norms_distribution.png', dpi=300)
    plt.close()
    
    # 2. Cosine similarity histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(cosine_sims, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(cosine_sims.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {cosine_sims.mean():.3f}')
    ax.set_xlabel('Cosine Similarity (Old vs New)')
    ax.set_ylabel('Frequency')
    ax.set_title('Entity Representation Stability Across Embeddings')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'cosine_similarity_distribution.png', dpi=300)
    plt.close()
    
    # 3. PCA visualization
    pca = PCA(n_components=2)
    old_pca = pca.fit_transform(old_emb_real)
    new_pca = pca.fit_transform(new_emb_real)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(old_pca[:, 0], old_pca[:, 1], alpha=0.5, s=10)
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    axes[0].set_title('Full KG Embeddings (PCA)')
    
    axes[1].scatter(new_pca[:, 0], new_pca[:, 1], alpha=0.5, s=10, color='orange')
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    axes[1].set_title('Optimized KG Embeddings (PCA)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pca_visualization.png', dpi=300)
    plt.close()
    
    print(f"\n✓ Saved plots to {output_dir}/")

def main():
    parser = argparse.ArgumentParser(description='Compare embeddings from full vs optimized KG')
    parser.add_argument('--old-emb', type=str, required=True, help='Path to old (full KG) embeddings .npy')
    parser.add_argument('--old-map', type=str, required=True, help='Path to old entity mapping .csv')
    parser.add_argument('--new-emb', type=str, required=True, help='Path to new (optimized KG) embeddings .npy')
    parser.add_argument('--new-map', type=str, required=True, help='Path to new entity mapping .csv')
    parser.add_argument('--output_dir', type=str, default='./comparison_results', help='Output directory for plots')
    args = parser.parse_args()
    
    print("="*60)
    print("EMBEDDING COMPARISON: Full KG vs Optimized KG")
    print("="*60)
    
    # Load embeddings
    print("\nLoading embeddings...")
    old_emb, old_map = load_embeddings(args.old_emb, args.old_map)
    new_emb, new_map = load_embeddings(args.new_emb, args.new_map)
    
    # Compare
    compare_shapes(old_emb, new_emb, old_map, new_map)
    old_norms, new_norms, old_var, new_var = compare_statistics(old_emb, new_emb)
    shared, old_entities, new_entities = find_shared_entities(old_map, new_map)
    cosine_sims, shared_old_emb, shared_new_emb, shared_labels = compare_shared_embeddings(
        old_emb, old_map, new_emb, new_map, shared
    )
    
    # Plot
    plot_comparison(old_emb, new_emb, shared_labels, cosine_sims, args.output_dir)
    
    print("\n" + "="*60)
    print("Comparison complete!")
    print("="*60)

if __name__ == '__main__':
    main()
