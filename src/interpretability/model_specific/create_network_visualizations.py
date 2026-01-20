#!/usr/bin/env python3
"""
Create advanced network visualizations for ComplEx+SVM biomarkers.

Generates:
1. Protein-Pathway interaction network
2. Feature-Entity bipartite network
3. Biomarker co-occurrence network
4. Hierarchical clustering dendrogram
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import networkx as nx
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns.set_style('white')
plt.rcParams['figure.figsize'] = (14, 10)


def create_protein_pathway_network(biomarker_df: pd.DataFrame, output_dir: Path, top_n=15):
    """Create network showing connections between top proteins and pathways."""
    
    # Get top proteins and pathways
    top_proteins = biomarker_df[biomarker_df['entity_type'] == 'Protein'].nlargest(top_n, 'entity_score')
    top_pathways = biomarker_df[biomarker_df['entity_type'] == 'Pathway'].nlargest(top_n, 'entity_score')
    
    # Create graph
    G = nx.Graph()
    
    # Add protein nodes
    for _, row in top_proteins.iterrows():
        protein = row['entity'].replace('Protein_', '')
        G.add_node(protein, node_type='Protein', score=row['entity_score'], 
                  importance=row['feature_importance'])
    
    # Add pathway nodes
    for _, row in top_pathways.iterrows():
        pathway = row['entity'].replace('Pathway_', 'P:')[:20]
        G.add_node(pathway, node_type='Pathway', score=row['entity_score'],
                  importance=row['feature_importance'])
    
    # Add edges based on shared features
    protein_features = {}
    pathway_features = {}
    
    for _, row in top_proteins.iterrows():
        protein = row['entity'].replace('Protein_', '')
        protein_features[protein] = row['feature']
    
    for _, row in top_pathways.iterrows():
        pathway = row['entity'].replace('Pathway_', 'P:')[:20]
        pathway_features[pathway] = row['feature']
    
    # Connect proteins and pathways that share features
    for protein, p_feat in protein_features.items():
        for pathway, pw_feat in pathway_features.items():
            if p_feat == pw_feat:
                G.add_edge(protein, pathway, weight=1.0)
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Separate nodes by type
    protein_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'Protein']
    pathway_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'Pathway']
    
    # Node sizes based on score
    protein_sizes = [G.nodes[n]['score'] * 30 for n in protein_nodes]
    pathway_sizes = [G.nodes[n]['score'] * 30 for n in pathway_nodes]
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.2, width=2, edge_color='gray', ax=ax)
    
    # Draw protein nodes
    nx.draw_networkx_nodes(G, pos, nodelist=protein_nodes, 
                          node_color='#e74c3c', node_size=protein_sizes,
                          alpha=0.8, label='Proteins', ax=ax)
    
    # Draw pathway nodes
    nx.draw_networkx_nodes(G, pos, nodelist=pathway_nodes,
                          node_color='#3498db', node_size=pathway_sizes,
                          alpha=0.8, label='Pathways', ax=ax)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
    
    ax.set_title('Protein-Pathway Interaction Network\n(Top 15 of each, connected by shared features)',
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='upper right')
    ax.axis('off')
    
    plt.tight_layout()
    output_file = output_dir / 'protein_pathway_network.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"Saved: {output_file}")
    plt.close()


def create_feature_entity_bipartite(biomarker_df: pd.DataFrame, mapped_df: pd.DataFrame, output_dir: Path):
    """Create bipartite network between features and entities."""
    
    # Get top features
    top_features = mapped_df.nlargest(10, 'importance')
    
    # Create bipartite graph
    G = nx.Graph()
    
    # Add feature nodes
    for _, row in top_features.iterrows():
        feature = row['feature']
        G.add_node(feature, bipartite=0, importance=row['importance'])
    
    # Add entity nodes and edges
    import ast
    for _, row in top_features.iterrows():
        feature = row['feature']
        try:
            entities = ast.literal_eval(str(row['top_entities']))
            scores = ast.literal_eval(str(row['entity_scores']))
            
            for entity, score in zip(entities, scores):
                entity_label = str(entity).replace('Protein_', '').replace('Pathway_', 'P:')[:20]
                G.add_node(entity_label, bipartite=1, score=float(score))
                G.add_edge(feature, entity_label, weight=float(score))
        except:
            continue
    
    # Separate node sets
    feature_nodes = [n for n, d in G.nodes(data=True) if d.get('bipartite') == 0]
    entity_nodes = [n for n, d in G.nodes(data=True) if d.get('bipartite') == 1]
    
    # Layout - bipartite
    pos = {}
    feature_y_spacing = 1.0 / (len(feature_nodes) + 1)
    entity_y_spacing = 1.0 / (len(entity_nodes) + 1)
    
    for i, node in enumerate(feature_nodes):
        pos[node] = (0, (i + 1) * feature_y_spacing)
    
    for i, node in enumerate(entity_nodes):
        pos[node] = (2, (i + 1) * entity_y_spacing)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Draw edges with varying thickness
    edges = G.edges(data=True)
    edge_widths = [d['weight'] / 5 for _, _, d in edges]
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.3, edge_color='gray', ax=ax)
    
    # Draw feature nodes
    feature_sizes = [G.nodes[n]['importance'] * 1000 for n in feature_nodes]
    nx.draw_networkx_nodes(G, pos, nodelist=feature_nodes,
                          node_color='#9b59b6', node_size=feature_sizes,
                          alpha=0.8, node_shape='s', label='Features', ax=ax)
    
    # Draw entity nodes
    entity_sizes = [G.nodes[n]['score'] * 15 for n in entity_nodes]
    nx.draw_networkx_nodes(G, pos, nodelist=entity_nodes,
                          node_color='#e74c3c', node_size=entity_sizes,
                          alpha=0.8, label='Entities', ax=ax)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
    
    ax.set_title('Feature-Entity Bipartite Network\n(Top 10 Features and Their Associated Entities)',
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='upper right')
    ax.axis('off')
    ax.set_xlim(-0.5, 2.5)
    
    plt.tight_layout()
    output_file = output_dir / 'feature_entity_bipartite.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"Saved: {output_file}")
    plt.close()


def create_circular_network(biomarker_df: pd.DataFrame, output_dir: Path):
    """Create circular network layout showing all entity types."""
    
    # Get top entities of each type
    proteins = biomarker_df[biomarker_df['entity_type'] == 'Protein'].nlargest(10, 'entity_score')
    pathways = biomarker_df[biomarker_df['entity_type'] == 'Pathway'].nlargest(10, 'entity_score')
    go_terms = biomarker_df[biomarker_df['entity_type'] == 'GO'].nlargest(5, 'entity_score')
    
    # Create graph
    G = nx.Graph()
    
    # Add center node
    G.add_node('ComplEx+SVM\nModel', node_type='Center', layer=0)
    
    # Add entity nodes
    for _, row in proteins.iterrows():
        label = row['entity'].replace('Protein_', '')
        G.add_node(label, node_type='Protein', score=row['entity_score'], layer=1)
        G.add_edge('ComplEx+SVM\nModel', label)
    
    for _, row in pathways.iterrows():
        label = row['entity'].replace('Pathway_', 'P:')[:15]
        G.add_node(label, node_type='Pathway', score=row['entity_score'], layer=1)
        G.add_edge('ComplEx+SVM\nModel', label)
    
    for _, row in go_terms.iterrows():
        label = row['entity'].replace('GO_', 'GO:')[:15]
        G.add_node(label, node_type='GO', score=row['entity_score'], layer=1)
        G.add_edge('ComplEx+SVM\nModel', label)
    
    # Circular layout
    pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 14))
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.15, width=1.5, edge_color='gray', ax=ax)
    
    # Draw center node
    nx.draw_networkx_nodes(G, pos, nodelist=['ComplEx+SVM\nModel'],
                          node_color='#f39c12', node_size=3000,
                          alpha=0.9, ax=ax)
    
    # Draw entity nodes by type
    node_colors = {'Protein': '#e74c3c', 'Pathway': '#3498db', 'GO': '#2ecc71'}
    
    for node_type, color in node_colors.items():
        nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == node_type]
        if nodes:
            sizes = [G.nodes[n]['score'] * 25 for n in nodes]
            nx.draw_networkx_nodes(G, pos, nodelist=nodes,
                                  node_color=color, node_size=sizes,
                                  alpha=0.8, label=node_type, ax=ax)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=7, font_weight='bold', ax=ax)
    
    ax.set_title('Biomarker Relationship Network\n(Radial Layout from Model Center)',
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='upper right')
    ax.axis('off')
    
    plt.tight_layout()
    output_file = output_dir / 'circular_biomarker_network.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"Saved: {output_file}")
    plt.close()


def create_hierarchical_clustering(biomarker_df: pd.DataFrame, output_dir: Path):
    """Create dendrogram showing hierarchical clustering of entities."""
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist
    
    # Pivot data for clustering
    pivot_data = biomarker_df.pivot_table(
        index='entity',
        columns='feature',
        values='entity_score',
        fill_value=0
    )
    
    # Limit to entities with sufficient data
    entity_counts = (pivot_data > 0).sum(axis=1)
    pivot_data = pivot_data[entity_counts >= 1]
    
    if len(pivot_data) < 3:
        logger.warning("Not enough entities for clustering")
        return
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(pivot_data.values, method='ward')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create dendrogram
    dendro = dendrogram(
        linkage_matrix,
        labels=[e.replace('Protein_', '').replace('Pathway_', 'P:')[:20] 
                for e in pivot_data.index],
        ax=ax,
        leaf_font_size=8,
        color_threshold=0.7*max(linkage_matrix[:,2])
    )
    
    ax.set_title('Hierarchical Clustering of Biomarkers\n(Based on Feature Score Patterns)',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Entity', fontsize=12, fontweight='bold')
    ax.set_ylabel('Distance', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_file = output_dir / 'biomarker_clustering_dendrogram.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_file}")
    plt.close()


def create_chord_diagram(biomarker_df: pd.DataFrame, output_dir: Path):
    """Create chord-like diagram showing feature-entity connections."""
    
    # Get top features and entities
    top_entities = biomarker_df.nlargest(20, 'entity_score')['entity'].unique()
    entity_feature_map = biomarker_df[biomarker_df['entity'].isin(top_entities)]
    
    # Create adjacency matrix
    features = entity_feature_map['feature'].unique()[:8]
    entities = top_entities[:15]
    
    matrix = np.zeros((len(features), len(entities)))
    
    for i, feat in enumerate(features):
        for j, ent in enumerate(entities):
            match = entity_feature_map[(entity_feature_map['feature'] == feat) & 
                                       (entity_feature_map['entity'] == ent)]
            if not match.empty:
                matrix[i, j] = match['entity_score'].values[0]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Create circular positions
    n_total = len(features) + len(entities)
    angles = np.linspace(0, 2*np.pi, n_total, endpoint=False)
    
    feature_angles = angles[:len(features)]
    entity_angles = angles[len(features):]
    
    radius = 1.0
    feature_pos = [(radius * np.cos(a), radius * np.sin(a)) for a in feature_angles]
    entity_pos = [(radius * np.cos(a), radius * np.sin(a)) for a in entity_angles]
    
    # Draw connections
    for i, feat in enumerate(features):
        for j, ent in enumerate(entities):
            if matrix[i, j] > 0:
                x1, y1 = feature_pos[i]
                x2, y2 = entity_pos[j]
                
                # Bezier curve
                control_x, control_y = 0, 0
                
                from matplotlib.path import Path
                import matplotlib.patches as patches
                
                verts = [(x1, y1), (control_x, control_y), (x2, y2)]
                codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
                
                path = Path(verts, codes)
                patch = patches.PathPatch(path, facecolor='none', 
                                         edgecolor='gray',
                                         alpha=matrix[i, j]/matrix.max() * 0.5,
                                         lw=2)
                ax.add_patch(patch)
    
    # Draw nodes
    for i, (x, y) in enumerate(feature_pos):
        circle = plt.Circle((x, y), 0.08, color='#9b59b6', alpha=0.8, zorder=10)
        ax.add_patch(circle)
        ax.text(x*1.15, y*1.15, features[i], ha='center', va='center', 
               fontsize=8, fontweight='bold')
    
    for i, (x, y) in enumerate(entity_pos):
        circle = plt.Circle((x, y), 0.06, color='#e74c3c', alpha=0.8, zorder=10)
        ax.add_patch(circle)
        label = entities[i].replace('Protein_', '').replace('Pathway_', 'P:')[:12]
        ax.text(x*1.15, y*1.15, label, ha='center', va='center',
               fontsize=7, fontweight='bold')
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Feature-Entity Connection Diagram\n(Chord-Style Visualization)',
                fontsize=16, fontweight='bold', pad=20)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#9b59b6', alpha=0.8, label='Features'),
        Patch(facecolor='#e74c3c', alpha=0.8, label='Entities')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    plt.tight_layout()
    output_file = output_dir / 'feature_entity_chord.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"Saved: {output_file}")
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Create network visualizations')
    parser.add_argument('--biomarker-csv', type=str,
                       default='results/interpretability/complex_svm_mapped/complex_svm_biomarkers.csv')
    parser.add_argument('--mapped-csv', type=str,
                       default='results/interpretability/complex_svm_mapped/complex_svm_features_mapped.csv')
    parser.add_argument('--output-dir', type=str,
                       default='results/interpretability/complex_svm_mapped/network_visualizations')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading data...")
    biomarker_df = pd.read_csv(args.biomarker_csv)
    mapped_df = pd.read_csv(args.mapped_csv)
    
    logger.info("Creating network visualizations...")
    
    create_protein_pathway_network(biomarker_df, output_dir)
    create_feature_entity_bipartite(biomarker_df, mapped_df, output_dir)
    create_circular_network(biomarker_df, output_dir)
    create_chord_diagram(biomarker_df, output_dir)
    create_hierarchical_clustering(biomarker_df, output_dir)
    
    logger.info(f"\nAll network visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()
