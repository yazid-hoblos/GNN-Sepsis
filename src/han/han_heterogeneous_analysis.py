"""
Fixed HAN attention analysis with proper heterogeneous neighborhood extraction.
Shows all node types (Protein, Pathway, GO_Term, Reaction) connected to each patient.
"""

import torch
import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
import networkx as nx

from src.han.model import SepsisHANClassifier
from src.han.owl_data_loader_with_features import load_hetero_graph_from_owl
import xml.etree.ElementTree as ET


def extract_node_names_from_owl(owl_path):
    """
    Load node names from pre-converted CSV files (from kg_conversion).
    Proteins have gene symbols in their IDs: Protein_STX12, Protein_SUCLG1 etc.
    """
    
    node_names = defaultdict(dict)
    
    try:
        # Load from kg_exports (already converted)
        nodes_csv = Path('kg_exports/nodes.csv')
        if nodes_csv.exists():
            df_nodes = pd.read_csv(nodes_csv)
            
            for _, row in df_nodes.iterrows():
                node_id = str(row.get('node_id', ''))
                node_category = str(row.get('node_category', ''))
                
                if not node_id or node_id == 'nan':
                    continue
                
                # Extract name based on node type
                if 'Protein' in node_category or node_id.startswith('Protein_'):
                    # For proteins, the gene name is in the ID: Protein_STX12 -> STX12
                    gene_name = node_id.replace('Protein_', '').strip()
                    if gene_name and len(gene_name) > 0:
                        # Use counter as ID (since we don't have numeric IDs for proteins with names)
                        # Instead, create a mapping based on order
                        node_names['Protein'][len(node_names['Protein'])] = gene_name
                
                elif 'Pathway' in node_category or node_id.startswith('Pathway_'):
                    name = str(row.get('hasName', node_id))
                    if pd.isna(name) or name == 'nan':
                        name = node_id.replace('Pathway_', '').strip()
                    node_names['Pathway'][len(node_names['Pathway'])] = str(name)[:40]
                
                elif 'GO' in node_category or node_id.startswith('GO_'):
                    name = str(row.get('hasName', node_id))
                    if pd.isna(name) or name == 'nan':
                        name = node_id.replace('GO_Term_', '').replace('GO_', '').strip()
                    node_names['GO_Term'][len(node_names['GO_Term'])] = str(name)[:40]
                
                elif 'Reaction' in node_category or node_id.startswith('Reaction_'):
                    name = str(row.get('hasName', node_id))
                    if pd.isna(name) or name == 'nan':
                        name = node_id.replace('Reaction_', '').strip()
                    node_names['Reaction'][len(node_names['Reaction'])] = str(name)[:40]
        
        else:
            print("  ⚠️  kg_exports/nodes.csv not found, using defaults")
    
    except Exception as e:
        print(f"  ⚠️  Could not load node names: {e}")
    
    # Print what we loaded
    print(f"  ✓ Loaded node names:")
    for ntype, names_dict in node_names.items():
        if names_dict:
            samples = list(names_dict.items())[:2]
            print(f"    {ntype}: {len(names_dict)} nodes (e.g., {samples})")
    
    return node_names


def extract_heterogeneous_neighborhood(data, patient_idx):
    """
    Extract patient's heterogeneous neighborhood including 2-hop neighbors.
    
    Structure:
    - 1-hop: Sample -> Proteins (direct gene expression)
    - 2-hop: Sample -> Protein -> Pathway/GO_Term/Reaction (biological context)
    """
    
    neighbors_by_type = defaultdict(lambda: defaultdict(list))
    relation_summary = defaultdict(int)
    
    # STEP 1: Get direct protein neighbors (1-hop)
    protein_neighbors = set()
    for edge_type in data.edge_types:
        src_type, rel_type, dst_type = edge_type
        if src_type == 'Sample' and dst_type == 'Protein':
            edge_index = data[edge_type].edge_index
            src_nodes = edge_index[0].numpy()
            dst_nodes = edge_index[1].numpy()
            mask = src_nodes == patient_idx
            
            if mask.any():
                connected = dst_nodes[mask].tolist()
                protein_neighbors.update(connected)
                edge_type_str = f"Sample-[{rel_type}]->Protein"
                neighbors_by_type['Protein'][edge_type_str].extend(connected)
                relation_summary[edge_type_str] += len(connected)
    
    # STEP 2: Get 2-hop neighbors (Protein -> other node types)
    for edge_type in data.edge_types:
        src_type, rel_type, dst_type = edge_type
        if src_type == 'Protein' and dst_type != 'Sample':
            edge_index = data[edge_type].edge_index
            src_nodes = edge_index[0].numpy()
            dst_nodes = edge_index[1].numpy()
            
            # Find edges from our protein neighbors
            for protein_id in protein_neighbors:
                mask = src_nodes == protein_id
                if mask.any():
                    connected = dst_nodes[mask].tolist()
                    edge_type_str = f"Protein-[{rel_type}]->{dst_type}"
                    neighbors_by_type[dst_type][edge_type_str].extend(connected)
                    relation_summary[edge_type_str] += len(connected)
    
    return neighbors_by_type, relation_summary


def visualize_heterogeneous_subgraph(patient_idx, neighbors_by_type, relation_summary, 
                                     predictions_df, output_dir, node_names=None):
    """
    Visualize patient's heterogeneous neighborhood with all node types.
    Includes actual node names (not just IDs).
    """
    
    if node_names is None:
        node_names = defaultdict(dict)
    
    pred_row = predictions_df[predictions_df['sample_idx'] == patient_idx].iloc[0]
    
    # Create graph
    G = nx.DiGraph()
    
    # Node type colors
    color_map = {
        'Sample': '#2ecc71',      # Green
        'Protein': '#e74c3c',     # Red
        'Pathway': '#3498db',     # Blue
        'GO_Term': '#9b59b6',     # Purple
        'Reaction': '#f39c12'     # Orange
    }
    
    # Add patient node (center)
    patient_node = f"Patient_{patient_idx}"
    G.add_node(patient_node, node_type='Sample')
    
    # Add neighbors by type (showing heterogeneous structure)
    node_colors_list = []
    node_sizes_list = []
    
    # Add patient (large, center)
    node_colors_list.append(color_map['Sample'])
    node_sizes_list.append(4000)
    
    # Track edges for visualization
    edge_type_counts = defaultdict(int)
    
    # Layer 1: Proteins (direct from sample)
    protein_nodes = set()
    protein_relations = defaultdict(int)
    
    for edge_type_str, node_indices in neighbors_by_type.get('Protein', {}).items():
        unique_neighbors = list(set(node_indices))[:6]  # Top 6 proteins
        for node_idx in unique_neighbors:
            node_id = f"Protein_{node_idx}"
            protein_nodes.add(node_id)
            
            if node_id not in G:
                G.add_node(node_id, node_type='Protein')
                node_colors_list.append(color_map['Protein'])
                node_sizes_list.append(1500)
            
            G.add_edge(patient_node, node_id, relation='hasGeneExpressionOA')
            protein_relations['hasGeneExpressionOA'] += 1
    
    # Layer 2: Other node types (from proteins)
    for node_type in ['Pathway', 'GO_Term', 'Reaction']:
        if node_type in neighbors_by_type:
            for edge_type_str, node_indices in neighbors_by_type[node_type].items():
                unique_neighbors = list(set(node_indices))[:4]  # Top 4 of each type
                for node_idx in unique_neighbors:
                    node_id = f"{node_type}_{node_idx}"
                    
                    if node_id not in G:
                        G.add_node(node_id, node_type=node_type)
                        node_colors_list.append(color_map.get(node_type, '#95a5a6'))
                        node_sizes_list.append(1000)
                    
                    # Connect to a sample of proteins (to avoid clutter)
                    sample_proteins = list(protein_nodes)[:2]
                    for prot_node in sample_proteins:
                        if not G.has_edge(prot_node, node_id):
                            G.add_edge(prot_node, node_id, relation=edge_type_str)
                            edge_type_counts[edge_type_str] += 1
    
    # Layout with patient at center
    pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)
    pos[patient_node] = [0, 0]  # Force patient to center
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Draw edges with different colors per relation type
    edge_colors_list = []
    edge_widths_list = []
    relation_colors = {
        'hasGeneExpressionOA': '#e74c3c',    # Red - gene expression
        'isAssociatedWithGO': '#9b59b6',     # Purple - GO terms
        'participatesIn': '#3498db',         # Blue - pathways
        'other': '#95a5a6'
    }
    
    for src, dst, attr in G.edges(data=True):
        relation = attr.get('relation', 'other')
        # Extract base relation name
        if '[' in relation:
            base_rel = relation.split('[')[1].split(']')[0]
        else:
            base_rel = relation
        
        color = relation_colors.get(base_rel, relation_colors['other'])
        edge_colors_list.append(color)
        
        # Thicker edges for direct patient connections
        if src == patient_node:
            edge_widths_list.append(2.5)
        else:
            edge_widths_list.append(1.0)
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, 
        node_color=node_colors_list,
        node_size=node_sizes_list,
        ax=ax, alpha=0.9, edgecolors='black', linewidths=2
    )
    
    # Draw edges
    edges = G.edges()
    nx.draw_networkx_edges(
        G, pos,
        edge_color=edge_colors_list,
        width=edge_widths_list,
        ax=ax, alpha=0.6,
        arrowsize=15, arrowstyle='->',
        connectionstyle='arc3,rad=0.1'
    )
    
    # Draw labels with actual names
    labels = {}
    for node in G.nodes():
        if node.startswith('Patient'):
            labels[node] = f"Patient\n{patient_idx}"
        else:
            parts = node.split('_', 1)  # Split on first underscore only
            node_type = parts[0]
            node_id_str = parts[1] if len(parts) > 1 else '0'
            
            # Get actual name if available
            try:
                node_idx = int(node_id_str)
                if node_type in node_names and node_idx in node_names[node_type]:
                    name = node_names[node_type][node_idx]
                else:
                    name = node_id_str
            except:
                name = node_id_str
            
            # Shorten display
            display_name = name if len(name) <= 15 else name[:12] + '...'
            labels[node] = f"{node_type}\n{display_name}"
    
    nx.draw_networkx_labels(G, pos, labels, font_size=7, font_weight='bold', ax=ax)
    
    # Title with emphasis on heterogeneous structure
    pred_class = "🔴 SEPTIC" if pred_row['predicted_label'] == 1 else "🟢 HEALTHY"
    prob_septic = pred_row['prob_septic']
    prob_healthy = 1.0 - prob_septic
    
    title = f"Patient {patient_idx} - Heterogeneous Knowledge Graph (2-hop Neighborhood)\n"
    title += f"Prediction: {pred_class} | P(Septic)={prob_septic:.3f}, P(Healthy)={prob_healthy:.3f}\n"
    title += f"Layer 1: Gene Expression (Proteins) → Layer 2: Biological Context (Pathways/GO/Reactions)"
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    
    # Legend for node types
    node_legend = [
        Patch(facecolor='#2ecc71', edgecolor='black', linewidth=1.5, label='Sample'),
        Patch(facecolor='#e74c3c', edgecolor='black', linewidth=1.5, label='Protein'),
        Patch(facecolor='#3498db', edgecolor='black', linewidth=1.5, label='Pathway'),
        Patch(facecolor='#9b59b6', edgecolor='black', linewidth=1.5, label='GO_Term'),
        Patch(facecolor='#f39c12', edgecolor='black', linewidth=1.5, label='Reaction'),
    ]
    ax.legend(handles=node_legend, loc='upper left', fontsize=10, title='Node Types', framealpha=0.95)
    
    # Summary statistics box
    total_neighbors = sum(len(nodes) for nt_dict in neighbors_by_type.values() for nodes in nt_dict.values())
    node_type_counts = {}
    for node_type, relations_dict in neighbors_by_type.items():
        node_type_counts[node_type] = len(set(n for nodes in relations_dict.values() for n in nodes))
    
    info_text = f"Total Neighbors: {total_neighbors}\n"
    info_text += f"Unique Node Types Connected: {len(node_type_counts)}\n\n"
    for ntype in ['Protein', 'Pathway', 'GO_Term', 'Reaction']:
        if ntype in node_type_counts:
            info_text += f"{ntype}: {node_type_counts[ntype]}\n"
    
    ax.text(0.98, 0.98, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.axis('off')
    plt.tight_layout()
    
    subgraph_path = f'{output_dir}/patient_{patient_idx:03d}_heterogeneous_subgraph.png'
    plt.savefig(subgraph_path, dpi=150, bbox_inches='tight')
    print(f"✓ {subgraph_path}")
    plt.close()
    
    return node_type_counts


def create_heterogeneous_analysis_report(patient_idx, neighbors_by_type, relation_summary, 
                                         predictions_df, output_dir):
    """
    Create detailed report showing heterogeneous neighborhood composition.
    """
    
    pred_row = predictions_df[predictions_df['sample_idx'] == patient_idx].iloc[0]
    
    report_path = f'{output_dir}/patient_{patient_idx:03d}_heterogeneous_analysis.txt'
    with open(report_path, 'w') as f:
        f.write(f"{'='*90}\n")
        f.write(f"PATIENT {patient_idx} - HETEROGENEOUS NEIGHBORHOOD ANALYSIS\n")
        f.write(f"{'='*90}\n\n")
        
        # Prediction
        pred_class = "🔴 SEPTIC" if pred_row['predicted_label'] == 1 else "🟢 HEALTHY"
        f.write(f"PREDICTION: {pred_class}\n")
        f.write(f"  Probability (septic): {pred_row['prob_septic']:.4f}\n")
        f.write(f"  Probability (healthy): {1.0 - pred_row['prob_septic']:.4f}\n\n")
        
        # Node type breakdown
        f.write(f"NODE TYPE DISTRIBUTION\n")
        f.write(f"{'-'*90}\n")
        f.write(f"{'Node Type':<20} {'Total Connected':>20} {'Via Relations':>45}\n")
        f.write(f"{'-'*90}\n")
        
        for node_type, relations_dict in sorted(neighbors_by_type.items()):
            all_neighbors = set(n for nodes in relations_dict.values() for n in nodes)
            relation_list = ', '.join(relations_dict.keys())
            f.write(f"{node_type:<20} {len(all_neighbors):>20} {relation_list:>45}\n")
        
        # Relation breakdown
        f.write(f"\n\nRELATION TYPE BREAKDOWN\n")
        f.write(f"{'-'*90}\n")
        f.write(f"{'Relation':<50} {'Edge Count':>35}\n")
        f.write(f"{'-'*90}\n")
        
        for rel, count in sorted(relation_summary.items(), key=lambda x: -x[1]):
            f.write(f"{rel:<50} {count:>35}\n")
        
        # Interpretation
        f.write(f"\n\nINTERPRETATION\n")
        f.write(f"{'-'*90}\n")
        
        total_neighbors = sum(len(set(n for nodes in rd.values() for n in nodes)) 
                            for rd in neighbors_by_type.values())
        f.write(f"This patient is connected to {total_neighbors} entities across ")
        f.write(f"{len(neighbors_by_type)} different node types through ")
        f.write(f"{len(relation_summary)} relation types.\n\n")
        
        # Most important types
        node_type_counts = {}
        for node_type, relations_dict in neighbors_by_type.items():
            node_type_counts[node_type] = len(set(n for nodes in relations_dict.values() for n in nodes))
        
        if node_type_counts:
            f.write(f"Node types by importance (count):\n")
            for ntype, count in sorted(node_type_counts.items(), key=lambda x: -x[1]):
                f.write(f"  1. {ntype}: {count} nodes\n" if node_type_counts[ntype] == max(node_type_counts.values()) else f"  • {ntype}: {count} nodes\n")
        
        # Graph structure insight
        f.write(f"\nGRAPH STRUCTURE INSIGHT:\n")
        f.write(f"This heterogeneous neighborhood captures:\n")
        f.write(f"  • Gene expression signature (Proteins via hasGeneExpressionOA)\n")
        f.write(f"  • Biological context (Pathways, GO Terms, Reactions)\n")
        f.write(f"  • Multiple levels of annotation and interaction\n")
        f.write(f"\nThe model uses this rich heterogeneous structure to determine\n")
        f.write(f"whether the patient's expression signature matches septic or healthy\n")
        f.write(f"patterns learned during training.\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--owl_path', default='output/new_outputs/GSE54514_enriched_ontology_degfilter_v2.11.owl')
    parser.add_argument('--predictions_path', default='results/han_model_with_expression/sample_predictions.csv')
    parser.add_argument('--output_dir', default='results/han_attention_analysis')
    args = parser.parse_args()
    
    print("="*90)
    print("HETEROGENEOUS NEIGHBORHOOD ANALYSIS FOR HAN")
    print("="*90)
    
    print("\nExtracting node names from OWL...")
    node_names = extract_node_names_from_owl(args.owl_path)
    print(f"✓ Extracted names for:")
    for ntype, names_dict in node_names.items():
        print(f"  {ntype}: {len(names_dict)} nodes")
    
    print("\nLoading data...")
    data = load_hetero_graph_from_owl(args.owl_path)
    print(f"✓ Graph: {data}")
    print(f"  Node types: {list(data.node_types)}")
    print(f"  Edge types: {len(data.edge_types)}")
    
    preds_df = pd.read_csv(args.predictions_path)
    
    # Select diverse patients
    septic_patients = preds_df[preds_df['predicted_label'] == 1].nlargest(3, 'prob_septic')['sample_idx'].values
    healthy_patients = preds_df[preds_df['predicted_label'] == 0].nlargest(3, 'prob_septic')['sample_idx'].values
    
    focus_patients = list(septic_patients) + list(healthy_patients)
    
    print(f"\nAnalyzing {len(focus_patients)} patients...")
    print(f"  Septic: {list(septic_patients)}")
    print(f"  Healthy: {list(healthy_patients)}")
    
    for patient_idx in focus_patients:
        patient_idx = int(patient_idx)
        print(f"\n  Patient {patient_idx}...")
        
        # Extract heterogeneous neighborhood
        neighbors_by_type, relation_summary = extract_heterogeneous_neighborhood(data, patient_idx)
        
        # Visualize (with node names)
        visualize_heterogeneous_subgraph(
            patient_idx, neighbors_by_type, relation_summary, preds_df, args.output_dir, node_names
        )
        
        # Report
        create_heterogeneous_analysis_report(
            patient_idx, neighbors_by_type, relation_summary, preds_df, args.output_dir
        )
    
    print(f"\n{'='*90}")
    print(f"✓ HETEROGENEOUS ANALYSIS COMPLETE")
    print(f"  Output: {args.output_dir}/")
    print(f"  - patient_XXX_heterogeneous_subgraph.png: Full heterogeneous network with node names")
    print(f"  - patient_XXX_heterogeneous_analysis.txt: Detailed breakdown")
    print(f"{'='*90}")
