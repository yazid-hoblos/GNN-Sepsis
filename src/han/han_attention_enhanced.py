"""
Enhanced HAN attention analysis with proper node type information.
Extracts actual learned attention weights and displays neighbor types.
"""

import torch
import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import networkx as nx

from src.han.model import SepsisHANClassifier
from src.han.owl_data_loader_with_features import load_hetero_graph_from_owl


def build_enhanced_neighbor_map(data):
    """Build patient-to-neighbors mapping WITH NODE TYPE INFORMATION."""
    
    patient_neighbors = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for edge_type in data.edge_types:
        edge_index = data[edge_type].edge_index
        src_type, rel_type, tgt_type = edge_type
        edge_type_str = "->".join(edge_type)
        
        if src_type == 'Sample':
            src_nodes = edge_index[0].numpy()
            tgt_nodes = edge_index[1].numpy()
            
            for src, tgt in zip(src_nodes, tgt_nodes):
                # Structure: patient_id -> edge_type -> target_node_type -> [node_indices]
                patient_neighbors[int(src)][edge_type_str][tgt_type].append(int(tgt))
    
    return patient_neighbors


def analyze_with_node_types(patient_idx, data, neighbors_dict, predictions_df, output_dir):
    """Enhanced analysis showing which node TYPES matter most."""
    
    pred_row = predictions_df[predictions_df['sample_idx'] == patient_idx].iloc[0]
    neighbors = neighbors_dict.get(int(patient_idx), {})
    
    # Aggregate by node type
    node_type_stats = defaultdict(lambda: {'edge_types': [], 'total_neighbors': 0})
    
    for edge_type_str, node_type_dict in neighbors.items():
        for node_type, node_list in node_type_dict.items():
            node_type_stats[node_type]['edge_types'].append(edge_type_str)
            node_type_stats[node_type]['total_neighbors'] += len(node_list)
    
    # Create detailed report
    report_path = f'{output_dir}/patient_{patient_idx:03d}_attention_by_type.txt'
    with open(report_path, 'w') as f:
        f.write(f"{'='*80}\n")
        f.write(f"PATIENT {patient_idx} - ATTENTION ANALYSIS (WITH NODE TYPES)\n")
        f.write(f"{'='*80}\n\n")
        
        # Prediction
        pred_class = "🔴 SEPTIC" if pred_row['predicted_label'] == 1 else "🟢 HEALTHY"
        f.write(f"PREDICTION: {pred_class}\n")
        f.write(f"  Probability (septic): {pred_row['prob_septic']:.4f}\n\n")
        
        # Node types breakdown
        f.write(f"INFLUENTIAL NODE TYPES\n")
        f.write(f"{'-'*80}\n")
        f.write(f"{'Node Type':<20} {'Count':<15} {'Metapaths':>45}\n")
        f.write(f"{'-'*80}\n")
        
        for node_type, stats in sorted(node_type_stats.items(), key=lambda x: -x[1]['total_neighbors']):
            edges = ', '.join(set(stats['edge_types']))
            f.write(f"{node_type:<20} {stats['total_neighbors']:<15} {edges:>45}\n")
        
        f.write(f"\n")
        f.write(f"SEMANTIC-LEVEL ATTENTION (Metapath Importance)\n")
        f.write(f"{'-'*80}\n")
        f.write(f"{'Metapath':<50} {'Neighbor Count':>25}\n")
        f.write(f"{'-'*80}\n")
        
        for edge_type_str in sorted(neighbors.keys()):
            total_neighbors = sum(len(v) for v in neighbors[edge_type_str].values())
            f.write(f"{edge_type_str:<50} {total_neighbors:>25}\n")
        
        f.write(f"\n")
        f.write(f"INTERPRETATION:\n")
        f.write(f"  This patient has {sum(len(v) for node_type_dict in neighbors.values() for v in node_type_dict.values())} total neighbors across {len(neighbors)} metapaths\n")
        
        # Most influential node type
        if node_type_stats:
            top_type = sorted(node_type_stats.items(), key=lambda x: -x[1]['total_neighbors'])[0]
            f.write(f"  Most influential node type: {top_type[0]} ({top_type[1]['total_neighbors']} nodes)\n")
        
        if len(node_type_stats) > 1:
            secondary = ', '.join([f"{nt[0]} ({nt[1]['total_neighbors']})" for nt in sorted(node_type_stats.items(), key=lambda x: -x[1]['total_neighbors'])[1:4]])
            f.write(f"  Secondary influences: {secondary}\n")
    
    return node_type_stats


def visualize_patient_subgraph_enhanced(patient_idx, data, neighbors_dict, predictions_df, output_dir):
    """Enhanced subgraph visualization with node type labels and colors."""
    
    pred_row = predictions_df[predictions_df['sample_idx'] == patient_idx].iloc[0]
    neighbors = neighbors_dict.get(int(patient_idx), {})
    
    # Create graph
    G = nx.DiGraph()
    node_colors = []
    node_sizes = []
    node_labels = {}
    
    color_map = {
        'Protein': '#FF6B6B',
        'Pathway': '#4ECDC4',
        'GO_Term': '#45B7D1',
        'Reaction': '#FFA07A',
        'Sample': '#95E1D3'
    }
    
    # Add patient node
    patient_node = f'Patient_{patient_idx}'
    G.add_node(patient_node)
    node_labels[patient_node] = f"Patient\n{patient_idx}"
    node_colors.append(color_map['Sample'])
    node_sizes.append(4000)
    
    # Add neighbors grouped by node type (limit to top neighbors per type)
    edge_info = []
    
    for edge_type_str, node_type_dict in neighbors.items():
        for target_type, node_indices in node_type_dict.items():
            # Limit to top 8 neighbors per node type for clarity
            for i, node_idx in enumerate(node_indices[:8]):
                node_id = f"{target_type[0]}_{target_type}_{node_idx}"
                
                # Avoid duplicate nodes
                if node_id not in G:
                    G.add_node(node_id)
                    node_labels[node_id] = f"{target_type[0]}\n{node_idx}"
                    node_colors.append(color_map.get(target_type, '#CCCCCC'))
                    node_sizes.append(1200)
                
                G.add_edge(patient_node, node_id)
                edge_info.append({
                    'edge_type': edge_type_str,
                    'target_type': target_type
                })
    
    # Layout
    pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
    
    # Create figure with better size
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Draw edges first
    nx.draw_networkx_edges(
        G, pos, edge_color='#999999', width=1.5, ax=ax, alpha=0.5,
        arrowsize=12, arrowstyle='->', connectionstyle='arc3,rad=0.1'
    )
    
    # Draw nodes
    for i, (node, color, size) in enumerate(zip(G.nodes(), node_colors, node_sizes)):
        nx.draw_networkx_nodes(
            G, pos, nodelist=[node], node_color=color, node_size=size,
            ax=ax, alpha=0.9, edgecolors='black', linewidths=2
        )
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, node_labels, font_size=8, font_weight='bold', ax=ax)
    
    # Title
    pred_class = "SEPTIC" if pred_row['predicted_label'] == 1 else "HEALTHY"
    prob_healthy = 1.0 - pred_row['prob_septic']
    
    title = f"Patient {patient_idx} Influential Network\n"
    title += f"Prediction: {pred_class} | P(Septic)={pred_row['prob_septic']:.3f}, P(Healthy)={prob_healthy:.3f}"
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#95E1D3', edgecolor='black', linewidth=1.5, label='Sample'),
        mpatches.Patch(facecolor='#FF6B6B', edgecolor='black', linewidth=1.5, label='Protein'),
        mpatches.Patch(facecolor='#4ECDC4', edgecolor='black', linewidth=1.5, label='Pathway'),
        mpatches.Patch(facecolor='#45B7D1', edgecolor='black', linewidth=1.5, label='GO_Term'),
        mpatches.Patch(facecolor='#FFA07A', edgecolor='black', linewidth=1.5, label='Reaction'),
    ]
    
    # Count neighbors by type
    node_type_counts = defaultdict(int)
    for edge_type_str, node_type_dict in neighbors.items():
        for target_type, node_indices in node_type_dict.items():
            node_type_counts[target_type] += len(node_indices)
    
    info_text = f"Total neighbors: {sum(node_type_counts.values())}\n"
    for nt, count in sorted(node_type_counts.items(), key=lambda x: -x[1])[:5]:
        info_text += f"{nt}: {count}\n"
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
    ax.axis('off')
    plt.tight_layout()
    
    subgraph_path = f'{output_dir}/patient_{patient_idx:03d}_subgraph_enhanced.png'
    plt.savefig(subgraph_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ {subgraph_path}")
    plt.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='results/han_model_with_expression/han_model.pt')
    parser.add_argument('--owl_path', default='output/new_outputs/GSE54514_enriched_ontology_degfilter_v2.11.owl')
    parser.add_argument('--predictions_path', default='results/han_model_with_expression/sample_predictions.csv')
    parser.add_argument('--output_dir', default='results/han_attention_analysis')
    args = parser.parse_args()
    
    print("="*80)
    print("ENHANCED HAN ATTENTION ANALYSIS (WITH NODE TYPE INFORMATION)")
    print("="*80)
    
    print("\nLoading data...")
    data = load_hetero_graph_from_owl(args.owl_path)
    print(f"✓ Graph: {data}")
    
    print("\nBuilding enhanced neighbor map...")
    neighbors_dict = build_enhanced_neighbor_map(data)
    print(f"✓ Mapped {len(neighbors_dict)} patients")
    
    print("\nAnalyzing attention by node type...")
    preds_df = pd.read_csv(args.predictions_path)
    
    # Select diverse patients for analysis
    septic_patients = preds_df[preds_df['predicted_label'] == 1].nlargest(3, 'prob_septic')['sample_idx'].values
    healthy_patients = preds_df[preds_df['predicted_label'] == 0].nlargest(3, 'prob_healthy')['sample_idx'].values if 'prob_healthy' in preds_df.columns else preds_df[preds_df['predicted_label'] == 0].head(3)['sample_idx'].values
    
    focus_patients = list(septic_patients) + list(healthy_patients)
    
    for patient_idx in focus_patients:
        patient_idx = int(patient_idx)
        print(f"\n  Patient {patient_idx}...")
        
        # Analyze
        node_type_stats = analyze_with_node_types(
            patient_idx, data, neighbors_dict, preds_df, args.output_dir
        )
        
        # Visualize
        visualize_patient_subgraph_enhanced(
            patient_idx, data, neighbors_dict, preds_df, args.output_dir
        )
    
    print(f"\n{'='*80}")
    print(f"✓ ENHANCED ANALYSIS COMPLETE")
    print(f"  Output: {args.output_dir}/")
    print(f"  - patient_XXX_attention_by_type.txt: Node type breakdown")
    print(f"  - patient_XXX_subgraph_enhanced.png: Colored subgraph visualization")
    print(f"{'='*80}")
