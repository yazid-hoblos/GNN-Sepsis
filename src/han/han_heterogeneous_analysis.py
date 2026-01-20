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
import requests

from src.han.model import SepsisHANClassifier
from src.han.owl_data_loader_with_features import load_hetero_graph_from_owl
from src.han.patient_gradient_analysis import GradientBasedAttentionAnalyzer
import xml.etree.ElementTree as ET
import logging

logger = logging.getLogger(__name__)

def parse_go_obo(go_file_path):
    """Parse go.obo file and extract GO term definitions (multiple id formats)."""
    go_defs = {}
    try:
        current_id = None
        current_name = None
        current_def = None
        in_term = False
        with open(go_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line == '[Term]':
                    if current_id and (current_name or current_def):
                        go_defs[current_id] = {
                            'name': current_name or 'Unknown',
                            'definition': current_def or 'No definition'
                        }
                        go_num = current_id.replace('GO:', '')
                        go_defs[f'GO_{go_num}'] = go_defs[current_id]
                        go_defs[f'GO_{go_num}_instance'] = go_defs[current_id]
                    current_id = None
                    current_name = None
                    current_def = None
                    in_term = True
                    continue
                if not in_term or not line:
                    continue
                if line.startswith('id: GO:'):
                    go_num = line.split('GO:')[1].strip()
                    current_id = f'GO:{go_num}'
                elif line.startswith('name: '):
                    current_name = line.replace('name: ', '')
                elif line.startswith('def: "'):
                    def_text = line.split('def: "')[1].split('" [')[0]
                    current_def = def_text
        # Last term
        if current_id and (current_name or current_def):
            go_defs[current_id] = {
                'name': current_name or 'Unknown',
                'definition': current_def or 'No definition'
            }
            go_num = current_id.replace('GO:', '')
            go_defs[f'GO_{go_num}'] = go_defs[current_id]
            go_defs[f'GO_{go_num}_instance'] = go_defs[current_id]
        return go_defs
    except Exception as e:
        logger.warning(f"Error parsing go.obo: {e}")
        return {}

def find_go_file():
    """Search for go.obo file in common locations."""
    search_paths = [
        Path('OntoKGCreation/go.obo'),
        Path('F:/yaz/OntoKGCreation/go.obo'),
        Path('/usr/share/ontologies/go.obo'),
        Path('data/go.obo'),
    ]
    for p in search_paths:
        if p.exists():
            return p
    return None

def get_reactome_info(reaction_id: str) -> dict:
    """Get Reactome reaction display name if possible."""
    base = f'https://reactome.org/ContentService/data/query/{reaction_id}'
    url = f'https://reactome.org/content/detail/{reaction_id}'
    try:
        resp = requests.get(base, timeout=5)
        if resp.ok:
            data = resp.json()
            name = data.get('displayName', reaction_id)
        else:
            name = reaction_id
    except Exception:
        name = reaction_id
    return {'name': name, 'type': 'Reactome Reaction', 'url': url}


def extract_node_names_from_owl(owl_path):
    """
    Load node names from pre-converted CSV files (from kg_conversion).
    Proteins have gene symbols in their IDs: Protein_STX12, Protein_SUCLG1 etc.
    """
    
    node_names = defaultdict(dict)
    
    try:
        # Load GO term definitions once
        go_file = find_go_file()
        go_defs = parse_go_obo(go_file) if go_file else {}
        if not go_defs:
            logger.info("GO definitions not found; GO labels will fall back to IDs")
        # Load from kg_exports (already converted)
        nodes_csv = Path('OntoKGCreation/converted/nodes.csv')
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
                    # Example node_id: GO_0000974_instance
                    go_term = node_id
                    name = None
                    if go_defs:
                        # Try direct
                        info = go_defs.get(go_term)
                        # print(info)
                        if not info:
                            # Try without _instance
                            base_id = go_term.replace('_instance', '')
                            info = go_defs.get(base_id)
                        if not info:
                            # Try GO:XXXX format
                            go_num = go_term.replace('GO_', '').replace('_instance', '')
                            info = go_defs.get(f'GO:{go_num}')
                        if info:
                            name = info.get('name')
                    if not name:
                        # Fallback to trimmed ID
                        name = go_term.replace('GO_Term_', '').replace('GO_', '').strip()
                    node_names['GO_Term'][len(node_names['GO_Term'])] = str(name)[:40]
                
                elif 'Reaction' in node_category or node_id.startswith('Reaction_'):
                    # Extract raw reaction id (e.g., R-HSA-XXXX or numeric)
                    raw_id = node_id.replace('Reaction_', '').strip()
                    # If it's numeric, prefix with R-HSA-
                    reaction_id = raw_id if raw_id.startswith('R-') else (f'R-HSA-{raw_id}' if raw_id.isdigit() else raw_id)
                    info = get_reactome_info(reaction_id)
                    name = info.get('name', raw_id)
                    node_names['Reaction'][len(node_names['Reaction'])] = str(name)[:60]
        
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


def compute_protein_importance_scores(model, data, patient_idx, device='cpu'):
    """
    Compute importance scores for proteins based on gradient attribution.
    """
    try:
        analyzer = GradientBasedAttentionAnalyzer(model, data, device)
        gradients = analyzer.compute_input_gradients(patient_idx)
        
        protein_scores = {}
        if 'Protein' in gradients:
            grad_tensor = gradients['Protein']
            # Use L2 norm of gradient per protein as importance
            importance = (grad_tensor ** 2).sum(dim=1).sqrt().cpu().numpy()
            for idx, score in enumerate(importance):
                protein_scores[idx] = float(score)
        
        return protein_scores
    except Exception as e:
        logger.warning(f"Could not compute gradient importance: {e}")
        return {}


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
                                     predictions_df, output_dir, node_names=None,
                                     model=None, data=None, device='cpu'):
    """
    Visualize patient's heterogeneous neighborhood with all node types.
    Includes actual node names (not just IDs).
    Proteins can be ranked by gradient importance if model is provided.
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
    
    # Layer 1: Proteins (direct from sample) - rank by gradient importance or frequency
    protein_nodes = set()
    protein_relations = defaultdict(int)
    protein_freq = defaultdict(int)
    
    # Count protein frequencies
    for edge_type_str, node_indices in neighbors_by_type.get('Protein', {}).items():
        for node_idx in node_indices:
            protein_freq[node_idx] += 1
    
    # Compute gradient-based importance if model/data provided
    protein_scores = {}
    if model is not None and data is not None:
        logger.info(f"Computing gradient importance for patient {patient_idx}...")
        protein_scores = compute_protein_importance_scores(model, data, patient_idx, device)
    
    # Rank proteins: by gradient score if available, else by frequency
    if protein_scores:
        top_proteins = [idx for idx, _ in sorted(protein_scores.items(), key=lambda x: -x[1])[:10]]
        logger.info(f"Ranked proteins by gradient importance: {top_proteins}")
    else:
        top_proteins = [idx for idx, _ in sorted(protein_freq.items(), key=lambda x: -x[1])[:6]]
        logger.info(f"Ranked proteins by frequency: {top_proteins}")
    
    # Add top proteins to graph
    for node_idx in top_proteins:
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
                unique_neighbors = list(set(node_indices))[:6]  # Top 6 of each type
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
            print(node_names.keys())
            print(node_type, node_id_str)
            try:
                if node_type in node_names and node_idx in node_names[node_type]:
                    node_idx = int(node_id_str)
                    name = node_names[node_type][node_idx]
                elif node_type == 'GO':
                    name = node_names['GO_Term'][int(node_id_str.split('_')[-1])]
                    print(name)
                else:
                    name = node_id_str
            except:
                print('Error parsing node id:', node_id_str)
                name = node_id_str
            
            labels[node] = name
    
    # Set global font for legend
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13})
    # Draw node labels with smaller font
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold', font_family='DejaVu Sans', ax=ax)
    
    # Create combined legend for node types and edge types
    node_legend = [
        Patch(facecolor='#2ecc71', edgecolor='black', linewidth=1.5, label='Sample'),
        Patch(facecolor='#e74c3c', edgecolor='black', linewidth=1.5, label='Protein'),
        Patch(facecolor='#3498db', edgecolor='black', linewidth=1.5, label='Pathway'),
        Patch(facecolor='#9b59b6', edgecolor='black', linewidth=1.5, label='GO_Term'),
        Patch(facecolor='#f39c12', edgecolor='black', linewidth=1.5, label='Reaction'),
    ]
    
    edge_legend = [
        mpatches.Patch(color='#e74c3c', label='Gene Expression'),
        mpatches.Patch(color='#9b59b6', label='GO Association'),
        mpatches.Patch(color='#3498db', label='Pathway Link'),
    ]
    
    # Combine legends
    all_legend = node_legend + edge_legend
    all_labels = [
        'Sample', 'Protein', 'Pathway', 'GO_Term', 'Reaction',
        'Gene Expression', 'GO Association', 'Pathway Link', 'Other'
    ]
    
    # Create two separate legends
    legend1 = ax.legend(handles=node_legend, loc='upper left', fontsize=13, 
                        title='Node Types', framealpha=0.98, title_fontsize=13,
                        edgecolor='black', fancybox=True, prop={'family': 'DejaVu Sans', 'size': 13})

    ax.add_artist(legend1)
    legend2 = ax.legend(handles=edge_legend, loc='lower left', fontsize=13, 
                        title='Edge Types', framealpha=0.98, title_fontsize=13,
                        edgecolor='black', fancybox=True, prop={'family': 'DejaVu Sans', 'size': 13})
    
    ax.axis('off')
    plt.tight_layout()
    
    subgraph_path = f'{output_dir}/patient_{patient_idx:03d}_heterogeneous_subgraph.png'
    plt.savefig(subgraph_path, dpi=1000, bbox_inches='tight', facecolor='white')
    print(f"✓ {subgraph_path}")
    plt.close()
    
    # Compute node type counts for return
    node_type_counts = {}
    for node_type, relations_dict in neighbors_by_type.items():
        node_type_counts[node_type] = len(set(n for nodes in relations_dict.values() for n in nodes))
    
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
    parser.add_argument('--model_path', default='results/han_model_with_expression/han_model.pt', help='Path to trained HAN model')
    parser.add_argument('--output_dir', default='results/han_attention_analysis')
    parser.add_argument('--use_gradient', action='store_true', help='Use gradient-based protein ranking (requires model)')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
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
    
    # Load model if using gradient-based ranking
    model = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.use_gradient and Path(args.model_path).exists():
        print(f"\nLoading HAN model from {args.model_path}...")
        model = SepsisHANClassifier(
            in_channels_dict={ntype: data[ntype].x.size(1) for ntype in data.node_types},
            hidden_channels=64,
            out_channels=32,
            num_layers=2,
            num_heads=8,
            dropout=0.3,
            metadata=data.metadata()
        ).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        print(f"✓ Model loaded on device: {device}")
    elif args.use_gradient:
        print(f"⚠️  --use_gradient specified but model not found at {args.model_path}; using frequency-based ranking")
    
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
        
        # Visualize (with optional gradient-based ranking)
        visualize_heterogeneous_subgraph(
            patient_idx, neighbors_by_type, relation_summary, preds_df, args.output_dir, node_names,
            model=model, data=data, device=device
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