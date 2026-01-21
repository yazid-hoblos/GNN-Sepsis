#!/usr/bin/env python3
"""
Build a poster-ready PPI network for consensus biomarker proteins.

- Nodes: consensus proteins (optionally hub connectors)
- Edges: protein-protein links from precomputed subgraph analysis
- Size: consensus avg_score
- Color: immune-tagged vs other vs hubs
- Edge Color: Based on Edge Centrality (Sum of degrees of connected nodes) if types are uniform
- Outputs: PNG + SVG plots, node/edge tables for reproducibility
"""

import argparse
import logging
import re
import json
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import networkx as nx
import pandas as pd
from pyvis.network import Network

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


def _parse_neighbors(neighbor_str: str) -> Set[str]:
    if pd.isna(neighbor_str):
        return set()
    return {n for n in str(neighbor_str).split(';') if n}


def load_consensus(consensus_csv: Path) -> pd.DataFrame:
    if not consensus_csv.exists():
        raise FileNotFoundError(f"Consensus CSV not found: {consensus_csv}")
    df = pd.read_csv(consensus_csv)
    required = {'protein', 'protein_name', 'avg_score'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Consensus CSV missing columns: {missing}")
    df['avg_score'] = pd.to_numeric(df['avg_score'], errors='coerce').fillna(0)
    return df


def load_ppi(ppi_csv: Path) -> pd.DataFrame:
    if not ppi_csv.exists():
        raise FileNotFoundError(f"PPI CSV not found: {ppi_csv}")
    df = pd.read_csv(ppi_csv)
    required = {'protein', 'protein_neighbors'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"PPI CSV missing columns: {missing}")
    return df


def load_hubs(hubs_csv: Path) -> pd.DataFrame:
    if not hubs_csv.exists():
        logger.warning(f"Hub file not found: {hubs_csv}; continuing without hubs")
        return pd.DataFrame(columns=['protein', 'protein_name', 'num_biomarkers'])
    df = pd.read_csv(hubs_csv)
    if 'protein' not in df.columns:
        raise ValueError(f"Hub CSV missing 'protein' column: {hubs_csv}")
    return df


def load_immune_set(go_enriched_csv: Path, keywords: Iterable[str]) -> Set[str]:
    if go_enriched_csv is None or not go_enriched_csv.exists():
        logger.warning("GO enriched file not provided/found; immune tagging skipped")
        return set()
    df = pd.read_csv(go_enriched_csv)
    text_cols = [c for c in ['go_name', 'go_definition'] if c in df.columns]
    if not text_cols or 'biomarkers' not in df.columns:
        logger.warning("GO enriched file lacks expected columns; immune tagging skipped")
        return set()
    pattern = re.compile('|'.join([re.escape(k) for k in keywords]), re.IGNORECASE)
    immune = set()
    for _, row in df.iterrows():
        text = ' '.join([str(row[c]) for c in text_cols])
        if pattern.search(text):
            biomarker_tokens = re.split(r'[;,]', str(row['biomarkers']))
            for token in biomarker_tokens:
                token = token.strip()
                if not token:
                    continue
                protein_id = token if token.startswith('Protein_') else f'Protein_{token}'
                immune.add(protein_id)
    logger.info(f"Tagged {len(immune)} immune-related proteins from GO terms")
    return immune


def build_edges(ppi_df: pd.DataFrame, allowed: Set[str]) -> Dict[Tuple[str, str], Dict]:
    """Build edges with attributes (type, weight) if available."""
    edges: Dict[Tuple[str, str], Dict] = {}
    for _, row in ppi_df.iterrows():
        src = row['protein']
        if src not in allowed:
            continue
        # Parse edge types if available (format: protein:type:weight or just protein)
        neighbor_str = str(row.get('protein_neighbors', ''))
        if pd.isna(neighbor_str):
            continue
        for token in neighbor_str.split(';'):
            if not token:
                continue
            parts = token.split(':')
            nbr = parts[0].strip()
            edge_type = parts[1] if len(parts) > 1 else 'interaction'
            weight = float(parts[2]) if len(parts) > 2 else 1.0
            
            if nbr in allowed:
                a, b = sorted([src, nbr])
                edge_key = (a, b)
                # Keep the edge with highest weight if duplicate
                if edge_key not in edges or edges[edge_key]['weight'] < weight:
                    edges[edge_key] = {'type': edge_type, 'weight': weight}
    logger.info(f"Kept {len(edges)} PPI edges among {len(allowed)} nodes")
    return edges


def build_graph(consensus_df: pd.DataFrame, hubs_df: pd.DataFrame, ppi_df: pd.DataFrame, include_hubs: bool, max_hubs: int = 20) -> Tuple[nx.Graph, pd.DataFrame]:
    """Build graph with filtered hubs and edge attributes."""
    allowed = set(consensus_df['protein'])
    
    # First pass to compute hub connectivity
    filtered_hubs_df = hubs_df
    if include_hubs and not hubs_df.empty:
        # Build temporary graph to compute hub degrees
        temp_allowed = allowed.union(hubs_df['protein'])
        temp_edges = build_edges(ppi_df, temp_allowed)
        temp_G = nx.Graph()
        temp_G.add_edges_from(temp_edges.keys())
        
        # Filter hubs: keep top N by degree (connectivity to consensus)
        hub_degrees = []
        for hub in hubs_df['protein']:
            if hub in temp_G:
                # Count connections to consensus proteins only
                consensus_neighbors = sum(1 for n in temp_G.neighbors(hub) if n in allowed)
                hub_degrees.append((hub, consensus_neighbors))
        
        hub_degrees.sort(key=lambda x: x[1], reverse=True)
        top_hubs = {h for h, _ in hub_degrees[:max_hubs]}
        filtered_hubs_df = hubs_df[hubs_df['protein'].isin(top_hubs)]
        allowed.update(top_hubs)
        logger.info(f"Filtered to {len(filtered_hubs_df)} hub proteins (from {len(hubs_df)})")
    
    edges = build_edges(ppi_df, allowed)
    G = nx.Graph()
    G.add_nodes_from(allowed)
    
    # Add edges with attributes
    for (u, v), attrs in edges.items():
        G.add_edge(u, v, **attrs)
    
    return G, filtered_hubs_df


def compute_node_attributes(consensus_df: pd.DataFrame, hubs_df: pd.DataFrame, immune_set: Set[str]) -> Dict[str, Dict]:
    attrs: Dict[str, Dict] = {}
    consensus_scores = dict(zip(consensus_df['protein'], consensus_df['avg_score']))
    min_score = min(consensus_scores.values()) if consensus_scores else 0
    max_score = max(consensus_scores.values()) if consensus_scores else 1
    score_range = max(max_score - min_score, 1e-6)
    for protein, score in consensus_scores.items():
        attrs[protein] = {
            'label': protein.replace('Protein_', ''),
            'score': score,
            'kind': 'consensus',
            'immune': protein in immune_set,
        }
    for _, row in hubs_df.iterrows():
        protein = row['protein']
        attrs.setdefault(protein, {
            'label': protein.replace('Protein_', ''),
            'score': min_score,
            'kind': 'hub',
            'immune': protein in immune_set,
        })
    for protein, data in attrs.items():
        norm = (data['score'] - min_score) / score_range
        # Larger size range for better visual distinction (200-800)
        if data['kind'] == 'hub':
            data['size'] = 150  # Smaller size for hubs
        else:
            data['size'] = 300 + 500 * norm
    return attrs


def pick_labels(G: nx.Graph, attrs: Dict[str, Dict], top_n: int) -> Dict[str, str]:
    """Pick top-N highest degree nodes for labeling, excluding hubs."""
    if G.number_of_nodes() == 0:
        return {}
    # Only consider non-hub nodes for labeling
    non_hub_nodes = [n for n in G.nodes() if attrs.get(n, {}).get('kind') != 'hub']
    degrees = [(n, G.degree(n)) for n in non_hub_nodes]
    ranked = sorted(degrees, key=lambda x: x[1], reverse=True)
    selected = {node for node, _ in ranked[:top_n]}
    labels = {n: attrs[n]['label'] for n in selected if n in attrs}
    logger.info(f"Labeling {len(labels)} non-hub nodes (top {top_n} by degree)")
    return labels


def draw_network(G: nx.Graph, attrs: Dict[str, Dict], output_dir: Path, title: str, label_top: int):
    if G.number_of_nodes() == 0:
        logger.warning("Graph is empty; skipping plot")
        return
    pos = nx.spring_layout(G, seed=42, k=None)
    plt.figure(figsize=(14, 14))
    
    # Define edge type colors and styles
    edge_colors_map = {
        'physical': '#ef4444',      # Red
        'genetic': '#3b82f6',       # Blue
        'pathway': '#8b5cf6',       # Purple
        'coexpression': '#10b981',  # Green
        'interaction': '#9ca3af',   # Gray (default)
    }

    edge_types_all = [str(data.get('type', 'interaction')).lower() for _, _, data in G.edges(data=True)]
    diverse_edge_types = len(set(edge_types_all)) > 1
    
    # Group edges by type for rendering
    edges_by_type = {}
    for u, v, data in G.edges(data=True):
        edge_type = data.get('type', 'interaction')
        if edge_type not in edges_by_type:
            edges_by_type[edge_type] = []
        edges_by_type[edge_type].append((u, v))
    
    # Setup colormap for degree-based coloring (if types are not diverse)
    cmap = plt.cm.plasma  # 'plasma' looks good on white: Blue -> Red -> Yellow
    
    # Calculate degree sums for normalization
    all_edges = G.edges()
    degree_sums = [G.degree(u) + G.degree(v) for u, v in all_edges]
    if not degree_sums:
        d_min, d_max = 0, 1
    else:
        d_min, d_max = min(degree_sums), max(degree_sums)
    if d_max - d_min < 1e-6:
        d_max += 1

    # Draw edges
    for edge_type, edge_list in edges_by_type.items():
        # Get width based on weights (optional)
        weights = [G[u][v].get('weight', 1.0) for u, v in edge_list]
        widths = [0.5 + 2.0 * min(w, 1.0) for w in weights]

        if diverse_edge_types:
            color = edge_colors_map.get(edge_type, edge_colors_map['interaction'])
            nx.draw_networkx_edges(G, pos, edgelist=edge_list, width=widths, alpha=0.4, edge_color=color)
        else:
            # Color by Degree Sum (Centrality)
            # Higher degree sum = brighter/warmer color
            current_degree_sums = [G.degree(u) + G.degree(v) for u, v in edge_list]
            edge_colors = [mcolors.to_hex(cmap((ds - d_min) / (d_max - d_min))) for ds in current_degree_sums]
            nx.draw_networkx_edges(G, pos, edgelist=edge_list, width=widths, alpha=0.6, edge_color=edge_colors)

    groups = {
        'consensus_immune': {'nodes': [], 'color': '#D63262'}, #D63262
        'consensus_other': {'nodes': [], 'color': '#218396'},
        'hub': {'nodes': [], 'color': '#519E69'},
    }
    for node in G.nodes():
        info = attrs.get(node, {})
        if info.get('kind') == 'hub':
            groups['hub']['nodes'].append(node)
        elif info.get('immune'):
            groups['consensus_immune']['nodes'].append(node)
        else:
            groups['consensus_other']['nodes'].append(node)
    
    for group_name, group in groups.items():
        if not group['nodes']:
            continue
        sizes = [attrs[n]['size'] for n in group['nodes']]
        nx.draw_networkx_nodes(group['nodes'], pos, node_color=group['color'], node_size=sizes, alpha=0.95, label=group_name.replace('_', ' ').title())
    
    labels = pick_labels(G, attrs, label_top)
    if labels:
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight='bold', font_family='DejaVu Sans')
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    
    # Add legends
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    
    # Edge Legend
    if diverse_edge_types:
        edge_legend_elements = [
            Line2D([0], [0], color='#ef4444', linewidth=2, label='Physical'),
            Line2D([0], [0], color='#3b82f6', linewidth=2, label='Genetic'),
            Line2D([0], [0], color='#8b5cf6', linewidth=2, label='Pathway'),
            Line2D([0], [0], color='#10b981', linewidth=2, label='Coexpression'),
            Line2D([0], [0], color='#9ca3af', linewidth=2, label='Other'),
        ]
        edge_title = "Edge Types"
    else:
        # Create a gradient legend for degree centrality
        edge_legend_elements = [
            Line2D([0], [0], color=mcolors.to_hex(cmap(0.1)), linewidth=2, label='Low Connectivity'),
            Line2D([0], [0], color=mcolors.to_hex(cmap(0.5)), linewidth=2, label='Medium'),
            Line2D([0], [0], color=mcolors.to_hex(cmap(0.9)), linewidth=2, label='High (Backbone)'),
        ]
        edge_title = "Edge Centrality"

    # Node type legend
    node_legend_elements = [
        Patch(facecolor='#d84b4b', label='Consensus (Immune)'),
        Patch(facecolor='#3b82f6', label='Consensus (Other)'),
        Patch(facecolor='#7e2cad', label='Hub Connector'),
    ]
    
    # Position legends
    edge_legend = plt.legend(handles=edge_legend_elements, loc='upper left', 
                             title=edge_title, fontsize=11, title_fontsize=12, framealpha=0.95)
    plt.gca().add_artist(edge_legend)
    plt.legend(handles=node_legend_elements, loc='lower left', 
               title='Node Types', fontsize=11, title_fontsize=12, framealpha=0.95)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / 'consensus_ppi_poster.png'
    svg_path = output_dir / 'consensus_ppi_poster.svg'
    plt.savefig(png_path, dpi=400, bbox_inches='tight', facecolor='white')
    plt.savefig(svg_path, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved plot: {png_path}")
    logger.info(f"Saved plot: {svg_path}")


def draw_pyvis(G: nx.Graph, attrs: Dict[str, Dict], output_dir: Path, title: str, label_top: int, height: str = "800px"):
    if G.number_of_nodes() == 0:
        logger.warning("Graph is empty; skipping pyvis plot")
        return

    net = Network(height=height, width="100%", directed=False, notebook=False)
    
    # Compute a good layout using NetworkX
    try:
        pos = nx.kamada_kawai_layout(G, scale=1000)
    except:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42, scale=1000)
    
    color_map = {
        'consensus_immune': '#D63262',
        'consensus_other': '#519E69',
        'hub': "#B69D23",
    }

    visible_labels = pick_labels(G, attrs, label_top)

    for node in G.nodes():
        info = attrs.get(node, {})
        kind = info.get('kind', 'consensus')
        immune = info.get('immune', False)
        if kind == 'hub':
            group = 'hub'
        elif immune:
            group = 'consensus_immune'
        else:
            group = 'consensus_other'

        x, y = pos[node]
        display_label = info.get('label', '') if node in visible_labels else ' '
        
        net.add_node(
            node,
            label=display_label,
            size=info.get('size', 300) * 0.05,
            color=color_map[group],
            title=f"{info.get('label', node)} | score={info.get('score', 0):.3f} | degree={G.degree(node)} | {group}",
            group=group,
            x=float(x),
            y=float(y),
        )

    # Prepare for edge coloring
    def _normalize_edge_type(raw: str) -> str:
        t = raw.lower().strip()
        if 'physic' in t: return 'physical'
        if 'genetic' in t: return 'genetic'
        if 'pathway' in t: return 'pathway'
        if 'coexp' in t or 'co-exp' in t: return 'coexpression'
        return 'interaction'

    edge_types_all = [_normalize_edge_type(str(data.get('type', 'interaction'))) for _, _, data in G.edges(data=True)]
    diverse_edge_types = len(set(edge_types_all)) > 1
    
    edge_colors_map = {
        'physical': '#ef4444',
        'genetic': '#3b82f6',
        'pathway': '#8b5cf6',
        'coexpression': '#10b981',
        'interaction': '#9ca3af',
    }
    
    # Pre-calculate degree stats for coloring
    all_edges = G.edges()
    degree_sums = [G.degree(u) + G.degree(v) for u, v in all_edges]
    if not degree_sums:
        d_min, d_max = 0, 1
    else:
        d_min, d_max = min(degree_sums), max(degree_sums)
    if d_max - d_min < 1e-6:
        d_max += 1
    
    cmap = plt.cm.plasma # Match the static plot

    for u, v, data in G.edges(data=True):
        edge_type_raw = str(data.get('type', 'interaction'))
        edge_type = _normalize_edge_type(edge_type_raw)
        weight = data.get('weight', 1.0)
        width = 1 + 3 * min(weight, 1.0)

        # Logic: If types are diverse, use type-color. Else, use degree-sum heat map.
        if diverse_edge_types:
            color_val = edge_colors_map.get(edge_type, edge_colors_map['interaction'])
        else:
            # Calculate local degree sum
            deg_sum = G.degree(u) + G.degree(v)
            norm = (deg_sum - d_min) / (d_max - d_min)
            color_val = mcolors.to_hex(cmap(norm))

        net.add_edge(u, v,
                     color=color_val,
                     width=width,
                     title=f"Type: {edge_type_raw} | Weight: {weight:.2f} | Connectivity: {G.degree(u)+G.degree(v)}")

    # IMPORTANT: Disable global inheritance so specific edge colors work
    net.set_options(json.dumps({
        "physics": {"enabled": False},
        "interaction": {"hover": True, "navigationButtons": True, "dragNodes": True, "dragView": True, "zoomView": True},
        "nodes": {"font": {"size": 30, "face": "arial", "bold": {"color": "#111"}}},
        "edges": {
            "smooth": {"type": "continuous"},
            "color": {"inherit": False}
        },
        "groups": {
            "consensus_immune": {"color": {"background": color_map['consensus_immune'], "border": "#111", "highlight": {"background": color_map['consensus_immune'], "border": "#111"}}},
            "consensus_other": {"color": {"background": color_map['consensus_other'], "border": "#111", "highlight": {"background": color_map['consensus_other'], "border": "#111"}}},
            "hub": {"color": {"background": color_map['hub'], "border": "#111", "highlight": {"background": color_map['hub'], "border": "#111"}}}
        }
    }))

    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / 'consensus_ppi_poster.html'
    if hasattr(net, 'options') and hasattr(net.options, 'configure'):
        net.show_buttons(filter_=['physics'])
    net.write_html(str(html_path), open_browser=False, notebook=False)
    logger.info(f"Saved interactive plot: {html_path}")


def save_tables(G: nx.Graph, attrs: Dict[str, Dict], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    nodes_records = []
    for node in G.nodes():
        info = attrs.get(node, {})
        nodes_records.append({
            'protein': node,
            'label': info.get('label', node.replace('Protein_', '')),
            'score': info.get('score', 0),
            'kind': info.get('kind', 'unknown'),
            'immune': info.get('immune', False),
            'size': info.get('size', 300),
            'degree': G.degree(node),
        })
    edges_records = [{'source': u, 'target': v} for u, v in G.edges()]
    pd.DataFrame(nodes_records).to_csv(output_dir / 'consensus_ppi_nodes.csv', index=False)
    pd.DataFrame(edges_records).to_csv(output_dir / 'consensus_ppi_edges.csv', index=False)
    logger.info("Saved node and edge tables")


def parse_args():
    parser = argparse.ArgumentParser(description='Generate consensus biomarker PPI network plot')
    parser.add_argument('--consensus-csv', type=Path, default=Path('results/interpretability/complex_svm_mapped/complex_consensus_proteins.csv'))
    parser.add_argument('--ppi-csv', type=Path, default=Path('results/interpretability/complex_svm_mapped/kg_subgraph_analysis_consensus/protein_interactions.csv'))
    parser.add_argument('--hubs-csv', type=Path, default=Path('results/interpretability/complex_svm_mapped/kg_subgraph_analysis_consensus/hub_proteins.csv'))
    parser.add_argument('--go-enriched-csv', type=Path, default=Path('results/interpretability/complex_svm_mapped/kg_subgraph_analysis_consensus/biomarker_go_terms_enriched.csv'))
    parser.add_argument('--output-dir', type=Path, default=Path('results/interpretability/complex_svm_mapped/plots'))
    parser.add_argument('--include-hubs', action=argparse.BooleanOptionalAction, default=True, help='Include hub connectors in the plot')
    parser.add_argument('--max-hubs', type=int, default=65, help='Maximum number of hub nodes to include')
    parser.add_argument('--immune-keywords', type=str, default='immune,macrophage,tlr,toll-like,interferon,cytokine,leukocyte,lymph,innate,adaptive,antigen,mhc,nod2,tlr9')
    parser.add_argument('--label-top', type=int, default=28, help='Number of highest-degree non-hub nodes to label')
    parser.add_argument('--no-static', action='store_true', help='Skip static PNG/SVG rendering')
    parser.add_argument('--pyvis-height', type=str, default='800px', help='Height for pyvis canvas (e.g., 800px)')
    return parser.parse_args()


def main():
    args = parse_args()
    consensus_df = load_consensus(args.consensus_csv)
    ppi_df = load_ppi(args.ppi_csv)
    hubs_df = load_hubs(args.hubs_csv) if args.include_hubs else pd.DataFrame(columns=['protein'])
    keywords = [k.strip() for k in args.immune_keywords.split(',') if k.strip()]
    immune_set = load_immune_set(args.go_enriched_csv, keywords)
    
    G, filtered_hubs_df = build_graph(consensus_df, hubs_df, ppi_df, include_hubs=args.include_hubs, max_hubs=args.max_hubs)
    attrs = compute_node_attributes(consensus_df, filtered_hubs_df, immune_set)
    if not args.no_static:
        draw_network(G, attrs, args.output_dir, title='Consensus biomarkers PPI (ComplEx + 4 models)', label_top=args.label_top)
    
    draw_pyvis(G, attrs, args.output_dir, title='Consensus biomarkers PPI (ComplEx + 4 models)', 
               label_top=args.label_top, height=args.pyvis_height)
    save_tables(G, attrs, args.output_dir)


if __name__ == '__main__':
    main()