#!/usr/bin/env python3
"""
Extract and analyze biomarker subgraph from Knowledge Graph (PROTEINS ONLY).

This script:
1. Loads the KG and identifies biomarker proteins (consensus or SVM)
2. Extracts 2-hop neighborhood subgraph around biomarker proteins
3. Analyzes protein-protein interactions
4. Identifies hub proteins connecting multiple biomarkers
5. Calculates centrality metrics
6. Detects protein functional modules (communities)
7. Generates visualizations and reports
"""

import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
import logging
import json
from typing import Dict, Set, List, Tuple
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


def convert_owl_to_csv(owl_path: str, csv_output: str) -> bool:
    """Convert OWL to CSV using OntoKGCreation toolkit."""
    try:
        if Path(csv_output).exists():
            logger.info(f"CSV already exists: {csv_output}")
            return True
        
        cmd = [
            'python', 
            'src/preprocessing/graph_to_csv.py',
            '--input', owl_path,
            '--output', csv_output
        ]
        logger.info(f"Converting OWL to CSV: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            logger.error(f"Conversion failed: {result.stderr}")
            return False
        
        logger.info(f"Conversion successful: {csv_output}")
        return True
    
    except Exception as e:
        logger.error(f"Error converting OWL: {e}")
        return False


def load_knowledge_graph(owl_path: str = None, csv_path: str = None) -> Dict:
    """Load KG from CSV directly (assuming pre-converted)."""
    node_info = {}
    edges = []
    
    if csv_path and Path(csv_path).exists():
        logger.info(f"Loading from CSV: {csv_path}")
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"CSV columns: {df.columns.tolist()}")
            
            # Try different column name formats
            if all(col in df.columns for col in ['subject', 'predicate', 'object']):
                for _, row in df.iterrows():
                    edges.append({
                        'source': row['subject'],
                        'target': row['object'],
                        'relation': row['predicate']
                    })
            elif all(col in df.columns for col in ['source', 'relation', 'target']):
                for _, row in df.iterrows():
                    edges.append({
                        'source': row['source'],
                        'target': row['target'],
                        'relation': row['relation']
                    })
            else:
                logger.error(f"CSV format not recognized. Available columns: {df.columns.tolist()}")
                return {'node_info': {}, 'edges': []}
        
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            return {'node_info': {}, 'edges': []}
    
    logger.info(f"Loaded {len(edges)} edges")
    return {'node_info': node_info, 'edges': edges}


def build_networkx_graph(kg_data: Dict) -> nx.Graph:
    """Build NetworkX graph from KG data."""
    G = nx.Graph()  # Undirected for connectivity analysis
    
    for edge in kg_data['edges']:
        source = edge['source']
        target = edge['target']
        relation = edge.get('relation', 'unknown')
        
        G.add_edge(source, target, relation=relation)
    
    logger.info(f"Built graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def extract_protein_biomarkers(biomarker_csv: str = None) -> Set[str]:
    """Load protein biomarkers from consensus or individual feature mapping."""
    if biomarker_csv is None:
        # Try consensus first, fall back to SVM
        consensus_path = Path('results/interpretability/complex_svm_mapped/complex_consensus_proteins.csv')
        svm_path = Path('results/interpretability/complex_svm_mapped/complex_svm_biomarkers.csv')
        
        if consensus_path.exists():
            biomarker_csv = str(consensus_path)
            logger.info("Using consensus proteins")
        elif svm_path.exists():
            biomarker_csv = str(svm_path)
            logger.info("Using SVM proteins")
        else:
            logger.error(f"Biomarker file not found")
            return set()
    
    df = pd.read_csv(biomarker_csv)
    
    # Handle different CSV formats
    if 'protein' in df.columns:
        # Consensus format
        proteins = set(df['protein'].unique())
    elif 'entity' in df.columns:
        # SVM format
        proteins = set(df[df['entity_type'] == 'Protein']['entity'].unique())
    else:
        logger.error(f"Unknown biomarker CSV format. Columns: {df.columns.tolist()}")
        return set()
    
    logger.info(f"Loaded {len(proteins)} protein biomarkers")
    return proteins


def extract_subgraph_around_biomarkers(G: nx.Graph, biomarkers: Set[str], hop_distance: int = 1) -> nx.Graph:
    """Extract subgraph around biomarkers (k-hop neighborhood)."""
    subgraph_nodes = set()
    
    for biomarker in biomarkers:
        if biomarker in G:
            # Get k-hop neighborhood
            neighborhood = nx.single_source_shortest_path_length(G, biomarker, cutoff=hop_distance)
            subgraph_nodes.update(neighborhood.keys())
        else:
            logger.warning(f"Biomarker not in graph: {biomarker}")
    
    # Extract subgraph
    subgraph = G.subgraph(subgraph_nodes).copy()
    logger.info(f"Extracted subgraph: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")
    
    return subgraph


def analyze_protein_interactions(subgraph: nx.Graph, biomarkers: Set[str]) -> pd.DataFrame:
    """Analyze protein-protein interactions (edges between proteins)."""
    results = []
    protein_nodes = {node for node in subgraph.nodes() if node.startswith('Protein_')}
    
    for node in protein_nodes:
        neighbors = set(subgraph.neighbors(node))
        protein_neighbors = neighbors & protein_nodes
        
        if protein_neighbors:
            results.append({
                'protein': node,
                'protein_name': node.replace('Protein_', ''),
                'num_protein_neighbors': len(protein_neighbors),
                'protein_neighbors': ';'.join(sorted(protein_neighbors)),
            })
    
    df = pd.DataFrame(results)
    logger.info(f"Found {len(df)} proteins with protein-protein interactions: {df['num_protein_neighbors'].sum()} total connections")
    
    return df.sort_values('num_protein_neighbors', ascending=False)


def find_common_protein_hubs(subgraph: nx.Graph, biomarkers: Set[str]) -> pd.DataFrame:
    """Find proteins connecting multiple biomarkers (hub proteins)."""
    results = []
    protein_nodes = {node for node in subgraph.nodes() if node.startswith('Protein_')}
    
    for protein in protein_nodes:
        if protein in biomarkers:
            continue  # Skip if it's already a biomarker
        
        # Find all paths from this protein to biomarkers
        connected_biomarkers = []
        for biomarker in biomarkers:
            if biomarker in subgraph and nx.has_path(subgraph, protein, biomarker):
                connected_biomarkers.append(biomarker)
        
        # Only include proteins connecting to multiple biomarkers
        if len(connected_biomarkers) > 1:
            results.append({
                'protein': protein,
                'protein_name': protein.replace('Protein_', ''),
                'num_biomarkers': len(connected_biomarkers),
                'biomarkers': ','.join(sorted(connected_biomarkers)),
                'biomarker_names': ','.join([b.replace('Protein_', '') for b in sorted(connected_biomarkers)])
            })
    
    df = pd.DataFrame(results)
    logger.info(f"Found {len(df)} hub proteins connecting multiple biomarkers")
    
    return df.sort_values('num_biomarkers', ascending=False)


def calculate_centrality_metrics(subgraph: nx.Graph, biomarkers: Set[str]) -> pd.DataFrame:
    """Calculate centrality metrics for biomarker proteins."""
    protein_biomarkers = {b for b in biomarkers if b.startswith('Protein_')}
    
    degree = nx.degree_centrality(subgraph)
    betweenness = nx.betweenness_centrality(subgraph)
    closeness = nx.closeness_centrality(subgraph)
    
    results = []
    for protein in protein_biomarkers:
        if protein in subgraph:
            results.append({
                'protein': protein,
                'protein_name': protein.replace('Protein_', ''),
                'degree_centrality': degree.get(protein, 0),
                'betweenness_centrality': betweenness.get(protein, 0),
                'closeness_centrality': closeness.get(protein, 0),
            })
    
    df = pd.DataFrame(results)
    logger.info(f"Calculated centrality for {len(df)} biomarker proteins")
    
    return df.sort_values('degree_centrality', ascending=False)


def detect_communities(subgraph: nx.Graph) -> Dict:
    """Detect functional modules using Louvain method."""
    try:
        from networkx.algorithms import community
        communities_gen = community.greedy_modularity_communities(subgraph)
        
        communities_dict = {}
        protein_communities = {}
        
        for i, comm in enumerate(communities_gen):
            proteins = {n for n in comm if n.startswith('Protein_')}
            if proteins:
                community_id = f"Module_{i}"
                communities_dict[community_id] = {
                    'num_nodes': len(comm),
                    'num_proteins': len(proteins),
                    'proteins': sorted(list(proteins)),
                }
                for protein in proteins:
                    protein_communities[protein] = community_id
        
        logger.info(f"Detected {len(communities_dict)} functional modules")
        return {'modules': communities_dict, 'protein_to_module': protein_communities}
    
    except Exception as e:
        logger.warning(f"Could not detect communities: {e}")
        return {'modules': {}, 'protein_to_module': {}}


def analyze_connected_entity_types(subgraph: nx.Graph, biomarkers: Set[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Analyze pathways, GO terms, and reactions connected to biomarker proteins."""
    
    # Extract connected entities
    pathway_connections = defaultdict(set)
    go_connections = defaultdict(set)
    reaction_connections = defaultdict(set)
    
    for biomarker in biomarkers:
        if biomarker not in subgraph:
            continue
        
        neighbors = set(subgraph.neighbors(biomarker))
        
        for neighbor in neighbors:
            if neighbor.startswith('Pathway_'):
                pathway_connections[neighbor].add(biomarker)
            elif neighbor.startswith('GO_'):
                go_connections[neighbor].add(biomarker)
            elif neighbor.startswith('Reaction_'):
                reaction_connections[neighbor].add(biomarker)
    
    # Create DataFrames
    pathway_df = pd.DataFrame([
        {
            'pathway': pathway,
            'pathway_name': pathway.replace('Pathway_', ''),
            'num_biomarkers': len(biomarkers_set),
            'biomarkers': ','.join(sorted([b.replace('Protein_', '') for b in biomarkers_set]))
        }
        for pathway, biomarkers_set in pathway_connections.items()
    ]).sort_values('num_biomarkers', ascending=False)
    
    go_df = pd.DataFrame([
        {
            'go_term': go_id,
            'go_id': go_id.replace('GO_', '').replace('_instance', ''),
            'num_biomarkers': len(biomarkers_set),
            'biomarkers': ','.join(sorted([b.replace('Protein_', '') for b in biomarkers_set]))
        }
        for go_id, biomarkers_set in go_connections.items()
    ]).sort_values('num_biomarkers', ascending=False)
    
    reaction_df = pd.DataFrame([
        {
            'reaction': reaction,
            'reaction_name': reaction.replace('Reaction_', ''),
            'num_biomarkers': len(biomarkers_set),
            'biomarkers': ','.join(sorted([b.replace('Protein_', '') for b in biomarkers_set]))
        }
        for reaction, biomarkers_set in reaction_connections.items()
    ]).sort_values('num_biomarkers', ascending=False)
    
    logger.info(f"Found {len(pathway_df)} pathways connected to biomarkers")
    logger.info(f"Found {len(go_df)} GO terms connected to biomarkers")
    logger.info(f"Found {len(reaction_df)} reactions connected to biomarkers")
    
    return pathway_df, go_df, reaction_df


def visualize_protein_pathway_network(subgraph: nx.Graph, biomarkers: Set[str], 
                                      pathway_df: pd.DataFrame, output_path: str):
    """Visualize bipartite network of proteins and pathways."""
    try:
        if len(pathway_df) == 0:
            logger.warning("No pathways to visualize")
            return
        
        # Create bipartite graph
        B = nx.Graph()
        
        # Add protein and pathway nodes
        B.add_nodes_from(biomarkers, bipartite=0, node_type='protein')
        
        top_pathways = pathway_df.head(20)  # Limit to top 20 for readability
        pathway_nodes = set(top_pathways['pathway'].tolist())
        B.add_nodes_from(pathway_nodes, bipartite=1, node_type='pathway')
        
        # Add edges
        for pathway, protein_set in [(row['pathway'], row['biomarkers'].split(',')) 
                                     for _, row in top_pathways.iterrows()]:
            protein_names = [f"Protein_{p}" for p in protein_set]
            for protein in protein_names:
                if protein in biomarkers:
                    B.add_edge(protein, pathway)
        
        # Layout using bipartite layout
        pos = {}
        proteins_list = sorted(list(biomarkers))
        pathways_list = sorted(list(pathway_nodes))
        
        # Proteins on left, pathways on right
        for i, protein in enumerate(proteins_list):
            pos[protein] = (0, i)
        for i, pathway in enumerate(pathways_list):
            pos[pathway] = (2, i)
        
        # Draw
        plt.figure(figsize=(12, 10))
        
        # Protein nodes (red)
        nx.draw_networkx_nodes([n for n in B.nodes() if n.startswith('Protein_')], pos, 
                              node_color='red', node_size=400, alpha=0.8, label='Biomarkers')
        
        # Pathway nodes (blue)
        nx.draw_networkx_nodes([n for n in B.nodes() if n.startswith('Pathway_')], pos, 
                              node_color='lightblue', node_size=300, alpha=0.8, label='Pathways')
        
        # Edges
        nx.draw_networkx_edges(B, pos, alpha=0.3, width=1)
        
        # Labels
        labels = {n: n.replace('Protein_', '').replace('Pathway_', '') for n in B.nodes()}
        nx.draw_networkx_labels(B, pos, labels, font_size=8)
        
        plt.title("Biomarker Protein - Pathway Network\n(Top 20 pathways)", 
                 fontsize=14, fontweight='bold')
        plt.legend(loc='upper left')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved pathway network visualization: {output_path}")
        plt.close()
    
    except Exception as e:
        logger.error(f"Error visualizing pathway network: {e}")


def visualize_go_term_enrichment(go_df: pd.DataFrame, output_path: str):
    """Visualize GO term enrichment as bubble plot."""
    try:
        if len(go_df) == 0:
            logger.warning("No GO terms to visualize")
            return
        
        top_go = go_df.head(20)
        
        plt.figure(figsize=(12, 8))
        plt.scatter(range(len(top_go)), top_go['num_biomarkers'], 
                   s=top_go['num_biomarkers']*100, alpha=0.6, c=range(len(top_go)), cmap='viridis')
        
        plt.xlabel('GO Terms', fontsize=12)
        plt.ylabel('Number of Biomarkers', fontsize=12)
        plt.title('GO Term Enrichment in Biomarker Proteins\n(Top 20 by biomarker count)', 
                 fontsize=14, fontweight='bold')
        
        # Shorten labels for readability
        go_labels = []
        for go_id in top_go['go_id']:
            # Handle different formats
            if isinstance(go_id, str):
                # Extract just the number part
                num = go_id.replace('GO:', '').replace('GO_', '').split('_')[0]
                go_labels.append(f"GO:{num}")
            else:
                go_labels.append(f"GO:{go_id}")
        
        plt.xticks(range(len(top_go)), go_labels, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved GO term enrichment plot: {output_path}")
        plt.close()
    
    except Exception as e:
        logger.error(f"Error visualizing GO enrichment: {e}")


def visualize_multilayer_network(subgraph: nx.Graph, biomarkers: Set[str], 
                                 pathway_df: pd.DataFrame, go_df: pd.DataFrame, 
                                 output_path: str):
    """Visualize multi-layer network with proteins, pathways, and GO terms."""
    try:
        # Build network with all entity types
        M = nx.Graph()
        
        # Add biomarker proteins
        M.add_nodes_from(biomarkers, entity_type='protein')
        
        # Add top pathways and GO terms
        top_pathways = pathway_df.head(10)['pathway'].tolist()
        top_go = go_df.head(10)['go_term'].tolist()
        
        M.add_nodes_from(top_pathways, entity_type='pathway')
        M.add_nodes_from(top_go, entity_type='go_term')
        
        # Add edges from subgraph
        for protein in biomarkers:
            if protein in subgraph:
                for neighbor in subgraph.neighbors(protein):
                    if neighbor in top_pathways or neighbor in top_go:
                        M.add_edge(protein, neighbor)
        
        # Layout
        pos = nx.spring_layout(M, k=1, iterations=50, seed=42)
        
        # Draw
        plt.figure(figsize=(16, 12))
        
        # Node colors by type
        protein_nodes = [n for n in M.nodes() if n.startswith('Protein_')]
        pathway_nodes = [n for n in M.nodes() if n.startswith('Pathway_')]
        go_nodes = [n for n in M.nodes() if n.startswith('GO_')]
        
        nx.draw_networkx_nodes(protein_nodes, pos, node_color='red', node_size=600, 
                              alpha=0.8, label='Biomarkers')
        nx.draw_networkx_nodes(pathway_nodes, pos, node_color='lightblue', node_size=400, 
                              alpha=0.8, label='Pathways')
        nx.draw_networkx_nodes(go_nodes, pos, node_color='lightgreen', node_size=300, 
                              alpha=0.8, label='GO Terms')
        
        # Edges
        nx.draw_networkx_edges(M, pos, alpha=0.2, width=1)
        
        # Labels
        labels = {}
        for n in protein_nodes:
            labels[n] = n.replace('Protein_', '')
        for n in pathway_nodes:
            labels[n] = 'P:' + n.replace('Pathway_', '')[:10]
        for n in go_nodes:
            labels[n] = 'GO:' + n.split('_')[1]
        
        nx.draw_networkx_labels(M, pos, labels, font_size=8)
        
        plt.title("Multi-layer Biomarker Network\n(Proteins + Pathways + GO Terms)", 
                 fontsize=14, fontweight='bold')
        plt.legend(loc='upper left', fontsize=11)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved multi-layer network visualization: {output_path}")
        plt.close()
    
    except Exception as e:
        logger.error(f"Error visualizing multi-layer network: {e}")


def visualize_biomarker_subgraph(subgraph: nx.Graph, biomarkers: Set[str], output_path: str):
    """Visualize protein biomarker network."""
    try:
        # Filter to protein nodes only for cleaner visualization
        protein_nodes = {node for node in subgraph.nodes() if node.startswith('Protein_')}
        protein_subgraph = subgraph.subgraph(protein_nodes).copy()
        
        if protein_subgraph.number_of_nodes() < 2:
            logger.warning("Not enough protein nodes to visualize")
            return
        
        plt.figure(figsize=(14, 10))
        
        # Layout
        pos = nx.spring_layout(protein_subgraph, k=0.5, iterations=50, seed=42)
        
        # Node colors: biomarkers vs hubs
        node_colors = []
        for node in protein_subgraph.nodes():
            if node in biomarkers:
                node_colors.append('red')
            else:
                node_colors.append('lightblue')
        
        # Draw
        nx.draw_networkx_nodes(protein_subgraph, pos, node_color=node_colors, 
                              node_size=500, alpha=0.9)
        nx.draw_networkx_edges(protein_subgraph, pos, alpha=0.3, width=1)
        
        # Labels (shortened protein names)
        labels = {node: node.replace('Protein_', '') for node in protein_subgraph.nodes()}
        nx.draw_networkx_labels(protein_subgraph, pos, labels, font_size=8)
        
        plt.title("Protein Biomarker Network (Proteins Only)\nRed: Biomarkers, Blue: Hub Connectors", 
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization: {output_path}")
        plt.close()
    
    except Exception as e:
        logger.error(f"Error visualizing: {e}")


def create_analysis_report(biomarkers: Set[str], ppi_df: pd.DataFrame, 
                          hub_df: pd.DataFrame, centrality_df: pd.DataFrame,
                          communities: Dict, output_path: str):
    """Create comprehensive analysis report."""
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CONSENSUS PROTEIN BIOMARKER SUBGRAPH ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total biomarker proteins: {len(biomarkers)}\n")
        f.write(f"Biomarker proteins:\n")
        for protein in sorted(biomarkers):
            f.write(f"  - {protein.replace('Protein_', '')}\n")
        
        f.write(f"\n" + "-"*80 + "\n")
        f.write("PROTEIN-PROTEIN INTERACTIONS\n")
        f.write("-"*80 + "\n")
        
        if len(ppi_df) > 0:
            f.write(f"Proteins with PPI: {len(ppi_df)}\n")
            f.write(f"Total protein-protein connections: {ppi_df['num_protein_neighbors'].sum()}\n\n")
            f.write(ppi_df.to_string(index=False))
        else:
            f.write("No protein-protein interactions found in subgraph.\n")
        
        f.write(f"\n" + "-"*80 + "\n")
        f.write("HUB PROTEINS (connecting multiple biomarkers)\n")
        f.write("-"*80 + "\n")
        
        if len(hub_df) > 0:
            f.write(f"Hub proteins found: {len(hub_df)}\n\n")
            for _, row in hub_df.head(20).iterrows():
                f.write(f"\n{row['protein']} ({row['protein_name']})\n")
                f.write(f"  Connects {row['num_biomarkers']} biomarkers: {row['biomarker_names']}\n")
        else:
            f.write("No hub proteins found.\n")
        
        f.write(f"\n" + "-"*80 + "\n")
        f.write("BIOMARKER PROTEIN CENTRALITY METRICS\n")
        f.write("-"*80 + "\n")
        
        f.write(centrality_df.to_string(index=False))
        
        f.write(f"\n" + "-"*80 + "\n")
        f.write("FUNCTIONAL MODULES\n")
        f.write("-"*80 + "\n")
        
        if communities['modules']:
            f.write(f"Modules detected: {len(communities['modules'])}\n\n")
            for module_id, module_info in communities['modules'].items():
                f.write(f"\n{module_id}:\n")
                f.write(f"  Total nodes: {module_info['num_nodes']}\n")
                f.write(f"  Proteins: {module_info['num_proteins']}\n")
                f.write(f"  Protein members: {', '.join([p.replace('Protein_', '') for p in module_info['proteins'][:5]])}")
                if len(module_info['proteins']) > 5:
                    f.write(f" ... +{len(module_info['proteins']) - 5}")
                f.write("\n")
        else:
            f.write("No functional modules detected.\n")
    
    logger.info(f"Saved report: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze protein biomarker subgraph from Knowledge Graph'
    )
    parser.add_argument('--kg-csv', type=str,
                       default='output/v2.11/GSE54514_enriched_ontology_degfilter_v2.11_csv/edges.csv',
                       help='Path to KG CSV file (pre-converted)')
    parser.add_argument('--biomarkers-csv', type=str, default=None,
                       help='Path to consensus biomarkers CSV (auto-detected if not provided)')
    parser.add_argument('--output-dir', type=str,
                       default='results/interpretability/complex_svm_mapped/kg_subgraph_analysis_consensus',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    
    # Load KG
    logger.info("Loading knowledge graph...")
    kg_data = load_knowledge_graph(csv_path=args.kg_csv)
    
    if not kg_data['edges']:
        logger.error("No edges loaded from KG")
        return
    
    # Build graph
    G = build_networkx_graph(kg_data)
    
    # Load protein biomarkers
    biomarkers = extract_protein_biomarkers(args.biomarkers_csv)
    
    if not biomarkers:
        logger.error("No biomarkers found")
        return
    
    # Extract subgraph
    logger.info("Extracting 2-hop neighborhood around biomarkers...")
    subgraph = extract_subgraph_around_biomarkers(G, biomarkers, hop_distance=1)
    
    # Analyses
    logger.info("\n" + "="*80)
    logger.info("CONSENSUS PROTEIN ANALYSIS")
    logger.info("="*80 + "\n")
    
    # Protein-protein interactions
    logger.info("Analyzing protein-protein interactions...")
    ppi_df = analyze_protein_interactions(subgraph, biomarkers)
    ppi_csv = output_dir / 'protein_interactions.csv'
    ppi_df.to_csv(ppi_csv, index=False)
    logger.info(f"Saved: {ppi_csv}")
    
    # Hub proteins
    logger.info("Finding hub proteins...")
    hub_df = find_common_protein_hubs(subgraph, biomarkers)
    hub_csv = output_dir / 'hub_proteins.csv'
    hub_df.to_csv(hub_csv, index=False)
    logger.info(f"Saved: {hub_csv}")
    
    # Centrality metrics
    logger.info("Calculating centrality metrics...")
    centrality_df = calculate_centrality_metrics(subgraph, biomarkers)
    centrality_csv = output_dir / 'biomarker_centrality_metrics.csv'
    centrality_df.to_csv(centrality_csv, index=False)
    logger.info(f"Saved: {centrality_csv}")
    
    # Communities
    logger.info("Detecting functional modules...")
    communities = detect_communities(subgraph)
    communities_json = output_dir / 'protein_communities.json'
    with open(communities_json, 'w') as f:
        # Convert to JSON-serializable format
        json_data = {
            'modules': {k: {'num_nodes': v['num_nodes'], 'num_proteins': v['num_proteins'], 
                           'proteins': v['proteins']} 
                       for k, v in communities['modules'].items()},
            'protein_to_module': communities['protein_to_module']
        }
        json.dump(json_data, f, indent=2)
    logger.info(f"Saved: {communities_json}")
    
    # Extended analysis: Pathways, GO terms, Reactions
    logger.info("\n" + "="*80)
    logger.info("EXTENDED ANALYSIS: Entity Types Connected to Biomarkers")
    logger.info("="*80 + "\n")
    
    logger.info("Analyzing connected entity types...")
    pathway_df, go_df, reaction_df = analyze_connected_entity_types(subgraph, biomarkers)
    
    pathway_csv = output_dir / 'biomarker_pathways.csv'
    pathway_df.to_csv(pathway_csv, index=False)
    logger.info(f"Saved: {pathway_csv}")
    
    go_csv = output_dir / 'biomarker_go_terms.csv'
    go_df.to_csv(go_csv, index=False)
    logger.info(f"Saved: {go_csv}")
    
    reaction_csv = output_dir / 'biomarker_reactions.csv'
    reaction_df.to_csv(reaction_csv, index=False)
    logger.info(f"Saved: {reaction_csv}")
    
    # Extended visualizations
    logger.info("\nCreating extended visualizations...")
    
    pathway_viz = output_dir / 'protein_pathway_bipartite_network.png'
    visualize_protein_pathway_network(subgraph, biomarkers, pathway_df, str(pathway_viz))
    
    go_viz = output_dir / 'go_term_enrichment.png'
    visualize_go_term_enrichment(go_df, str(go_viz))
    
    multilayer_viz = output_dir / 'multilayer_network.png'
    visualize_multilayer_network(subgraph, biomarkers, pathway_df, go_df, str(multilayer_viz))
    
    # Original visualizations
    logger.info("Creating protein-centric visualizations...")
    viz_path = output_dir / 'protein_biomarker_network.png'
    visualize_biomarker_subgraph(subgraph, biomarkers, str(viz_path))
    
    # Report
    logger.info("Creating analysis report...")
    report_path = output_dir / 'protein_analysis_report.txt'
    create_analysis_report(biomarkers, ppi_df, hub_df, centrality_df, communities, str(report_path))
    
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*80)
    logger.info(f"\nOutput files in: {output_dir}")
    logger.info("\nGenerated files:")
    logger.info("\n  PROTEIN-CENTRIC ANALYSIS:")
    logger.info("    - protein_interactions.csv")
    logger.info("    - hub_proteins.csv")
    logger.info("    - biomarker_centrality_metrics.csv")
    logger.info("    - protein_communities.json")
    logger.info("    - protein_biomarker_network.png")
    logger.info("\n  EXTENDED ANALYSIS (Entity Types Connected to Biomarkers):")
    logger.info("    - biomarker_pathways.csv")
    logger.info("    - biomarker_go_terms.csv")
    logger.info("    - biomarker_reactions.csv")
    logger.info("\n  VISUALIZATIONS:")
    logger.info("    - protein_pathway_bipartite_network.png (POSTER QUALITY)")
    logger.info("    - go_term_enrichment.png (POSTER QUALITY)")
    logger.info("    - multilayer_network.png (POSTER QUALITY)")
    logger.info("\n  REPORT:")
    logger.info("    - protein_analysis_report.txt")


if __name__ == '__main__':
    main()
