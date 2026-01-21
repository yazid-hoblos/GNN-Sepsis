#!/usr/bin/env python3
"""
Extract and analyze biomarker subgraphs from the Knowledge Graph.

This script:
1. Loads the KG (edges and nodes)
2. Extracts local neighborhoods around top biomarkers
3. Finds paths between biomarkers and sepsis/healthy samples
4. Identifies intermediate entities (hidden connections)
5. Detects functional modules/communities
6. Visualizes biomarker subgraph
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from collections import defaultdict, Counter
import json
import subprocess
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns.set_style('white')


def convert_owl_to_csv(owl_file: str, output_dir: str):
    """Convert OWL file to CSV using the conversion script."""
    logger.info(f"Converting OWL file: {owl_file}")
    
    try:
        cmd = [sys.executable, 'src/utils/convert_owl_to_csv.py', '--owl-file', owl_file, '--output-dir', output_dir]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info("OWL conversion successful")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"OWL conversion failed: {e.stderr}")
        return False


def load_knowledge_graph(kg_source: str):
    """Load KG from OWL file or CSV directory."""
    kg_path = Path(kg_source)
    
    # Check if it's an OWL file
    if kg_path.suffix == '.owl':
        logger.info(f"Detected OWL file: {kg_path}")
        
        # Create output directory for CSV files
        csv_output_dir = kg_path.parent / f"{kg_path.stem}_csv"
        csv_output_dir.mkdir(exist_ok=True)
        
        # Check if already converted
        edges_file = csv_output_dir / 'edges.csv'
        nodes_file = csv_output_dir / 'nodes.csv'
        
        if not edges_file.exists() or not nodes_file.exists():
            logger.info("Converting OWL to CSV...")
            if not convert_owl_to_csv(str(kg_path), str(csv_output_dir)):
                raise RuntimeError("Failed to convert OWL to CSV")
        else:
            logger.info("Using existing CSV files")
        
        # Load CSV files
        logger.info(f"Loading edges from: {edges_file}")
        edges_df = pd.read_csv(edges_file)
        
        logger.info(f"Loading nodes from: {nodes_file}")
        nodes_df = pd.read_csv(nodes_file)
        
    else:
        # Assume it's a directory with CSV files
        edges_file = kg_path / 'edge_attributes.csv'
        nodes_file = kg_path / 'node_features.csv'
        
        if not edges_file.exists():
            edges_file = kg_path / 'edges.csv'
        if not nodes_file.exists():
            nodes_file = kg_path / 'nodes.csv'
        
        logger.info(f"Loading edges from: {edges_file}")
        edges_df = pd.read_csv(edges_file)
        
        logger.info(f"Loading nodes from: {nodes_file}")
        nodes_df = pd.read_csv(nodes_file)
    
    logger.info(f"Loaded {len(edges_df)} edges")
    logger.info(f"Loaded {len(nodes_df)} node entries")
    
    return edges_df, nodes_df


def build_networkx_graph(edges_df: pd.DataFrame):
    """Build NetworkX graph from edge dataframe."""
    G = nx.DiGraph()
    
    # Handle different edge column naming conventions
    if 'subject' in edges_df.columns and 'predicate' in edges_df.columns and 'object' in edges_df.columns:
        # Standard format from edge_attributes.csv
        for _, row in edges_df.iterrows():
            G.add_edge(row['subject'], row['object'], relation=row['predicate'])
    elif 'source' in edges_df.columns and 'relation' in edges_df.columns and 'target' in edges_df.columns:
        # Format from OWL conversion (edges.csv)
        for _, row in edges_df.iterrows():
            G.add_edge(row['source'], row['target'], relation=row['relation'])
    else:
        raise ValueError(f"Unrecognized edge format. Columns: {edges_df.columns.tolist()}")
    
    logger.info(f"Built graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Convert to undirected for some analyses
    G_undirected = G.to_undirected()
    
    return G, G_undirected


def get_node_info(nodes_df: pd.DataFrame):
    """Extract node information from nodes dataframe (handles both formats)."""
    node_disease = {}
    all_nodes = set()
    
    # Check format
    if 'name_feature' in nodes_df.columns:
        # Long format (node_features.csv)
        disease_status = nodes_df[nodes_df['name_feature'] == 'hasDiseaseStatus'].copy()
        node_disease = dict(zip(disease_status['node_id'], disease_status['value_feature']))
        all_nodes = set(nodes_df['node_id'].unique())
    elif 'node_id' in nodes_df.columns and 'node_type' in nodes_df.columns:
        # Wide format (nodes.csv from OWL conversion)
        all_nodes = set(nodes_df['node_id'].unique())
        # Try to find disease status column
        disease_cols = [col for col in nodes_df.columns if 'disease' in col.lower() or 'status' in col.lower()]
        if disease_cols:
            node_disease = dict(zip(nodes_df['node_id'], nodes_df[disease_cols[0]]))
    else:
        # Fallback: just get node IDs
        if 'node_id' in nodes_df.columns:
            all_nodes = set(nodes_df['node_id'].unique())
        elif 'source' in nodes_df.columns:
            all_nodes = set(nodes_df['source'].unique())
    
    logger.info(f"Found {len(all_nodes)} unique nodes")
    logger.info(f"Disease status available for {len(node_disease)} nodes")
    
    return all_nodes, node_disease


def extract_biomarker_neighborhood(G, biomarker_entities, k_hops=2):
    """Extract k-hop neighborhood around biomarker entities."""
    subgraph_nodes = set()
    
    for entity in biomarker_entities:
        if entity in G:
            # Get k-hop neighbors
            for node in nx.single_source_shortest_path_length(G, entity, cutoff=k_hops).keys():
                subgraph_nodes.add(node)
    
    subgraph = G.subgraph(subgraph_nodes).copy()
    
    logger.info(f"Extracted subgraph: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")
    
    return subgraph


def analyze_biomarker_type_interactions(G, biomarker_entities):
    """Analyze interactions across biomarker entity types."""
    # Categorize biomarkers by type
    biomarker_types = defaultdict(list)
    for entity in biomarker_entities:
        entity_type = entity.split('_')[0]
        biomarker_types[entity_type].append(entity)
    
    logger.info(f"Biomarker types: {dict((k, len(v)) for k, v in biomarker_types.items())}")
    
    # Find edges between different biomarker types
    type_interactions = defaultdict(list)
    
    for entity1 in biomarker_entities:
        if entity1 not in G:
            continue
        type1 = entity1.split('_')[0]
        
        for neighbor in G.neighbors(entity1):
            if neighbor in biomarker_entities:
                type2 = neighbor.split('_')[0]
                if type1 != type2:  # Different types
                    edge_data = G[entity1][neighbor]
                    relation = edge_data.get('relation', 'unknown')
                    type_interactions[f"{type1}-{type2}"].append({
                        'entity1': entity1,
                        'entity2': neighbor,
                        'relation': relation
                    })
    
    logger.info(f"Inter-type biomarker connections: {dict((k, len(v)) for k, v in type_interactions.items())}")
    
    # Flatten for output
    interactions = []
    for conn_list in type_interactions.values():
        interactions.extend(conn_list)
    
    return interactions


def find_common_neighbors(G, biomarker_entities):
    """Find common neighbors of biomarkers (entities connected to multiple biomarkers)."""
    neighbor_counts = defaultdict(list)
    
    for biomarker in biomarker_entities:
        if biomarker not in G:
            continue
        for neighbor in G.neighbors(biomarker):
            if neighbor not in biomarker_entities:  # Not a biomarker itself
                neighbor_counts[neighbor].append(biomarker)
    
    # Find neighbors connected to multiple biomarkers
    common_neighbors = [(neighbor, biomarkers, len(biomarkers)) 
                       for neighbor, biomarkers in neighbor_counts.items() 
                       if len(biomarkers) > 1]
    common_neighbors.sort(key=lambda x: x[2], reverse=True)
    
    logger.info(f"Found {len(common_neighbors)} entities connected to multiple biomarkers")
    if common_neighbors:
        logger.info(f"Top connector: {common_neighbors[0][0]} connected to {common_neighbors[0][2]} biomarkers")
    
    return common_neighbors


def analyze_pathway_enrichment(G, biomarker_entities):
    """Analyze how biomarkers of different types are connected."""
    biomarker_proteins = [b for b in biomarker_entities if b.startswith('Protein_')]
    biomarker_pathways = [b for b in biomarker_entities if b.startswith('Pathway_')]
    biomarker_go = [b for b in biomarker_entities if b.startswith('GO_')]
    
    # Track connections between types
    protein_pathway_edges = []
    protein_go_edges = []
    
    # Proteins to pathways
    for protein in biomarker_proteins:
        if protein not in G:
            continue
        for neighbor in G.neighbors(protein):
            if neighbor in biomarker_pathways:
                edge_data = G[protein][neighbor]
                relation = edge_data.get('relation', 'unknown')
                protein_pathway_edges.append({
                    'protein': protein,
                    'pathway': neighbor,
                    'relation': relation
                })
    
    # Pathways to proteins (reverse)
    for pathway in biomarker_pathways:
        if pathway not in G:
            continue
        for neighbor in G.neighbors(pathway):
            if neighbor in biomarker_proteins:
                edge_data = G[pathway][neighbor]
                relation = edge_data.get('relation', 'unknown')
                if not any(pe['protein'] == neighbor and pe['pathway'] == pathway for pe in protein_pathway_edges):
                    protein_pathway_edges.append({
                        'protein': neighbor,
                        'pathway': pathway,
                        'relation': relation
                    })
    
    # Proteins to GO terms
    for protein in biomarker_proteins:
        if protein not in G:
            continue
        for neighbor in G.neighbors(protein):
            if neighbor in biomarker_go:
                edge_data = G[protein][neighbor]
                relation = edge_data.get('relation', 'unknown')
                protein_go_edges.append({
                    'protein': protein,
                    'go_term': neighbor,
                    'relation': relation
                })
    
    # GO to proteins (reverse)
    for go_term in biomarker_go:
        if go_term not in G:
            continue
        for neighbor in G.neighbors(go_term):
            if neighbor in biomarker_proteins:
                edge_data = G[go_term][neighbor]
                relation = edge_data.get('relation', 'unknown')
                if not any(pe['protein'] == neighbor and pe['go_term'] == go_term for pe in protein_go_edges):
                    protein_go_edges.append({
                        'protein': neighbor,
                        'go_term': go_term,
                        'relation': relation
                    })
    
    logger.info(f"Protein-Pathway edges: {len(protein_pathway_edges)}")
    logger.info(f"Protein-GO edges: {len(protein_go_edges)}")
    
    return protein_pathway_edges, protein_go_edges


def calculate_centrality_metrics(G, biomarker_entities):
    """Calculate centrality metrics for biomarkers in the subgraph."""
    # Only compute for biomarkers present in graph
    biomarkers_in_graph = [b for b in biomarker_entities if b in G]
    
    if not biomarkers_in_graph:
        return {}
    
    # Degree centrality
    degree_cent = nx.degree_centrality(G)
    
    # Betweenness centrality (on subset for performance)
    try:
        betweenness_cent = nx.betweenness_centrality(G, k=min(100, G.number_of_nodes()))
    except:
        betweenness_cent = {}
    
    # Closeness centrality (on subset)
    try:
        closeness_cent = nx.closeness_centrality(G)
    except:
        closeness_cent = {}
    
    centrality_data = []
    for biomarker in biomarkers_in_graph:
        centrality_data.append({
            'entity': biomarker,
            'degree_centrality': degree_cent.get(biomarker, 0),
            'betweenness_centrality': betweenness_cent.get(biomarker, 0),
            'closeness_centrality': closeness_cent.get(biomarker, 0)
        })
    
    logger.info(f"Calculated centrality metrics for {len(centrality_data)} biomarkers")
    return centrality_data


def detect_communities(G):
    """Detect communities/modules in biomarker subgraph."""
    # Convert to undirected for community detection
    G_undirected = G.to_undirected()
    
    # Use Louvain method
    try:
        from networkx.algorithms import community
        communities = community.greedy_modularity_communities(G_undirected)
        
        logger.info(f"Detected {len(communities)} communities")
        
        # Convert to dict
        node_community = {}
        for i, comm in enumerate(communities):
            for node in comm:
                node_community[node] = i
        
        return communities, node_community
    except Exception as e:
        logger.warning(f"Community detection failed: {e}")
        return [], {}


def visualize_biomarker_subgraph(G, biomarker_entities, node_disease, output_dir: Path):
    """Visualize the biomarker subgraph."""
    
    # Limit to manageable size
    if G.number_of_nodes() > 200:
        logger.info("Graph too large, sampling nodes...")
        # Keep biomarkers and their 1-hop neighbors
        nodes_to_keep = set(biomarker_entities)
        for biomarker in biomarker_entities:
            if biomarker in G:
                nodes_to_keep.update(G.neighbors(biomarker))
        G = G.subgraph(list(nodes_to_keep)[:200]).copy()
    
    fig, ax = plt.subplots(figsize=(20, 16))
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Color nodes by type
    node_colors = []
    node_sizes = []
    
    for node in G.nodes():
        if node.startswith('Sample_'):
            if node in node_disease:
                if node_disease[node] == 'healthy':
                    node_colors.append('#2ecc71')  # Green for healthy
                else:
                    node_colors.append('#e74c3c')  # Red for sepsis
            else:
                node_colors.append('#95a5a6')
            node_sizes.append(300)
        elif node.startswith('Protein_'):
            if node in biomarker_entities:
                node_colors.append('#f39c12')  # Orange for biomarker proteins
                node_sizes.append(600)
            else:
                node_colors.append('#3498db')  # Blue for other proteins
                node_sizes.append(200)
        elif node.startswith('Pathway_'):
            if node in biomarker_entities:
                node_colors.append('#9b59b6')  # Purple for biomarker pathways
                node_sizes.append(500)
            else:
                node_colors.append('#c39bd3')
                node_sizes.append(200)
        elif node.startswith('GO_'):
            node_colors.append('#1abc9c')  # Teal for GO terms
            node_sizes.append(150)
        else:
            node_colors.append('#bdc3c7')  # Gray for others
            node_sizes.append(100)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.1, width=0.5, edge_color='gray',
                          arrows=True, arrowsize=5, ax=ax)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                          alpha=0.8, ax=ax)
    
    # Draw labels for biomarkers only
    biomarker_labels = {n: n.replace('Protein_', '').replace('Pathway_', 'P:')[:15] 
                       for n in G.nodes() if n in biomarker_entities}
    nx.draw_networkx_labels(G, pos, labels=biomarker_labels, font_size=7,
                           font_weight='bold', ax=ax)
    
    ax.set_title('Biomarker Subgraph from Knowledge Graph\n(Top ComplEx+SVM Features)',
                fontsize=18, fontweight='bold', pad=20)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#f39c12', label='Biomarker Proteins'),
        Patch(facecolor='#9b59b6', label='Biomarker Pathways'),
        Patch(facecolor='#3498db', label='Other Proteins'),
        Patch(facecolor='#e74c3c', label='Sepsis Samples'),
        Patch(facecolor='#2ecc71', label='Healthy Samples'),
        Patch(facecolor='#1abc9c', label='GO Terms'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
    ax.axis('off')
    
    plt.tight_layout()
    output_file = output_dir / 'biomarker_subgraph_from_kg.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"Saved: {output_file}")
    plt.close()


def create_biomarker_analysis_report(interactions, pathway_enrichment, go_enrichment, common_neighbors, centrality_data, communities, output_dir: Path):
    """Create comprehensive biomarker analysis report."""
    
    report_lines = [
        "=" * 80,
        "BIOMARKER KNOWLEDGE GRAPH ANALYSIS",
        "=" * 80,
        "",
        "=" * 80,
        "BIOMARKER TYPE INTERACTIONS",
        "=" * 80,
        "",
        f"Found {len(interactions)} connections between different biomarker types:\n",
    ]
    
    if interactions:
        # Group by relation type
        relations_count = defaultdict(int)
        for inter in interactions:
            relations_count[inter['relation']] += 1
        
        for relation, count in sorted(relations_count.items(), key=lambda x: x[1], reverse=True)[:10]:
            report_lines.append(f"  {relation}: {count}")
    else:
        report_lines.append("  No inter-type biomarker interactions found.")
    
    report_lines.extend([
        "",
        "=" * 80,
        "COMMON NEIGHBORS (HUB CONNECTORS)",
        "=" * 80,
        "",
        "Entities connected to multiple biomarkers:",
        ""
    ])
    
    if common_neighbors:
        for neighbor, biomarkers, count in common_neighbors[:15]:
            neighbor_name = neighbor.replace('Protein_', '').replace('Pathway_', '').replace('GO_', '')[:40]
            biomarker_names = [b.replace('Protein_', '').replace('Pathway_', '')[:15] for b in biomarkers[:3]]
            report_lines.append(f"  {neighbor_name:40}  ({count} biomarkers)")
            report_lines.append(f"    Connected to: {', '.join(biomarker_names)}")
    else:
        report_lines.append("  No common neighbors found.")
    
    report_lines.extend([
        "",
        "=" * 80,
        "PROTEIN-PATHWAY CONNECTIONS",
        "=" * 80,
        ""
    ])
    
    if pathway_enrichment:
        relation_types = defaultdict(int)
        for edge in pathway_enrichment:
            relation_types[edge['relation']] += 1
        
        report_lines.append(f"Found {len(pathway_enrichment)} protein-pathway connections")
        for relation, count in sorted(relation_types.items(), key=lambda x: x[1], reverse=True)[:5]:
            report_lines.append(f"  {relation}: {count}")
    else:
        report_lines.append("  No protein-pathway connections found.")
    
    report_lines.extend([
        "",
        "=" * 80,
        "PROTEIN-GO CONNECTIONS",
        "=" * 80,
        ""
    ])
    
    if go_enrichment:
        relation_types = defaultdict(int)
        for edge in go_enrichment:
            relation_types[edge['relation']] += 1
        
        report_lines.append(f"Found {len(go_enrichment)} protein-GO term connections")
        for relation, count in sorted(relation_types.items(), key=lambda x: x[1], reverse=True)[:5]:
            report_lines.append(f"  {relation}: {count}")
    else:
        report_lines.append("  No protein-GO connections found.")
    
    report_lines.extend([
        "",
        "=" * 80,
        "CENTRALITY ANALYSIS",
        "=" * 80,
        "",
        "Top biomarkers by network centrality (hub nodes):",
        ""
    ])
    
    if centrality_data:
        # Sort by degree centrality
        sorted_by_degree = sorted(centrality_data, key=lambda x: x['degree_centrality'], reverse=True)
        report_lines.append("Top by Degree Centrality (most connections):")
        for item in sorted_by_degree[:10]:
            entity_name = item['entity'].replace('Protein_', '').replace('Pathway_', 'P:')[:40]
            report_lines.append(f"  {entity_name:40}  {item['degree_centrality']:.4f}")
        
        report_lines.append("")
        
        # Sort by betweenness centrality
        sorted_by_betweenness = sorted(centrality_data, key=lambda x: x['betweenness_centrality'], reverse=True)
        report_lines.append("Top by Betweenness Centrality (bridge biomarkers):")
        for item in sorted_by_betweenness[:10]:
            entity_name = item['entity'].replace('Protein_', '').replace('Pathway_', 'P:')[:40]
            report_lines.append(f"  {entity_name:40}  {item['betweenness_centrality']:.4f}")
    
    report_lines.extend([
        "",
        "=" * 80,
        "COMMUNITY STRUCTURE",
        "=" * 80,
        "",
        f"Detected {len(communities)} functional modules/communities",
        ""
    ])
    
    # Save report
    report_file = output_dir / 'biomarker_kg_analysis_report.txt'
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Saved biomarker analysis report: {report_file}")
    
    # Print to console
    print('\n'.join(report_lines))


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract biomarker subgraphs from KG')
    parser.add_argument('--kg-source', type=str,
                       default='output/v2.11/GSE54514_enriched_ontology_degfilter_v2.11.owl',
                       help='Knowledge graph source (OWL file or CSV directory)')
    parser.add_argument('--biomarker-csv', type=str,
                       default='results/interpretability/complex_svm_mapped/complex_svm_biomarkers.csv',
                       help='Biomarker CSV file')
    parser.add_argument('--output-dir', type=str,
                       default='results/interpretability/complex_svm_mapped/kg_subgraph_analysis',
                       help='Output directory')
    parser.add_argument('--k-hops', type=int, default=2,
                       help='Number of hops for neighborhood extraction')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load biomarkers
    logger.info("Loading biomarkers...")
    biomarker_df = pd.read_csv(args.biomarker_csv)
    top_biomarkers = biomarker_df.nlargest(20, 'entity_score')
    biomarker_entities = top_biomarkers['entity'].unique().tolist()
    
    logger.info(f"Analyzing {len(biomarker_entities)} top biomarkers")
    
    # Load KG
    logger.info("Loading knowledge graph...")
    edges_df, nodes_df = load_knowledge_graph(args.kg_source)
    
    # Build graph
    logger.info("Building NetworkX graph...")
    G, G_undirected = build_networkx_graph(edges_df)
    
    # Get node info
    logger.info("Extracting node information...")
    all_nodes, node_disease = get_node_info(nodes_df)
    
    # Extract biomarker neighborhood
    logger.info(f"Extracting {args.k_hops}-hop neighborhood around biomarkers...")
    biomarker_subgraph = extract_biomarker_neighborhood(G_undirected, biomarker_entities, args.k_hops)
    
    # Save subgraph info
    subgraph_stats = {
        'num_nodes': biomarker_subgraph.number_of_nodes(),
        'num_edges': biomarker_subgraph.number_of_edges(),
        'biomarker_entities': biomarker_entities,
        'k_hops': args.k_hops
    }
    
    with open(output_dir / 'subgraph_stats.json', 'w') as f:
        json.dump(subgraph_stats, f, indent=2)
    
    # Analyze protein-protein interactions
    logger.info("Analyzing biomarker type interactions...")
    interactions = analyze_biomarker_type_interactions(G_undirected, biomarker_entities)
    
    # Save interactions
    if interactions:
        interactions_df = pd.DataFrame(interactions)
        interactions_df.to_csv(output_dir / 'biomarker_type_interactions.csv', index=False)
        logger.info(f"Saved biomarker type interactions: {output_dir / 'biomarker_type_interactions.csv'}")
    
    # Find common neighbors
    logger.info("Finding common neighbors of biomarkers...")
    common_neighbors = find_common_neighbors(G_undirected, biomarker_entities)
    
    # Save common neighbors
    if common_neighbors:
        common_neighbor_df = pd.DataFrame([
            {'entity': neighbor, 'num_biomarkers': count, 
             'biomarkers': ','.join([b.replace('Protein_', '').replace('Pathway_', '') for b in biomarkers[:5]])}
            for neighbor, biomarkers, count in common_neighbors[:100]
        ])
        common_neighbor_df.to_csv(output_dir / 'common_neighbors.csv', index=False)
        logger.info(f"Saved common neighbors: {output_dir / 'common_neighbors.csv'}")
    
    # Analyze pathway enrichment
    logger.info("Analyzing biomarker entity networks...")
    pathway_enrichment, go_enrichment = analyze_pathway_enrichment(G_undirected, biomarker_entities)
    
    # Calculate centrality metrics
    logger.info("Calculating centrality metrics...")
    centrality_data = calculate_centrality_metrics(biomarker_subgraph, biomarker_entities)
    
    # Save centrality data
    if centrality_data:
        centrality_df = pd.DataFrame(centrality_data)
        centrality_df.to_csv(output_dir / 'centrality_metrics.csv', index=False)
        logger.info(f"Saved centrality metrics: {output_dir / 'centrality_metrics.csv'}")
    
    # Detect communities
    logger.info("Detecting communities in biomarker subgraph...")
    communities, node_community = detect_communities(biomarker_subgraph)
    
    if communities:
        community_info = []
        for i, comm in enumerate(communities):
            community_info.append({
                'community_id': i,
                'size': len(comm),
                'members': list(comm)[:20]  # Limit for readability
            })
        
        with open(output_dir / 'communities.json', 'w') as f:
            json.dump(community_info, f, indent=2)
        logger.info(f"Saved community info: {output_dir / 'communities.json'}")
    
    # Visualize
    logger.info("Creating biomarker subgraph visualization...")
    visualize_biomarker_subgraph(biomarker_subgraph, biomarker_entities, node_disease, output_dir)
    
    # Create comprehensive analysis report
    logger.info("Creating biomarker analysis report...")
    create_biomarker_analysis_report(interactions, pathway_enrichment, go_enrichment, common_neighbors, centrality_data, communities, output_dir)
    
    logger.info(f"\nAll results saved to: {output_dir}")
    logger.info("\nKey findings:")
    logger.info(f"- Biomarker subgraph: {biomarker_subgraph.number_of_nodes()} nodes, {biomarker_subgraph.number_of_edges()} edges")
    logger.info(f"- Biomarker type interactions: {len(interactions)}")
    logger.info(f"- Common neighbor entities: {len(common_neighbors)}")
    logger.info(f"- Protein-Pathway connections: {len(pathway_enrichment)}")
    logger.info(f"- Protein-GO connections: {len(go_enrichment)}")
    logger.info(f"- Communities detected: {len(communities)}")


if __name__ == '__main__':
    main()
