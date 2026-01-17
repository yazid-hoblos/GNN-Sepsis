"""
Graph Structure Importance Analysis
------------------------------------

Analyze which edges, nodes, and subgraphs contribute most to RGCN's predictive power.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import networkx as nx
import argparse


class GraphStructureAnalyzer:
    """
    Analyze importance of graph structure elements:
    - Node importance: Which entities are most central?
    - Edge type importance: Which relationship types matter?
    - Subgraph patterns: What local structures are predictive?
    """
    
    def __init__(self, 
                 entity_mapping_path: str,
                 relation_mapping_path: str = None,
                 triples_path: str = None):
        """
        Initialize graph structure analyzer.
        
        Parameters
        ----------
        entity_mapping_path : str
            Path to entity mapping CSV
        relation_mapping_path : str
            Path to relation mapping CSV
        triples_path : str
            Path to knowledge graph triples file
        """
        self.entities = pd.read_csv(entity_mapping_path) if entity_mapping_path else pd.DataFrame()
        self.relations = pd.read_csv(relation_mapping_path) if relation_mapping_path else pd.DataFrame(columns=['relation', 'label'])
        
        # Load triples (edges)
        if triples_path:
            print(f"Loading knowledge graph from {triples_path}")
            self.triples = self._load_triples(triples_path)
            print(f"  Loaded {len(self.triples)} triples")
        else:
            raise ValueError("triples_path is required")
        
        # Build NetworkX graph for analysis
        self.graph = self._build_networkx_graph()
    
    def _load_triples(self, triples_path: str) -> pd.DataFrame:
        """Load KG triples from file."""
        # Try different formats
        path = Path(triples_path)
        
        if path.suffix == '.tsv':
            df = pd.read_csv(path, sep='\t', names=['head', 'relation', 'tail'])
        elif path.suffix == '.csv':
            df = pd.read_csv(path)
            # Rename columns if using edge_attributes format
            if 'subject' in df.columns:
                df = df.rename(columns={'subject': 'head', 'predicate': 'relation', 'object': 'tail'})
            elif 'head' not in df.columns:
                # Generic CSV - assume first 3 columns are head, relation, tail
                df.columns = ['head', 'relation', 'tail'] + list(df.columns[3:])
        else:
            # Try pickle
            import pickle
            with open(path, 'rb') as f:
                triples = pickle.load(f)
            df = pd.DataFrame(triples, columns=['head', 'relation', 'tail'])
        
        return df
    
    def _build_networkx_graph(self) -> nx.MultiDiGraph:
        """Build NetworkX graph from triples."""
        G = nx.MultiDiGraph()
        
        for _, row in self.triples.iterrows():
            G.add_edge(row['head'], row['tail'], relation=row['relation'])
        
        print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    
    def analyze_node_centrality(self, top_k: int = 50) -> pd.DataFrame:
        """
        Compute centrality metrics for all nodes.
        
        Parameters
        ----------
        top_k : int
            Number of top nodes to return
            
        Returns
        -------
        centrality_df : pd.DataFrame
            Node centrality metrics
        """
        print("\nComputing node centrality...")
        
        # Degree centrality
        degree_cent = nx.degree_centrality(self.graph)
        
        # Betweenness centrality (on simplified graph for speed)
        simple_graph = nx.Graph(self.graph)
        if simple_graph.number_of_nodes() < 5000:
            betweenness_cent = nx.betweenness_centrality(simple_graph, k=min(100, simple_graph.number_of_nodes()))
        else:
            betweenness_cent = {n: 0 for n in simple_graph.nodes()}
        
        # PageRank
        pagerank = nx.pagerank(self.graph, alpha=0.85, max_iter=100)
        
        # Combine
        nodes_data = []
        for node in self.graph.nodes():
            # Get entity label if available
            label = f"Entity_{node}"
            if 'entity_id' in self.entities.columns:
                entity_row = self.entities[self.entities['entity_id'] == node]
                if len(entity_row) > 0 and 'label' in entity_row.columns:
                    label = entity_row['label'].iloc[0]
            elif 'node_id' in self.entities.columns:
                entity_row = self.entities[self.entities['node_id'] == node]
                if len(entity_row) > 0:
                    label = entity_row.get('label', pd.Series([node])).iloc[0]
            
            nodes_data.append({
                'entity_id': node,
                'label': label,
                'degree': self.graph.degree(node),
                'degree_centrality': degree_cent[node],
                'betweenness_centrality': betweenness_cent.get(node, 0),
                'pagerank': pagerank[node]
            })
        
        df = pd.DataFrame(nodes_data)
        df = df.sort_values('pagerank', ascending=False)
        
        return df.head(top_k)
    
    def analyze_edge_type_importance(self) -> pd.DataFrame:
        """
        Analyze importance of different edge types.
        
        Returns
        -------
        edge_stats : pd.DataFrame
            Statistics per edge type
        """
        print("\nAnalyzing edge types...")
        
        edge_stats = self.triples.groupby('relation').agg({
            'head': 'count',
        }).rename(columns={'head': 'count'})
        
        edge_stats = edge_stats.reset_index()
        
        # Relations are already labels in the triples, so just sort
        edge_stats = edge_stats.sort_values('count', ascending=False)
        edge_stats = edge_stats.rename(columns={'relation': 'label'})
        
        return edge_stats
    
    def find_patient_connected_entities(self, 
                                       patient_prefix: str = "Sample",
                                       top_k: int = 100) -> pd.DataFrame:
        """
        Find entities most connected to patient nodes.
        
        These are likely genes/proteins directly measured or
        strongly associated with patient outcomes.
        
        Parameters
        ----------
        patient_prefix : str
            Prefix for patient nodes
        top_k : int
            Number of top entities to return
            
        Returns
        -------
        connected_entities : pd.DataFrame
            Entities sorted by connection count to patients
        """
        print(f"\nFinding entities connected to patients...")
        
        # Find patient nodes
        patient_nodes = [n for n in self.graph.nodes() 
                        if str(n).startswith(patient_prefix)]
        print(f"  Found {len(patient_nodes)} patient nodes")
        
        # Count connections
        entity_counts = {}
        for patient in patient_nodes:
            # Get neighbors
            neighbors = list(self.graph.neighbors(patient))
            for neighbor in neighbors:
                if not str(neighbor).startswith(patient_prefix):
                    entity_counts[neighbor] = entity_counts.get(neighbor, 0) + 1
        
        # Convert to DataFrame
        results = []
        for entity_id, count in entity_counts.items():
            label = f"Entity_{entity_id}"
            if 'entity_id' in self.entities.columns:
                entity_row = self.entities[self.entities['entity_id'] == entity_id]
                if len(entity_row) > 0 and 'label' in entity_row.columns:
                    label = entity_row['label'].iloc[0]
            elif 'node_id' in self.entities.columns:
                entity_row = self.entities[self.entities['node_id'] == entity_id]
                if len(entity_row) > 0:
                    label = entity_row.get('label', pd.Series([entity_id])).iloc[0]
            
            results.append({
                'entity_id': entity_id,
                'label': label,
                'patient_connections': count
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values('patient_connections', ascending=False)
        
        return df.head(top_k)
    
    def plot_edge_type_distribution(self, 
                                    output_path: str = None,
                                    show: bool = True) -> plt.Figure:
        """Plot distribution of edge types."""
        edge_stats = self.analyze_edge_type_importance()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        top_edges = edge_stats.head(20)
        
        ax.barh(range(len(top_edges)), top_edges['count'], alpha=0.7)
        ax.set_yticks(range(len(top_edges)))
        ax.set_yticklabels(top_edges['label'].fillna('Unknown'), fontsize=10)
        ax.set_xlabel('Edge Count', fontsize=12, fontweight='bold')
        ax.set_title('Top 20 Edge Types in Knowledge Graph', 
                    fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved: {output_path}")
        
        if show:
            plt.show()
        
        return fig


def main():
    """Run graph structure analysis."""
    parser = argparse.ArgumentParser(description="Graph structure analysis for RGCN")
    parser.add_argument('--version', default='v2.11', help='Model version tag (e.g., v2.11)')
    parser.add_argument('--use-owl', action='store_true', help='Convert OWL to CSV and use it instead of edge_attributes.csv')
    parser.add_argument('--owl-path', default='output/GSE54514_enriched_ontology_degfilterv2.11.owl',
                        help='Path to OWL file (v2.11) for conversion')
    args = parser.parse_args()

    print("="*80)
    print("GRAPH STRUCTURE ANALYSIS")
    print(f"Version={args.version}")
    print("="*80)
    
    model_dir = Path(f"models/executions/GSE54514_enriched_ontology_degfilter{args.version}")
    output_dir = Path(f"results/graph_structure/{args.version}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.use_owl:
        # Convert OWL to CSV
        try:
            from kg_conversion.convert_owl_to_csv import parse_owl_to_csv
        except Exception:
            print("✗ Could not import kg_conversion.convert_owl_to_csv; ensure it exists")
            return

        owl_export_dir = output_dir / "owl_export"
        owl_export_dir.mkdir(parents=True, exist_ok=True)
        df_nodes, df_edges, _ = parse_owl_to_csv(args.owl_path, output_dir=str(owl_export_dir))
        if df_edges is None:
            print("✗ OWL conversion failed")
            return
        triples_path = owl_export_dir / "edges.csv"
        entity_mapping_path = owl_export_dir / "nodes.csv"
        relation_mapping_path = None
    else:
        triples_path = model_dir / "edge_attributes.csv"
        if not triples_path.exists():
            print("✗ Could not find edge_attributes.csv. Looked for:")
            print(f"  - {triples_path}")
            return
        entity_mapping_path = model_dir / "outputmodel_RGCN_entity_mapping.csv"
        relation_mapping_path = model_dir / "outputmodel_RGCN_relation_mapping.csv"

    analyzer = GraphStructureAnalyzer(
        entity_mapping_path=str(entity_mapping_path),
        relation_mapping_path=str(relation_mapping_path) if relation_mapping_path else None,
        triples_path=str(triples_path)
    )
    
    # Node centrality
    centrality = analyzer.analyze_node_centrality(top_k=50)
    centrality.to_csv(output_dir / "node_centrality.csv", index=False)
    print(f"\n✓ Node centrality saved: {output_dir / 'node_centrality.csv'}")
    print("\nTop 10 central nodes:")
    print(centrality.head(10)[['label', 'degree', 'pagerank']].to_string(index=False))
    
    # Edge types
    edge_stats = analyzer.analyze_edge_type_importance()
    edge_stats.to_csv(output_dir / "edge_type_stats.csv", index=False)
    print(f"\n✓ Edge statistics saved: {output_dir / 'edge_type_stats.csv'}")
    
    analyzer.plot_edge_type_distribution(
        output_path=str(output_dir / "edge_type_distribution.png"),
        show=False
    )
    
    # Patient-connected entities
    patient_entities = analyzer.find_patient_connected_entities(top_k=100)
    patient_entities.to_csv(output_dir / "patient_connected_entities.csv", index=False)
    print(f"\n✓ Patient connections saved: {output_dir / 'patient_connected_entities.csv'}")
    print("\nTop 10 patient-connected entities:")
    print(patient_entities.head(10).to_string(index=False))
    
    print("\n" + "="*80)
    print("Analysis complete! Results in:", output_dir)
    print("="*80)


if __name__ == '__main__':
    main()
