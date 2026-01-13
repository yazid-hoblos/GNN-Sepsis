"""
Graph Structure Importance Analysis
------------------------------------

Analyze which edges, nodes, and subgraphs contribute most to 
RGCN's predictive power.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import networkx as nx


class GraphStructureAnalyzer:
    """
    Analyze importance of graph structure elements:
    - Node importance: Which entities are most central?
    - Edge type importance: Which relationship types matter?
    - Subgraph patterns: What local structures are predictive?
    """
    
    def __init__(self, 
                 entity_mapping_path: str,
                 relation_mapping_path: str,
                 triples_path: str):
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
        self.entities = pd.read_csv(entity_mapping_path)
        self.relations = pd.read_csv(relation_mapping_path)
        
        # Load triples (edges)
        print(f"Loading knowledge graph from {triples_path}")
        self.triples = self._load_triples(triples_path)
        print(f"  Loaded {len(self.triples)} triples")
        
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
            entity_row = self.entities[self.entities['entity_id'] == node]
            label = entity_row['label'].iloc[0] if len(entity_row) > 0 else f"Entity_{node}"
            
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
            entity_row = self.entities[self.entities['entity_id'] == entity_id]
            label = entity_row['label'].iloc[0] if len(entity_row) > 0 else f"Entity_{entity_id}"
            
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
    print("="*80)
    print("GRAPH STRUCTURE ANALYSIS")
    print("="*80)
    
    # Configuration
    model_dir = Path("models/executions/GSE54514_enriched_ontology_degfilterv2.11")
    
    # The edge attributes file is the main KG triples source
    triples_path = model_dir / "edge_attributes.csv"
    
    if not triples_path.exists():
        print("✗ Could not find edge_attributes.csv. Looked for:")
        print(f"  - {triples_path}")
        return
    
    analyzer = GraphStructureAnalyzer(
        entity_mapping_path=str(model_dir / "outputmodel_RGCN_entity_mapping.csv"),
        relation_mapping_path=str(model_dir / "outputmodel_RGCN_relation_mapping.csv"),
        triples_path=str(triples_path)
    )
    
    output_dir = Path("results/graph_structure")
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
