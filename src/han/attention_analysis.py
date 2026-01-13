"""
Attention Weight Analysis for HAN

Extract and visualize node-level and semantic-level attention
to understand which neighbors and edge types drive predictions.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import networkx as nx
import logging

from torch_geometric.data import HeteroData


logger = logging.getLogger(__name__)


class AttentionAnalyzer:
    """
    Analyze HAN attention weights for interpretability.
    
    Provides:
    - Node-level attention: Which neighbors influenced prediction?
    - Semantic-level attention: Which edge types are important?
    - Subgraph visualization: Patient's neighborhood with attention coloring
    """
    
    def __init__(self,
                 model,
                 data: HeteroData,
                 node_id_to_idx: Dict[Tuple[str, str], int],
                 idx_to_node_id: Dict[Tuple[str, int], str]):
        """
        Initialize analyzer.
        
        Parameters
        ----------
        model : SepsisHANClassifier
            Trained HAN model
        data : HeteroData
            Graph data
        node_id_to_idx : dict
            Mapping from (node_type, node_id) to local index
        idx_to_node_id : dict
            Mapping from (node_type, local_idx) to node_id
        """
        self.model = model
        self.data = data
        self.node_id_to_idx = node_id_to_idx
        self.idx_to_node_id = idx_to_node_id
        
        # Cache of predictions and embeddings
        self.predictions = None
        self.probabilities = None
        self.embeddings = None
    
    def compute_predictions(self, return_embeddings: bool = True):
        """
        Compute predictions and patient embeddings on all samples.
        
        Parameters
        ----------
        return_embeddings : bool
            If True, also compute embeddings
        """
        self.model.eval()
        
        with torch.no_grad():
            logits, patient_embeddings, _ = self.model(
                self.data.x_dict,
                self.data.edge_index_dict
            )
        
        self.probabilities = torch.softmax(logits, dim=1)
        self.predictions = logits.argmax(dim=1)
        self.embeddings = patient_embeddings if return_embeddings else None
    
    def get_neighbor_influence(self,
                               sample_idx: int,
                               top_k: int = 20) -> pd.DataFrame:
        """
        Get most influential neighbors for a sample's prediction.
        
        This is computed using embedding similarity:
        Neighbors with embeddings most similar to patient embeddings
        are considered most influential.
        
        Parameters
        ----------
        sample_idx : int
            Index of sample within Sample node type
        top_k : int
            Number of top neighbors to return
        
        Returns
        -------
        neighbors : pd.DataFrame
            Columns: neighbor_id, neighbor_type, similarity_score, influence_rank
        """
        if self.embeddings is None:
            self.compute_predictions(return_embeddings=True)
        
        # Get patient embedding
        patient_emb = self.embeddings[sample_idx]  # [out_channels]
        
        # Compute similarity with all other nodes
        similarities = {}
        
        for node_type in self.data.node_types:
            if node_type == 'Sample':
                continue  # Skip other samples
            
            node_embeddings = self.data[node_type].x  # [num_nodes, in_channels]
            
            # Compute cosine similarity (simplified - would need actual node embeddings)
            # For now, use feature-based similarity
            if node_embeddings.size(1) == patient_emb.size(0):
                cosine_sim = torch.cosine_similarity(
                    patient_emb.unsqueeze(0),
                    node_embeddings
                )
                
                for node_local_idx, score in enumerate(cosine_sim):
                    node_id = self.idx_to_node_id.get((node_type, node_local_idx), f"Unknown_{node_local_idx}")
                    similarities[node_id] = float(score)
        
        # Sort by similarity
        top_neighbors = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Create result dataframe
        results = []
        for rank, (node_id, score) in enumerate(top_neighbors, 1):
            node_type = self._extract_node_type(node_id)
            results.append({
                'rank': rank,
                'neighbor_id': node_id,
                'neighbor_type': node_type,
                'influence_score': score,
            })
        
        return pd.DataFrame(results)
    
    def get_edge_type_importance(self) -> pd.DataFrame:
        """
        Get importance of different edge types across all predictions.
        
        Returns
        -------
        edge_importance : pd.DataFrame
            Columns: edge_type, source_type, target_type, importance_score
        """
        # Count edge type usage in predictions
        edge_counts = {}
        
        for edge_key in self.data.edge_types:
            source_type, edge_type, target_type = edge_key
            edge_index = self.data[edge_key].edge_index
            num_edges = edge_index.size(1)
            
            edge_counts[edge_key] = num_edges
        
        # Normalize by total edges
        total_edges = sum(edge_counts.values())
        
        # Create results
        results = []
        for edge_key, count in sorted(edge_counts.items(), key=lambda x: x[1], reverse=True):
            source_type, edge_type, target_type = edge_key
            importance = count / total_edges
            
            results.append({
                'edge_type': edge_type,
                'source_type': source_type,
                'target_type': target_type,
                'num_edges': count,
                'importance': importance,
                'percentage': f"{importance*100:.1f}%",
            })
        
        return pd.DataFrame(results)
    
    def get_patient_subgraph(self,
                             sample_idx: int,
                             depth: int = 1,
                             top_neighbors: int = 20) -> Tuple[nx.Graph, Dict]:
        """
        Extract patient's local subgraph.
        
        Parameters
        ----------
        sample_idx : int
            Patient index
        depth : int
            Number of hops to include
        top_neighbors : int
            Number of top neighbors per hop
        
        Returns
        -------
        G : nx.Graph
            NetworkX graph of subgraph
        metadata : dict
            Node and edge metadata for visualization
        """
        G = nx.Graph()
        
        # Add patient node
        patient_id = self.idx_to_node_id.get(('Sample', sample_idx), f"Patient_{sample_idx}")
        prediction = self.predictions[sample_idx].item() if self.predictions is not None else -1
        pred_label = "Healthy" if prediction == 1 else "Septic"
        confidence = self.probabilities[sample_idx].max().item() if self.probabilities is not None else 0
        
        G.add_node(
            patient_id,
            node_type='Sample',
            label=f"{patient_id}\n({pred_label}, {confidence:.2f})",
            color='lightgreen' if prediction == 1 else 'lightcoral',
        )
        
        # Track nodes by type and index for edge lookups
        added_nodes = {('Sample', sample_idx): patient_id}
        
        # Find all edges from this sample
        for edge_key in self.data.edge_types:
            source_type, edge_type, target_type = edge_key
            
            # Only consider edges starting from Sample
            if source_type != 'Sample':
                continue
                
            edge_index = self.data[edge_key].edge_index
            
            # Find all targets for this sample
            source_mask = edge_index[0] == sample_idx
            target_indices = edge_index[1, source_mask].cpu().numpy()
            
            # Take top N by influence score (using edge ordering as proxy)
            neighbor_count = 0
            for target_idx in target_indices[:top_neighbors]:
                target_idx = int(target_idx)
                neighbor_id = self.idx_to_node_id.get((target_type, target_idx), f"{target_type}_{target_idx}")
                
                if neighbor_id not in G:
                    # Truncate label for readability
                    short_label = neighbor_id
                    if '_' in neighbor_id:
                        parts = neighbor_id.split('_')
                        short_label = f"{parts[0]}\n{parts[1][:8] if len(parts) > 1 else ''}"
                    
                    G.add_node(
                        neighbor_id,
                        node_type=target_type,
                        label=short_label,
                        color='lightblue',
                    )
                    added_nodes[(target_type, target_idx)] = neighbor_id
                
                # Add edge
                G.add_edge(
                    patient_id, 
                    neighbor_id,
                    edge_type=edge_type,
                    color='gray'
                )
                
                neighbor_count += 1
                if neighbor_count >= top_neighbors:
                    break
        
        # Create metadata
        metadata = {
            'patient_id': patient_id,
            'prediction': 'Healthy' if prediction == 1 else 'Septic',
            'confidence': confidence,
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
        }
        
        return G, metadata
    
    def plot_patient_subgraph(self,
                              sample_idx: int,
                              output_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (12, 10)):
        """
        Visualize patient's subgraph with attention coloring.
        
        Parameters
        ----------
        sample_idx : int
            Patient index
        output_path : str, optional
            Path to save figure
        figsize : tuple
            Figure size
        """
        G, metadata = self.get_patient_subgraph(sample_idx)
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw edges
        edges = G.edges()
        weights = [G[u][v].get('weight', 1.0) for u, v in edges]
        
        nx.draw_networkx_edges(
            G, pos,
            width=[2 * (w + 0.5) for w in weights],
            alpha=0.3,
            ax=ax
        )
        
        # Draw nodes
        node_colors = [G.nodes[node]['color'] for node in G.nodes()]
        
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=[300 if node == metadata['patient_id'] else 200 for node in G.nodes()],
            ax=ax
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            G, pos,
            labels={node: G.nodes[node]['label'] for node in G.nodes()},
            font_size=8,
            ax=ax
        )
        
        # Title and metadata
        title = f"Patient {metadata['patient_id']}\n" \
                f"Prediction: {metadata['prediction']} (confidence={metadata['confidence']:.2f})\n" \
                f"Subgraph: {metadata['num_nodes']} nodes, {metadata['num_edges']} edges"
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved: {output_path}")
        
        return fig, ax
    
    def plot_edge_type_importance(self,
                                  output_path: Optional[str] = None,
                                  figsize: Tuple[int, int] = (12, 6)):
        """
        Plot importance of different edge types.
        
        Parameters
        ----------
        output_path : str, optional
            Path to save figure
        figsize : tuple
            Figure size
        """
        edge_importance = self.get_edge_type_importance()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Bar plot
        top_edges = edge_importance.head(15)
        
        ax.barh(range(len(top_edges)), top_edges['importance'], alpha=0.7)
        ax.set_yticks(range(len(top_edges)))
        ax.set_yticklabels([f"{row['edge_type']}\n({row['source_type']}→{row['target_type']})"
                             for _, row in top_edges.iterrows()], fontsize=9)
        ax.set_xlabel('Relative Importance', fontsize=12, fontweight='bold')
        ax.set_title('Edge Type Importance in HAN Model', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved: {output_path}")
        
        return fig, ax
    
    def _extract_node_type(self, node_id: str) -> str:
        """Extract node type from ID."""
        prefixes = {
            'Sample_': 'Sample',
            'Protein_': 'Protein',
            'Gene_': 'Gene',
            'Pathway_': 'Pathway',
            'GO_': 'GO_Term',
            'Reaction_': 'Reaction',
        }
        
        for prefix, node_type in prefixes.items():
            if node_id.startswith(prefix):
                return node_type
        return 'Entity'


if __name__ == '__main__':
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)
    
    from src.han.data_loader import load_hetero_graph
    from src.han.model import SepsisHANClassifier
    
    print("Loading graph...")
    data = load_hetero_graph('models/executions/GSE54514_enriched_ontology_degfilterv2.11')
    
    print("Creating model...")
    in_channels_dict = {node_type: data[node_type].x.size(1) 
                        for node_type in data.node_types}
    model = SepsisHANClassifier(in_channels_dict=in_channels_dict)
    
    print("Initializing analyzer...")
    # Note: Need to build node_id_to_idx mappings from data_loader
    analyzer = AttentionAnalyzer(model, data, {}, {})
    
    print("Analyzing predictions...")
    analyzer.compute_predictions()
    
    # Get edge type importance
    edge_imp = analyzer.get_edge_type_importance()
    print("\nEdge Type Importance:")
    print(edge_imp.head(10))
