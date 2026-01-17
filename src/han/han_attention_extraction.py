"""
Enhanced HAN Attention Extraction for Interpretability

Extracts attention weights at multiple levels:
1. Node-level: Which neighbors contribute most to each node's representation
2. Layer-level: Attention distribution across GNN layers
3. Edge-type-level: Which relation types (metapaths) are most important
4. Sample-level: Aggregate attention patterns for patient predictions
"""

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from collections import defaultdict

from torch_geometric.data import HeteroData
from torch_geometric.nn import MessagePassing


logger = logging.getLogger(__name__)


class HANAttentionExtractor:
    """
    Extract and analyze attention weights from trained HAN model.
    
    Supports:
    - GAT-style attention (if using GATConv layers)
    - Semantic-level attention (if using HANConv)
    - Custom attention hooks for any MessagePassing layer
    """
    
    def __init__(self,
                 model,
                 data: HeteroData,
                 device: str = 'cpu'):
        """
        Initialize attention extractor.
        
        Parameters
        ----------
        model : nn.Module
            Trained HAN model with attention mechanisms
        data : HeteroData
            Graph data
        device : str
            Device for computation
        """
        self.model = model.to(device)
        self.data = data.to(device)
        self.device = device
        
        # Storage for attention weights
        self.attention_weights = {}
        self.attention_hooks = []
        
        # Predictions cache
        self.predictions = None
        self.probabilities = None
        self.embeddings = None
        
    def register_attention_hooks(self):
        """
        Register forward hooks to capture attention weights during forward pass.
        Works with HANConv and GATConv layers.
        """
        logger.info("Registering attention hooks...")
        
        def attention_hook(module, input, output):
            """Hook to capture attention weights from MessagePassing layers."""
            module_name = module.__class__.__name__
            
            # GATConv stores attention in .alpha or ._alpha
            if hasattr(module, '_alpha') and module._alpha is not None:
                if module_name not in self.attention_weights:
                    self.attention_weights[module_name] = []
                self.attention_weights[module_name].append(module._alpha.detach().cpu())
            elif hasattr(module, 'alpha') and module.alpha is not None:
                if module_name not in self.attention_weights:
                    self.attention_weights[module_name] = []
                self.attention_weights[module_name].append(module.alpha.detach().cpu())
            
            # HANConv: Check internal projection layers
            if module_name == 'HANConv' and hasattr(module, 'proj'):
                han_attentions = {}
                for attr_name in dir(module.proj):
                    if not attr_name.startswith('_'):
                        submodule = getattr(module.proj, attr_name, None)
                        if submodule is not None and isinstance(submodule, MessagePassing):
                            if hasattr(submodule, '_alpha') and submodule._alpha is not None:
                                han_attentions[attr_name] = submodule._alpha.detach().cpu()
                            elif hasattr(submodule, 'alpha') and submodule.alpha is not None:
                                han_attentions[attr_name] = submodule.alpha.detach().cpu()
                
                if han_attentions:
                    if 'HANConv' not in self.attention_weights:
                        self.attention_weights['HANConv'] = []
                    self.attention_weights['HANConv'].append(han_attentions)
        
        # Register hooks for all MessagePassing layers (includes HANConv, GATConv)
        for name, module in self.model.named_modules():
            if isinstance(module, MessagePassing):
                hook = module.register_forward_hook(attention_hook)
                self.attention_hooks.append(hook)
                logger.info(f"  Registered hook for {name}: {module.__class__.__name__}")
        
        return len(self.attention_hooks)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.attention_hooks:
            hook.remove()
        self.attention_hooks = []
        logger.info("Removed all attention hooks")
    
    def extract_attention(self, return_embeddings: bool = True) -> Dict[str, Any]:
        """
        Perform forward pass and extract attention weights.
        
        Parameters
        ----------
        return_embeddings : bool
            Whether to return node embeddings
        
        Returns
        -------
        results : dict
            Dictionary containing:
            - predictions: Predicted labels
            - probabilities: Class probabilities
            - embeddings: Node embeddings (if return_embeddings=True)
            - attention_weights: Attention weights per layer
        """
        logger.info("Extracting attention weights...")
        
        # Clear previous attention weights
        self.attention_weights = {}
        
        # Register hooks
        self.register_attention_hooks()
        
        # Forward pass with attention extraction
        self.model.eval()
        with torch.no_grad():
            # Call with return_attention=True
            outputs = self.model(self.data.x_dict, self.data.edge_index_dict, return_attention=True)
            
            if isinstance(outputs, tuple):
                logits = outputs[0]
                embeddings = outputs[1] if len(outputs) > 1 else None
                attention_from_model = outputs[2] if len(outputs) > 2 else None
                
                # Merge model-returned attention with hook-captured attention
                if attention_from_model and 'layer_attentions' in attention_from_model:
                    self.attention_weights['model_attention'] = attention_from_model['layer_attentions']
            else:
                logits = outputs
                embeddings = None
        
        # Remove hooks
        self.remove_hooks()
        
        # Compute predictions
        self.probabilities = torch.softmax(logits, dim=1)
        self.predictions = logits.argmax(dim=1)
        self.embeddings = embeddings
        
        results = {
            'predictions': self.predictions.cpu().numpy(),
            'probabilities': self.probabilities.cpu().numpy(),
            'attention_weights': self.attention_weights,
        }
        
        if return_embeddings and embeddings is not None:
            results['embeddings'] = embeddings.cpu().numpy()
        
        logger.info(f"Extracted attention from {len(self.attention_weights)} layer types")
        
        return results
    
    def compute_node_level_attention(self, 
                                     sample_idx: int,
                                     edge_type: Tuple[str, str, str],
                                     top_k: int = 20) -> pd.DataFrame:
        """
        Get top-K neighbors with highest attention for a given sample.
        
        Parameters
        ----------
        sample_idx : int
            Index of sample node
        edge_type : tuple
            (src_type, relation, dst_type)
        top_k : int
            Number of top neighbors to return
        
        Returns
        -------
        neighbors_df : pd.DataFrame
            DataFrame with columns: neighbor_idx, attention_score, neighbor_type
        """
        if not self.attention_weights:
            raise ValueError("No attention weights available. Run extract_attention() first.")
        
        # Find attention weights for this edge type
        # This is a simplified version - actual implementation depends on model architecture
        attention_data = []
        
        for layer_name, attention_list in self.attention_weights.items():
            for attn in attention_list:
                # attn shape typically: [num_edges, num_heads] or [num_edges]
                if attn.dim() > 1:
                    attn = attn.mean(dim=-1)  # Average over heads
                
                # Get edge_index for this edge type
                if edge_type in self.data.edge_types:
                    edge_index = self.data[edge_type].edge_index
                    
                    # Find edges originating from sample_idx
                    mask = edge_index[0] == sample_idx
                    neighbor_indices = edge_index[1, mask].cpu().numpy()
                    neighbor_attention = attn[mask].cpu().numpy()
                    
                    for neighbor_idx, att_score in zip(neighbor_indices, neighbor_attention):
                        attention_data.append({
                            'neighbor_idx': int(neighbor_idx),
                            'attention_score': float(att_score),
                            'edge_type': f"{edge_type[0]}-{edge_type[1]}-{edge_type[2]}",
                            'layer': layer_name
                        })
        
        if not attention_data:
            logger.warning(f"No attention data found for sample {sample_idx} and edge type {edge_type}")
            return pd.DataFrame()
        
        df = pd.DataFrame(attention_data)
        
        # Aggregate attention across layers (mean)
        df_agg = df.groupby(['neighbor_idx', 'edge_type'])['attention_score'].mean().reset_index()
        df_agg = df_agg.sort_values('attention_score', ascending=False).head(top_k)
        
        return df_agg
    
    def compute_edge_type_importance(self) -> pd.DataFrame:
        """
        Compute importance of each edge type based on aggregate attention.
        If no attention weights available, compute based on edge structure.
        
        Returns
        -------
        importance_df : pd.DataFrame
            DataFrame with columns: edge_type, mean_attention, std_attention, num_edges
        """
        if not self.attention_weights:
            logger.warning("No attention weights captured. Computing edge importance based on graph structure.")
            return self._compute_structural_importance()
        
        edge_type_stats = defaultdict(list)
        
        for edge_type in self.data.edge_types:
            edge_index = self.data[edge_type].edge_index
            num_edges = edge_index.size(1)
            
            # Collect attention scores for this edge type
            for layer_name, attention_list in self.attention_weights.items():
                for attn in attention_list:
                    if attn.dim() > 1:
                        attn = attn.mean(dim=-1)
                    
                    if attn.size(0) == num_edges:
                        edge_type_stats[edge_type].append(attn.cpu().numpy())
        
        # Compute statistics
        results = []
        for edge_type, attention_scores in edge_type_stats.items():
            if attention_scores:
                all_scores = np.concatenate(attention_scores)
                results.append({
                    'edge_type': f"{edge_type[0]}-{edge_type[1]}-{edge_type[2]}",
                    'src_type': edge_type[0],
                    'relation': edge_type[1],
                    'dst_type': edge_type[2],
                    'mean_attention': np.mean(all_scores),
                    'std_attention': np.std(all_scores),
                    'median_attention': np.median(all_scores),
                    'num_edges': len(all_scores)
                })
        
        df = pd.DataFrame(results)
        df = df.sort_values('mean_attention', ascending=False)
        
        return df
    
    def _compute_structural_importance(self) -> pd.DataFrame:
        """
        Fallback: Compute edge type importance based on graph structure.
        Uses edge count and connectivity as proxy for importance.
        
        Returns
        -------
        importance_df : pd.DataFrame
            DataFrame with structural importance scores
        """
        results = []
        
        for edge_type in self.data.edge_types:
            edge_index = self.data[edge_type].edge_index
            num_edges = edge_index.size(1)
            
            src_type, relation, dst_type = edge_type
            
            # Compute structural metrics
            src_nodes = edge_index[0].unique()
            dst_nodes = edge_index[1].unique()
            
            # Normalize by total possible edges
            max_possible = len(src_nodes) * len(dst_nodes)
            density = num_edges / max_possible if max_possible > 0 else 0
            
            # Use normalized edge count as importance score
            results.append({
                'src_type': src_type,
                'relation': relation,
                'dst_type': dst_type,
                'edge_type': f"{src_type}__{relation}__{dst_type}",
                'mean_attention': density,  # Use density as proxy for attention
                'std_attention': 0.0,  # No variation in structural measure
                'median_attention': density,
                'num_edges': num_edges
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values('num_edges', ascending=False)  # Sort by edge count
        
        logger.info(f"Computed structural importance for {len(df)} edge types (no attention weights available)")
        
        return df
    
    def visualize_attention_distribution(self, 
                                        output_path: str = 'results/han_attention/attention_distribution.png'):
        """
        Create visualization of attention distribution across edge types.
        
        Parameters
        ----------
        output_path : str
            Path to save figure
        """
        df = self.compute_edge_type_importance()
        
        if df.empty:
            logger.warning("No attention data to visualize")
            return
        
        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Mean attention by edge type
        ax = axes[0]
        df_top = df.head(15)
        ax.barh(range(len(df_top)), df_top['mean_attention'], color='steelblue', alpha=0.7)
        ax.set_yticks(range(len(df_top)))
        ax.set_yticklabels(df_top['edge_type'], fontsize=8)
        ax.set_xlabel('Mean Attention Score', fontsize=11, fontweight='bold')
        ax.set_title('Top 15 Edge Types by Attention', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        # Plot 2: Attention by source node type
        ax = axes[1]
        src_type_attention = df.groupby('src_type')['mean_attention'].mean().sort_values(ascending=False)
        ax.bar(range(len(src_type_attention)), src_type_attention.values, color='coral', alpha=0.7)
        ax.set_xticks(range(len(src_type_attention)))
        ax.set_xticklabels(src_type_attention.index, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Mean Attention Score', fontsize=11, fontweight='bold')
        ax.set_title('Attention by Source Node Type', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved attention distribution to {output_path}")
        plt.close()
    
    def export_attention_summary(self, 
                                 output_dir: str = 'results/han_attention'):
        """
        Export comprehensive attention analysis to CSV files.
        
        Parameters
        ----------
        output_dir : str
            Directory to save outputs
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Exporting attention analysis to {output_dir}")
        
        # 1. Edge type importance
        df_edge_importance = self.compute_edge_type_importance()
        df_edge_importance.to_csv(output_path / 'edge_type_importance.csv', index=False)
        logger.info(f"  ✓ Saved edge_type_importance.csv")
        
        # 2. Sample predictions
        if self.predictions is not None:
            df_predictions = pd.DataFrame({
                'sample_idx': range(len(self.predictions)),
                'predicted_label': self.predictions.cpu().numpy(),
                'prob_septic': self.probabilities[:, 0].cpu().numpy(),
                'prob_healthy': self.probabilities[:, 1].cpu().numpy(),
            })
            df_predictions.to_csv(output_path / 'sample_predictions.csv', index=False)
            logger.info(f"  ✓ Saved sample_predictions.csv")
        
        # 3. Attention visualization
        self.visualize_attention_distribution(str(output_path / 'attention_distribution.png'))
        
        logger.info(f"✓ Attention analysis complete! Results in {output_dir}")
        
        return output_path


if __name__ == '__main__':
    """Test attention extraction on trained HAN model."""
    
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # This would need a trained model to work
    print("AttentionExtractor created. Load a trained HAN model to use.")
    print("\nExample usage:")
    print("  from src.han.han_attention_extraction import HANAttentionExtractor")
    print("  extractor = HANAttentionExtractor(model, data)")
    print("  results = extractor.extract_attention()")
    print("  extractor.export_attention_summary('results/han_attention')")
