"""
GNN Interpretability Module
----------------------------

Tools for interpreting Graph Neural Network models (RGCN, ComplEx)
Focuses on understanding:
- Which graph structures are important
- Which dimensions capture what biological information
- Edge/node importance in predictions
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class GNNInterpreter:
    """
    Interpret GNN embeddings and their biological meaning.
    
    Maps embedding dimensions back to biological entities (genes, proteins, pathways)
    and analyzes which graph structures contribute to predictions.
    """
    
    def __init__(self, 
                 embeddings_path: str,
                 entity_mapping_path: str,
                 node_features_path: Optional[str] = None,
                 embedding_type: str = 'RGCN'):
        """
        Initialize GNN interpreter.
        
        Parameters
        ----------
        embeddings_path : str
            Path to .npy file with entity embeddings
        entity_mapping_path : str
            Path to entity-to-id mapping (TSV or CSV)
        node_features_path : str, optional
            Path to node features CSV
        embedding_type : str
            Type of embedding: 'RGCN' or 'ComplEx'
        """
        self.embedding_type = embedding_type
        
        # Load embeddings
        print(f"Loading {embedding_type} embeddings from {embeddings_path}")
        self.embeddings = np.load(embeddings_path, allow_pickle=True)
        
        # Handle ComplEx (complex-valued embeddings)
        if np.iscomplexobj(self.embeddings):
            print("  ComplEx embeddings detected, splitting real+imag")
            self.embeddings = np.concatenate([self.embeddings.real, self.embeddings.imag], axis=1)
        
        print(f"  Embeddings shape: {self.embeddings.shape}")
        
        # Load entity mapping
        print(f"Loading entity mapping from {entity_mapping_path}")
        if entity_mapping_path.endswith('.tsv'):
            self.entity_mapping = pd.read_csv(entity_mapping_path, sep='\t')
        else:
            self.entity_mapping = pd.read_csv(entity_mapping_path)
        
        print(f"  Loaded {len(self.entity_mapping)} entities")
        print(f"  Columns: {list(self.entity_mapping.columns)}")
        
        # Load node features if available
        self.node_features = None
        if node_features_path and Path(node_features_path).exists():
            self.node_features = pd.read_csv(node_features_path)
            print(f"  Loaded node features: {len(self.node_features)} nodes")
    
    def get_dimension_top_entities(self, dimension: int, top_k: int = 20) -> pd.DataFrame:
        """
        Get entities with highest values in a specific embedding dimension.
        
        Parameters
        ----------
        dimension : int
            Embedding dimension to analyze
        top_k : int
            Number of top entities to return
            
        Returns
        -------
        top_entities : pd.DataFrame
            DataFrame with entity info and embedding values
        """
        dim_values = self.embeddings[:, dimension]
        
        # Get top positive and negative
        top_positive_idx = np.argsort(dim_values)[-top_k:][::-1]
        top_negative_idx = np.argsort(dim_values)[:top_k]
        
        results = []
        
        for idx in top_positive_idx:
            entity_info = self.entity_mapping.iloc[idx].to_dict()
            entity_info['embedding_value'] = float(dim_values[idx])
            entity_info['direction'] = 'positive'
            results.append(entity_info)
        
        for idx in top_negative_idx:
            entity_info = self.entity_mapping.iloc[idx].to_dict()
            entity_info['embedding_value'] = float(dim_values[idx])
            entity_info['direction'] = 'negative'
            results.append(entity_info)
        
        return pd.DataFrame(results)
    
    def analyze_important_dimensions(self, 
                                    important_dims: List[int],
                                    top_k: int = 10) -> Dict[int, pd.DataFrame]:
        """
        Analyze multiple important dimensions at once.
        
        Parameters
        ----------
        important_dims : list of int
            List of dimension indices to analyze
        top_k : int
            Number of top entities per dimension
            
        Returns
        -------
        results : dict
            Mapping from dimension to top entities DataFrame
        """
        results = {}
        
        print(f"\nAnalyzing {len(important_dims)} important dimensions...")
        print("="*80)
        
        for dim in important_dims:
            print(f"\nDimension {dim}:")
            print("-"*80)
            
            entities = self.get_dimension_top_entities(dim, top_k=top_k)
            results[dim] = entities
            
            # Print summary
            if 'label' in entities.columns:
                print(f"Top 5 positive entities:")
                for _, row in entities[entities['direction'] == 'positive'].head(5).iterrows():
                    print(f"  {row['label']:40s} {row['embedding_value']:+.4f}")
            else:
                print(f"Top 5 positive entities:")
                for _, row in entities[entities['direction'] == 'positive'].head(5).iterrows():
                    print(f"  Entity {row.get('entity_id', row.get('id', 'unknown')):40s} {row['embedding_value']:+.4f}")
        
        return results
    
    def get_patient_embeddings(self, patient_prefix: str = "Sample") -> Tuple[np.ndarray, List[str]]:
        """
        Extract patient/sample embeddings from entity embeddings.
        
        Parameters
        ----------
        patient_prefix : str
            Prefix used for patient nodes (default: "Sample")
            
        Returns
        -------
        patient_embeddings : np.ndarray
            Shape (n_patients, embedding_dim)
        patient_ids : list of str
            Patient identifiers
        """
        # Find patient entities
        if 'label' in self.entity_mapping.columns:
            patient_mask = self.entity_mapping['label'].str.startswith(patient_prefix)
        elif 'node_id' in self.entity_mapping.columns:
            patient_mask = self.entity_mapping['node_id'].str.startswith(patient_prefix)
        else:
            raise ValueError("Cannot find patient entities - no 'label' or 'node_id' column")
        
        patient_indices = self.entity_mapping[patient_mask].index.tolist()
        patient_embeddings = self.embeddings[patient_indices]
        
        if 'label' in self.entity_mapping.columns:
            patient_ids = self.entity_mapping[patient_mask]['label'].tolist()
        else:
            patient_ids = self.entity_mapping[patient_mask]['node_id'].tolist()
        
        print(f"Extracted {len(patient_ids)} patient embeddings")
        
        return patient_embeddings, patient_ids
    
    def plot_dimension_distribution(self, dimension: int, 
                                   entity_type: Optional[str] = None,
                                   output_path: Optional[str] = None,
                                   show: bool = True) -> plt.Figure:
        """
        Plot distribution of values in an embedding dimension.
        
        Parameters
        ----------
        dimension : int
            Dimension to plot
        entity_type : str, optional
            Filter by entity type (e.g., 'Gene', 'Pathway')
        output_path : str, optional
            Save plot to this path
        show : bool
            Whether to display the plot
            
        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        dim_values = self.embeddings[:, dimension]
        
        if entity_type and 'type' in self.entity_mapping.columns:
            mask = self.entity_mapping['type'] == entity_type
            dim_values = dim_values[mask]
            title_suffix = f" ({entity_type} entities)"
        else:
            title_suffix = ""
        
        ax.hist(dim_values, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel(f'Embedding Value (Dimension {dimension})', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'{self.embedding_type} - Dimension {dimension} Distribution{title_suffix}', 
                    fontsize=14, fontweight='bold')
        ax.axvline(0, color='red', linestyle='--', alpha=0.5)
        
        # Add statistics
        stats_text = f'Mean: {dim_values.mean():.4f}\nStd: {dim_values.std():.4f}'
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved to: {output_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_dimension_correlation_matrix(self, 
                                         dimensions: List[int],
                                         output_path: Optional[str] = None,
                                         show: bool = True) -> plt.Figure:
        """
        Plot correlation matrix for multiple important dimensions.
        
        Parameters
        ----------
        dimensions : list of int
            Dimensions to correlate
        output_path : str, optional
            Save plot to this path
        show : bool
            Whether to display the plot
            
        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        # Extract relevant dimensions
        emb_subset = self.embeddings[:, dimensions]
        
        # Compute correlation
        corr = np.corrcoef(emb_subset.T)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(corr, 
                   xticklabels=[f'Dim_{d}' for d in dimensions],
                   yticklabels=[f'Dim_{d}' for d in dimensions],
                   annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   ax=ax, cbar_kws={'label': 'Correlation'})
        
        ax.set_title(f'{self.embedding_type} - Dimension Correlation Matrix', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved to: {output_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def compare_patient_groups(self, 
                              patient_labels: np.ndarray,
                              dimension: int,
                              label_names: Dict[int, str] = {0: 'Control', 1: 'Septic'}) -> Dict:
        """
        Compare embedding dimension values between patient groups.
        
        Parameters
        ----------
        patient_labels : np.ndarray
            Binary labels for patients (0=control, 1=disease)
        dimension : int
            Dimension to compare
        label_names : dict
            Mapping from label to name
            
        Returns
        -------
        comparison : dict
            Statistics for each group
        """
        patient_embeddings, _ = self.get_patient_embeddings()
        
        dim_values = patient_embeddings[:, dimension]
        
        results = {}
        for label, name in label_names.items():
            mask = patient_labels == label
            values = dim_values[mask]
            
            results[name] = {
                'mean': float(values.mean()),
                'std': float(values.std()),
                'median': float(np.median(values)),
                'n': int(mask.sum())
            }
        
        # Statistical test
        from scipy.stats import mannwhitneyu
        group0 = dim_values[patient_labels == 0]
        group1 = dim_values[patient_labels == 1]
        stat, pval = mannwhitneyu(group0, group1)
        
        results['test'] = {
            'statistic': float(stat),
            'p_value': float(pval),
            'significant': bool(pval < 0.05)
        }
        
        return results
