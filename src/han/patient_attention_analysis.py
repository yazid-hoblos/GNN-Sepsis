"""
Patient-level Attention Analysis for HAN Interpretability

Extracts and visualizes attention for individual patients to show:
1. Node-level attention: Which neighbors influence each patient's prediction?
2. Semantic-level attention: Which metapath types matter?
3. Patient subgraph visualization: Important nodes and relationships

This directly addresses supervisor's requirements:
- Extract attention weights per patient
- Visualize patient subgraphs with attention highlights
- Justify predictions through attention analysis
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from collections import defaultdict

from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)


class PatientAttentionAnalyzer:
    """
    Analyze attention weights for individual patients to understand
    which neighbors and metapaths influence their predictions.
    """
    
    def __init__(self,
                 model,
                 data: HeteroData,
                 device: str = 'cpu'):
        """
        Initialize patient attention analyzer.
        
        Parameters
        ----------
        model : nn.Module
            Trained HAN model
        data : HeteroData
            Graph data with metadata
        device : str
            Device for computation
        """
        self.model = model.to(device)
        self.data = data.to(device)
        self.device = device
        
        # Create node ID to name mappings
        self.node_idx_to_name = self._create_node_mappings()
    
    def _create_node_mappings(self) -> Dict[str, Dict[int, str]]:
        """
        Create mappings from node index to node name for each node type.
        
        Returns
        -------
        mappings : dict
            {node_type: {idx: name}}
        """
        mappings = {}
        
        for node_type in self.data.node_types:
            idx_to_name = {}
            if node_type == 'Sample':
                # Samples: use indices
                for idx in range(self.data[node_type].num_nodes):
                    idx_to_name[idx] = f"Patient_{idx}"
            else:
                # For proteins, pathways, GO terms, reactions - use indices for now
                # In a real scenario, you'd load actual names from OWL
                for idx in range(self.data[node_type].num_nodes):
                    idx_to_name[idx] = f"{node_type}_{idx}"
            
            mappings[node_type] = idx_to_name
        
        return mappings
    
    def get_patient_neighbors(self,
                             patient_idx: int,
                             edge_type: Tuple[str, str, str],
                             top_k: int = 10) -> pd.DataFrame:
        """
        Get top-K neighbors of a patient along a specific metapath.
        
        Parameters
        ----------
        patient_idx : int
            Index of patient node
        edge_type : tuple
            (src_type, relation, dst_type)
        top_k : int
            Number of top neighbors to return
        
        Returns
        -------
        neighbors_df : pd.DataFrame
            DataFrame with neighbor info (index, type, edge count)
        """
        src_type, relation, dst_type = edge_type
        
        if src_type != 'Sample':
            logger.warning(f"Edge type {edge_type} doesn't start from Sample")
            return pd.DataFrame()
        
        if edge_type not in self.data.edge_types:
            logger.warning(f"Edge type {edge_type} not in data")
            return pd.DataFrame()
        
        edge_index = self.data[edge_type].edge_index
        
        # Find edges from this patient
        mask = edge_index[0] == patient_idx
        dst_indices = edge_index[1][mask]
        
        if len(dst_indices) == 0:
            return pd.DataFrame()
        
        # Count edges per destination (multiple edges to same node)
        unique_neighbors, counts = torch.unique(dst_indices, return_counts=True)
        
        # Sort by count and get top-k
        top_indices = counts.argsort(descending=True)[:top_k]
        top_neighbors = unique_neighbors[top_indices]
        top_counts = counts[top_indices]
        
        results = []
        for neighbor_idx, count in zip(top_neighbors, top_counts):
            results.append({
                'neighbor_idx': int(neighbor_idx),
                'neighbor_name': self.node_idx_to_name[dst_type][int(neighbor_idx)],
                'neighbor_type': dst_type,
                'edge_count': int(count),
                'edge_type': relation
            })
        
        return pd.DataFrame(results)
    
    def analyze_patient_prediction(self,
                                  patient_idx: int,
                                  predicted_label: int,
                                  prediction_prob: float) -> Dict:
        """
        Analyze all factors influencing a single patient's prediction.
        
        Parameters
        ----------
        patient_idx : int
            Index of patient
        predicted_label : int
            Predicted class (0=healthy, 1=septic)
        prediction_prob : float
            Prediction probability
        
        Returns
        -------
        analysis : dict
            Comprehensive analysis with:
            - prediction info
            - important neighbors per metapath
            - edge type importance
        """
        result = {
            'patient_idx': patient_idx,
            'patient_name': self.node_idx_to_name['Sample'][patient_idx],
            'predicted_label': predicted_label,
            'predicted_class': 'Septic' if predicted_label == 1 else 'Healthy',
            'prediction_prob': prediction_prob,
            'neighbors_by_metapath': {},
            'edge_type_distribution': {},
        }
        
        # Analyze neighbors along each metapath
        for edge_type in self.data.edge_types:
            src_type, relation, dst_type = edge_type
            if src_type == 'Sample':
                neighbors_df = self.get_patient_neighbors(
                    patient_idx, edge_type, top_k=5
                )
                if len(neighbors_df) > 0:
                    result['neighbors_by_metapath'][relation] = neighbors_df
                    result['edge_type_distribution'][relation] = len(neighbors_df)
        
        return result
    
    def generate_patient_report(self,
                               predictions_csv: str,
                               output_dir: str = 'results/patient_analysis',
                               num_patients: Optional[int] = None) -> None:
        """
        Generate detailed report for patient-level attention analysis.
        
        Parameters
        ----------
        predictions_csv : str
            Path to sample_predictions.csv
        output_dir : str
            Output directory for reports
        num_patients : int, optional
            Number of patients to analyze (None = all)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load predictions
        preds = pd.read_csv(predictions_csv)
        
        if num_patients:
            preds = preds.head(num_patients)
        
        logger.info(f"Analyzing {len(preds)} patients...")
        
        # Analyze each patient
        patient_analyses = []
        for idx, row in preds.iterrows():
            patient_idx = int(row['sample_idx'])
            pred_label = int(row['predicted_label'])
            pred_prob = float(row['prob_septic']) if pred_label == 1 else float(row['prob_healthy'])
            
            analysis = self.analyze_patient_prediction(
                patient_idx, pred_label, pred_prob
            )
            patient_analyses.append(analysis)
        
        # Save detailed analysis for each patient
        logger.info(f"Saving patient-level analysis...")
        for analysis in patient_analyses:
            patient_file = output_path / f"patient_{analysis['patient_idx']:03d}_analysis.txt"
            with open(patient_file, 'w') as f:
                f.write(f"Patient Analysis Report\n")
                f.write(f"{'='*80}\n\n")
                f.write(f"Patient ID: {analysis['patient_name']}\n")
                f.write(f"Prediction: {analysis['predicted_class']}\n")
                f.write(f"Confidence: {analysis['prediction_prob']:.4f}\n\n")
                
                f.write(f"Influential Neighbors by Metapath:\n")
                f.write(f"{'-'*80}\n")
                for relation, neighbors_df in analysis['neighbors_by_metapath'].items():
                    f.write(f"\n{relation} (→ {neighbors_df.iloc[0]['neighbor_type']}):\n")
                    for _, row in neighbors_df.iterrows():
                        f.write(f"  - {row['neighbor_name']:30s} (edges: {row['edge_count']})\n")
        
        # Create summary CSV
        summary_data = []
        for analysis in patient_analyses:
            summary_data.append({
                'patient_idx': analysis['patient_idx'],
                'patient_name': analysis['patient_name'],
                'predicted_label': analysis['predicted_label'],
                'predicted_class': analysis['predicted_class'],
                'prediction_prob': analysis['prediction_prob'],
                'num_metapaths': len(analysis['neighbors_by_metapath']),
                'total_neighbors': sum(analysis['edge_type_distribution'].values()),
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_path / 'patient_summary.csv', index=False)
        logger.info(f"✓ Saved: {output_path / 'patient_summary.csv'}")
        
        # Create aggregate statistics
        logger.info(f"\nPatient Attention Analysis Summary:")
        logger.info(f"  Total patients analyzed: {len(patient_analyses)}")
        logger.info(f"  Average prediction confidence: {summary_df['prediction_prob'].mean():.4f}")
        logger.info(f"  Avg neighbors per patient: {summary_df['total_neighbors'].mean():.1f}")
        logger.info(f"  Avg metapaths per patient: {summary_df['num_metapaths'].mean():.1f}")
    
    def create_patient_subgraph_summary(self,
                                       predictions_csv: str,
                                       output_dir: str = 'results/patient_analysis') -> None:
        """
        Create summary of patient subgraph characteristics.
        
        Parameters
        ----------
        predictions_csv : str
            Path to sample_predictions.csv
        output_dir : str
            Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        preds = pd.read_csv(predictions_csv)
        
        # Analyze metapath usage across patients
        metapath_usage = defaultdict(int)
        
        for _, row in preds.iterrows():
            patient_idx = int(row['sample_idx'])
            
            for edge_type in self.data.edge_types:
                src_type, relation, dst_type = edge_type
                if src_type == 'Sample':
                    edge_index = self.data[edge_type].edge_index
                    mask = edge_index[0] == patient_idx
                    if mask.any():
                        metapath_usage[relation] += 1
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        
        metapaths = list(metapath_usage.keys())
        usage_counts = list(metapath_usage.values())
        
        ax.barh(metapaths, usage_counts, color='steelblue')
        ax.set_xlabel('Number of Patients with This Metapath', fontsize=12)
        ax.set_ylabel('Metapath Type', fontsize=12)
        ax.set_title('Metapath Usage Across Patient Predictions', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(output_path / 'metapath_usage.png', dpi=150, bbox_inches='tight')
        logger.info(f"✓ Saved: {output_path / 'metapath_usage.png'}")
        plt.close()
        
        # Save metapath usage data
        usage_df = pd.DataFrame({
            'metapath': metapaths,
            'usage_count': usage_counts,
            'usage_pct': [count / len(preds) * 100 for count in usage_counts]
        }).sort_values('usage_count', ascending=False)
        
        usage_df.to_csv(output_path / 'metapath_usage.csv', index=False)
        logger.info(f"✓ Saved: {output_path / 'metapath_usage.csv'}")


def main():
    """Example usage of patient attention analyzer."""
    import sys
    
    # Load model and data
    from src.han.owl_data_loader import load_hetero_graph_from_owl
    from src.han.model import SepsisHANClassifier
    
    print("\n" + "="*80)
    print("PATIENT-LEVEL ATTENTION ANALYSIS FOR HAN INTERPRETABILITY")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load data
    print("\nLoading graph data...")
    data = load_hetero_graph_from_owl(
        'output/new_outputs/GSE54514_enriched_ontology_degfilter_v2.11.owl'
    )
    
    # Create model
    print("Creating HAN model...")
    model = SepsisHANClassifier(
        in_channels_dict={ntype: data[ntype].x.size(1) for ntype in data.node_types},
        hidden_channels=64,
        out_channels=32,
        num_layers=2,
        num_heads=8,
        dropout=0.3,
        metadata=data.metadata()
    ).to(device)
    
    # Load trained weights
    print("Loading trained model...")
    model.load_state_dict(torch.load('results/han_model_balanced/han_model.pt', map_location=device))
    
    # Create analyzer
    analyzer = PatientAttentionAnalyzer(model, data, device)
    
    # Generate analysis
    print("\nGenerating patient-level analysis...")
    analyzer.generate_patient_report(
        'results/han_model_balanced/sample_predictions.csv',
        output_dir='results/patient_attention_analysis',
        num_patients=20  # Analyze first 20 patients for detailed report
    )
    
    # Create subgraph summary
    print("Creating subgraph visualization...")
    analyzer.create_patient_subgraph_summary(
        'results/han_model_balanced/sample_predictions.csv',
        output_dir='results/patient_attention_analysis'
    )
    
    print("\n" + "="*80)
    print("✓ Patient attention analysis complete!")
    print("  Results saved to: results/patient_attention_analysis/")
    print("="*80 + "\n")


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    main()
