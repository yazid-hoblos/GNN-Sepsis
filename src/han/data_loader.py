"""
Heterogeneous Graph Data Loader for Sepsis Prediction

Converts node_features.csv and edge_attributes.csv into PyTorch Geometric
HeteroData format suitable for HAN training.
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging

from torch_geometric.data import HeteroData


logger = logging.getLogger(__name__)


class HeteroGraphBuilder:
    """
    Build heterogeneous graph from node and edge CSV files.
    
    Input files:
    - node_features.csv: (node_id, name_feature, value_feature)
    - edge_attributes.csv: (subject, predicate, object, nameAnnotation, value)
    
    Output:
    - HeteroData with node types, edge types, and patient labels
    """
    
    # Node type prefixes
    NODE_TYPE_PREFIXES = {
        'Sample_': 'Sample',
        'Protein_': 'Protein',
        'Gene_': 'Gene',
        'Pathway_': 'Pathway',
        'GO_': 'GO_Term',
        'Reaction_': 'Reaction',
    }
    
    def __init__(self, 
                 node_features_path: str,
                 edge_attributes_path: str,
                 entity_mapping_path: Optional[str] = None,
                 relation_mapping_path: Optional[str] = None):
        """
        Initialize graph builder.
        
        Parameters
        ----------
        node_features_path : str
            Path to node_features.csv (node properties including disease status)
        edge_attributes_path : str
            Path to edge_attributes.csv (graph edges)
        entity_mapping_path : str, optional
            Path to entity mapping (for reference)
        relation_mapping_path : str, optional
            Path to relation mapping (for reference)
        """
        logger.info(f"Loading node features from {node_features_path}")
        self.node_features = pd.read_csv(node_features_path, dtype={'node_id': str})
        
        logger.info(f"Loading edge attributes from {edge_attributes_path}")
        self.edge_attributes = pd.read_csv(edge_attributes_path, dtype={'subject': str, 'object': str})
        
        self.entity_mapping = None
        if entity_mapping_path and Path(entity_mapping_path).exists():
            self.entity_mapping = pd.read_csv(entity_mapping_path)
            
        self.relation_mapping = None
        if relation_mapping_path and Path(relation_mapping_path).exists():
            self.relation_mapping = pd.read_csv(relation_mapping_path)
        
        # Initialize node mappings
        self.node_id_to_idx = {}  # Maps (node_type, node_name) -> index within that type
        self.idx_to_node_id = {}
        self.node_type_to_indices = defaultdict(list)
        self.node_types = set()
        self.edge_types = set()
        
    def _extract_node_type(self, node_id: str) -> str:
        """Extract node type from node ID prefix."""
        for prefix, node_type in self.NODE_TYPE_PREFIXES.items():
            if node_id.startswith(prefix):
                return node_type
        # Default to generic 'Entity' if no prefix matches
        return 'Entity'
    
    def _extract_disease_label(self, node_id: str) -> Optional[int]:
        """
        Extract disease label for sample nodes.
        
        Returns
        -------
        label : int or None
            0 = Septic, 1 = Healthy, None = Not a sample or unknown status
        """
        if not node_id.startswith('Sample_'):
            return None
            
        # Look up node features
        node_rows = self.node_features[self.node_features['node_id'] == node_id]
        
        for _, row in node_rows.iterrows():
            if row['name_feature'] == 'hasDiseaseStatus':
                status = row['value_feature'].lower().strip()
                if status == 'healthy':
                    return 1
                elif status == 'septic':
                    return 0
        
        return None
    
    def _build_node_mappings(self) -> Dict[str, torch.Tensor]:
        """
        Build node indices and type mappings.
        
        Returns
        -------
        node_features_dict : dict
            Mapping of node_type -> node features tensor
        """
        logger.info("Building node mappings...")
        
        # First pass: extract all unique nodes from features
        unique_nodes = self.node_features['node_id'].unique()
        
        # Second pass: extract nodes from edges
        unique_nodes_from_edges = set(self.edge_attributes['subject'].unique()) | \
                                  set(self.edge_attributes['object'].unique())
        
        all_nodes = set(unique_nodes) | unique_nodes_from_edges
        logger.info(f"Total unique nodes: {len(all_nodes)}")
        
        # Group nodes by type and assign indices
        nodes_by_type = defaultdict(list)
        for node_id in sorted(all_nodes):
            node_type = self._extract_node_type(node_id)
            nodes_by_type[node_type].append(node_id)
        
        # Create mappings
        global_idx = 0
        for node_type in sorted(nodes_by_type.keys()):
            self.node_types.add(node_type)
            node_ids = sorted(nodes_by_type[node_type])
            
            for local_idx, node_id in enumerate(node_ids):
                self.node_id_to_idx[(node_type, node_id)] = local_idx
                self.idx_to_node_id[(node_type, local_idx)] = node_id
                self.node_type_to_indices[node_type].append(global_idx)
                global_idx += 1
            
            logger.info(f"  {node_type}: {len(node_ids)} nodes")
        
        return None
    
    def _extract_sample_labels(self) -> Tuple[torch.Tensor, torch.BoolTensor]:
        """
        Extract disease labels for sample nodes.
        
        Returns
        -------
        y : torch.Tensor
            Disease labels (0=septic, 1=healthy)
        valid_mask : torch.BoolTensor
            Mask for samples with valid labels
        """
        logger.info("Extracting sample labels...")
        
        sample_nodes = sorted([node_id for node_type, node_id in self.node_id_to_idx.keys() 
                              if node_type == 'Sample'])
        
        labels = []
        valid_mask = []
        
        for node_id in sample_nodes:
            label = self._extract_disease_label(node_id)
            if label is not None:
                labels.append(label)
                valid_mask.append(True)
            else:
                labels.append(0)  # Placeholder
                valid_mask.append(False)
        
        y = torch.tensor(labels, dtype=torch.long)
        valid_mask = torch.tensor(valid_mask, dtype=torch.bool)
        
        n_septic = (y == 0).sum().item()
        n_healthy = (y == 1).sum().item()
        logger.info(f"  Septic samples: {n_septic}")
        logger.info(f"  Healthy samples: {n_healthy}")
        logger.info(f"  Samples with valid labels: {valid_mask.sum().item()}")
        
        return y, valid_mask
    
    def _build_edge_indices(self) -> Dict[Tuple[str, str, str], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Build edge indices for each edge type.
        
        Returns
        -------
        edge_index_dict : dict
            Mapping of (source_type, edge_type, target_type) -> (source_idx, target_idx)
        """
        logger.info("Building edge indices...")
        
        edge_index_dict = {}
        edge_counts = defaultdict(int)
        
        for _, row in self.edge_attributes.iterrows():
            subject = row['subject']
            predicate = row['predicate']
            obj = row['object']
            
            source_type = self._extract_node_type(subject)
            target_type = self._extract_node_type(obj)
            
            # Skip if nodes not in mapping (shouldn't happen)
            if (source_type, subject) not in self.node_id_to_idx or \
               (target_type, obj) not in self.node_id_to_idx:
                continue
            
            source_idx = self.node_id_to_idx[(source_type, subject)]
            target_idx = self.node_id_to_idx[(target_type, obj)]
            
            edge_key = (source_type, predicate, target_type)
            self.edge_types.add(edge_key)
            
            if edge_key not in edge_index_dict:
                edge_index_dict[edge_key] = {'sources': [], 'targets': []}
            
            edge_index_dict[edge_key]['sources'].append(source_idx)
            edge_index_dict[edge_key]['targets'].append(target_idx)
            edge_counts[edge_key] += 1
        
        # Convert to tensors
        edge_index_tensors = {}
        for edge_key, indices in edge_index_dict.items():
            sources = torch.tensor(indices['sources'], dtype=torch.long)
            targets = torch.tensor(indices['targets'], dtype=torch.long)
            edge_index = torch.stack([sources, targets])
            edge_index_tensors[edge_key] = edge_index
        
        logger.info(f"Edge types: {len(self.edge_types)}")
        for edge_key, count in sorted(edge_counts.items()):
            logger.info(f"  {edge_key[0]} -[{edge_key[1]}]-> {edge_key[2]}: {count} edges")
        
        return edge_index_tensors
    
    def _create_node_features(self) -> Dict[str, torch.Tensor]:
        """
        Create node feature tensors.
        
        For now, uses one-hot encoding of node types + available features.
        
        Returns
        -------
        x_dict : dict
            Mapping of node_type -> feature tensor
        """
        logger.info("Creating node feature tensors...")
        
        x_dict = {}
        
        for node_type in sorted(self.node_types):
            num_nodes = len(self.node_type_to_indices[node_type])
            
            # Start with simple feature: node index (one-hot or embedding)
            # This can be enhanced with actual biological features
            x = torch.eye(num_nodes, dtype=torch.float32)
            
            x_dict[node_type] = x
            logger.info(f"  {node_type}: {x.shape}")
        
        return x_dict
    
    def build_hetero_data(self, train_ratio: float = 0.7, val_ratio: float = 0.1) -> HeteroData:
        """
        Build complete HeteroData object.
        
        Parameters
        ----------
        train_ratio : float
            Fraction of samples for training
        val_ratio : float
            Fraction of samples for validation
            (test_ratio = 1 - train_ratio - val_ratio)
        
        Returns
        -------
        data : HeteroData
            PyTorch Geometric heterogeneous data object
        """
        logger.info("="*80)
        logger.info("Building Heterogeneous Graph")
        logger.info("="*80)
        
        # Step 1: Build node mappings
        self._build_node_mappings()
        
        # Step 2: Extract labels and create masks
        y, valid_labels_mask = self._extract_sample_labels()
        
        # Step 3: Build edge indices
        edge_index_dict = self._build_edge_indices()
        
        # Step 4: Create node features
        x_dict = self._create_node_features()
        
        # Step 5: Create HeteroData object
        data = HeteroData()
        
        # Add node features
        for node_type, x in x_dict.items():
            data[node_type].x = x
        
        # Add sample labels
        data['Sample'].y = y
        data['Sample'].valid_mask = valid_labels_mask
        
        # Add edge indices
        for edge_key, edge_index in edge_index_dict.items():
            data[edge_key].edge_index = edge_index
        
        # Step 6: Create train/val/test masks
        num_samples = data['Sample'].num_nodes
        indices = torch.arange(num_samples)
        
        # Shuffle indices
        perm = torch.randperm(num_samples)
        indices = indices[perm]
        
        # Split indices
        train_size = int(num_samples * train_ratio)
        val_size = int(num_samples * val_ratio)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
        
        # Create masks
        train_mask = torch.zeros(num_samples, dtype=torch.bool)
        val_mask = torch.zeros(num_samples, dtype=torch.bool)
        test_mask = torch.zeros(num_samples, dtype=torch.bool)
        
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        
        # Only keep valid samples in masks
        train_mask &= valid_labels_mask
        val_mask &= valid_labels_mask
        test_mask &= valid_labels_mask
        
        data['Sample'].train_mask = train_mask
        data['Sample'].val_mask = val_mask
        data['Sample'].test_mask = test_mask
        
        logger.info("\nDataset split:")
        logger.info(f"  Train: {train_mask.sum().item()} samples")
        logger.info(f"  Val: {val_mask.sum().item()} samples")
        logger.info(f"  Test: {test_mask.sum().item()} samples")
        
        logger.info("\nHeteroData Summary:")
        logger.info(f"  Node types: {list(data.node_types)}")
        logger.info(f"  Edge types: {list(data.edge_types)}")
        logger.info(f"  Total nodes: {sum(data[node_type].num_nodes for node_type in data.node_types)}")
        logger.info(f"  Total edges: {sum(data[edge_type].num_edges for edge_type in data.edge_types)}")
        
        return data


def load_hetero_graph(data_dir: str, 
                      train_ratio: float = 0.7, 
                      val_ratio: float = 0.1) -> HeteroData:
    """
    Convenience function to load heterogeneous graph from standard directory.
    
    Parameters
    ----------
    data_dir : str
        Path to models/executions/GSE54514_enriched_ontology_degfilterv2.11/
    train_ratio : float
        Training set ratio
    val_ratio : float
        Validation set ratio
    
    Returns
    -------
    data : HeteroData
        Heterogeneous graph ready for HAN training
    """
    data_path = Path(data_dir)
    
    builder = HeteroGraphBuilder(
        node_features_path=str(data_path / 'node_features.csv'),
        edge_attributes_path=str(data_path / 'edge_attributes.csv'),
        entity_mapping_path=str(data_path / 'outputmodel_RGCN_entity_mapping.csv'),
        relation_mapping_path=str(data_path / 'outputmodel_RGCN_relation_mapping.csv'),
    )
    
    data = builder.build_hetero_data(train_ratio=train_ratio, val_ratio=val_ratio)
    
    return data


if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    # Load and build graph
    print("Loading data...")
    data = load_hetero_graph('models/executions/GSE54514_enriched_ontology_degfilterv2.11')
    
    print("\n" + "="*80)
    print("Success! Heterogeneous graph is ready for HAN training.")
    print("="*80)
