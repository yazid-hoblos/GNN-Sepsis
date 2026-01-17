"""
OWL-based Heterogeneous Graph Data Loader for Sepsis Prediction

Directly parses OWL ontology files into PyTorch Geometric HeteroData format,
eliminating dependency on intermediate CSV files.
"""

import xml.etree.ElementTree as ET
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging

from torch_geometric.data import HeteroData


logger = logging.getLogger(__name__)


class OWLHeteroGraphBuilder:
    """
    Build heterogeneous graph directly from OWL ontology file.
    
    Input:
    - OWL file (e.g., GSE54514_enriched_ontology_degfilterv2.11.owl)
    
    Output:
    - HeteroData with node types, edge types, and patient labels
    """
    
    # Node type prefixes for classification
    NODE_TYPE_PREFIXES = {
        'Sample_': 'Sample',
        'GSM': 'Sample',
        'Protein_': 'Protein',
        'Gene_': 'Gene',
        'Pathway_': 'Pathway',
        'GO_': 'GO_Term',
        'GO:': 'GO_Term',
        'Reaction_': 'Reaction',
        'R-HSA': 'Pathway',
        'REACT:': 'Pathway',
        'MONDO:': 'Disease',
        'HP:': 'Phenotype',
    }
    
    def __init__(self, owl_path: str):
        """
        Initialize OWL graph builder.
        
        Parameters
        ----------
        owl_path : str
            Path to OWL ontology file
        """
        self.owl_path = Path(owl_path)
        if not self.owl_path.exists():
            raise FileNotFoundError(f"OWL file not found: {owl_path}")
        
        logger.info(f"Loading OWL ontology from {owl_path}")
        
        # Parse OWL file
        self.tree = ET.parse(str(self.owl_path))
        self.root = self.tree.getroot()
        
        # OWL/RDF namespaces
        self.ns = {
            'owl': 'http://www.w3.org/2002/07/owl#',
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
            'obo': 'http://purl.obolibrary.org/obo/',
        }
        
        # Initialize mappings
        self.node_id_to_idx = {}  # Maps (node_type, node_id) -> index within that type
        self.idx_to_node_id = {}
        self.node_type_to_indices = defaultdict(list)
        self.node_types = set()
        self.edge_types = set()
        
        # Node attributes storage
        self.node_attributes = defaultdict(lambda: defaultdict(dict))  # node_id -> {attr_name: attr_value}
        
    def _clean_uri(self, uri: str) -> str:
        """Clean URI by removing namespace prefixes."""
        if not uri:
            return ''
        # Remove common prefixes
        for prefix in ['http://www.w3.org/2002/07/owl#',
                      'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
                      'http://www.w3.org/2000/01/rdf-schema#',
                      'http://purl.obolibrary.org/obo/']:
            uri = uri.replace(prefix, '')
        # Remove leading # and {} namespace markers
        uri = uri.lstrip('#')
        uri = uri.replace('{go.owl#}', '')
        uri = uri.replace('{', '').replace('}', '')
        return uri
    
    def _extract_node_type(self, node_id: str) -> str:
        """Extract node type from node ID prefix."""
        for prefix, node_type in self.NODE_TYPE_PREFIXES.items():
            if node_id.startswith(prefix):
                return node_type
        # Default to 'Entity' if no prefix matches
        return 'Entity'
    
    def _parse_individuals(self) -> Dict[str, str]:
        """
        Parse OWL individuals and extract node attributes.
        
        Returns
        -------
        node_to_type : dict
            Mapping of node_id -> node_type
        """
        logger.info("Parsing OWL individuals...")
        
        all_individuals = self.root.findall('.//owl:NamedIndividual', self.ns)
        logger.info(f"Found {len(all_individuals)} individuals")
        
        node_to_type = {}
        
        for individual in all_individuals:
            # Extract node ID from rdf:about attribute
            node_uri = individual.get(f'{{{self.ns["rdf"]}}}about', '')
            if not node_uri:
                continue
            
            node_id = self._clean_uri(node_uri)
            node_type = self._extract_node_type(node_id)
            node_to_type[node_id] = node_type
            
            # Extract all data properties for this individual
            for child in individual:
                prop_name = self._clean_uri(child.tag)
                
                # Skip RDF type declarations
                if prop_name in ['type', 'Class', 'NamedIndividual']:
                    continue
                
                # Extract property value
                prop_value = child.text if child.text else child.get(f'{{{self.ns["rdf"]}}}resource', '')
                if prop_value:
                    prop_value = self._clean_uri(prop_value)
                    self.node_attributes[node_id][prop_name] = prop_value
        
        logger.info(f"Extracted {len(node_to_type)} nodes with attributes")
        
        return node_to_type
    
    def _parse_edges(self) -> List[Tuple[str, str, str]]:
        """
        Parse OWL object properties to extract edges.
        
        Returns
        -------
        edges : list of (subject, predicate, object) tuples
        """
        logger.info("Parsing OWL edges...")
        
        edges = []
        all_individuals = self.root.findall('.//owl:NamedIndividual', self.ns)
        
        for individual in all_individuals:
            subject = self._clean_uri(individual.get(f'{{{self.ns["rdf"]}}}about', ''))
            if not subject:
                continue
            
            # Find all object properties (relationships)
            for child in individual:
                # Clean the tag to get predicate name
                tag = child.tag
                # Remove namespace first
                if '}' in tag:
                    tag = tag.split('}', 1)[-1]
                # Clean remaining URI components
                predicate = self._clean_uri(tag)
                
                # Skip data properties and type declarations
                if predicate in ['type', 'label', 'comment', 'hasDiseaseStatus', 'hasValue', '']:
                    continue
                
                # Extract object (target node)
                obj = child.get(f'{{{self.ns["rdf"]}}}resource', '')
                if obj:
                    obj = self._clean_uri(obj)
                    # Sanitize predicate name for PyTorch (no dots, no special chars)
                    predicate = predicate.replace('.', '_').replace(':', '_').replace('#', '_')
                    edges.append((subject, predicate, obj))
        
        logger.info(f"Extracted {len(edges)} edges")
        
        return edges
    
    def _extract_disease_label(self, node_id: str) -> Optional[int]:
        """
        Extract disease label for sample nodes.
        
        Returns
        -------
        label : int or None
            0 = Septic, 1 = Healthy, None = Not a sample or unknown status
        """
        if not node_id.startswith('Sample_') and not node_id.startswith('GSM'):
            return None
        
        # Check node attributes for disease status
        attrs = self.node_attributes.get(node_id, {})
        
        # Try multiple attribute names (case-insensitive)
        for key in attrs.keys():
            if 'disease' in key.lower() or 'status' in key.lower():
                status = str(attrs[key]).lower().strip()
                if 'healthy' in status or 'control' in status:
                    return 1
                elif 'septic' in status or 'sepsis' in status:
                    return 0
        
        return None
    
    def _build_node_mappings(self, node_to_type: Dict[str, str]):
        """
        Build node indices and type mappings.
        
        Parameters
        ----------
        node_to_type : dict
            Mapping of node_id -> node_type
        """
        logger.info("Building node mappings...")
        
        # Group nodes by type
        nodes_by_type = defaultdict(list)
        for node_id, node_type in sorted(node_to_type.items()):
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
    
    def _create_node_features(self) -> Dict[str, torch.Tensor]:
        """
        Create feature tensors for each node type.
        Uses one-hot encoding if no numeric features available.
        
        Returns
        -------
        node_features_dict : dict
            Mapping of node_type -> node features tensor
        """
        logger.info("Creating node feature tensors...")
        
        node_features_dict = {}
        
        for node_type in self.node_types:
            # Get all nodes of this type
            node_indices = sorted(
                [local_idx for (nt, _), local_idx in self.node_id_to_idx.items() if nt == node_type]
            )
            num_nodes = len(node_indices)
            
            # For now, use one-hot encoding (can be enhanced with embeddings later)
            # Each node gets a unique one-hot vector within its type
            features = torch.eye(num_nodes, dtype=torch.float)
            
            node_features_dict[node_type] = features
            logger.info(f"  {node_type}: {features.shape}")
        
        return node_features_dict
    
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
        
        sample_nodes = sorted([node_id for (node_type, node_id) in self.node_id_to_idx.keys() 
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
    
    def _build_edge_indices(self, edges: List[Tuple[str, str, str]]) -> Dict[Tuple[str, str, str], torch.Tensor]:
        """
        Build edge indices for heterogeneous graph.
        
        Parameters
        ----------
        edges : list
            List of (subject, predicate, object) tuples
        
        Returns
        -------
        edge_index_dict : dict
            Mapping of (src_type, edge_type, dst_type) -> edge_index tensor
        """
        logger.info("Building edge indices...")
        
        # Group edges by type
        edges_by_type = defaultdict(list)
        
        for subject, predicate, obj in edges:
            # Skip if either node not in mappings
            src_type = self._extract_node_type(subject)
            dst_type = self._extract_node_type(obj)
            
            if (src_type, subject) not in self.node_id_to_idx:
                continue
            if (dst_type, obj) not in self.node_id_to_idx:
                continue
            
            src_idx = self.node_id_to_idx[(src_type, subject)]
            dst_idx = self.node_id_to_idx[(dst_type, obj)]
            
            edge_type = (src_type, predicate, dst_type)
            edges_by_type[edge_type].append((src_idx, dst_idx))
        
        # Convert to PyTorch tensors
        edge_index_dict = {}
        for edge_type, edge_list in edges_by_type.items():
            if len(edge_list) > 0:
                src_indices = [src for src, dst in edge_list]
                dst_indices = [dst for src, dst in edge_list]
                edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
                edge_index_dict[edge_type] = edge_index
                self.edge_types.add(edge_type)
                logger.info(f"  {edge_type}: {edge_index.shape[1]} edges")
        
        return edge_index_dict
    
    def _create_train_val_test_masks(self, 
                                     num_samples: int,
                                     valid_mask: torch.BoolTensor,
                                     train_ratio: float = 0.7,
                                     val_ratio: float = 0.1) -> Tuple[torch.BoolTensor, torch.BoolTensor, torch.BoolTensor]:
        """
        Create train/val/test masks for sample nodes.
        
        Parameters
        ----------
        num_samples : int
            Total number of sample nodes
        valid_mask : torch.BoolTensor
            Mask indicating which samples have valid labels
        train_ratio : float
            Fraction for training
        val_ratio : float
            Fraction for validation
        
        Returns
        -------
        train_mask, val_mask, test_mask : torch.BoolTensor
            Boolean masks for train/val/test splits
        """
        # Initialize masks
        train_mask = torch.zeros(num_samples, dtype=torch.bool)
        val_mask = torch.zeros(num_samples, dtype=torch.bool)
        test_mask = torch.zeros(num_samples, dtype=torch.bool)
        
        # Get indices of valid samples
        valid_indices = torch.where(valid_mask)[0].numpy()
        
        # Shuffle and split
        np.random.seed(42)
        np.random.shuffle(valid_indices)
        
        n_valid = len(valid_indices)
        n_train = int(n_valid * train_ratio)
        n_val = int(n_valid * val_ratio)
        
        train_indices = valid_indices[:n_train]
        val_indices = valid_indices[n_train:n_train + n_val]
        test_indices = valid_indices[n_train + n_val:]
        
        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True
        
        logger.info(f"Split sizes - Train: {n_train}, Val: {n_val}, Test: {len(test_indices)}")
        
        return train_mask, val_mask, test_mask
    
    def build_hetero_data(self, 
                         train_ratio: float = 0.7, 
                         val_ratio: float = 0.1) -> HeteroData:
        """
        Build complete heterogeneous graph.
        
        Parameters
        ----------
        train_ratio : float
            Training set ratio
        val_ratio : float
            Validation set ratio
        
        Returns
        -------
        data : HeteroData
            Heterogeneous graph ready for HAN training
        """
        logger.info("\n" + "="*80)
        logger.info("Building Heterogeneous Graph from OWL")
        logger.info("="*80)
        
        # Step 1: Parse individuals and edges
        node_to_type = self._parse_individuals()
        edges = self._parse_edges()
        
        # Step 2: Build node mappings
        self._build_node_mappings(node_to_type)
        
        # Step 3: Create node features
        node_features_dict = self._create_node_features()
        
        # Step 4: Extract sample labels
        y, valid_mask = self._extract_sample_labels()
        
        # Step 5: Build edge indices
        edge_index_dict = self._build_edge_indices(edges)
        
        # Step 6: Create train/val/test masks
        num_samples = len([k for k in self.node_id_to_idx.keys() if k[0] == 'Sample'])
        train_mask, val_mask, test_mask = self._create_train_val_test_masks(
            num_samples, valid_mask, train_ratio, val_ratio
        )
        
        # Step 7: Build HeteroData object
        logger.info("\nAssembling HeteroData...")
        data = HeteroData()
        
        # Add node features
        for node_type, features in node_features_dict.items():
            data[node_type].x = features
            data[node_type].num_nodes = features.size(0)
        
        # Add edges
        for edge_type, edge_index in edge_index_dict.items():
            data[edge_type].edge_index = edge_index
        
        # Add sample labels and masks
        data['Sample'].y = y
        data['Sample'].train_mask = train_mask
        data['Sample'].val_mask = val_mask
        data['Sample'].test_mask = test_mask
        
        logger.info("\n" + "="*80)
        logger.info("Graph Construction Complete")
        logger.info("="*80)
        logger.info(f"Node types: {list(data.node_types)}")
        logger.info(f"Edge types: {len(list(data.edge_types))} types")
        logger.info(f"Total nodes: {sum(data[node_type].num_nodes for node_type in data.node_types)}")
        logger.info(f"Total edges: {sum(data[edge_type].num_edges for edge_type in data.edge_types)}")
        
        return data


def load_hetero_graph_from_owl(owl_path: str,
                                train_ratio: float = 0.7,
                                val_ratio: float = 0.1) -> HeteroData:
    """
    Convenience function to load heterogeneous graph from OWL file.
    
    Parameters
    ----------
    owl_path : str
        Path to OWL ontology file
    train_ratio : float
        Training set ratio
    val_ratio : float
        Validation set ratio
    
    Returns
    -------
    data : HeteroData
        Heterogeneous graph ready for HAN training
    """
    builder = OWLHeteroGraphBuilder(owl_path)
    data = builder.build_hetero_data(train_ratio=train_ratio, val_ratio=val_ratio)
    return data


if __name__ == '__main__':
    import sys
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Default OWL path
    owl_path = 'output/new_outputs/GSE54514_enriched_ontology_degfilter_v2.11.owl'
    
    if len(sys.argv) > 1:
        owl_path = sys.argv[1]
    
    print(f"\nLoading graph from: {owl_path}")
    data = load_hetero_graph_from_owl(owl_path)
    
    print("\n" + "="*80)
    print("✓ Success! Heterogeneous graph is ready for HAN training.")
    print("="*80)
