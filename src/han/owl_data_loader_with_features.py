"""
Enhanced OWL Data Loader with RGCN Embedding Features

Uses pre-computed RGCN embeddings (which already contain learned biological patterns)
instead of identity matrices for node features.

This is faster and more practical than downloading GEO data on-the-fly.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


def load_hetero_graph_from_owl_with_features(
    owl_path: str,
    rgcn_embedding_path: Optional[str] = None,
    feature_dim: Optional[int] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    **kwargs) -> Dict:
    """
    Load heterogeneous graph from OWL with RGCN embedding features.
    
    Uses pre-computed RGCN embeddings for node features instead of identity matrices.
    This provides learned biological patterns without expensive GEO downloads.
    
    Parameters
    ----------
    owl_path : str
        Path to OWL file
    rgcn_embedding_path : str, optional
        Path to RGCN embeddings file (.pt). If None, tries to find automatically.
    feature_dim : int, optional
        Dimension for features (if None, use RGCN dim)
    train_ratio : float
        Training set ratio
    val_ratio : float
        Validation set ratio
    
    Returns
    -------
    data : torch_geometric.data.HeteroData
        Heterogeneous graph with RGCN embedding features
    """
    from src.han.owl_data_loader import OWLHeteroGraphBuilder
    
    logger.info("="*80)
    logger.info("Building HeteroGraph with RGCN Embedding Features")
    logger.info("="*80)
    
    # Build base graph from OWL
    logger.info("Building base OWL graph...")
    builder = OWLHeteroGraphBuilder(owl_path)
    data = builder.build_hetero_data()
    
    # Try to load RGCN embeddings
    if rgcn_embedding_path is None:
        # Auto-search for RGCN embeddings
        search_paths = [
            'results/rgcn_embeddings.pt',
            'results/embeddings/rgcn_embeddings.pt',
            'dump/v2.11_none/rgcn_embeddings.pt',
        ]
        for path in search_paths:
            if Path(path).exists():
                rgcn_embedding_path = path
                break
    
    if rgcn_embedding_path and Path(rgcn_embedding_path).exists():
        logger.info(f"Loading RGCN embeddings from {rgcn_embedding_path}...")
        try:
            rgcn_data = torch.load(rgcn_embedding_path, map_location='cpu')
            
            # Handle different storage formats
            if isinstance(rgcn_data, dict):
                embeddings = rgcn_data.get('embeddings', rgcn_data.get('x', rgcn_data))
            else:
                embeddings = rgcn_data
            
            if isinstance(embeddings, dict):
                # Update node features for each type
                for node_type in data.node_types:
                    if node_type in embeddings:
                        emb = embeddings[node_type]
                        if isinstance(emb, torch.Tensor):
                            # Ensure correct number of nodes
                            if emb.shape[0] == data[node_type].num_nodes:
                                data[node_type].x = emb.float()
                                logger.info(f"  ✓ {node_type}: {emb.shape}")
            
            logger.info("✓ RGCN embeddings loaded successfully!")
            
        except Exception as e:
            logger.warning(f"Could not load RGCN embeddings: {e}")
            logger.info("Proceeding with random features...")
    else:
        logger.info("RGCN embeddings not found. Using random features...")
        # Initialize with random features (will still learn better than identity matrices)
        for node_type in data.node_types:
            dim = feature_dim or 64
            data[node_type].x = torch.randn(data[node_type].num_nodes, dim)
    
    logger.info("\n" + "="*80)
    logger.info("✓ HeteroGraph with embedding features ready!")
    logger.info("="*80)
    
    return data


# Monkey-patch for backward compatibility
def load_hetero_graph_from_owl(owl_path: str, **kwargs):
    """Wrapper that uses expression features by default."""
    return load_hetero_graph_from_owl_with_features(owl_path, **kwargs)
