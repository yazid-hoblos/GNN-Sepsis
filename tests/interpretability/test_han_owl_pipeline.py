#!/usr/bin/env python
"""
Quick test script for HAN OWL pipeline
Tests: OWL loading → HAN data preparation → (optional) small training run
"""

import torch
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_owl_loader():
    """Test OWL data loader."""
    logger.info("="*80)
    logger.info("TEST 1: OWL Data Loader")
    logger.info("="*80)
    
    from src.han.owl_data_loader import load_hetero_graph_from_owl
    
    owl_path = 'output/new_outputs/GSE54514_enriched_ontology_degfilter_v2.11.owl'
    
    logger.info(f"Loading graph from: {owl_path}")
    data = load_hetero_graph_from_owl(owl_path, train_ratio=0.7, val_ratio=0.1)
    
    logger.info("\n✓ OWL loader test passed!")
    logger.info(f"  Node types: {list(data.node_types)}")
    logger.info(f"  Edge types: {len(list(data.edge_types))} types")
    logger.info(f"  Sample nodes: {data['Sample'].num_nodes}")
    logger.info(f"  Train samples: {data['Sample'].train_mask.sum().item()}")
    logger.info(f"  Val samples: {data['Sample'].val_mask.sum().item()}")
    logger.info(f"  Test samples: {data['Sample'].test_mask.sum().item()}")
    
    return data


def test_model_creation(data):
    """Test HAN model creation."""
    logger.info("\n" + "="*80)
    logger.info("TEST 2: HAN Model Creation")
    logger.info("="*80)
    
    from src.han.model import SepsisHANClassifier
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Get input channels for each node type
    in_channels_dict = {
        node_type: data[node_type].x.size(1)
        for node_type in data.node_types
    }
    
    logger.info(f"Input channels: {in_channels_dict}")
    
    # Create model
    model = SepsisHANClassifier(
        in_channels_dict=in_channels_dict,
        metadata=data.metadata(),
        hidden_channels=64,
        out_channels=32,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"\n✓ Model created successfully!")
    logger.info(f"  Parameters: {num_params:,}")
    
    return model, device


def test_forward_pass(model, data, device):
    """Test forward pass."""
    logger.info("\n" + "="*80)
    logger.info("TEST 3: Forward Pass")
    logger.info("="*80)
    
    data = data.to(device)
    model.eval()
    
    with torch.no_grad():
        outputs = model(data.x_dict, data.edge_index_dict)
        
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
    
    logger.info(f"\n✓ Forward pass successful!")
    logger.info(f"  Logits shape: {logits.shape}")
    logger.info(f"  Predictions: {logits.argmax(dim=1)[:10].cpu().tolist()}")
    
    return logits


def test_attention_extraction(model, data, device):
    """Test attention extraction."""
    logger.info("\n" + "="*80)
    logger.info("TEST 4: Attention Extraction")
    logger.info("="*80)
    
    from src.han.han_attention_extraction import HANAttentionExtractor
    
    extractor = HANAttentionExtractor(model, data, device=str(device))
    
    logger.info("Extracting attention weights...")
    results = extractor.extract_attention(return_embeddings=True)
    
    logger.info(f"\n✓ Attention extraction successful!")
    logger.info(f"  Predictions: {results['predictions'][:10]}")
    logger.info(f"  Probabilities shape: {results['probabilities'].shape}")
    logger.info(f"  Attention layers captured: {len(results['attention_weights'])}")
    
    # Try to compute edge type importance
    try:
        df_edge_importance = extractor.compute_edge_type_importance()
        if not df_edge_importance.empty:
            logger.info(f"\n  Top 5 Edge Types by Attention:")
            for _, row in df_edge_importance.head(5).iterrows():
                logger.info(f"    {row['edge_type']}: {row['mean_attention']:.4f}")
    except Exception as e:
        logger.warning(f"  Edge importance computation skipped: {e}")
    
    return extractor


def main():
    """Run all tests."""
    logger.info("\n" + "="*80)
    logger.info("HAN OWL PIPELINE TEST SUITE")
    logger.info("="*80)
    
    try:
        # Test 1: Load data
        data = test_owl_loader()
        
        # Test 2: Create model
        model, device = test_model_creation(data)
        
        # Test 3: Forward pass
        logits = test_forward_pass(model, data, device)
        
        # Test 4: Attention extraction
        extractor = test_attention_extraction(model, data, device)
        
        logger.info("\n" + "="*80)
        logger.info("✓ ALL TESTS PASSED!")
        logger.info("="*80)
        logger.info("\nNext steps:")
        logger.info("  1. Train HAN model: python src/han/run_han.py")
        logger.info("  2. Analyze attention: Results will be in results/han_model/")
        logger.info("  3. Integrate with SHAP interpretability pipeline")
        
    except Exception as e:
        logger.error(f"\n✗ TEST FAILED: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
