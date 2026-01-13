"""
Main HAN Training Script for Sepsis Prediction

Trains HAN model end-to-end and performs attention analysis.
"""

import argparse
import logging
import torch
from pathlib import Path

from src.han.data_loader import load_hetero_graph
from src.han.model import SepsisHANClassifier, HANTrainer
from src.han.attention_analysis import AttentionAnalyzer


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(args):
    """Main training pipeline."""
    
    logger.info("="*80)
    logger.info("HAN Model Training for Sepsis Prediction")
    logger.info("="*80)
    
    # Device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Step 1: Load data
    logger.info("\n" + "="*80)
    logger.info("Step 1: Loading Heterogeneous Graph")
    logger.info("="*80)
    
    data = load_hetero_graph(
        data_dir=args.data_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    
    # Move to device
    data = data.to(device)
    
    logger.info(f"\nGraph loaded successfully!")
    logger.info(f"Node types: {list(data.node_types)}")
    logger.info(f"Edge types: {list(data.edge_types)}")
    
    # Step 2: Create model
    logger.info("\n" + "="*80)
    logger.info("Step 2: Creating HAN Model")
    logger.info("="*80)
    
    in_channels_dict = {
        node_type: data[node_type].x.size(1)
        for node_type in data.node_types
    }
    
    logger.info(f"Input channels: {in_channels_dict}")
    
    model = SepsisHANClassifier(
        in_channels_dict=in_channels_dict,
        metadata=data.metadata(),
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")
    
    # Step 3: Create trainer
    logger.info("\n" + "="*80)
    logger.info("Step 3: Setting up Trainer")
    logger.info("="*80)
    
    trainer = HANTrainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    # Step 4: Train
    logger.info("\n" + "="*80)
    logger.info("Step 4: Training")
    logger.info("="*80)
    
    history = trainer.train(
        data=data,
        epochs=args.epochs,
        early_stopping_patience=args.patience,
        verbose=args.verbose,
    )
    
    # Step 5: Attention Analysis
    logger.info("\n" + "="*80)
    logger.info("Step 5: Attention Analysis")
    logger.info("="*80)
    
    analyzer = AttentionAnalyzer(model, data, {}, {})
    analyzer.compute_predictions(return_embeddings=True)
    
    # Edge type importance
    logger.info("\nEdge Type Importance:")
    edge_imp = analyzer.get_edge_type_importance()
    logger.info(edge_imp.to_string())
    
    # Save edge importance
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    edge_imp.to_csv(output_dir / 'edge_type_importance.csv', index=False)
    logger.info(f"Saved: {output_dir / 'edge_type_importance.csv'}")
    
    # Visualizations
    logger.info("\nGenerating visualizations...")
    
    analyzer.plot_edge_type_importance(
        output_path=str(output_dir / 'edge_type_importance.png')
    )
    
    # Visualize a few patient subgraphs
    for sample_idx in range(min(3, data['Sample'].num_nodes)):
        if data['Sample'].valid_mask[sample_idx]:
            analyzer.plot_patient_subgraph(
                sample_idx=sample_idx,
                output_path=str(output_dir / f'patient_{sample_idx}_subgraph.png')
            )
    
    # Step 6: Save results
    logger.info("\n" + "="*80)
    logger.info("Step 6: Saving Results")
    logger.info("="*80)
    
    # Save model
    model_path = output_dir / 'han_model.pt'
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved model: {model_path}")
    
    # Save history
    import json
    history_dict = {
        'train_loss': history['train_loss'],
        'val_acc': history['val_acc'],
        'test_acc': float(history['test_acc']),
        'test_loss': float(history['test_loss']),
    }
    
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history_dict, f, indent=2)
    logger.info(f"Saved history: {output_dir / 'training_history.json'}")
    
    logger.info("\n" + "="*80)
    logger.info("Training Complete!")
    logger.info("="*80)
    logger.info(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train HAN model for sepsis prediction'
    )
    
    # Data arguments
    parser.add_argument('--data_dir',
                       type=str,
                       default='models/executions/GSE54514_enriched_ontology_degfilterv2.11',
                       help='Path to graph data directory')
    
    # Model arguments
    parser.add_argument('--hidden_channels',
                       type=int,
                       default=64,
                       help='Hidden dimension')
    parser.add_argument('--out_channels',
                       type=int,
                       default=32,
                       help='Output dimension')
    parser.add_argument('--num_layers',
                       type=int,
                       default=2,
                       help='Number of HAN layers')
    parser.add_argument('--num_heads',
                       type=int,
                       default=8,
                       help='Number of attention heads')
    parser.add_argument('--dropout',
                       type=float,
                       default=0.1,
                       help='Dropout probability')
    
    # Training arguments
    parser.add_argument('--epochs',
                       type=int,
                       default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate',
                       type=float,
                       default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay',
                       type=float,
                       default=1e-5,
                       help='Weight decay')
    parser.add_argument('--patience',
                       type=int,
                       default=20,
                       help='Early stopping patience')
    parser.add_argument('--verbose',
                       type=int,
                       default=10,
                       help='Logging frequency')
    
    # Data split arguments
    parser.add_argument('--train_ratio',
                       type=float,
                       default=0.7,
                       help='Training set ratio')
    parser.add_argument('--val_ratio',
                       type=float,
                       default=0.1,
                       help='Validation set ratio')
    
    # Misc arguments
    parser.add_argument('--device',
                       type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device (cuda or cpu)')
    parser.add_argument('--output_dir',
                       type=str,
                       default='results/han_model',
                       help='Output directory for results')
    
    args = parser.parse_args()
    main(args)
