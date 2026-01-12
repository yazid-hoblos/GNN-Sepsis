#!/usr/bin/env python3
"""
Retrain Knowledge Graph Embeddings using edge lists from models/executions

This script reads the edge attributes CSV files and retrains KGE models
to generate new embeddings for the GENIOMHE sepsis research project.
"""

import argparse
import os
import sys
import logging
import pandas as pd
import numpy as np
import torch
from pathlib import Path

# PyKEEN imports
from pykeen.models import ComplEx, RGCN, TransE, DistMult, RotatE, ConvE
from pykeen.training import SLCWATrainingLoop
from pykeen.triples import TriplesFactory
from pykeen.evaluation import RankBasedEvaluator
from pykeen.stoppers import EarlyStopper
from pykeen.optimizers import Adam

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Retrain Knowledge Graph Embeddings from edge lists")
    
    # Model parameters
    parser.add_argument('--model', type=str, default='ComplEx',
                       choices=['ComplEx', 'RGCN', 'TransE', 'DistMult', 'RotatE', 'ConvE'],
                       help='KGE model to train')
    parser.add_argument('--embedding_dim', type=int, default=100,
                       help='Embedding dimension')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=1000,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=512,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=50,
                       help='Early stopping patience')
    
    # Data parameters
    parser.add_argument('--dataset', type=str, default='GSE54514_enriched_ontology_degfilterv2.9',
                       help='Dataset identifier for edge file')
    parser.add_argument('--input_dir', type=str, default='../models/executions',
                       help='Input directory containing edge attributes CSV')
    parser.add_argument('--output_dir', type=str, default='./new_embeddings',
                       help='Output directory for retrained embeddings')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training')
    
    return parser.parse_args()


def setup_device(device_arg):
    """Setup computation device."""
    if device_arg == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_arg
    
    logger.info(f"Using device: {device}")
    return device


def setup_output_directory(output_dir, model_name, dataset):
    """Create output directory structure early."""
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")
    
    # Pre-define file paths
    file_paths = {
        'entity_embeddings': os.path.join(output_dir, f"{model_name}_entity_embeddings.npy"),
        'relation_embeddings': os.path.join(output_dir, f"{model_name}_relation_embeddings.npy"),
        'entity_mapping': os.path.join(output_dir, f"{model_name}_entity_mapping.csv"),
        'relation_mapping': os.path.join(output_dir, f"{model_name}_relation_mapping.csv"),
        'checkpoint': os.path.join(output_dir, f"{model_name}_checkpoint.pt"),
        'losses': os.path.join(output_dir, f"{model_name}_losses.csv")
    }
    
    return file_paths


def load_edge_data(input_dir, dataset):
    """Load edge data from CSV file in models/executions."""
    edge_file = os.path.join(input_dir, f"{dataset}_edge_attributes.csv")
    
    if not os.path.exists(edge_file):
        raise FileNotFoundError(f"Edge attributes file not found: {edge_file}")
    
    logger.info(f"Loading edge data from: {edge_file}")
    df = pd.read_csv(edge_file)
    
    logger.info(f"Loaded {len(df)} edges")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    # Extract triples (subject, predicate, object)
    triples = df[['subject', 'predicate', 'object']].values
    
    # Remove any rows with NaN values
    original_count = len(triples)
    triples_df = pd.DataFrame(triples, columns=['head', 'relation', 'tail'])
    triples_df = triples_df.dropna()
    triples = triples_df.values
    
    logger.info(f"Cleaned triples: {len(triples)} (removed {original_count - len(triples)} NaN entries)")
    
    # Show some statistics
    unique_entities = set(triples[:, 0]) | set(triples[:, 2])
    unique_relations = set(triples[:, 1])
    
    logger.info(f"Unique entities: {len(unique_entities)}")
    logger.info(f"Unique relations: {len(unique_relations)}")
    logger.info(f"Sample relations: {list(unique_relations)[:5]}")
    
    return triples


def create_model(model_name, triples_factory, embedding_dim, device):
    """Create KGE model instance."""
    model_kwargs = {
        'triples_factory': triples_factory,
        'embedding_dim': embedding_dim,
        'random_seed': 42
    }
    
    if model_name == 'ComplEx':
        model = ComplEx(**model_kwargs)
    elif model_name == 'RGCN':
        model = RGCN(**model_kwargs)
    elif model_name == 'TransE':
        model = TransE(**model_kwargs)
    elif model_name == 'DistMult':
        model = DistMult(**model_kwargs)
    elif model_name == 'RotatE':
        model = RotatE(**model_kwargs)
    elif model_name == 'ConvE':
        model = ConvE(**model_kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = model.to(device)
    logger.info(f"Created {model_name} model with {embedding_dim}D embeddings")
    return model


def train_model(model, triples_factory, args, device, file_paths):
    """Train the KGE model."""
    logger.info("Starting model training...")
    
    # Create optimizer
    optimizer = Adam(params=model.parameters(), lr=args.learning_rate)
    
    # Create training loop
    training_loop = SLCWATrainingLoop(
        model=model,
        triples_factory=triples_factory,
        optimizer=optimizer,
    )
    
    # Split triples for evaluation (80/20 split)
    training_tf, testing_tf = triples_factory.split([0.8, 0.2])
    logger.info(f"Split data: {len(training_tf.mapped_triples)} training, {len(testing_tf.mapped_triples)} testing")
    
    # Save initial mappings
    save_mappings(triples_factory, file_paths)
    
    # Create early stopper with evaluation triples
    stopper = EarlyStopper(
        model=model,
        evaluator=RankBasedEvaluator(),
        training_triples_factory=training_tf,
        evaluation_triples_factory=testing_tf,
        patience=args.patience,
        frequency=10,  # Evaluate every 10 epochs
    )
    
    # Train with the training subset
    losses = training_loop.train(
        triples_factory=training_tf,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        stopper=stopper,
        use_tqdm=True,
    )
    
    # Save losses incrementally during training
    save_losses_incremental(losses, file_paths['losses'])
    
    logger.info("Training completed")
    return losses


def save_mappings(triples_factory, file_paths):
    """Save entity and relation mappings early."""
    # Save entity mapping
    entity_to_id = triples_factory.entity_to_id
    entity_mapping = pd.DataFrame([
        {'entity_id': idx, 'label': entity}
        for entity, idx in entity_to_id.items()
    ])
    entity_mapping.to_csv(file_paths['entity_mapping'], index=False)
    logger.info(f"Saved entity mapping: {os.path.basename(file_paths['entity_mapping'])} ({len(entity_mapping)} entities)")
    
    # Save relation mapping
    relation_to_id = triples_factory.relation_to_id
    relation_mapping = pd.DataFrame([
        {'relation_id': idx, 'label': relation}
        for relation, idx in relation_to_id.items()
    ])
    relation_mapping.to_csv(file_paths['relation_mapping'], index=False)
    logger.info(f"Saved relation mapping: {os.path.basename(file_paths['relation_mapping'])} ({len(relation_mapping)} relations)")


def save_losses_incremental(losses, losses_file):
    """Save training losses incrementally."""
    if losses:
        losses_df = pd.DataFrame({'epoch': range(len(losses)), 'loss': losses})
        losses_df.to_csv(losses_file, index=False)
        logger.info(f"Saved training losses: {os.path.basename(losses_file)} ({len(losses)} epochs)")


def save_embeddings(model, file_paths):
    """Save retrained embeddings."""
    # Save entity embeddings - handle different PyKEEN model structures
    try:
        # Try direct weight access first
        if hasattr(model.entity_representations[0], 'weight'):
            entity_embeddings = model.entity_representations[0].weight.data.cpu().numpy()
        else:
            # For newer PyKEEN versions or after training, use the embedding tensor directly
            entity_embeddings = model.entity_representations[0](torch.arange(model.num_entities)).detach().cpu().numpy()
    except:
        # Fallback: get embeddings through forward pass
        device = next(model.parameters()).device
        entity_ids = torch.arange(model.num_entities, device=device)
        entity_embeddings = model.entity_representations[0](entity_ids).detach().cpu().numpy()
    
    np.save(file_paths['entity_embeddings'], entity_embeddings)
    logger.info(f"Saved entity embeddings: {os.path.basename(file_paths['entity_embeddings'])}")
    logger.info(f"Entity embeddings shape: {entity_embeddings.shape}")
    
    # Save relation embeddings - similar handling
    if hasattr(model, 'relation_representations') and model.relation_representations:
        try:
            if hasattr(model.relation_representations[0], 'weight'):
                relation_embeddings = model.relation_representations[0].weight.data.cpu().numpy()
            else:
                relation_embeddings = model.relation_representations[0](torch.arange(model.num_relations)).detach().cpu().numpy()
        except:
            device = next(model.parameters()).device
            relation_ids = torch.arange(model.num_relations, device=device)
            relation_embeddings = model.relation_representations[0](relation_ids).detach().cpu().numpy()
        
        np.save(file_paths['relation_embeddings'], relation_embeddings)
        logger.info(f"Saved relation embeddings: {os.path.basename(file_paths['relation_embeddings'])}")
        logger.info(f"Relation embeddings shape: {relation_embeddings.shape}")
    
    # Save model checkpoint
    torch.save(model.state_dict(), file_paths['checkpoint'])
    logger.info(f"Saved model checkpoint: {os.path.basename(file_paths['checkpoint'])}")
    
    return file_paths


def main():
    """Main retraining function."""
    args = parse_arguments()
    
    logger.info("="*60)
    logger.info("Starting KGE retraining from edge lists...")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Embedding dimension: {args.embedding_dim}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info("="*60)
    
    # Setup device and output directory
    device = setup_device(args.device)
    file_paths = setup_output_directory(args.output_dir, args.model, args.dataset)
    
    # Load edge data
    triples = load_edge_data(args.input_dir, args.dataset)
    
    # Create triples factory
    logger.info("Creating triples factory...")
    triples_factory = TriplesFactory.from_labeled_triples(
        triples=triples,
        create_inverse_triples=True,
    )
    
    logger.info(f"Triples factory statistics:")
    logger.info(f"  - Entities: {triples_factory.num_entities}")
    logger.info(f"  - Relations: {triples_factory.num_relations}")
    logger.info(f"  - Triples: {triples_factory.num_triples}")
    
    # Create model
    model = create_model(args.model, triples_factory, args.embedding_dim, device)
    
    # Train model (saves mappings and losses incrementally)
    losses = train_model(model, triples_factory, args, device, file_paths)
    
    # Save final embeddings and checkpoint
    output_files = save_embeddings(model, file_paths)
    
    # Final summary
    logger.info("="*60)
    logger.info("KGE retraining completed successfully!")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("Generated files:")
    for desc, path in file_paths.items():
        if os.path.exists(path):
            logger.info(f"  - {desc}: {os.path.basename(path)}")
    logger.info("="*60)


if __name__ == "__main__":
    main()