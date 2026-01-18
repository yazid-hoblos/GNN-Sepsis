#!/usr/bin/env python3
"""
Retrain Knowledge Graph Embeddings from edge lists (CSV) or OWL files.

This script reads edge data from either:
  - CSV files (from models/executions or OntoKGCreation/converted/optimized/lean)
  - OWL files (from OntoKGCreation/output or models/executions)

And retrains KGE models (ComplEx, RGCN, etc.) to generate new embeddings.
"""

import argparse
import os
import sys
import logging
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from rdflib import Graph

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
    parser = argparse.ArgumentParser(description="Retrain Knowledge Graph Embeddings from CSV or OWL")
    
    # Input source (not mutually exclusive at parse time, validated in main)
    parser.add_argument('--csv-nodes', type=str, default=None,
                       help='Path to nodes.csv (use with --csv-edges)')
    parser.add_argument('--csv-edges', type=str, default=None,
                       help='Path to edges.csv (use with --csv-nodes)')
    parser.add_argument('--owl-file', type=str, default=None,
                       help='Path to OWL file')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Dataset identifier for edge file in models/executions/')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='ComplEx',
                       choices=['ComplEx', 'RGCN', 'TransE', 'DistMult', 'RotatE', 'ConvE'],
                       help='KGE model to train')
    parser.add_argument('--embedding_dim', type=int, default=100,
                       help='Embedding dimension')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=1000,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training (default: 32 to fit in typical GPU memory)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=50,
                       help='Early stopping patience')
    
    # Data parameters
    parser.add_argument('--input_dir', type=str, default='../models/executions',
                       help='Input directory for --dataset option (default: models/executions)')
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


def load_from_csv(nodes_file, edges_file):
    """Load KG edges from CSV files (nodes.csv and edges.csv).
    
    Handles both column naming conventions:
    - subject, predicate, object (lean version)
    - source, relation, target (non-lean version)
    """
    logger.info(f"Loading nodes from: {nodes_file}")
    nodes_df = pd.read_csv(nodes_file)
    logger.info(f"Loaded {len(nodes_df)} nodes")
    
    logger.info(f"Loading edges from: {edges_file}")
    edges_df = pd.read_csv(edges_file)
    logger.info(f"Loaded {len(edges_df)} edges")
    logger.info(f"Edge CSV columns: {edges_df.columns.tolist()}")
    
    # Determine which column names are used
    if 'subject' in edges_df.columns and 'predicate' in edges_df.columns and 'object' in edges_df.columns:
        # Lean version naming
        triple_cols = ['subject', 'predicate', 'object']
    elif 'source' in edges_df.columns and 'relation' in edges_df.columns and 'target' in edges_df.columns:
        # Non-lean version naming
        triple_cols = ['source', 'relation', 'target']
    else:
        raise KeyError(f"Could not find expected columns. Available columns: {edges_df.columns.tolist()}. " 
                      f"Expected either (subject, predicate, object) or (source, relation, target)")
    
    logger.info(f"Using columns: {triple_cols}")
    
    # Extract triples
    triples = edges_df[triple_cols].values
    
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


def load_from_owl(owl_file):
    """Load KG edges from OWL file using RDFlib."""
    logger.info(f"Loading OWL from: {owl_file}")
    g = Graph()
    g.parse(str(owl_file), format='xml')
    logger.info(f"Parsed OWL file with {len(g)} triples")
    
    # Extract triples from RDF graph
    triples_list = []
    for subject, predicate, obj in g:
        subject_str = str(subject).split('#')[-1].split('/')[-1]
        predicate_str = str(predicate).split('#')[-1].split('/')[-1]
        obj_str = str(obj).split('#')[-1].split('/')[-1]
        triples_list.append([subject_str, predicate_str, obj_str])
    
    triples = np.array(triples_list)
    
    # Remove any rows with NaN values
    triples_df = pd.DataFrame(triples, columns=['head', 'relation', 'tail'])
    triples_df = triples_df.dropna()
    triples = triples_df.values
    
    # Show some statistics
    unique_entities = set(triples[:, 0]) | set(triples[:, 2])
    unique_relations = set(triples[:, 1])
    
    logger.info(f"Extracted {len(triples)} triples from OWL")
    logger.info(f"Unique entities: {len(unique_entities)}")
    logger.info(f"Unique relations: {len(unique_relations)}")
    logger.info(f"Sample relations: {list(unique_relations)[:5]}")
    
    return triples


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
    """Train the KGE model with CUDA OOM fallback."""
    logger.info("Starting model training...")
    
    try:
        return _train_model_impl(model, triples_factory, args, device, file_paths)
    except RuntimeError as e:
        if 'CUDA' in str(e) and 'out of memory' in str(e).lower():
            logger.warning(f"CUDA out of memory error: {e}")
            logger.warning("Falling back to CPU training with batch size 8...")
            
            # Clear CUDA cache aggressively
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            # Use very small batch size and switch to CPU
            args.batch_size = 8
            logger.info(f"Reduced batch size to {args.batch_size}")
            
            # Move model to CPU and clear gradients
            device = 'cpu'
            model = model.to(device)
            model.zero_grad()
            logger.info("Switched to CPU training")
            
            return _train_model_impl(model, triples_factory, args, device, file_paths)
        else:
            raise


def _train_model_impl(model, triples_factory, args, device, file_paths):
    """Implementation of model training with memory management."""
    logger.info(f"Starting model training on {device}...")
    logger.info(f"Batch size: {args.batch_size}")
    
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
    try:
        losses = training_loop.train(
            triples_factory=training_tf,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            stopper=stopper,
            use_tqdm=True,
        )
    finally:
        # Always clean up CUDA memory after training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    
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
    
    # Validate input arguments
    num_sources = sum([bool(args.csv_nodes or args.csv_edges), bool(args.owl_file), bool(args.dataset)])
    if num_sources == 0:
        logger.error("Error: Must specify one of: (--csv-nodes and --csv-edges), --owl-file, or --dataset")
        sys.exit(1)
    if num_sources > 1:
        logger.error("Error: Cannot use multiple input sources together. Choose one: (--csv-nodes and --csv-edges), --owl-file, or --dataset")
        sys.exit(1)
    
    # Validate CSV pair
    if args.csv_nodes or args.csv_edges:
        if not (args.csv_nodes and args.csv_edges):
            logger.error("Error: Both --csv-nodes and --csv-edges are required")
            sys.exit(1)
        if not os.path.exists(args.csv_nodes) or not os.path.exists(args.csv_edges):
            logger.error(f"Error: CSV files not found")
            sys.exit(1)
    
    logger.info("="*60)
    logger.info("Starting KGE retraining...")
    logger.info(f"Model: {args.model}")
    logger.info(f"Embedding dimension: {args.embedding_dim}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info("="*60)
    
    # Setup device and output directory
    device = setup_device(args.device)
    file_paths = setup_output_directory(args.output_dir, args.model, 
                                       args.dataset or (args.csv_nodes and "csv") or "owl")
    
    # Load edge data from appropriate source
    if args.dataset:
        logger.info(f"Loading from dataset: {args.dataset}")
        triples = load_edge_data(args.input_dir, args.dataset)
    elif args.csv_nodes and args.csv_edges:
        logger.info(f"Loading from CSV files: nodes={args.csv_nodes}, edges={args.csv_edges}")
        triples = load_from_csv(args.csv_nodes, args.csv_edges)
    elif args.owl_file:
        logger.info(f"Loading from OWL file: {args.owl_file}")
        triples = load_from_owl(args.owl_file)
    
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