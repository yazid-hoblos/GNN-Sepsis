"""
HAN Model for Sepsis Prediction

Implements Heterogeneous Attention Network with interpretable
attention weights for node and semantic-level analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Iterable
import logging

from torch_geometric.nn import HeteroConv, GATConv, HANConv
from torch_geometric.data import HeteroData


logger = logging.getLogger(__name__)


class SimpleHANFallback(nn.Module):
    """Lightweight hetero GAT stack used when torch_geometric.nn.HAN is absent."""

    def __init__(self,
                 in_channels_dict: Dict[str, int],
                 hidden_channels: int,
                 out_channels: int,
                 num_layers: int,
                 num_heads: int,
                 dropout: float):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        # Project each node type to a common hidden size so hetero convs can share dims.
        self.projectors = nn.ModuleDict({
            ntype: nn.Linear(in_dim, hidden_channels)
            for ntype, in_dim in in_channels_dict.items()
        })

        # Lazy-created conv stack, built on first forward once edge types are known.
        self.convs: Optional[nn.ModuleList] = None
        self.post_projectors: Optional[nn.ModuleDict] = None

    def _build_convs(self,
                     edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
                     device: torch.device):
        edge_types = list(edge_index_dict.keys())
        self.convs = nn.ModuleList()
        for _ in range(self.num_layers):
            conv = HeteroConv({
                etype: GATConv(
                    (-1, -1),
                    self.hidden_channels,
                    heads=self.num_heads,
                    concat=False,
                    dropout=self.dropout,
                    add_self_loops=False,
                )
                for etype in edge_types
            }, aggr='mean')
            self.convs.append(conv.to(device))

    def _build_post(self, node_types, device: torch.device):
        self.post_projectors = nn.ModuleDict({
            ntype: nn.Linear(self.hidden_channels, self.out_channels)
            for ntype in node_types
        }).to(device)

    def forward(self,
                x_dict: Dict[str, torch.Tensor],
                edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]):
        # Initial projection to shared hidden size
        device = next(iter(x_dict.values())).device
        h_dict = {
            ntype: F.dropout(self.projectors[ntype](x), p=self.dropout, training=self.training)
            for ntype, x in x_dict.items()
        }

        if self.convs is None:
            self._build_convs(edge_index_dict, device)
        # Convolutional stack
        for conv in self.convs:
            updated = conv(h_dict, edge_index_dict)
            # Keep old representations for node types without incoming messages
            merged = {}
            for ntype in h_dict.keys():
                if ntype in updated and updated[ntype] is not None:
                    merged[ntype] = updated[ntype]
                else:
                    merged[ntype] = h_dict[ntype]
            h_dict = {k: F.elu(v) for k, v in merged.items()}
            h_dict = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in h_dict.items()}

        if self.post_projectors is None:
            self._build_post(h_dict.keys(), device)

        out_dict = {k: self.post_projectors[k](v) for k, v in h_dict.items()}
        return out_dict


class SepsisHANClassifier(nn.Module):
    """
    HAN-based classifier for sepsis prediction.
    
    Architecture:
    - HAN layers: Learn node embeddings considering all metapaths
    - Semantic attention: Which edge types matter?
    - Node-level attention: Which neighbors matter?
    - Classification head: Patient embeddings -> septic/healthy
    """
    
    def __init__(self,
                 in_channels_dict: Dict[str, int],
                 metadata: Tuple[List[str], List[Tuple[str, str, str]]],
                 hidden_channels: int = 64,
                 out_channels: int = 32,
                 num_layers: int = 2,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        """
        Initialize SepsisHANClassifier.
        
        Parameters
        ----------
        in_channels_dict : dict
            Input channel dimensions for each node type
        hidden_channels : int
            Hidden dimension for HAN layers
        out_channels : int
            Output dimension for HAN embeddings
        num_layers : int
            Number of HAN layers
        num_heads : int
            Number of attention heads
        dropout : float
            Dropout probability
        """
        super().__init__()
        
        self.in_channels_dict = in_channels_dict
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.metadata = metadata
        self.metadata_aug = self._augment_metadata(metadata)
        
        # Store for later access to attention weights
        self.attention_weights_node = None
        self.attention_weights_semantic = None
        
        # HAN-style conv stack using HANConv
        self.han_convs = nn.ModuleList()
        # First layer: derive input dims from dict
        self.han_convs.append(
            HANConv(
                in_channels=in_channels_dict,
                out_channels=hidden_channels,
                metadata=self.metadata_aug,
                heads=num_heads,
                dropout=dropout,
            )
        )
        # Additional layers: use prior layer's out_channels as next in_channels
        in_dim = hidden_channels
        for layer_idx in range(num_layers - 1):
            self.han_convs.append(
                HANConv(
                    in_channels=in_dim,
                    out_channels=out_channels,
                    metadata=self.metadata_aug,
                    heads=num_heads,
                    dropout=dropout,
                )
            )
            in_dim = out_channels

        logger.info("Using HANConv stack (meta-path based)")
        
        # Classification head: Patient embeddings -> logits
        self.classifier = nn.Sequential(
            nn.Linear(out_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 2),  # Binary classification
        )
        
        logger.info(f"Classifier: {out_channels} -> {hidden_channels//2} -> 2")
    
    def forward(self,
                x_dict: Dict[str, torch.Tensor],
                edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
                return_attention: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        """
        Forward pass.
        
        Parameters
        ----------
        x_dict : dict
            Node feature dictionary
        edge_index_dict : dict
            Edge index dictionary
        return_attention : bool
            If True, return attention weights
        
        Returns
        -------
        logits : torch.Tensor
            Classification logits for patient samples [num_samples, 2]
        patient_embeddings : torch.Tensor
            Learned embeddings for patient nodes [num_samples, out_channels]
        attention_dict : dict or None
            Attention weights if return_attention=True
        """
        # Augment edges with reverse + self loops to enable Sample updates
        edge_index_aug = self._augment_edges(edge_index_dict, x_dict)

        h_dict = x_dict
        for conv in self.han_convs:
            h_dict = conv(h_dict, edge_index_aug)
        
        # Extract patient embeddings
        patient_embeddings = h_dict['Sample']
        
        # Classification
        logits = self.classifier(patient_embeddings)
        
        # Return attention weights if requested
        attention_dict = None
        if return_attention:
            attention_dict = self._extract_attention_weights()
        
        return logits, patient_embeddings, attention_dict
    
    def _extract_attention_weights(self) -> Dict:
        """
        Extract attention weights from HAN layers.
        
        Returns
        -------
        attention_dict : dict
            Contains:
            - node_attention: Which neighbors influenced prediction?
            - semantic_attention: Which edge types matter?
        """
        attention_dict = {
            'node_attention': self.attention_weights_node,
            'semantic_attention': self.attention_weights_semantic,
        }
        return attention_dict

    def _augment_metadata(self, metadata: Tuple[List[str], List[Tuple[str, str, str]]]) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        node_types, edge_types = metadata
        augmented_edges: List[Tuple[str, str, str]] = list(edge_types)

        # Add reverse edges
        for (src, rel, dst) in edge_types:
            rev = (dst, f"{rel}_rev", src)
            if rev not in augmented_edges:
                augmented_edges.append(rev)

        # Add self-loop edge types for each node type
        for ntype in node_types:
            loop = (ntype, 'self', ntype)
            if loop not in augmented_edges:
                augmented_edges.append(loop)

        return (list(node_types), augmented_edges)

    def _augment_edges(self,
                       edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
                       x_dict: Dict[str, torch.Tensor]) -> Dict[Tuple[str, str, str], torch.Tensor]:
        """Add reverse edges and self-loops required by the meta-paths."""
        augmented = dict(edge_index_dict)

        # Reverse edges
        for (src, rel, dst), edge_index in edge_index_dict.items():
            rev_key = (dst, f"{rel}_rev", src)
            if rev_key not in augmented:
                augmented[rev_key] = edge_index.flip(0)

        # Self loops per node type
        for ntype, x in x_dict.items():
            n = x.size(0)
            idx = torch.arange(n, device=x.device)
            augmented[(ntype, 'self', ntype)] = torch.stack([idx, idx], dim=0)

        return augmented


class HANTrainer:
    """
    Trainer for HAN model with attention weight tracking.
    """
    
    def __init__(self,
                 model: SepsisHANClassifier,
                 device: torch.device = torch.device('cpu'),
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-5):
        """
        Initialize trainer.
        
        Parameters
        ----------
        model : SepsisHANClassifier
            Model to train
        device : torch.device
            Device for training
        learning_rate : float
            Learning rate
        weight_decay : float
            L2 regularization weight
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Compute class weights to handle imbalance
        # This helps the model learn the minority class better
        class_weights = torch.tensor([1.0, 3.5], device=device)  # Septic (1) is 3.5x more important
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        self.train_losses = []
        self.val_accs = []
        
        logger.info(f"Trainer initialized (device={device})")
    
    def train_epoch(self, data: HeteroData) -> float:
        """
        Single training epoch.
        
        Parameters
        ----------
        data : HeteroData
            Graph data
        
        Returns
        -------
        loss : float
            Training loss
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        logits, _, _ = self.model(data.x_dict, data.edge_index_dict)
        
        # Get training mask and labels
        train_mask = data['Sample'].train_mask
        y = data['Sample'].y
        
        # Compute loss only on training samples
        loss = self.criterion(logits[train_mask], y[train_mask])
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        self.train_losses.append(loss.item())
        
        return loss.item()
    
    def evaluate(self, data: HeteroData, mask_type: str = 'val') -> Tuple[float, float]:
        """
        Evaluate model on validation or test set.
        
        Parameters
        ----------
        data : HeteroData
            Graph data
        mask_type : str
            'val' or 'test'
        
        Returns
        -------
        accuracy : float
            Accuracy on set
        loss : float
            Loss on set
        """
        self.model.eval()
        
        with torch.no_grad():
            logits, _, _ = self.model(data.x_dict, data.edge_index_dict)
        
        # Get mask and labels
        mask = data['Sample'][f'{mask_type}_mask']
        y = data['Sample'].y
        
        # Compute accuracy
        pred = logits[mask].argmax(dim=1)
        acc = (pred == y[mask]).float().mean().item()
        
        # Compute loss
        loss = self.criterion(logits[mask], y[mask]).item()
        
        if mask_type == 'val':
            self.val_accs.append(acc)
        
        return acc, loss
    
    def train(self,
              data: HeteroData,
              epochs: int = 100,
              early_stopping_patience: int = 20,
              verbose: int = 10) -> Dict[str, List[float]]:
        """
        Full training loop with early stopping.
        
        Parameters
        ----------
        data : HeteroData
            Graph data
        epochs : int
            Number of epochs to train
        early_stopping_patience : int
            Early stopping patience (epochs without improvement)
        verbose : int
            Print every N epochs
        
        Returns
        -------
        history : dict
            Training history (losses, accuracies)
        """
        logger.info(f"Starting training for {epochs} epochs...")
        
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            # Train
            train_loss = self.train_epoch(data)
            
            # Evaluate
            val_acc, val_loss = self.evaluate(data, mask_type='val')
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                self.best_state_dict = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            # Logging
            if epoch % verbose == 0 or epoch == 1:
                logger.info(
                    f"Epoch {epoch:3d} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Acc: {val_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f}"
                )
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                # Restore best model
                self.model.load_state_dict(self.best_state_dict)
                break
        
        # Final evaluation
        test_acc, test_loss = self.evaluate(data, mask_type='test')
        logger.info(f"\nFinal Test Accuracy: {test_acc:.4f}")
        logger.info(f"Final Test Loss: {test_loss:.4f}")
        
        history = {
            'train_loss': self.train_losses,
            'val_acc': self.val_accs,
            'test_acc': test_acc,
            'test_loss': test_loss,
        }
        
        return history
    
    def predict(self, data: HeteroData, return_embeddings: bool = False):
        """
        Get predictions on all samples.
        
        Parameters
        ----------
        data : HeteroData
            Graph data
        return_embeddings : bool
            If True, also return patient embeddings
        
        Returns
        -------
        predictions : torch.Tensor
            Class predictions [num_samples]
        probabilities : torch.Tensor
            Class probabilities [num_samples, 2]
        embeddings : torch.Tensor or None
            Patient embeddings if return_embeddings=True
        """
        self.model.eval()
        
        with torch.no_grad():
            logits, embeddings, _ = self.model(data.x_dict, data.edge_index_dict)
        
        probabilities = F.softmax(logits, dim=1)
        predictions = logits.argmax(dim=1)
        
        if return_embeddings:
            return predictions, probabilities, embeddings
        else:
            return predictions, probabilities


if __name__ == '__main__':
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)
    
    from src.han.data_loader import load_hetero_graph
    
    # Load data
    print("Loading graph...")
    data = load_hetero_graph('models/executions/GSE54514_enriched_ontology_degfilterv2.11')
    
    # Create model
    print("Creating model...")
    in_channels_dict = {node_type: data[node_type].x.size(1) 
                        for node_type in data.node_types}
    
    model = SepsisHANClassifier(
        in_channels_dict=in_channels_dict,
        hidden_channels=64,
        out_channels=32,
        num_layers=2,
        num_heads=8,
    )
    
    # Initialize trainer
    print("Creating trainer...")
    trainer = HANTrainer(model)
    
    # Train
    print("Training...")
    history = trainer.train(data, epochs=50, verbose=10)
    
    print("\nTraining complete!")
