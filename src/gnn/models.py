"""GNN models for heterogeneous graph embedding extraction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, Linear


# =============================================================================
# Custom Convolutions with Edge Weight Support (Lazy init compatible)
# =============================================================================

class WeightedConv(nn.Module):
    """Base weighted convolution for edge weight support (with degree normalization)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.root = Linear(in_channels, out_channels, bias=True)

    def forward(self, x, edge_index, edge_weight=None):
        if isinstance(x, tuple):
            x_src, x_dst = x
        else:
            x_src = x_dst = x

        src_idx, dst_idx = edge_index
        msg = self.lin(x_src[src_idx])

        if edge_weight is not None:
            msg = msg * edge_weight.view(-1, 1)

        out = torch.zeros(x_dst.size(0), self.out_channels, device=x_dst.device)
        out.scatter_add_(0, dst_idx.unsqueeze(-1).expand_as(msg), msg)

        # Normalize by degree (like SAGE mean aggregation)
        deg = torch.zeros(x_dst.size(0), device=x_dst.device)
        if edge_weight is not None:
            deg.scatter_add_(0, dst_idx, edge_weight)
        else:
            deg.scatter_add_(0, dst_idx, torch.ones(src_idx.size(0), device=x_dst.device))
        out = out / deg.clamp(min=1).unsqueeze(-1)

        return out + self.root(x_dst)


class WeightedSAGEConv(nn.Module):
    """SAGE convolution with edge weights (sum aggregation to avoid over-smoothing)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        self.lin_neigh = Linear(in_channels, out_channels, bias=False)
        self.lin_self = Linear(in_channels, out_channels, bias=True)

    def forward(self, x, edge_index, edge_weight=None):
        if isinstance(x, tuple):
            x_src, x_dst = x
        else:
            x_src = x_dst = x

        src_idx, dst_idx = edge_index
        msg = self.lin_neigh(x_src[src_idx])

        if edge_weight is not None:
            msg = msg * edge_weight.view(-1, 1)

        out = torch.zeros(x_dst.size(0), self.out_channels, device=x_dst.device)
        out.scatter_add_(0, dst_idx.unsqueeze(-1).expand_as(msg), msg)

        # Sum aggregation (no normalization - preserves node degree info)
        return out + self.lin_self(x_dst)


# =============================================================================
# Heterogeneous GNN Models
# =============================================================================

class HeteroGAT(nn.Module):
    """Heterogeneous GAT with edge weight support via edge_dim."""

    def __init__(self, metadata, hidden_channels: int = 100, heads: int = 4, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        _, edge_types = metadata

        # Layer 1: with edge_dim for weighted edges
        conv1 = {}
        for et in edge_types:
            conv1[et] = GATConv((-1, -1), hidden_channels // heads, heads=heads,
                                dropout=dropout, edge_dim=1, add_self_loops=False)
        self.conv1 = HeteroConv(conv1, aggr='sum')

        # Layer 2
        conv2 = {}
        for et in edge_types:
            conv2[et] = GATConv((-1, -1), hidden_channels, heads=1, concat=False,
                                dropout=dropout, add_self_loops=False)
        self.conv2 = HeteroConv(conv2, aggr='sum')

    def forward(self, x_dict, edge_index_dict, edge_weight_dict=None):
        # GAT uses edge_attr (shape [E, 1]) not edge_weight
        if edge_weight_dict:
            edge_attr = {k: v.unsqueeze(-1) if v.dim() == 1 else v for k, v in edge_weight_dict.items()}
            x_dict = self.conv1(x_dict, edge_index_dict, edge_attr)
        else:
            x_dict = self.conv1(x_dict, edge_index_dict)

        x_dict = {k: F.elu(F.dropout(v, p=self.dropout, training=self.training)) for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict


class HeteroGraphSAGE(nn.Module):
    """Heterogeneous GraphSAGE with edge weight support and LayerNorm."""

    def __init__(self, metadata, hidden_channels: int = 100, num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        node_types, edge_types = metadata

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            conv = {et: WeightedSAGEConv(-1, hidden_channels) for et in edge_types}
            self.convs.append(HeteroConv(conv, aggr='sum'))
            # LayerNorm for each node type
            self.norms.append(nn.ModuleDict({nt: nn.LayerNorm(hidden_channels) for nt in node_types}))

    def forward(self, x_dict, edge_index_dict, edge_weight_dict=None):
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict, edge_weight_dict) if edge_weight_dict else conv(x_dict, edge_index_dict)
            # Apply LayerNorm to re-spread embeddings
            x_dict = {k: self.norms[i][k](v) for k, v in x_dict.items()}
            if i < self.num_layers - 1:
                x_dict = {k: F.dropout(F.relu(v), p=self.dropout, training=self.training) for k, v in x_dict.items()}
        return x_dict


class HeteroRGCN(nn.Module):
    """Heterogeneous RGCN with edge weight support."""

    def __init__(self, metadata, hidden_channels: int = 100, num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        _, edge_types = metadata

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = {et: WeightedConv(-1, hidden_channels) for et in edge_types}
            self.convs.append(HeteroConv(conv, aggr='sum'))

    def forward(self, x_dict, edge_index_dict, edge_weight_dict=None):
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict, edge_weight_dict) if edge_weight_dict else conv(x_dict, edge_index_dict)
            if i < len(self.convs) - 1:
                x_dict = {k: F.dropout(F.relu(v), p=self.dropout, training=self.training) for k, v in x_dict.items()}
        return x_dict


# =============================================================================
# Model Factory
# =============================================================================

MODELS = {
    'gat': HeteroGAT,
    'sage': HeteroGraphSAGE,
    'rgcn': HeteroRGCN,
}

def create_model(name: str, metadata, hidden_channels: int = 100, **kwargs):
    """Create a model by name."""
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}. Choose from {list(MODELS.keys())}")
    return MODELS[name](metadata, hidden_channels=hidden_channels, **kwargs)
