"""Common utilities for GNN embedding extraction."""

from pathlib import Path
import torch
import torch_geometric.transforms as T
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Version configurations
DATA_VERSIONS = {
    'v2.9': PROJECT_ROOT / "data" / "hetero_graph" / "v2.9" / "hetero_graph.pt",
    'v2.10': PROJECT_ROOT / "data" / "hetero_graph" / "v2.10" / "hetero_graph.pt",
    'v2.11': PROJECT_ROOT / "data" / "hetero_graph" / "v2.11" / "hetero_graph.pt",
}
DATA_PATH = DATA_VERSIONS['v2.11']  # Default

def get_data_path(version: str = 'v2.11'):
    """Get data path for a specific version."""
    if version not in DATA_VERSIONS:
        raise ValueError(f"Unknown version: {version}. Choose from {list(DATA_VERSIONS.keys())}")
    return DATA_VERSIONS[version]

def get_output_dir(version: str = 'v2.11'):
    """Get output directory for embeddings."""
    return PROJECT_ROOT / "results" / "embeddings" / version


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
            self.counter = 0


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_hetero_data(data_path=DATA_PATH, add_reverse=True):
    """Load heterogeneous graph data (no normalization here - done later)."""
    data = torch.load(data_path, weights_only=False)
    print(f"Loaded: {data['sample'].x.shape[0]} samples, {data['protein'].x.shape[0]} proteins, {len(data.edge_types)} edge types")

    if add_reverse:
        data = T.ToUndirected()(data)
        print(f"After ToUndirected: {len(data.edge_types)} edge types")

    return data


def prepare_edge_weight_dict(data, device, normalize=True):
    """Prepare normalized edge weights for training."""
    edge_weight_dict = {}
    for edge_type in data.edge_types:
        store = data[edge_type]
        if hasattr(store, 'edge_weight') and store.edge_weight is not None:
            w = store.edge_weight
            if normalize:
                scaler = MinMaxScaler()
                w_np = w.cpu().numpy().reshape(-1, 1)
                w = torch.tensor(scaler.fit_transform(w_np).flatten(), dtype=torch.float)
            edge_weight_dict[edge_type] = w.to(device)
    return edge_weight_dict if edge_weight_dict else None


def sample_negative_edges(data, edge_type):
    """Sample negative edges for link prediction."""
    src, _, dst = edge_type
    n_neg = data[edge_type].edge_index.size(1)
    return torch.stack([
        torch.randint(0, data[src].x.size(0), (n_neg,)),
        torch.randint(0, data[dst].x.size(0), (n_neg,))
    ], dim=0)


def compute_link_loss(pos_score, neg_score):
    """Original BCE loss (stable version)."""
    pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-15).mean()
    neg_loss = -torch.log(1 - torch.sigmoid(neg_score) + 1e-15).mean()
    return pos_loss + neg_loss


def train_epoch(model, data, optimizer, device, edge_weight_dict):
    """Train one epoch."""
    model.train()
    optimizer.zero_grad()

    x_dict = {k: data[k].x.to(device) for k in data.node_types}
    edge_index_dict = {k: data[k].edge_index.to(device) for k in data.edge_types}

    z_dict = model(x_dict, edge_index_dict, edge_weight_dict)

    total_loss = 0.0
    for et in data.edge_types:
        src_type, _, dst_type = et
        edge_idx = data[et].edge_index.to(device)

        pos_score = (z_dict[src_type][edge_idx[0]] * z_dict[dst_type][edge_idx[1]]).sum(-1)
        neg_edge = sample_negative_edges(data, et).to(device)
        neg_score = (z_dict[src_type][neg_edge[0]] * z_dict[dst_type][neg_edge[1]]).sum(-1)

        total_loss += compute_link_loss(pos_score, neg_score)

    loss = total_loss / len(data.edge_types)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, device, edge_weight_dict):
    """Evaluate model."""
    model.eval()

    x_dict = {k: data[k].x.to(device) for k in data.node_types}
    edge_index_dict = {k: data[k].edge_index.to(device) for k in data.edge_types}

    z_dict = model(x_dict, edge_index_dict, edge_weight_dict)

    total_loss = 0.0
    for et in data.edge_types:
        src_type, _, dst_type = et
        edge_idx = data[et].edge_index.to(device)

        pos_score = (z_dict[src_type][edge_idx[0]] * z_dict[dst_type][edge_idx[1]]).sum(-1)
        neg_edge = sample_negative_edges(data, et).to(device)
        neg_score = (z_dict[src_type][neg_edge[0]] * z_dict[dst_type][neg_edge[1]]).sum(-1)

        total_loss += compute_link_loss(pos_score, neg_score)

    return (total_loss / len(data.edge_types)).item()


def plot_loss(train_losses, val_losses, output_path, model_name):
    """Plot training curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_embeddings(z_dict, output_dir):
    """Save sample and protein embeddings."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for n_type in ['sample', 'protein']:
        pd.DataFrame(z_dict[n_type].cpu().numpy()).to_csv(output_dir / f"{n_type}_embeddings.csv", index=False)
    print(f"Saved: samples {z_dict['sample'].shape}, proteins {z_dict['protein'].shape}")
