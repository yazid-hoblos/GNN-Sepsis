import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from torch_geometric.transforms import ToUndirected

class HeteroSAGE(torch.nn.Module):
    def __init__(self, metadata, hidden=64, out=2):
        super().__init__()
        # Build per-relation SAGEConv stacks
        self.conv1 = HeteroConv({edge_type: SAGEConv((-1, -1), hidden)
                                 for edge_type in metadata[1]}, aggr='sum')
        self.conv2 = HeteroConv({edge_type: SAGEConv((hidden, hidden), hidden)
                                 for edge_type in metadata[1]}, aggr='sum')
        self.lin = Linear(hidden, out)

    def forward(self, x_dict, edge_index_dict):
        # First layer: update only destination node types; keep originals for others
        out1 = self.conv1(x_dict, edge_index_dict)
        for k in x_dict.keys():
            out1[k] = out1.get(k, x_dict[k])
        out1 = {k: F.relu(v) for k, v in out1.items()}

        # Second layer with the same safeguard
        out2 = self.conv2(out1, edge_index_dict)
        for k in out1.keys():
            out2[k] = out2.get(k, out1[k])
        return out2


def train(data, epochs=50, lr=1e-3, weight_decay=1e-4):
    # Expect labels on 'sample' nodes
    if 'sample' not in data.node_types:
        raise ValueError("Graph must contain 'sample' node type with labels (data['sample'].y)")
    if getattr(data['sample'], 'y', None) is None:
        raise ValueError("Missing labels on 'sample' nodes: set data['sample'].y")

    # Ensure undirected graph so every node type can receive messages
    data = ToUndirected()(data)

    model = HeteroSAGE(data.metadata(), hidden=64, out=int(data['sample'].y.max().item()) + 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    y = data['sample'].y
    idx = torch.arange(y.size(0))
    # Simple split
    n = len(idx)
    train_mask = idx[: int(0.7 * n)]
    val_mask = idx[int(0.7 * n): int(0.85 * n)]
    test_mask = idx[int(0.85 * n):]

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        x_dict = model(data.x_dict, data.edge_index_dict)
        logits = model.lin(x_dict['sample'])
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            x_dict = model(data.x_dict, data.edge_index_dict)
            logits = model.lin(x_dict['sample'])
            val_acc = (logits[val_mask].argmax(dim=1) == y[val_mask]).float().mean().item()
        if epoch % 5 == 0:
            print(f"Epoch {epoch:03d} | loss {loss.item():.4f} | val_acc {val_acc:.3f}")

    model.eval()
    with torch.no_grad():
        x_dict = model(data.x_dict, data.edge_index_dict)
        logits = model.lin(x_dict['sample'])
        test_acc = (logits[test_mask].argmax(dim=1) == y[test_mask]).float().mean().item()
    print(f"Test accuracy: {test_acc:.3f}")
    return test_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', required=True, help='Path to hetero_graph.pt produced by load_heterodata.py')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    args = parser.parse_args()

    data = torch.load(Path(args.graph))
    train(data, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay)


if __name__ == '__main__':
    main()
