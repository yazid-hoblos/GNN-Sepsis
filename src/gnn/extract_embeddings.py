"""Extract GNN embeddings for heterogeneous graph.

Usage:
    python extract_embeddings.py --model gat
    python extract_embeddings.py --model sage
    python extract_embeddings.py --model rgcn
    python extract_embeddings.py --model all
"""

import argparse
import torch
from base import (
    PROJECT_ROOT, get_device, load_hetero_data,
    prepare_edge_weight_dict, train_epoch, evaluate,
    plot_loss, save_embeddings, EarlyStopping,
    get_data_path, get_output_dir, DATA_VERSIONS,
)
from models import create_model, MODELS


def run_model(name, data, device, output_dir, epochs=100, lr=0.01, patience=10):
    """Train model and extract embeddings."""
    print(f"\n{'='*50}\n  {name.upper()}\n{'='*50}")

    # Prepare edge weights (normalized)
    edge_weight_dict = prepare_edge_weight_dict(data, device, normalize=True)

    model = create_model(name, data.metadata()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=patience)

    train_losses, val_losses = [], []
    for epoch in range(epochs):
        train_loss = train_epoch(model, data, optimizer, device, edge_weight_dict)
        val_loss = evaluate(model, data, device, edge_weight_dict)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch} (best: {early_stopping.best_loss:.4f})")
            break

    if early_stopping.best_model_state:
        model.load_state_dict(early_stopping.best_model_state)

    # Save results
    model_dir = output_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)
    plot_loss(train_losses, val_losses, model_dir / "training_loss.png", name.upper())

    # Extract embeddings
    model.eval()
    with torch.no_grad():
        x_dict = {k: data[k].x.to(device) for k in data.node_types}
        edge_index_dict = {k: data[k].edge_index.to(device) for k in data.edge_types}
        z_dict = model(x_dict, edge_index_dict, edge_weight_dict)

    save_embeddings(z_dict, model_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()) + ["all"], default="all")
    parser.add_argument("--version", "-v", choices=list(DATA_VERSIONS.keys()), default="v2.11",
                        help="Data version to use (default: v2.11)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Version: {args.version}")

    data_path = get_data_path(args.version)
    data = load_hetero_data(data_path)
    output_dir = get_output_dir(args.version)

    models = list(MODELS.keys()) if args.model == "all" else [args.model]
    for name in models:
        run_model(name, data, device, output_dir, args.epochs, args.lr, args.patience)

    print(f"\n{'='*50}\nDone!\n{'='*50}")


if __name__ == "__main__":
    main()
