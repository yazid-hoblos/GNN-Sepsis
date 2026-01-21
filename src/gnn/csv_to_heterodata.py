import argparse
from pathlib import Path
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected
import torch.nn.functional as F


def infer_node_type(node_id: str, node_type: str | None, node_category: str | None) -> str:
    nid = str(node_id)
    if 'Sample_GSM' in nid or nid.startswith('Sample_'):
        return 'sample'
    if nid.startswith('Protein_'):
        return 'protein'
    if nid.startswith('Pathway_') or 'R-HSA-' in nid:
        return 'pathway'
    if nid.startswith('Reaction_'):
        return 'reaction'
    if nid.startswith('Gene_'):
        return 'gene'
    if nid.startswith('GO_') or (node_category and 'GO Term' in node_category):
        return 'goterm'
    if nid.startswith('Disease_') or (node_type and 'DOID' in node_type):
        return 'disease'
    if nid.startswith('Drug_'):
        return 'drug'
    # Fallbacks
    if node_category == 'Gene/Protein':
        return 'protein'
    return 'other'


def build_heterodata(nodes_csv: Path, edges_csv: Path, embedding_dim: int = 128, make_undirected: bool = True) -> HeteroData:
    nodes_df = pd.read_csv(nodes_csv)
    edges_df = pd.read_csv(edges_csv)

    # Infer node types and group ids per type
    nodes_df['ptype'] = nodes_df.apply(
        lambda r: infer_node_type(r.get('node_id'), r.get('node_type'), r.get('node_category')),
        axis=1
    )

    # Build indices per type
    node_to_idx = {}
    type_to_nodes = {}
    for ntype, group in nodes_df.groupby('ptype'):
        ids = list(group['node_id'])
        type_to_nodes[ntype] = ids
        node_to_idx[ntype] = {nid: i for i, nid in enumerate(ids)}

    # Create features per type (random normalized embeddings)
    torch.manual_seed(42)
    data = HeteroData()
    for ntype, ids in type_to_nodes.items():
        if len(ids) == 0:
            continue
        x = torch.randn(len(ids), embedding_dim, dtype=torch.float)
        x = F.normalize(x, dim=1)
        data[ntype].x = x

    # Optional: attach known numeric attributes for proteins
    if 'protein' in type_to_nodes and 'hasLog2_FC' in nodes_df.columns:
        prot = nodes_df[nodes_df['ptype'] == 'protein']
        # Coerce to numeric, fill NaN with 0.0
        log2_fc = torch.tensor(pd.to_numeric(prot['hasLog2_FC'], errors='coerce').fillna(0.0).values, dtype=torch.float)
        data['protein'].log2_fc = log2_fc

    # Sample labels from hasDiseaseStatus
    if 'sample' in type_to_nodes and 'hasDiseaseStatus' in nodes_df.columns:
        sample_df = nodes_df[nodes_df['ptype'] == 'sample']
        labels = []
        for v in sample_df['hasDiseaseStatus'].astype(str).str.lower().tolist():
            if 'sepsis' in v:
                labels.append(1)
            elif 'control' in v or 'healthy' in v:
                labels.append(0)
            else:
                labels.append(0)
        data['sample'].y = torch.tensor(labels, dtype=torch.long)

    # Build edge indices per (src_type, predicate, dst_type)
    edges = {}
    weights = {}
    for _, row in edges_df.iterrows():
        src_id = row['subject']
        rel = row['predicate']
        dst_id = row['object']
        w = row.get('hasExpressionValue')
        if pd.isna(w) or w == '' or w is None:
            w = row.get('weight')
        try:
            w = float(w)
        except (TypeError, ValueError):
            w = 1.0

        src_type = infer_node_type(src_id, None, None)
        dst_type = infer_node_type(dst_id, None, None)

        if src_type not in node_to_idx or dst_type not in node_to_idx:
            continue
        if src_id not in node_to_idx[src_type] or dst_id not in node_to_idx[dst_type]:
            continue

        key = (src_type, rel, dst_type)
        if key not in edges:
            edges[key] = [[], []]
            weights[key] = []
        edges[key][0].append(node_to_idx[src_type][src_id])
        edges[key][1].append(node_to_idx[dst_type][dst_id])
        weights[key].append(w)

    for key, (src_list, dst_list) in edges.items():
        src_type, rel, dst_type = key
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        data[src_type, rel, dst_type].edge_index = edge_index
        data[src_type, rel, dst_type].edge_weight = torch.tensor(weights[key], dtype=torch.float)

    # Make undirected to ensure messages reach all node types
    if make_undirected:
        data = ToUndirected()(data)

    return data


def main():
    parser = argparse.ArgumentParser(description='Build HeteroData from lean CSVs (nodes.csv, edges.csv)')
    parser.add_argument('--nodes', type=str, required=True, help='Path to nodes.csv')
    parser.add_argument('--edges', type=str, required=True, help='Path to edges.csv')
    parser.add_argument('--out', type=str, default=None, help='Output path for hetero_graph.pt')
    parser.add_argument('--dim', type=int, default=128, help='Embedding dimension per node type')
    parser.add_argument('--no-undirected', action='store_true', help='Do not add reverse edges')
    args = parser.parse_args()

    nodes_csv = Path(args.nodes)
    edges_csv = Path(args.edges)
    out_path = Path(args.out) if args.out else (Path(__file__).parent.parent.parent / 'data' / 'han' / 'csv' / 'hetero_graph.pt')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = build_heterodata(nodes_csv, edges_csv, embedding_dim=args.dim, make_undirected=not args.no_undirected)
    torch.save(data, out_path)

    print(f"Saved HeteroData: {out_path}")
    print(f"Node types: {data.node_types}")
    print(f"Edge types: {data.edge_types}")


if __name__ == '__main__':
    main()
