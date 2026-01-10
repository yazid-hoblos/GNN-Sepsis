from pathlib import Path
from rdflib import Graph
import torch
from torch_geometric.data import HeteroData
from collections import defaultdict

project_root = Path(__file__).parent.parent.parent
owl_dir = project_root / "models" / "executions" / "GSE54514_enriched_ontology_degfilterv2.9"
owl_filepath = owl_dir / "GSE54514_enriched_ontology_degfilterv2.9.owl"


def _get_node_type(uri):
    """Extract node type from URI (sample, protein, pathway, goterm, etc.)."""
    uri_str = str(uri)
    if 'Sample_GSM' in uri_str or 'Sample_' in uri_str:
        return 'sample'
    elif 'Protein_' in uri_str:
        return 'protein'
    elif 'Pathway_' in uri_str or 'R-HSA-' in uri_str:
        return 'pathway'
    elif 'Reaction_' in uri_str:
        return 'reaction'
    elif 'Gene_' in uri_str:
        return 'gene'
    elif 'GO:' in uri_str or '/GO_' in uri_str:
        return 'goterm'
    elif 'Disease_' in uri_str or 'DOID:' in uri_str:
        return 'disease'
    elif 'Drug_' in uri_str:
        return 'drug'
    else:
        return 'other'


def _get_node_id(uri):
    """Extract clean node ID from URI."""
    uri_str = str(uri)
    if '#' in uri_str:
        return uri_str.split('#')[-1]
    else:
        return uri_str.split('/')[-1]


def _load_owl_graph(owl_path=None):
    """Load OWL file with RDFlib."""
    if owl_path is None:
        owl_path = owl_filepath
    g = Graph()
    g.parse(str(owl_path), format='xml')
    return g


def _extract_nodes_and_edges(g, node_types_filter=None):
    """Extract nodes and edges from RDF graph."""
    nodes = defaultdict(set)
    edges = defaultdict(list)
    sample_labels = {}

    if node_types_filter is None:
        node_types_filter = {'sample', 'protein', 'pathway', 'goterm'}

    for subject, predicate, obj in g:
        subj_type = _get_node_type(subject)
        obj_type = _get_node_type(obj)
        pred_name = str(predicate).split('#')[-1].split('/')[-1]
        subj_id = _get_node_id(subject)
        obj_id = _get_node_id(obj)

        if subj_type in node_types_filter:
            nodes[subj_type].add(subj_id)
        if obj_type in node_types_filter:
            nodes[obj_type].add(obj_id)

        if subj_type == 'sample' and pred_name == 'hasDiseaseStatus':
            status = str(obj).lower()
            label = 1 if 'sepsis' in status else 0
            sample_labels[subj_id] = label

        if subj_type in node_types_filter and obj_type in node_types_filter:
            edge_key = (subj_type, pred_name, obj_type)
            edges[edge_key].append((subj_id, obj_id))

    for node_type in nodes:
        nodes[node_type] = sorted(list(nodes[node_type]))

    return nodes, edges, sample_labels


def _create_node_features(nodes):
    """Create one-hot encoded features for each node type."""
    node_to_idx = {}
    node_features = {}

    for node_type, node_list in nodes.items():
        n_nodes = len(node_list)
        features = torch.eye(n_nodes, dtype=torch.float)
        node_features[node_type] = features
        node_to_idx[node_type] = {node_id: idx for idx, node_id in enumerate(node_list)}

    return node_features, node_to_idx


def _create_edge_indices(edges, node_to_idx):
    """Convert edge lists to PyTorch Geometric format."""
    edge_indices = {}

    for edge_key, edge_list in edges.items():
        src_type, rel, dst_type = edge_key

        if src_type not in node_to_idx or dst_type not in node_to_idx:
            continue

        src_indices = []
        dst_indices = []

        for src_id, dst_id in edge_list:
            if src_id in node_to_idx[src_type] and dst_id in node_to_idx[dst_type]:
                src_indices.append(node_to_idx[src_type][src_id])
                dst_indices.append(node_to_idx[dst_type][dst_id])

        if len(src_indices) > 0:
            edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
            edge_indices[edge_key] = edge_index

    return edge_indices


def load_heterodata(owl_path=None, save_path=None):
    """Load OWL ontology and convert to HeteroData with all node types (sample, protein, pathway, goterm)."""
    g = _load_owl_graph(owl_path)
    nodes, edges, sample_labels = _extract_nodes_and_edges(g, node_types_filter={'sample', 'protein', 'pathway', 'goterm'})

    node_features, node_to_idx = _create_node_features(nodes)
    edge_indices = _create_edge_indices(edges, node_to_idx)

    n_samples = len(nodes['sample'])
    sample_label_tensor = torch.zeros(n_samples, dtype=torch.long)
    for sample_id, label in sample_labels.items():
        if sample_id in node_to_idx['sample']:
            idx = node_to_idx['sample'][sample_id]
            sample_label_tensor[idx] = label

    data = HeteroData()

    for node_type, features in node_features.items():
        data[node_type].x = features

    data['sample'].y = sample_label_tensor

    for (src_type, rel, dst_type), edge_index in edge_indices.items():
        data[src_type, rel, dst_type].edge_index = edge_index

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, save_path)

    return data


if __name__ == "__main__":
    print("="*80)
    print("Testing load_heterodata module")
    print("="*80)

    data = load_heterodata(save_path=project_root / "data" / "han" / "hetero_graph.pt")

    print(f"\n✓ HeteroData loaded:")
    print(data)
    print(f"\nNode types: {data.node_types}")
    print(f"Edge types: {data.edge_types}")
    print("\n" + "="*80)
    print("Test passed!")
    print("="*80)
