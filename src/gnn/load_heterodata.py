from pathlib import Path
from rdflib import Graph, Namespace, RDF, OWL
import torch
from torch_geometric.data import HeteroData
from collections import defaultdict
import xml.etree.ElementTree as ET
import numpy as np
import argparse

project_root = Path(__file__).parent.parent.parent

# Version configurations
OWL_VERSIONS = {
    'v2.9': {
        'owl_dir': 'GSE54514_enriched_ontology_degfilterv2.9',
        'owl_file': 'GSE54514_enriched_ontology_degfilterv2.9.owl',
    },
    'v2.10': {
        'owl_dir': 'GSE54514_enriched_ontology_degfilterv2.10',
        'owl_file': 'GSE54514_enriched_ontology_degfilter_v2.10_ovp0.2_ng3.owl',
    },
    'v2.11': {
        'owl_dir': 'GSE54514_enriched_ontology_degfilterv2.11',
        'owl_file': 'GSE54514_enriched_ontology_degfilter_v2.11.owl',
    },
}

def get_owl_path(version: str = 'v2.11') -> Path:
    """Get OWL file path for a specific version."""
    if version not in OWL_VERSIONS:
        raise ValueError(f"Unknown version: {version}. Choose from {list(OWL_VERSIONS.keys())}")
    config = OWL_VERSIONS[version]
    return project_root / "models" / "executions" / config['owl_dir'] / config['owl_file']

def get_save_path(version: str = 'v2.11') -> Path:
    """Get save path for HeteroData for a specific version."""
    return project_root / "data" / "han" / version / "hetero_graph.pt"


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


def _load_owl_graph(owl_path):
    """Load OWL file with RDFlib."""
    g = Graph()
    g.parse(str(owl_path), format='xml')
    return g


def _extract_edge_weights_from_axioms(owl_path):
    """Extract edge weights from OWL Axioms with hasExpressionValue."""
    print("  Extracting edge weights from OWL Axioms...")
    edge_weights = {}  # {(source, relation, target): weight}

    print("  Parsing XML file...")
    tree = ET.parse(owl_path)
    root = tree.getroot()
    print("  XML parsed successfully!")

    # Define namespaces
    ns = {
        'owl': 'http://www.w3.org/2002/07/owl#',
        'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'
    }

    print("  Finding axioms...")
    axiom_count = 0
    for i, axiom in enumerate(root.findall('.//owl:Axiom', ns)):
        if i % 50000 == 0 and i > 0:
            print(f"    Processed {i} axioms, found {axiom_count} with expression values...")

        # Get annotated source, property, and target
        source_elem = axiom.find('owl:annotatedSource', ns)
        prop_elem = axiom.find('owl:annotatedProperty', ns)
        target_elem = axiom.find('owl:annotatedTarget', ns)

        if source_elem is None or prop_elem is None or target_elem is None:
            continue

        source = source_elem.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource', '').split('#')[-1]
        relation = prop_elem.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource', '').split('#')[-1]
        target = target_elem.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource', '').split('#')[-1]

        # Get all hasExpressionValue elements (usually 3 replicates)
        expr_values = []
        for expr_elem in axiom.findall('.//{*}hasExpressionValue'):
            try:
                expr_values.append(float(expr_elem.text))
            except (ValueError, TypeError):
                pass

        if expr_values and source and relation and target:
            # Average the replicate values
            avg_weight = np.mean(expr_values)
            edge_weights[(source, relation, target)] = avg_weight
            axiom_count += 1

    print(f"  Found {axiom_count} edges with expression values")
    return edge_weights


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


def _create_node_features(nodes, embedding_dim=128):
    """Create fixed-size embeddings for each node type."""
    node_to_idx = {}
    node_features = {}

    torch.manual_seed(42)
    for node_type, node_list in nodes.items():
        n_nodes = len(node_list)
        features = torch.randn(n_nodes, embedding_dim, dtype=torch.float)
        features = torch.nn.functional.normalize(features, dim=1)
        node_features[node_type] = features
        node_to_idx[node_type] = {node_id: idx for idx, node_id in enumerate(node_list)}

    return node_features, node_to_idx


def _create_edge_indices(edges, node_to_idx, edge_weights_dict=None):
    """Convert edge lists to PyTorch Geometric format with optional edge weights."""
    edge_indices = {}
    edge_weights = {}

    for edge_key, edge_list in edges.items():
        src_type, rel, dst_type = edge_key

        if src_type not in node_to_idx or dst_type not in node_to_idx:
            continue

        src_indices = []
        dst_indices = []
        weights = []

        for src_id, dst_id in edge_list:
            if src_id in node_to_idx[src_type] and dst_id in node_to_idx[dst_type]:
                src_indices.append(node_to_idx[src_type][src_id])
                dst_indices.append(node_to_idx[dst_type][dst_id])

                # Get edge weight if available
                if edge_weights_dict:
                    weight_key = (src_id, rel, dst_id)
                    weight = edge_weights_dict.get(weight_key, 1.0)  # Default weight = 1.0
                    weights.append(weight)

        if len(src_indices) > 0:
            edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
            edge_indices[edge_key] = edge_index

            if edge_weights_dict and weights:
                edge_weights[edge_key] = torch.tensor(weights, dtype=torch.float)

    return edge_indices, edge_weights


def load_heterodata(owl_path=None, save_path=None, version: str = 'v2.11'):
    """Load OWL ontology and convert to HeteroData with edge weights.

    Args:
        owl_path: Path to OWL file (overrides version if provided)
        save_path: Path to save HeteroData (overrides version if provided)
        version: Version to use ('v2.9', 'v2.10', 'v2.11')
    """
    if owl_path is None:
        owl_path = get_owl_path(version)
    if save_path is None:
        save_path = get_save_path(version)

    print("Loading OWL graph with RDFlib...")
    g = _load_owl_graph(owl_path)
    print("  RDF graph loaded!")

    print("\nExtracting nodes and edges from RDF graph...")
    nodes, edges, sample_labels = _extract_nodes_and_edges(g, node_types_filter={'sample', 'protein', 'pathway', 'goterm'})
    print(f"  Found {len(nodes)} node types, {sum(len(e) for e in edges.values())} total edges")

    print("\nExtracting edge weights from OWL...")
    edge_weights_dict = _extract_edge_weights_from_axioms(owl_path)

    print("\nCreating node features and edge indices...")
    node_features, node_to_idx = _create_node_features(nodes)
    edge_indices, edge_weights = _create_edge_indices(edges, node_to_idx, edge_weights_dict)

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

    # Add edges and edge weights
    for (src_type, rel, dst_type), edge_index in edge_indices.items():
        data[src_type, rel, dst_type].edge_index = edge_index

        # Add edge weights if available
        if (src_type, rel, dst_type) in edge_weights:
            data[src_type, rel, dst_type].edge_weight = edge_weights[(src_type, rel, dst_type)]
            print(f"  Added edge weights for {src_type} -> {rel} -> {dst_type}: {edge_weights[(src_type, rel, dst_type)].shape}")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, save_path)

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load OWL ontology and convert to HeteroData')
    parser.add_argument('--version', '-v', type=str, default='v2.11',
                        choices=['v2.9', 'v2.10', 'v2.11'],
                        help='OWL version to use (default: v2.11)')
    args = parser.parse_args()

    print("="*80)
    print(f"Loading HeteroData from {args.version}")
    print("="*80)

    data = load_heterodata(version=args.version)

    print(f"\n✓ HeteroData loaded:")
    print(data)
    print(f"\nNode types: {data.node_types}")
    print(f"Edge types: {data.edge_types}")
    print("\n" + "="*80)
    print(f"Saved to: {get_save_path(args.version)}")
    print("="*80)
