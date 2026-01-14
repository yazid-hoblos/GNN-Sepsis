import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
import GEOparse
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

# Setup paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from load_dataset import create_expression_data
from load_embeddings import load_all_entities

# File paths
data_dir = project_root / "data"
geo_filepath = str(data_dir / "GSE54514_family.soft.gz")
gpl_filepath = str(data_dir / "GPL6947-13512.txt")


from functools import wraps
def prepare_df_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        df = func(*args, **kwargs)
        return df.set_index('label').iloc[:, 2:]
    return wrapper

@prepare_df_decorator
def load_df(key: str, folder_version: str = "v2.9", normalization: str = "robust") -> pd.DataFrame:
    """
    Generic interface to load expression data or Knowledge Graph embeddings.

    Parameters
    ----------
    key : str
        Type of data to load:

        - 'gene_expression' : Gene expression matrix (5000 probes) with sample metadata
        - 'RGCN_sample_embeddings' : Sample embeddings from RGCN model (128 dims)
        - 'Complex_sample_embeddings' : Sample embeddings from ComplEx model (128 dims)
        - 'concatenated_sample_embeddings' : Both RGCN + ComplEx concatenated (256 dims)
        - 'RGCN_protein_embeddings' : Protein embeddings from RGCN, weighted by gene expression
        - 'Complex_protein_embeddings' : Protein embeddings from ComplEx, weighted by gene expression
        - 'concatenated_protein_embeddings' : Both RGCN + ComplEx protein embeddings (256 dims)

        GNN models from results/embeddings/ (100 dims):
        - 'GraphSAGE_sample_embeddings' : Sample embeddings from GraphSAGE
        - 'weighted_RGCN_sample_embeddings' : Sample embeddings from RGCN with edge weights
        - 'GAT_sample_embeddings' : Sample embeddings from GAT
        - 'GraphSAGE_protein_embeddings' : Protein embeddings from GraphSAGE, weighted by expression
        - 'weighted_RGCN_protein_embeddings' : Protein embeddings from RGCN with edge weights, weighted by expression
        - 'GAT_protein_embeddings' : Protein embeddings from GAT, weighted by expression

    folder_version : str, optional
        Version of embeddings folder in models/executions/ (default: 'v2.9')
        Used for RGCN/ComplEx embeddings from Knowledge Graph.

    gnn_version : str, optional
        Version of GNN embeddings in results/embeddings/ (default: 'v2.9')
        Options: 'v2.9', 'v2.10', 'v2.11', etc.
        Used for GraphSAGE/weighted_RGCN/GAT embeddings.

    normalization : str, optional
        Normalization method to apply to gene expression data (default: 'robust')
        Options: 'robust', 'standard', 'minmax', 'log1p', 'none'

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: [label, hasAge, hasGender, <data_cols>, disease_status]


    Examples
    --------
    >>> df = load_df('gene_expression', 'v2.9', normalization='robust')
    >>> df = load_df('GraphSAGE_sample_embeddings', folder_version='v2.10')
    >>> df = load_df('GAT_protein_embeddings', folder_version='v2.9')
    """
    # GNN models
    gnn_sample_keys = [
        "GraphSAGE_sample_embeddings",
        "weighted_RGCN_sample_embeddings",
        "GAT_sample_embeddings"
    ]
    gnn_protein_keys = [
        "GraphSAGE_protein_embeddings",
        "weighted_RGCN_protein_embeddings",
        "GAT_protein_embeddings"
    ]

    if key == "gene_expression":
        return load_gene_expression(normalization=normalization)
    elif key in ("RGCN_sample_embeddings", "Complex_sample_embeddings"):
        return load_sample_embeddings(key.split("_")[0], folder_version)
    elif key == "concatenated_sample_embeddings":
        return load_concatenate_sample_embeddings(folder_version)
    elif key in ("RGCN_protein_embeddings", "Complex_protein_embeddings"):
        return load_protein_embeddings(key.split("_")[0], folder_version, normalization=normalization)
    elif key == "concatenated_protein_embeddings":
        return load_concatenate_protein_embeddings(folder_version, normalization=normalization)
    elif key in gnn_sample_keys:
        model_name = key.replace("_sample_embeddings", "")
        return load_gnn_sample_embeddings(model_name, gnn_version=folder_version)
    elif key in gnn_protein_keys:
        model_name = key.replace("_protein_embeddings", "")
        return load_gnn_protein_embeddings(model_name, gnn_version=folder_version, normalization=normalization)
    else:
        raise ValueError(f"Unknown data key: {key}")


# ============================================================================
# Gene Expression Loading
# ============================================================================

def load_gene_expression(normalization: str = "robust") -> pd.DataFrame:
    """
    Load gene expression matrix from GSE54514 with sample features and disease status."""
    # Load GEO dataset
    gse = GEOparse.get_GEO(filepath=geo_filepath, silent=True)

    # Extract expression matrix
    expression_data = create_expression_data(gse)

    # Apply normalization if requested
    if normalization != "none":
        expression_data = normalize_gene_expression(expression_data, method=normalization)

    # Load sample features
    df_features = _load_sample_features()

    # Combine features + expression + disease_status
    return _add_features_and_disease_status(expression_data, df_features)


# ============================================================================
# Common Embedding Utilities
# ============================================================================

def _get_embeddings_paths(model_name: str, folder_version: str = "v2.9") -> tuple:
    """Get file paths for embeddings data."""
    folder_name = f"GSE54514_enriched_ontology_degfilter{folder_version}"
    models_dir = project_root / "models" / "executions" / folder_name

    emb_path = str(models_dir / f"outputmodel_{model_name}_entity_embeddings.npy")
    map_csv = str(models_dir / f"outputmodel_{model_name}_entity_mapping.csv")
    node_features_csv = str(models_dir / "node_features.csv")

    return emb_path, map_csv, node_features_csv


def _expand_embedding_column(df: pd.DataFrame, prefix: str = 'emb') -> pd.DataFrame:
    """Transform embedding column (numpy arrays) into separate columns (emb_0, emb_1, ...)."""
    # Stack all embeddings into a 2D array
    embeddings_array = np.stack(df['embedding'].values)
    embedding_dim = embeddings_array.shape[1]

    # Create columns for each embedding dimension
    for i in range(embedding_dim):
        df[f'{prefix}_{i}'] = embeddings_array[:, i]

    # Drop the original embedding column
    df = df.drop(columns=['embedding'])

    return df


def _load_filtered_entities(model_name: str, folder_version: str, entity_prefix: str, pivot: bool = False) -> pd.DataFrame:
    """Load entities filtered by label prefix ('Sample_', 'Protein_', etc.) from Knowledge Graph."""

    emb_path, map_csv, node_features_csv = _get_embeddings_paths(model_name, folder_version)
    _, _, _, df_all = load_all_entities(emb_path, map_csv, node_features_csv)

    # Filter by prefix on label
    mask = df_all['label'].str.startswith(entity_prefix, na=False)
    df_filtered = df_all[mask].copy()

    if pivot and 'name_feature' in df_filtered.columns:
        # Extract embeddings for each entity
        df_embeddings = df_filtered.groupby(['entity_id', 'label'])['embedding'].first().reset_index()
        # Pivot the features to wide format (one row per entity)
        df_features_pivot = df_filtered.pivot_table(
            index=['entity_id', 'label'],
            columns='name_feature',
            values='value_feature',
            aggfunc='first'
        ).reset_index()
        # Merge embeddings back
        df_filtered = pd.merge(df_features_pivot, df_embeddings, on=['entity_id', 'label'], how='left')

    # Transform embedding column into separate columns (emb_0, emb_1, ...)
    df_filtered = _expand_embedding_column(df_filtered, prefix='emb')

    return df_filtered


def _concatenate_model_embeddings(load_func, folder_version: str, normalization: str = None) -> pd.DataFrame:
    """Generic function to concatenate RGCN and ComplEx embeddings."""
    # Load both models (pass normalization if provided)
    if normalization is not None:
        df_rgcn = load_func('RGCN', folder_version, normalization=normalization)
        df_complex = load_func('ComplEx', folder_version, normalization=normalization)
    else:
        df_rgcn = load_func('RGCN', folder_version)
        df_complex = load_func('ComplEx', folder_version)

    # Verify disease status matches
    assert (df_rgcn['disease_status'] == df_complex['disease_status']).all(), \
        "Disease status mismatch between RGCN and ComplEx"

    # Identify feature columns to keep
    feature_cols = ['label', 'hasAge', 'hasGender']

    # Extract features (same for both models)
    df_features = df_rgcn[feature_cols].copy()

    # Extract embedding columns (emb_* columns)
    rgcn_emb_cols = [c for c in df_rgcn.columns if c.startswith('emb_')]
    complex_emb_cols = [c for c in df_complex.columns if c.startswith('emb_')]

    # Rename with model prefix
    df_rgcn_emb = df_rgcn[rgcn_emb_cols].copy()
    df_complex_emb = df_complex[complex_emb_cols].copy()

    df_rgcn_emb.columns = [f'rgcn_{col}' for col in rgcn_emb_cols]
    df_complex_emb.columns = [f'complex_{col}' for col in complex_emb_cols]

    # Concatenate in order: features → rgcn_emb → complex_emb → disease_status
    df_concat = pd.concat([df_features, df_rgcn_emb, df_complex_emb], axis=1)
    df_concat['disease_status'] = df_rgcn['disease_status'].values

    return df_concat


# ============================================================================
# Common Sample Features Utilities
# ============================================================================

def _load_sample_features(folder_version: str = "v2.9") -> pd.DataFrame:
    """Load sample features (metadata) from KG."""
    # Load samples with pivot to get features (use RGCN arbitrarily, features are model-independent)
    df_samples = _load_filtered_entities('RGCN', folder_version, "Sample_", pivot=True)

    # Keep only feature columns (drop embedding columns emb_*)
    emb_cols = [c for c in df_samples.columns if c.startswith('emb_')]
    df_features = df_samples.drop(columns=emb_cols).copy()

    # Extract GSM ID from label for merging with GEO data
    df_features['gsm_id'] = df_features['label'].str.replace('Sample_', '', regex=False)

    return df_features


def _add_features_and_disease_status(df_data: pd.DataFrame, df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Combine sample features + data + disease_status in correct order.
    Order: features (label, hasAge, hasGender) → data columns → disease_status
    """
    # Reset index to get gsm_id as column
    df_data_reset = df_data.reset_index().rename(columns={'index': 'gsm_id'})

    # Merge features with data
    df_merged = pd.merge(df_features, df_data_reset, on='gsm_id', how='inner')

    # Create disease_status from hasDiseaseStatus
    df_merged['disease_status'] = df_merged['hasDiseaseStatus'].apply(
        lambda x: 0 if str(x).lower() == 'healthy' else 1
    )

    # Reorder columns: selected features → data → disease_status
    feature_cols = ['label', 'hasAge', 'hasGender']
    data_cols = [c for c in df_merged.columns if c not in df_features.columns and c != 'disease_status']

    df_result = df_merged[feature_cols + data_cols + ['disease_status']].copy()

    return df_result


# ============================================================================
# Sample Embeddings Loading
# ============================================================================

def load_sample_embeddings(model_name: str, folder_version: str = "v2.9") -> pd.DataFrame:
    """Load sample embeddings for a specific model with all patient features."""
    # Load samples once with embeddings already expanded into columns (emb_0, emb_1, ...)
    df_samples = _load_filtered_entities(model_name, folder_version, "Sample_", pivot=True)

    # Separate features and embeddings from the same DataFrame
    emb_cols = [c for c in df_samples.columns if c.startswith('emb_')]
    df_embeddings = df_samples[emb_cols].copy()
    df_embeddings['gsm_id'] = df_samples['label'].str.replace('Sample_', '', regex=False).values
    df_embeddings = df_embeddings.set_index('gsm_id')

    # Extract features directly from df_samples (no need to reload!)
    feature_col_names = [c for c in df_samples.columns if not c.startswith('emb_')]
    df_features = df_samples[feature_col_names].copy()
    df_features['gsm_id'] = df_features['label'].str.replace('Sample_', '', regex=False)

    # Combine features + embeddings + disease_status
    return _add_features_and_disease_status(df_embeddings, df_features)


def load_concatenate_sample_embeddings(folder_version: str = "v2.9") -> pd.DataFrame:
    """Concatenate RGCN and ComplEx sample embeddings."""
    return _concatenate_model_embeddings(load_sample_embeddings, folder_version)


# ============================================================================
# Protein Embeddings Loading
# ============================================================================

def _load_raw_protein_embeddings(model_name: str, folder_version: str = "v2.9"):
    """Load raw protein embeddings from KG and extract gene symbols."""
    # Load proteins without pivot (proteins don't have multiple features like samples)
    df_proteins = _load_filtered_entities(model_name, folder_version, "Protein_", pivot=False)

    # Keep only one row per protein (dedup by label)
    df_proteins = df_proteins.drop_duplicates(subset=['label']).reset_index(drop=True)

    # Extract gene symbol from protein label (e.g., "Protein_ACTB" → "ACTB")
    df_proteins['gene_symbol'] = df_proteins['label'].str.replace('Protein_', '', regex=False)

    return df_proteins


def load_probe_to_gene_mapping() -> pd.DataFrame:
    """
    Load mapping from probes to gene symbols and Entrez IDs.
    """
    gpl = pd.read_csv(gpl_filepath, sep="\t", comment='#')
    probe_to_gene = gpl[['ID', 'Entrez_Gene_ID', 'Symbol']].copy()

    # Clean Entrez IDs (keep first if multiple)
    probe_to_gene['Entrez_Gene_ID'] = probe_to_gene['Entrez_Gene_ID'].apply(
        lambda x: str(x).split('///')[0].strip() if pd.notna(x) else None
    )
    probe_to_gene['Entrez_Gene_ID'] = pd.to_numeric(probe_to_gene['Entrez_Gene_ID'], errors='coerce')
    probe_to_gene = probe_to_gene.dropna(subset=['Entrez_Gene_ID'])
    probe_to_gene['Entrez_Gene_ID'] = probe_to_gene['Entrez_Gene_ID'].astype(int)

    return probe_to_gene


def aggregate_probes_to_genes(expression_data: pd.DataFrame, probe_to_gene: pd.DataFrame) -> pd.DataFrame:
    """Aggregate probe-level expression to gene-level (mean)."""
    gene_expression = {}
    for gene_id in probe_to_gene['Entrez_Gene_ID'].unique():
        probes = probe_to_gene[probe_to_gene['Entrez_Gene_ID'] == gene_id]['ID'].values
        probes_in_data = [p for p in probes if p in expression_data.columns]
        if probes_in_data:
            gene_expression[gene_id] = expression_data[probes_in_data].mean(axis=1).values

    return pd.DataFrame(gene_expression, index=expression_data.index)


def normalize_gene_expression(gene_expr_matrix: pd.DataFrame, method: str = "robust") -> pd.DataFrame:
    """
    Normalize gene expression matrix using specified method.
        Normalization method (default: 'robust')
        - 'robust': RobustScaler (median and IQR, robust to outliers)
        - 'standard': StandardScaler (mean=0, std=1, Z-score)
        - 'minmax': MinMaxScaler (scale to [0, 1])
        - 'log1p': Log1p transformation (log(1 + x))
        - 'none': No normalization
    """
    if method == "none":
        return gene_expr_matrix.copy()

    normalized_matrix = gene_expr_matrix.copy()

    if method == "robust":
        scaler = RobustScaler()
        normalized_matrix[:] = scaler.fit_transform(normalized_matrix)
    elif method == "standard":
        scaler = StandardScaler()
        normalized_matrix[:] = scaler.fit_transform(normalized_matrix)
    elif method == "minmax":
        scaler = MinMaxScaler()
        normalized_matrix[:] = scaler.fit_transform(normalized_matrix)
    elif method == "log1p":
        # Log1p: log(1 + x), useful for count data
        normalized_matrix = np.log1p(normalized_matrix)
    else:
        raise ValueError(f"Unknown normalization method: {method}. "
                         f"Options: 'robust', 'standard', 'minmax', 'log1p', 'none'")

    return normalized_matrix


def map_proteins_to_genes(df_proteins: pd.DataFrame, probe_to_gene: pd.DataFrame) -> pd.DataFrame:
    """Map protein symbols to Entrez Gene IDs."""
    # Create symbol → Entrez ID mapping
    symbol_to_gene = probe_to_gene[['Symbol', 'Entrez_Gene_ID']].drop_duplicates()
    symbol_to_gene = symbol_to_gene.groupby('Symbol')['Entrez_Gene_ID'].first().to_dict()

    df_proteins['entrez_gene_id'] = df_proteins['gene_symbol'].map(symbol_to_gene)
    df_proteins = df_proteins.dropna(subset=['entrez_gene_id'])
    df_proteins['entrez_gene_id'] = df_proteins['entrez_gene_id'].astype(int)

    return df_proteins


def create_weighted_protein_embeddings(df_proteins: pd.DataFrame, gene_expr_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Create sample-level features by weighting protein embeddings with gene expression.
    For each sample: aggregated_feature_i = Σ(expression_gene × embedding_protein_i) / Σ(expression_gene)
    """
    n_samples = len(gene_expr_matrix)

    # Get embedding columns
    emb_cols = [c for c in df_proteins.columns if c.startswith('emb_')]
    embedding_dim = len(emb_cols)

    # Initialize aggregated features
    X_aggregated = np.zeros((n_samples, embedding_dim))
    total_expression = np.zeros(n_samples)

    for _, protein_row in df_proteins.iterrows():
        gene_id = protein_row['entrez_gene_id']

        if gene_id not in gene_expr_matrix.columns:
            continue

        protein_embedding = protein_row[emb_cols].to_numpy(dtype=float)  # (embedding_dim,)
        gene_expr = gene_expr_matrix[gene_id].values  # (n_samples,)

        # Weighted embedding: (n_samples,) × (embedding_dim,) → (n_samples, embedding_dim)
        weighted_emb = gene_expr[:, np.newaxis] * protein_embedding[np.newaxis, :]

        X_aggregated += weighted_emb
        total_expression += gene_expr

    # Normalize by total expression
    X_aggregated = X_aggregated / (total_expression[:, np.newaxis] + 1e-10)

    # Create DataFrame with embedding columns and GSM IDs as index
    df_embeddings = pd.DataFrame(X_aggregated, columns=emb_cols, index=gene_expr_matrix.index)

    return df_embeddings


def load_protein_embeddings(model_name: str, folder_version: str = "v2.9", normalization: str = "robust") -> pd.DataFrame:
    """ Load protein embeddings weighted by gene expression for each sample with features."""
    # Load protein embeddings
    df_proteins = _load_raw_protein_embeddings(model_name, folder_version)

    # Load gene expression data
    gse = GEOparse.get_GEO(filepath=geo_filepath, silent=True)
    expression_data = create_expression_data(gse)

    # Load probe-to-gene mapping
    probe_to_gene = load_probe_to_gene_mapping()

    # Aggregate probes to genes
    gene_expr_matrix = aggregate_probes_to_genes(expression_data, probe_to_gene)

    # Normalize gene expression
    gene_expr_matrix = normalize_gene_expression(gene_expr_matrix, method=normalization)

    # Map proteins to genes
    df_proteins = map_proteins_to_genes(df_proteins, probe_to_gene)

    # Filter proteins that have expression data
    df_proteins = df_proteins[df_proteins['entrez_gene_id'].isin(gene_expr_matrix.columns)]

    # Create weighted embeddings (with GSM IDs as index)
    df_protein_embeddings = create_weighted_protein_embeddings(df_proteins, gene_expr_matrix)

    # Load sample features
    df_features = _load_sample_features(folder_version)

    # Combine features + protein embeddings + disease_status
    return _add_features_and_disease_status(df_protein_embeddings, df_features)


def load_concatenate_protein_embeddings(folder_version: str = "v2.9", normalization: str = "robust") -> pd.DataFrame:
    """ Concatenate RGCN and ComplEx protein embeddings at sample level. """
    return _concatenate_model_embeddings(load_protein_embeddings, folder_version, normalization)


# ============================================================================
# GNN Embeddings Loading (from results/embeddings/)
# ============================================================================

def load_gnn_sample_embeddings(model_name: str, gnn_version: str = "v2.10") -> pd.DataFrame:
    """Load sample embeddings from GNN models (GraphSAGE, weighted_RGCN, GAT)."""
    # Remove weighted_ prefix if present
    clean_model_name = model_name.replace("weighted_", "")

    # Map model names to folder names
    model_folder_map = {
        "GraphSAGE": "sage",
        "RGCN": "rgcn",
        "GAT": "gat"
    }

    if clean_model_name not in model_folder_map:
        raise ValueError(f"Unknown GNN model: {model_name}")

    folder_name = model_folder_map[clean_model_name]
    embeddings_path = project_root / "results" / "embeddings" / gnn_version / folder_name / "sample_embeddings.csv"

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings not found at {embeddings_path}. Run the model first.")

    # Load embeddings array from CSV
    embeddings_array = pd.read_csv(embeddings_path).values
    n_samples, embedding_dim = embeddings_array.shape

    # Create DataFrame with embedding columns
    emb_cols = [f'emb_{i}' for i in range(embedding_dim)]
    df_embeddings = pd.DataFrame(embeddings_array, columns=emb_cols)

    # Load sample features to get GSM IDs and metadata
    df_features = _load_sample_features()

    # Match embeddings to samples (assuming same order as in hetero_graph.pt)
    if len(df_embeddings) != len(df_features):
        raise ValueError(f"Mismatch: {len(df_embeddings)} embeddings vs {len(df_features)} samples")

    df_embeddings['gsm_id'] = df_features['gsm_id'].values
    df_embeddings = df_embeddings.set_index('gsm_id')

    # Combine features + embeddings + disease_status
    return _add_features_and_disease_status(df_embeddings, df_features)


def _load_gnn_protein_node_ids(gnn_version: str = "v2.9") -> list:
    """Load protein node IDs from hetero_graph to get gene symbols."""
    import torch
    from src.han.load_heterodata import _load_owl_graph, _extract_nodes_and_edges, get_owl_path

    # Load OWL to get node IDs in correct order (same version as used in hetero_graph)
    owl_path = get_owl_path(gnn_version)

    g = _load_owl_graph(owl_path)
    nodes, _, _ = _extract_nodes_and_edges(g, node_types_filter={'sample', 'protein', 'pathway', 'goterm'})

    # Return protein node IDs in same order as in hetero_graph
    protein_node_ids = nodes['protein']
    return protein_node_ids


def load_gnn_protein_embeddings(model_name: str, gnn_version: str = "v2.10", normalization: str = "robust") -> pd.DataFrame:
    """Load protein embeddings from GNN models, weighted by gene expression."""
    # Remove weighted_ prefix if present
    clean_model_name = model_name.replace("weighted_", "")

    # Map model names to folder names
    model_folder_map = {
        "GraphSAGE": "sage",
        "RGCN": "rgcn",
        "GAT": "gat"
    }

    if clean_model_name not in model_folder_map:
        raise ValueError(f"Unknown GNN model: {model_name}")

    folder_name = model_folder_map[clean_model_name]
    embeddings_path = project_root / "results" / "embeddings" / gnn_version / folder_name / "protein_embeddings.csv"

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings not found at {embeddings_path}. Run the model first.")

    # Load protein embeddings array from CSV
    protein_embeddings_array = pd.read_csv(embeddings_path).values
    n_proteins, embedding_dim = protein_embeddings_array.shape

    # Create DataFrame with protein embeddings
    emb_cols = [f'emb_{i}' for i in range(embedding_dim)]

    # Load protein node IDs from hetero_graph to get gene symbols
    protein_node_ids = _load_gnn_protein_node_ids(gnn_version)

    if len(protein_node_ids) != n_proteins:
        raise ValueError(f"Mismatch: {n_proteins} embeddings vs {len(protein_node_ids)} protein nodes")

    # Create DataFrame with protein embeddings and gene symbols
    df_proteins = pd.DataFrame(protein_embeddings_array, columns=emb_cols)
    df_proteins['label'] = protein_node_ids
    df_proteins['gene_symbol'] = df_proteins['label'].str.replace('Protein_', '', regex=False)

    # Load gene expression and mapping
    gse = GEOparse.get_GEO(filepath=geo_filepath, silent=True)
    expression_data = create_expression_data(gse)
    probe_to_gene = load_probe_to_gene_mapping()
    gene_expr_matrix = aggregate_probes_to_genes(expression_data, probe_to_gene)
    gene_expr_matrix = normalize_gene_expression(gene_expr_matrix, method=normalization)

    # Map proteins to genes (same as RGCN/Complex)
    df_proteins = map_proteins_to_genes(df_proteins, probe_to_gene)

    # Filter proteins that have expression data
    df_proteins = df_proteins[df_proteins['entrez_gene_id'].isin(gene_expr_matrix.columns)]

    # Create weighted embeddings (same approach as RGCN/Complex)
    df_protein_embeddings = create_weighted_protein_embeddings(df_proteins, gene_expr_matrix)

    # Load sample features
    df_features = _load_sample_features()

    # Combine features + protein embeddings + disease_status
    return _add_features_and_disease_status(df_protein_embeddings, df_features)