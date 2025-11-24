"""
Export multi-level graphs for Gephi visualization.
Creates separate graph layers for different entity types and relationships.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

# Import from parent directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_embeddings import load_patient_data, load_all_entities


def export_patient_similarity_network(X_patients, y_patients, patient_ids, df_patients, 
                                      output_dir='gephi_exports', threshold=0.5):
    """
    Export patient similarity network.
    
    Parameters
    ----------
    X_patients : np.ndarray
        Patient embeddings
    y_patients : np.ndarray
        Patient labels
    patient_ids : list
        Patient entity IDs
    df_patients : pd.DataFrame
        Patient metadata
    output_dir : str
        Output directory
    threshold : float
        Similarity threshold for creating edges
    """
    print("\n" + "="*80)
    print("LAYER 1: Patient Similarity Network")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute similarity matrix
    sim_matrix = cosine_similarity(X_patients)
    
    # Create nodes file
    nodes = []
    for i, (pid, label) in enumerate(zip(patient_ids, y_patients)):
        node_id = df_patients.iloc[i]['node_id']
        nodes.append({
            'Id': node_id,
            'Label': node_id,
            'Disease_Status': 'Sepsis' if label == 1 else 'Healthy',
            'Disease_Status_Code': int(label),
            'Entity_ID': int(pid)
        })
    
    nodes_df = pd.DataFrame(nodes)
    nodes_df.to_csv(f"{output_dir}/layer1_patient_nodes.csv", index=False)
    
    # Create edges file (only similarities above threshold)
    edges = []
    for i in range(len(patient_ids)):
        for j in range(i+1, len(patient_ids)):
            similarity = sim_matrix[i, j]
            if similarity >= threshold:
                edges.append({
                    'Source': nodes_df.iloc[i]['Id'],
                    'Target': nodes_df.iloc[j]['Id'],
                    'Weight': float(similarity),
                    'Type': 'Undirected'
                })
    
    edges_df = pd.DataFrame(edges)
    edges_df.to_csv(f"{output_dir}/layer1_patient_edges.csv", index=False)
    
    print(f"  Nodes: {len(nodes)} patients")
    print(f"  Edges: {len(edges)} similarities (threshold={threshold})")
    print(f"  Files: layer1_patient_nodes.csv, layer1_patient_edges.csv")


def export_gene_interaction_network(df_edges, df_all, entity_labels, 
                                    output_dir='gephi_exports', top_genes=200):
    """
    Export gene-gene interaction network.
    
    Parameters
    ----------
    df_edges : pd.DataFrame
        Edge attributes from knowledge graph
    df_all : pd.DataFrame
        All entities dataframe
    entity_labels : list
        Entity labels
    output_dir : str
        Output directory
    top_genes : int
        Number of top genes to include
    """
    print("\n" + "="*80)
    print("LAYER 2: Gene Interaction Network")
    print("="*80)
    
    # Filter to protein-protein interactions only
    gene_edges = df_edges[
        (df_edges['subject'].str.startswith('Protein_', na=False)) &
        (df_edges['object'].str.startswith('Protein_', na=False)) &
        (df_edges['predicate'] == 'hasPhysicalInteractionWith')
    ].copy()
    
    if len(gene_edges) == 0:
        print("  ⚠️  No gene-gene interactions found in edges")
        return
    
    # Get top connected genes
    all_genes = pd.concat([gene_edges['subject'], gene_edges['object']]).value_counts()
    top_gene_ids = all_genes.head(top_genes).index.tolist()
    
    # Filter edges to top genes
    gene_edges = gene_edges[
        gene_edges['subject'].isin(top_gene_ids) &
        gene_edges['object'].isin(top_gene_ids)
    ]
    
    # Create nodes
    nodes = []
    for gene_id in top_gene_ids:
        nodes.append({
            'Id': gene_id,
            'Label': f"Gene_{gene_id}",
            'Type': 'Gene'
        })
    
    nodes_df = pd.DataFrame(nodes)
    nodes_df.to_csv(f"{output_dir}/layer2_gene_nodes.csv", index=False)
    
    # Create edges
    edges = []
    for _, row in gene_edges.iterrows():
        edges.append({
            'Source': row['subject'],
            'Target': row['object'],
            'Relation': row['predicate'],
            'Type': 'Directed' if row['predicate'] == 'regulates' else 'Undirected',
            'Weight': 1.0
        })
    
    edges_df = pd.DataFrame(edges)
    edges_df.to_csv(f"{output_dir}/layer2_gene_edges.csv", index=False)
    
    print(f"  Nodes: {len(nodes)} genes")
    print(f"  Edges: {len(edges)} interactions")
    print(f"  Files: layer2_gene_nodes.csv, layer2_gene_edges.csv")


def export_pathway_network(df_edges, output_dir='gephi_exports'):
    """
    Export pathway-pathway network.
    
    Parameters
    ----------
    df_edges : pd.DataFrame
        Edge attributes from knowledge graph
    output_dir : str
        Output directory
    """
    print("\n" + "="*80)
    print("LAYER 3: Pathway Network")
    print("="*80)
    
    # Filter to pathway nodes (GO terms, REACT terms)
    pathway_edges = df_edges[
        (df_edges['subject'].str.startswith('GO:', na=False) | 
         df_edges['subject'].str.startswith('REACT:', na=False)) &
        (df_edges['object'].str.startswith('GO:', na=False) | 
         df_edges['object'].str.startswith('REACT:', na=False))
    ].copy()
    
    if len(pathway_edges) == 0:
        print("  ⚠️  No pathway-pathway connections found")
        # Create pathway network from shared genes instead
        return export_pathway_gene_shared_network(df_edges, output_dir)
    
    # Get unique pathways
    pathways = pd.concat([pathway_edges['subject'], pathway_edges['object']]).unique()
    
    # Create nodes
    nodes = []
    for pathway in pathways:
        pathway_type = 'GO' if pathway.startswith('GO:') else 'Reactome'
        nodes.append({
            'Id': pathway,
            'Label': pathway,
            'Type': pathway_type
        })
    
    nodes_df = pd.DataFrame(nodes)
    nodes_df.to_csv(f"{output_dir}/layer3_pathway_nodes.csv", index=False)
    
    # Create edges
    edges = []
    for _, row in pathway_edges.iterrows():
        edges.append({
            'Source': row['subject'],
            'Target': row['object'],
            'Relation': row['predicate'],
            'Type': 'Directed',
            'Weight': 1.0
        })
    
    edges_df = pd.DataFrame(edges)
    edges_df.to_csv(f"{output_dir}/layer3_pathway_edges.csv", index=False)
    
    print(f"  Nodes: {len(nodes)} pathways")
    print(f"  Edges: {len(edges)} relationships")
    print(f"  Files: layer3_pathway_nodes.csv, layer3_pathway_edges.csv")


def export_pathway_gene_shared_network(df_edges, output_dir='gephi_exports', min_shared=3):
    """
    Create pathway network based on shared genes.
    
    Parameters
    ----------
    df_edges : pd.DataFrame
        Edge attributes
    output_dir : str
        Output directory
    min_shared : int
        Minimum number of shared genes to create edge
    """
    print("  Creating pathway network from shared genes...")
    
    # Get gene-pathway relationships (genes partOf pathways)
    pathway_genes = df_edges[
        ((df_edges['object'].str.startswith('GO:', na=False) | 
          df_edges['object'].str.startswith('REACT:', na=False)) &
         (df_edges['subject'].str.startswith('Protein_', na=False) |
          df_edges['subject'].str.match(r'^\d+$', na=False)))
    ].copy()
    
    if len(pathway_genes) == 0:
        print("  ⚠️  No pathway-gene relationships found")
        return
    
    # Group genes by pathway
    pathway_gene_dict = pathway_genes.groupby('object')['subject'].apply(set).to_dict()
    pathways = list(pathway_gene_dict.keys())
    
    # Create nodes
    nodes = []
    for pathway in pathways:
        pathway_type = 'GO' if pathway.startswith('GO:') else 'Reactome'
        nodes.append({
            'Id': pathway,
            'Label': pathway,
            'Type': pathway_type,
            'Gene_Count': len(pathway_gene_dict[pathway])
        })
    
    nodes_df = pd.DataFrame(nodes)
    nodes_df.to_csv(f"{output_dir}/layer3_pathway_nodes.csv", index=False)
    
    # Create edges based on shared genes
    edges = []
    for i, p1 in enumerate(pathways):
        for p2 in pathways[i+1:]:
            shared = len(pathway_gene_dict[p1] & pathway_gene_dict[p2])
            if shared >= min_shared:
                edges.append({
                    'Source': p1,
                    'Target': p2,
                    'Shared_Genes': shared,
                    'Type': 'Undirected',
                    'Weight': float(shared)
                })
    
    edges_df = pd.DataFrame(edges)
    edges_df.to_csv(f"{output_dir}/layer3_pathway_edges.csv", index=False)
    
    print(f"  Nodes: {len(nodes)} pathways")
    print(f"  Edges: {len(edges)} shared-gene relationships")
    print(f"  Files: layer3_pathway_nodes.csv, layer3_pathway_edges.csv")


def export_patient_gene_bipartite(df_edges, df_patients, output_dir='gephi_exports', top_genes=50):
    """
    Export patient-gene bipartite network.
    
    Parameters
    ----------
    df_edges : pd.DataFrame
        Edge attributes
    df_patients : pd.DataFrame
        Patient metadata
    output_dir : str
        Output directory
    top_genes : int
        Number of top expressed genes
    """
    print("\n" + "="*80)
    print("LAYER 4: Patient-Gene Bipartite Network")
    print("="*80)
    
    # Get patient-gene expression edges
    patient_ids_list = df_patients['node_id'].tolist()
    
    # Try different predicates for patient-gene relationships
    patient_gene_edges = df_edges[
        (df_edges['subject'].isin(patient_ids_list)) &
        ((df_edges['predicate'].str.contains('express', case=False, na=False)) |
         (df_edges['predicate'].str.contains('differential', case=False, na=False)) |
         (df_edges['predicate'] == 'hasExpression'))
    ].copy()
    
    if len(patient_gene_edges) == 0:
        print("  ⚠️  No patient-gene expression data found")
        return
    
    # Get top expressed genes
    top_gene_ids = patient_gene_edges['object'].value_counts().head(top_genes).index.tolist()
    patient_gene_edges = patient_gene_edges[patient_gene_edges['object'].isin(top_gene_ids)]
    
    # Create nodes (patients + genes)
    nodes = []
    
    # Patient nodes
    for _, patient in df_patients.iterrows():
        nodes.append({
            'Id': patient['node_id'],
            'Label': patient['node_id'],
            'Type': 'Patient',
            'Disease_Status': 'Sepsis' if patient.get('hasDiseaseStatus', '').lower() != 'healthy' else 'Healthy'
        })
    
    # Gene nodes
    for gene_id in top_gene_ids:
        nodes.append({
            'Id': gene_id,
            'Label': f"Gene_{gene_id}",
            'Type': 'Gene',
            'Disease_Status': ''
        })
    
    nodes_df = pd.DataFrame(nodes)
    nodes_df.to_csv(f"{output_dir}/layer4_bipartite_nodes.csv", index=False)
    
    # Create edges
    edges = []
    for _, row in patient_gene_edges.iterrows():
        edges.append({
            'Source': row['subject'],
            'Target': row['object'],
            'Type': 'Directed',
            'Weight': 1.0
        })
    
    edges_df = pd.DataFrame(edges)
    edges_df.to_csv(f"{output_dir}/layer4_bipartite_edges.csv", index=False)
    
    print(f"  Nodes: {len(nodes)} (patients + genes)")
    print(f"  Edges: {len(edges)} expression relationships")
    print(f"  Files: layer4_bipartite_nodes.csv, layer4_bipartite_edges.csv")


def main():
    """
    Main export function.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Export graphs for Gephi")
    parser.add_argument('--dataset', type=str, default='GSE54514')
    parser.add_argument('--model', type=str, default='ComplEx')
    parser.add_argument('--output_dir', type=str, default='gephi_exports')
    parser.add_argument('--patient_sim_threshold', type=float, default=0.5,
                        help='Similarity threshold for patient network')
    parser.add_argument('--top_genes', type=int, default=200,
                        help='Number of top genes for gene network')
    args = parser.parse_args()
    
    print("="*80)
    print("EXPORTING MULTI-LEVEL GRAPHS FOR GEPHI")
    print("="*80)
    
    # Load data
    control_label = "healthy"
    emb_path = f"./models/executions/{args.dataset}_enriched_ontology_degfilterv2.9_outputmodel_{args.model}_entity_embeddings.npy"
    node_features_csv = f"./models/executions/{args.dataset}_enriched_ontology_degfilterv2.9_node_features.csv"
    map_csv = f"./models/executions/{args.dataset}_enriched_ontology_degfilterv2.9_outputmodel_{args.model}_entity_mapping.csv"
    edge_csv = f"./models/executions/{args.dataset}_enriched_ontology_degfilterv2.9_edge_attributes.csv"
    
    # Load patient data
    print("\nLoading patient data...")
    patient_ids, X_patients, y_patients, df_patients = load_patient_data(
        emb_path, map_csv, node_features_csv, control_label
    )
    
    # Load all entities
    print("Loading all entities...")
    entity_ids, X_all, entity_labels, df_all = load_all_entities(
        emb_path, map_csv, node_features_csv
    )
    
    # Load edges
    print("Loading graph edges...")
    df_edges = pd.read_csv(edge_csv)
    
    # Export layers
    export_patient_similarity_network(
        X_patients, y_patients, patient_ids, df_patients,
        output_dir=args.output_dir,
        threshold=args.patient_sim_threshold
    )
    
    export_gene_interaction_network(
        df_edges, df_all, entity_labels,
        output_dir=args.output_dir,
        top_genes=args.top_genes
    )
    
    export_pathway_network(
        df_edges,
        output_dir=args.output_dir
    )
    
    export_patient_gene_bipartite(
        df_edges, df_patients,
        output_dir=args.output_dir,
        top_genes=50
    )
    
    print("\n" + "="*80)
    print("✅ EXPORT COMPLETE!")
    print("="*80)
    print(f"\nFiles saved to: {args.output_dir}/")
    print("\nGephi Import Instructions:")
    print("1. Open Gephi")
    print("2. File → Import Spreadsheet")
    print("3. For each layer:")
    print("   - Import layer*_nodes.csv as 'Nodes table'")
    print("   - Import layer*_edges.csv as 'Edges table'")
    print("4. Apply layouts:")
    print("   - Layer 1 (Patients): ForceAtlas 2")
    print("   - Layer 2 (Genes): Fruchterman Reingold")
    print("   - Layer 3 (Pathways): Yifan Hu")
    print("   - Layer 4 (Bipartite): Circular Layout")
    print("="*80)


if __name__ == "__main__":
    main()
