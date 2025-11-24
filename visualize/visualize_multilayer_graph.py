"""
Interactive Multi-Layer Graph Visualization
Visualize the knowledge graph with filtering options for different node and edge types.
"""

import pandas as pd
import numpy as np
import networkx as nx
from pyvis.network import Network
import argparse
from pathlib import Path
from load_embeddings import load_patient_data, load_all_entities


def load_graph_data(base_path='models/executions', prefix='GSE54514_enriched_ontology_degfilterv2.9'):
    """Load node features and edge attributes."""
    node_file = f"{base_path}/{prefix}_node_features.csv"
    edge_file = f"{base_path}/{prefix}_edge_attributes.csv"
    
    df_nodes = pd.read_csv(node_file)
    df_edges = pd.read_csv(edge_file)
    
    return df_nodes, df_edges


def classify_nodes(df_nodes):
    """Classify nodes into different types."""
    def get_node_type(node_id):
        if pd.isna(node_id):
            return 'Unknown'
        node_str = str(node_id)
        if node_str.startswith('Sample_GSM'):
            return 'Patient'
        elif node_str.startswith('Protein_'):
            return 'Gene/Protein'
        elif node_str.startswith('Pathway_'):
            return 'Pathway'
        elif node_str.startswith('GO:'):
            return 'GO Term'
        elif node_str.startswith('REACT:'):
            return 'Pathway'
        elif node_str.startswith('MONDO:') or node_str.startswith('HP:'):
            return 'Disease/Phenotype'
        elif node_str.isdigit():
            return 'Gene'
        else:
            return 'Other'
    
    df_nodes['node_type'] = df_nodes['node_id'].apply(get_node_type)
    return df_nodes


def create_filtered_graph(df_nodes, df_edges, 
                          node_types=None, 
                          edge_types=None,
                          max_nodes=1000,
                          max_edges=5000,
                          patient_sim_threshold=0.5,
                          include_patient_similarity=False):
    """
    Create a NetworkX graph with filtering options.
    
    Parameters:
    -----------
    df_nodes : DataFrame
        Node features
    df_edges : DataFrame
        Edge attributes
    node_types : list or None
        List of node types to include (e.g., ['Patient', 'Gene/Protein'])
    edge_types : list or None
        List of edge predicates to include (e.g., ['hasPhysicalInteractionWith'])
    max_nodes : int
        Maximum number of nodes to include (for performance)
    max_edges : int
        Maximum number of edges to include
    patient_sim_threshold : float
        Similarity threshold for patient-patient edges
    include_patient_similarity : bool
        Whether to add patient similarity edges
    """
    G = nx.Graph()
    
    # Expand shorthand edge types
    if edge_types:
        expanded_edge_types = []
        for et in edge_types:
            if et == 'gene_expression':
                expanded_edge_types.extend(['hasGeneEGroup_1T', 'hasGeneEGroup_2T', 
                                           'hasGeneEGroup_3T', 'hasGeneEGroup_4T'])
            elif et == 'protein_interaction':
                expanded_edge_types.extend(['hasPhysicalInteractionWith', 'hasGeneticInteractionWith'])
            else:
                expanded_edge_types.append(et)
        edge_types = expanded_edge_types
    
    # First, filter edges by type if specified
    if edge_types:
        df_edges_filtered = df_edges[df_edges['predicate'].isin(edge_types)].copy()
    else:
        df_edges_filtered = df_edges.copy()
    
    # Get all nodes from edges (genes only appear in edges, not node features)
    edge_nodes = set(df_edges_filtered['subject'].unique()) | set(df_edges_filtered['object'].unique())
    
    # Create a complete node list combining explicit nodes and edge nodes
    explicit_nodes = df_nodes[['node_id']].copy()
    explicit_nodes = classify_nodes(explicit_nodes)
    
    # Add implicit nodes from edges
    implicit_node_ids = edge_nodes - set(explicit_nodes['node_id'])
    if len(implicit_node_ids) > 0:
        implicit_nodes = pd.DataFrame({'node_id': list(implicit_node_ids)})
        implicit_nodes = classify_nodes(implicit_nodes)
        all_nodes = pd.concat([explicit_nodes, implicit_nodes], ignore_index=True)
    else:
        all_nodes = explicit_nodes
    
    # Filter nodes by type
    if node_types:
        all_nodes_filtered = all_nodes[all_nodes['node_type'].isin(node_types)].copy()
    else:
        all_nodes_filtered = all_nodes.copy()
    
    # Limit number of nodes for performance
    if len(all_nodes_filtered) > max_nodes:
        print(f"⚠️ Too many nodes ({len(all_nodes_filtered)}). Sampling {max_nodes} nodes.")
        all_nodes_filtered = all_nodes_filtered.sample(n=max_nodes, random_state=42)
    
    # Get node IDs
    node_ids = set(all_nodes_filtered['node_id'].values)
    
    # Re-filter edges to only include nodes we're keeping
    df_edges_filtered = df_edges_filtered[
        (df_edges_filtered['subject'].isin(node_ids)) &
        (df_edges_filtered['object'].isin(node_ids))
    ].copy()
    
    # Merge with full node features to get annotations
    df_nodes_with_features = all_nodes_filtered.merge(
        df_nodes, on='node_id', how='left'
    )
    
    # Add nodes
    print(f"Adding {len(all_nodes_filtered)} nodes...")
    for _, row in all_nodes_filtered.iterrows():
        node_id = row['node_id']
        node_type = row['node_type']
        
        # Get node label - try to find name from features
        label = str(node_id)
        node_features = df_nodes[df_nodes['node_id'] == node_id]
        if len(node_features) > 0:
            name_row = node_features[node_features['name_feature'] == 'hasName']
            if len(name_row) > 0:
                label = name_row.iloc[0]['value_feature']
        
        G.add_node(node_id, 
                  label=label,
                  type=node_type,
                  title=f"{node_type}: {label}")
    
    # Limit edges
    if len(df_edges_filtered) > max_edges:
        print(f"⚠️ Too many edges ({len(df_edges_filtered)}). Sampling {max_edges} edges.")
        df_edges_filtered = df_edges_filtered.sample(n=max_edges, random_state=42)
    
    # Add edges
    print(f"Adding {len(df_edges_filtered)} edges...")
    for _, row in df_edges_filtered.iterrows():
        source = row['subject']
        target = row['object']
        predicate = row['predicate']
        
        # Get edge weight if available
        weight = row.get('value', 1.0)
        if pd.isna(weight):
            weight = 1.0
        else:
            # Convert to float if it's a string
            try:
                weight = float(weight)
            except (ValueError, TypeError):
                weight = 1.0
        
        G.add_edge(source, target, 
                  type=predicate,
                  weight=weight,
                  title=f"{predicate} (weight: {weight:.3f})")
    
    # Optionally add patient similarity edges
    if include_patient_similarity:
        print("Computing patient similarity edges...")
        emb_path = f"./models/executions/GSE54514_enriched_ontology_degfilterv2.9_outputmodel_ComplEx_entity_embeddings.npy"
        node_features_csv = f"./models/executions/GSE54514_enriched_ontology_degfilterv2.9_node_features.csv"
        map_csv = f"./models/executions/GSE54514_enriched_ontology_degfilterv2.9_outputmodel_ComplEx_entity_mapping.csv"
        
        # Load patient data - returns (patient_ids, X_patients, y_patients, df_patients)
        patient_ids, X_patients, y_patients, df_patients = load_patient_data(emb_path, map_csv, node_features_csv, "healthy")
        
        # X_patients should already be 2D array of shape (n_patients, embedding_dim)
        print(f"  Loaded patient embeddings: {X_patients.shape}")
        
        # Filter to patients in the graph
        patient_mask = df_patients['node_id'].isin(node_ids).values
        df_patients_in_graph = df_patients[patient_mask].copy()
        X_patients_in_graph = X_patients[patient_mask]
        
        print(f"  Patients in graph: {len(df_patients_in_graph)}")
        print(f"  Computing similarities for {X_patients_in_graph.shape[0]} patients...")
        
        # Compute similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(X_patients_in_graph)
        
        n_patients = len(df_patients_in_graph)
        patient_ids_list = df_patients_in_graph['node_id'].tolist()
        
        edge_count = 0
        for i in range(n_patients):
            for j in range(i+1, n_patients):
                sim = similarities[i, j]
                if sim >= patient_sim_threshold:
                    G.add_edge(patient_ids_list[i], patient_ids_list[j],
                             type='similar_to',
                             weight=sim,
                             title=f"Similarity: {sim:.3f}")
                    edge_count += 1
        
        print(f"Added {edge_count} patient similarity edges (threshold: {patient_sim_threshold})")
    
    return G


def visualize_graph(G, output_file='graph_visualization.html', 
                   height='800px', width='100%',
                   physics=True,
                   notebook=False):
    """
    Create an interactive visualization using PyVis.
    
    Parameters:
    -----------
    G : NetworkX graph
        The graph to visualize
    output_file : str
        Output HTML file
    height : str
        Height of the visualization
    width : str
        Width of the visualization
    physics : bool
        Enable physics simulation
    notebook : bool
        Optimize for Jupyter notebook display
    """
    # Create PyVis network with enhanced options
    net = Network(height=height, width=width, 
                  bgcolor='#e8f1ff', font_color='#000c1f',
                  notebook=notebook,
                  select_menu=True,  # Enable selection menu
                  filter_menu=True,   # Enable filter menu
                  cdn_resources='remote')
    
    # Disable edge hiding on drag for better performance
    net.toggle_hide_edges_on_drag(False)
    
    # Set physics options with force atlas layout
    if physics:
        net.force_atlas_2based(
            gravity=-50,
            central_gravity=0.01,
            spring_length=100,
            spring_strength=0.08,
            damping=0.4,
            overlap=0
        )
    else:
        net.toggle_physics(False)
    
    # Node colors by type
    color_map = {
        'Patient': '#ff2146',           # Red/Pink
        'Gene/Protein': '#1772ff',      # Blue
        'Gene': '#45B7D1',              # Light Blue
        'GO Term': '#ff9500',           # Orange
        'Pathway': '#34c759',           # Green
        'Disease/Phenotype': '#ffcc00', # Yellow
        'Other': '#8e8e93'              # Gray
    }
    
    # Calculate node degrees for sizing
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 1
    
    # Add nodes with enhanced styling
    for node, attrs in G.nodes(data=True):
        node_type = attrs.get('type', 'Other')
        color = color_map.get(node_type, '#8e8e93')
        
        # Size based on degree (scaled between 10 and 50)
        degree = degrees.get(node, 0)
        size = 10 + (degree / max_degree) * 40
        
        label = attrs.get('label', str(node))
        
        # Enhanced title with more info
        title_text = f"""
        <b>{node_type}</b><br>
        Name: {label}<br>
        Degree: {degree}<br>
        ID: {node}
        """
        
        net.add_node(node,
                    label=label,
                    title=title_text.strip(),
                    color=color,
                    size=size,
                    value=degree,  # Used for built-in filtering
                    borderWidth=2,
                    borderWidthSelected=4,
                    font={'size': 12, 'color': '#000c1f'})
    
    # Edge colors by type with gradient support
    edge_color_map = {
        'hasPhysicalInteractionWith': {'min': '#34495e', 'max': '#1a252f'},
        'hasGeneticInteractionWith': {'min': '#2c3e50', 'max': '#1a252f'},
        'hasGeneEGroup_1T': {'min': '#ffcccb', 'max': '#e74c3c'},  # Red gradient (low expression)
        'hasGeneEGroup_2T': {'min': '#ffd7a8', 'max': '#f39c12'},  # Orange gradient
        'hasGeneEGroup_3T': {'min': '#fff3a8', 'max': '#f1c40f'},  # Yellow gradient
        'hasGeneEGroup_4T': {'min': '#b8e6b8', 'max': '#27ae60'},  # Green gradient (high expression)
        'similar_to': {'min': '#d7bfe6', 'max': '#9b59b6'},        # Purple gradient
        'partOf': {'min': '#a8d8ff', 'max': '#3498db'},            # Blue gradient
        'regulates': {'min': '#d7bfe6', 'max': '#8e44ad'}          # Purple gradient
    }
    
    # Get edge weights for normalization
    edge_weights = [attrs.get('weight', 1.0) for _, _, attrs in G.edges(data=True)]
    min_weight = min(edge_weights) if edge_weights else 0
    max_weight = max(edge_weights) if edge_weights else 1
    weight_range = max_weight - min_weight if max_weight > min_weight else 1
    
    # Add edges with enhanced styling
    for source, target, attrs in G.edges(data=True):
        edge_type = attrs.get('type', 'unknown')
        weight = attrs.get('weight', 1.0)
        
        # Ensure weight is numeric
        try:
            weight = float(weight)
        except (ValueError, TypeError):
            weight = 1.0
        
        # Normalize weight
        normalized_weight = (weight - min_weight) / weight_range if weight_range > 0 else 0.5
        
        # Get color gradient for this edge type
        colors = edge_color_map.get(edge_type, {'min': '#95a5a6', 'max': '#7f8c8d'})
        
        # Compute gradient color based on weight
        import matplotlib.colors
        min_rgb = np.array(matplotlib.colors.to_rgb(colors['min']))
        max_rgb = np.array(matplotlib.colors.to_rgb(colors['max']))
        edge_color = matplotlib.colors.to_hex(min_rgb + normalized_weight * (max_rgb - min_rgb))
        
        # Edge width based on weight
        edge_width = 1 + 4 * normalized_weight  # Range: 1-5
        
        # Enhanced title
        title_text = f"{edge_type}<br>Weight: {weight:.3f}<br>{source} → {target}"
        
        net.add_edge(source, target,
                    title=title_text,
                    color=edge_color,
                    value=weight,  # Used for built-in filtering
                    width=edge_width,
                    physics=True)
    
    # Add interactive control panel with filter options
    # PyVis built-in menus are now enabled via select_menu and filter_menu parameters
    
    # Add custom HTML for enhanced filtering
    custom_html = """
    <style>
        /* Match the disease network styling */
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            margin: 0;
            padding: 0;
        }
        
        .control-panel {
            position: fixed;
            top: 10px;
            left: 10px;
            background: white;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            z-index: 1000;
            max-width: 320px;
            max-height: 90vh;
            overflow-y: auto;
        }
        
        .control-panel h3 {
            margin: 0 0 15px 0;
            font-size: 18px;
            font-weight: 600;
            color: #1a1a1a;
            border-bottom: 2px solid #e8f1ff;
            padding-bottom: 10px;
        }
        
        .control-section {
            margin: 15px 0;
            padding: 12px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        
        .control-label {
            display: block;
            font-size: 13px;
            font-weight: 600;
            color: #495057;
            margin-bottom: 8px;
        }
        
        .filter-checkbox {
            margin: 6px 0;
            font-size: 12px;
            display: flex;
            align-items: center;
        }
        
        .filter-checkbox input {
            margin-right: 8px;
            cursor: pointer;
        }
        
        .filter-checkbox label {
            cursor: pointer;
            user-select: none;
        }
        
        .info-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px;
            border-radius: 8px;
            font-size: 12px;
            margin-top: 15px;
        }
        
        .info-box strong {
            display: block;
            margin-bottom: 8px;
            font-size: 13px;
        }
        
        .stat-item {
            margin: 4px 0;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            margin: 6px 0;
            font-size: 12px;
        }
        
        .legend-color {
            width: 18px;
            height: 18px;
            border-radius: 4px;
            margin-right: 8px;
            border: 1px solid rgba(0,0,0,0.1);
        }
        
        .toggle-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 8px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 11px;
            font-weight: 500;
            margin: 3px;
            transition: transform 0.1s, box-shadow 0.1s;
        }
        
        .toggle-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.4);
        }
        
        .toggle-btn:active {
            transform: translateY(0);
        }
        
        .btn-group {
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
            margin-top: 8px;
        }
        
        /* Scrollbar styling */
        .control-panel::-webkit-scrollbar {
            width: 6px;
        }
        
        .control-panel::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 3px;
        }
        
        .control-panel::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 3px;
        }
        
        .control-panel::-webkit-scrollbar-thumb:hover {
            background: #555;
        }
    </style>
    
    
    <script type="text/javascript">
        // Store original data
        let hiddenNodes = new Set();
        let hiddenEdges = new Set();
        
        // Initialize after network is loaded
        network.on("stabilizationIterationsDone", function () {
            updateStats();
        });
        
        function updateStats() {
            const nodes = network.body.data.nodes;
            const edges = network.body.data.edges;
            document.getElementById('nodeCount').textContent = nodes.length;
            document.getElementById('edgeCount').textContent = edges.length;
            document.getElementById('hiddenCount').textContent = hiddenNodes.size + hiddenEdges.size;
        }
        
        function filterNodeType(nodeType, show) {
            const nodes = network.body.data.nodes.get();
            const nodesToUpdate = nodes.filter(node => node.type === nodeType);
            
            nodesToUpdate.forEach(node => {
                if (show) {
                    hiddenNodes.delete(node.id);
                    network.body.data.nodes.update({id: node.id, hidden: false});
                } else {
                    hiddenNodes.add(node.id);
                    network.body.data.nodes.update({id: node.id, hidden: true});
                }
            });
            
            updateStats();
        }
        
        function filterEdgeType(edgeType, show) {
            const edges = network.body.data.edges.get();
            const edgesToUpdate = edges.filter(edge => edge.type === edgeType);
            
            edgesToUpdate.forEach(edge => {
                if (show) {
                    hiddenEdges.delete(edge.id);
                    network.body.data.edges.update({id: edge.id, hidden: false});
                } else {
                    hiddenEdges.add(edge.id);
                    network.body.data.edges.update({id: edge.id, hidden: true});
                }
            });
            
            updateStats();
        }
        
        function selectAllNodes() {
            const nodes = network.body.data.nodes.get().filter(n => !n.hidden);
            network.selectNodes(nodes.map(n => n.id));
            updateSelectedCount();
        }
        
        function deselectAll() {
            network.unselectAll();
            updateSelectedCount();
        }
        
        function resetFilters() {
            // Reset all checkboxes
            document.querySelectorAll('.filter-checkbox input').forEach(cb => cb.checked = true);
            
            // Show all nodes and edges
            const nodes = network.body.data.nodes.get();
            nodes.forEach(node => {
                network.body.data.nodes.update({id: node.id, hidden: false});
            });
            
            const edges = network.body.data.edges.get();
            edges.forEach(edge => {
                network.body.data.edges.update({id: edge.id, hidden: false});
            });
            
            hiddenNodes.clear();
            hiddenEdges.clear();
            updateStats();
        }
        
        function isolateSelected() {
            const selectedNodes = network.getSelectedNodes();
            if (selectedNodes.length === 0) {
                alert('Please select some nodes first!');
                return;
            }
            
            // Get connected edges
            const connectedEdges = new Set();
            selectedNodes.forEach(nodeId => {
                network.getConnectedEdges(nodeId).forEach(edgeId => {
                    connectedEdges.add(edgeId);
                });
            });
            
            // Get nodes connected to selected nodes
            const connectedNodes = new Set(selectedNodes);
            const edges = network.body.data.edges.get();
            edges.forEach(edge => {
                if (connectedEdges.has(edge.id)) {
                    connectedNodes.add(edge.from);
                    connectedNodes.add(edge.to);
                }
            });
            
            // Hide all other nodes
            const nodes = network.body.data.nodes.get();
            nodes.forEach(node => {
                if (!connectedNodes.has(node.id)) {
                    network.body.data.nodes.update({id: node.id, hidden: true});
                    hiddenNodes.add(node.id);
                }
            });
            
            // Hide edges not connected to selected nodes
            edges.forEach(edge => {
                if (!connectedEdges.has(edge.id)) {
                    network.body.data.edges.update({id: edge.id, hidden: true});
                    hiddenEdges.add(edge.id);
                }
            });
            
            updateStats();
        }
        
        function exportSelection() {
            const selectedNodes = network.getSelectedNodes();
            if (selectedNodes.length === 0) {
                alert('Please select some nodes first!');
                return;
            }
            
            const nodes = network.body.data.nodes.get(selectedNodes);
            const exportData = {
                nodes: nodes.map(n => ({
                    id: n.id,
                    label: n.label,
                    type: n.type
                })),
                count: nodes.length
            };
            
            console.log('Exported selection:', exportData);
            alert(`Exported ${nodes.length} nodes to console. Check browser developer tools.`);
        }
        
        function updateSelectedCount() {
            const count = network.getSelectedNodes().length;
            document.getElementById('selectedCount').textContent = count;
        }
        
        // Update selected count on selection change
        network.on("select", function(params) {
            updateSelectedCount();
        });
        
        // Log clicks for debugging
        network.on("click", function(params) {
            if (params.nodes.length > 0) {
                const nodeId = params.nodes[0];
                const node = network.body.data.nodes.get(nodeId);
                console.log('Clicked node:', node);
            }
        });
        
        // Update stats periodically
        setInterval(updateStats, 2000);
    </script>
    """
    
    # Save and inject custom HTML
    net.save_graph(output_file)
    
    # Read the generated HTML and inject custom controls
    with open(output_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Insert custom HTML before closing body tag
    html_content = html_content.replace('</body>', custom_html + '</body>')
    
    # Write back the modified HTML
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n✅ Interactive visualization saved to: {output_file}")
    print(f"   Open this file in your browser to interact with the graph.")
    print(f"\n📊 Graph statistics:")
    print(f"   Nodes: {G.number_of_nodes()}")
    print(f"   Edges: {G.number_of_edges()}")
    print(f"   Node types: {set(nx.get_node_attributes(G, 'type').values())}")
    print(f"   Edge types: {set(nx.get_edge_attributes(G, 'type').values())}")
    
    return net


def main():
    parser = argparse.ArgumentParser(
        description='Interactive multi-layer graph visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize protein-protein interactions only
  python visualize_multilayer_graph.py --node-types "Gene/Protein" --edge-types "protein_interaction"
  
  # Visualize patients and gene expression (all 4 quartiles)
  python visualize_multilayer_graph.py --node-types "Patient" "Gene/Protein" --edge-types "gene_expression" --max-nodes 200
  
  # High expression only (quartile 4)
  python visualize_multilayer_graph.py --node-types "Patient" "Gene/Protein" --edge-types "hasGeneEGroup_4T" --max-nodes 300
  
  # Add patient similarity network
  python visualize_multilayer_graph.py --node-types "Patient" --patient-similarity --threshold 0.4
  
  # Visualize pathways and genes
  python visualize_multilayer_graph.py --node-types "Pathway" "Gene/Protein" --max-edges 1000
        """
    )
    
    parser.add_argument('--node-types', nargs='+', 
                       choices=['Patient', 'Gene/Protein', 'Gene', 'GO Term', 
                               'Pathway', 'Disease/Phenotype', 'Other'],
                       help='Node types to include')
    
    parser.add_argument('--edge-types', nargs='+',
                       choices=['hasPhysicalInteractionWith', 'hasGeneticInteractionWith',
                               'hasGeneEGroup_1T', 'hasGeneEGroup_2T', 'hasGeneEGroup_3T', 
                               'hasGeneEGroup_4T', 'gene_expression', 'protein_interaction'],
                       help='Edge predicates to include. Use "gene_expression" for all hasGeneEGroup_* edges, "protein_interaction" for all interaction types')
    
    parser.add_argument('--max-nodes', type=int, default=5000,
                       help='Maximum number of nodes (default: 1000)')
    
    parser.add_argument('--max-edges', type=int, default=5000,
                       help='Maximum number of edges (default: 5000)')
    
    parser.add_argument('--patient-similarity', action='store_true',
                       help='Add patient similarity edges')
    
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Patient similarity threshold (default: 0.5)')
    
    parser.add_argument('--output', type=str, default='graph_visualization.html',
                       help='Output HTML file')
    
    parser.add_argument('--no-physics', action='store_true',
                       help='Disable physics simulation')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading graph data...")
    df_nodes, df_edges = load_graph_data()
    
    print(f"\nDataset summary:")
    df_nodes_classified = classify_nodes(df_nodes)
    print(f"  Total nodes: {len(df_nodes)}")
    print(f"  Total edges: {len(df_edges)}")
    print(f"\nNode types:")
    print(df_nodes_classified['node_type'].value_counts())
    print(f"\nEdge types:")
    print(df_edges['predicate'].value_counts())
    
    # Create filtered graph
    print(f"\nCreating filtered graph...")
    print(f"  Node types: {args.node_types if args.node_types else 'All'}")
    print(f"  Edge types: {args.edge_types if args.edge_types else 'All'}")
    
    G = create_filtered_graph(
        df_nodes, df_edges,
        node_types=args.node_types,
        edge_types=args.edge_types,
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
        patient_sim_threshold=args.threshold,
        include_patient_similarity=args.patient_similarity
    )
    
    # Visualize
    print(f"\nCreating interactive visualization...")
    visualize_graph(G, 
                   output_file=args.output,
                   physics=not args.no_physics)
    
    print("\n🎨 Legend:")
    print("  🔴 Red nodes: Patients")
    print("  🔵 Blue/Teal nodes: Genes/Proteins")
    print("  🟠 Orange nodes: GO Terms")
    print("  🟢 Green nodes: Pathways")
    print("  🟡 Yellow nodes: Diseases/Phenotypes")
    print("\n  Edge colors (expression quartiles):")
    print("  � Red: Low expression (Q1)")
    print("  🟠 Orange: Medium-low (Q2)")
    print("  🟡 Yellow: Medium-high (Q3)")
    print("  🟢 Green: High expression (Q4)")
    print("  ⚫ Dark gray: Protein interactions")
    print("\n�💡 Tips:")
    print("  - Click and drag nodes to rearrange")
    print("  - Scroll to zoom in/out")
    print("  - Click a node to highlight connections")
    print("  - Use the filter panel to customize view")
    print("\n📌 Quick filters:")
    print("  --edge-types gene_expression  (all expression quartiles)")
    print("  --edge-types protein_interaction  (physical + genetic interactions)")


if __name__ == '__main__':
    main()
