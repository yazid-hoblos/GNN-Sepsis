#!/usr/bin/env python3
"""
Dynamic Multi-Layer Network Visualization Web Application

A Flask-based web application for interactive exploration of multi-layer knowledge graphs
with real-time filtering, layer management, and dynamic visualization.
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import networkx as nx
import json
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_embeddings import load_patient_data, load_all_entities

app = Flask(__name__)

class MultiLayerNetworkManager:
    """Manages the multi-layer network data and provides filtering capabilities."""
    
    def __init__(self, kg_path='../kg_conversion', models_path='../models/executions'):
        self.kg_path = kg_path
        self.models_path = models_path
        self.df_nodes = None
        self.df_edges = None
        self.df_classes = None
        self.node_layers = {}
        self.edge_types = set()
        self.layer_definitions = {}
        self.patient_embeddings = None
        
        self.load_data()
        self.initialize_layers()
    
    def load_data(self):
        """Load knowledge graph data from CSV files."""
        # Load full knowledge graph data
        nodes_file = f"{self.kg_path}/nodes.csv"
        edges_file = f"{self.kg_path}/edges.csv"
        classes_file = f"{self.kg_path}/classes.csv"
        
        try:
            self.df_nodes = pd.read_csv(nodes_file)
            self.df_edges = pd.read_csv(edges_file)
            self.df_classes = pd.read_csv(classes_file)
            
            print(f"Loaded knowledge graph data:")
            print(f"  - Nodes: {len(self.df_nodes)}")
            print(f"  - Edges: {len(self.df_edges)}")
            print(f"  - Classes: {len(self.df_classes)}")
            
        except FileNotFoundError as e:
            print(f"Knowledge graph CSV files not found: {e}")
            print("Falling back to model execution data...")
            
            # Fallback to original data source
            node_file = f"{self.models_path}/GSE54514_enriched_ontology_degfilterv2.9_node_features.csv"
            edge_file = f"{self.models_path}/GSE54514_enriched_ontology_degfilterv2.9_edge_attributes.csv"
            
            self.df_nodes = pd.read_csv(node_file)
            self.df_edges = pd.read_csv(edge_file)
            self.df_classes = pd.DataFrame()  # Empty classes dataframe
        
        # Get unique edge types
        if 'predicate' in self.df_edges.columns:
            self.edge_types = set(self.df_edges['predicate'].unique())
        elif 'relation' in self.df_edges.columns:
            self.edge_types = set(self.df_edges['relation'].unique())
        else:
            # Try to find any column that might contain edge types
            potential_cols = ['type', 'edge_type', 'relationship']
            for col in potential_cols:
                if col in self.df_edges.columns:
                    self.edge_types = set(self.df_edges[col].unique())
                    break
            else:
                self.edge_types = {'unknown'}
        
        print(f"Found {len(self.edge_types)} edge types: {list(self.edge_types)[:5]}...")
    
    def initialize_layers(self):
        """Define and initialize network layers."""
        
        # Get all unique node IDs from different sources
        all_node_ids = set()
        
        # From edges
        edge_subject_col = 'subject' if 'subject' in self.df_edges.columns else ('source' if 'source' in self.df_edges.columns else 'from')
        edge_object_col = 'object' if 'object' in self.df_edges.columns else ('target' if 'target' in self.df_edges.columns else 'to')
        
        if edge_subject_col in self.df_edges.columns and edge_object_col in self.df_edges.columns:
            all_node_ids.update(self.df_edges[edge_subject_col].unique())
            all_node_ids.update(self.df_edges[edge_object_col].unique())
        
        # From explicit nodes
        node_id_col = 'node_id' if 'node_id' in self.df_nodes.columns else ('id' if 'id' in self.df_nodes.columns else 'name')
        if node_id_col in self.df_nodes.columns:
            all_node_ids.update(self.df_nodes[node_id_col].unique())
        
        # From classes if available
        if not self.df_classes.empty and 'class' in self.df_classes.columns:
            all_node_ids.update(self.df_classes['class'].unique())
        
        # Classify each node into layers
        for node_id in all_node_ids:
            layer_info = self._classify_node(node_id)
            self.node_layers[node_id] = layer_info
        
        # Layer colors matching the knowledge graph schema
        layer_colors = {
            'PatientSample_Layer': '#c8e6c9',  # Light green (as in schema diagram)
            'Protein_Layer': '#ffcdd2',        # Light red/pink (central hub in schema)
            'GO_Terms_Layer': '#e1bee7',       # Light purple (as in schema diagram)  
            'Pathway_Layer': '#bbdefb',        # Light blue (as in schema diagram)
            'Reaction_Layer': '#c5e1a5',       # Light green (as in schema diagram)
            'Other_Layer': '#8e8e93'           # Gray
        }
        
        self.layer_definitions = {}
        
        # Count nodes per layer
        layer_counts = {}
        for node_info in self.node_layers.values():
            layer = node_info['layer']
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
        
        # Define layers with descriptions matching knowledge graph schema
        layer_info = {
            'PatientSample_Layer': ('Patient Samples', 'Patient samples with clinical data and gene expression bins'),
            'Protein_Layer': ('Proteins', 'Protein entities with various interaction types (central hub)'),
            'GO_Terms_Layer': ('GO Terms', 'Gene Ontology terms (processes, functions, components)'),
            'Pathway_Layer': ('Pathways', 'Biological pathways containing reactions'),
            'Reaction_Layer': ('Reactions', 'Biochemical reactions within pathways'),
            'Other_Layer': ('Other', 'Unclassified entities')
        }
        
        # Create layer definitions only for layers that have nodes
        for layer_id, count in layer_counts.items():
            if count > 0:
                name, description = layer_info.get(layer_id, (layer_id, 'Unknown layer'))
                self.layer_definitions[layer_id] = {
                    'name': name,
                    'description': description,
                    'color': layer_colors.get(layer_id, '#8e8e93'),
                    'nodes': [nid for nid, info in self.node_layers.items() if info['layer'] == layer_id],
                    'count': count
                }
        
        print(f"Initialized {len(self.layer_definitions)} layers:")
        for layer_id, layer_info in self.layer_definitions.items():
            print(f"  {layer_info['name']}: {len(layer_info['nodes'])} nodes")
    
    def _classify_node(self, node_id):
        """Classify nodes based on knowledge graph schema entity types."""
        if pd.isna(node_id):
            return {'type': 'Unknown', 'layer': 'Other_Layer', 'level': 0}
        
        node_str = str(node_id).strip()
        
        # Check if we have node_type information from the nodes CSV
        if hasattr(self, 'df_nodes') and 'node_type' in self.df_nodes.columns:
            node_id_col = 'node_id' if 'node_id' in self.df_nodes.columns else ('id' if 'id' in self.df_nodes.columns else 'name')
            if node_id_col in self.df_nodes.columns:
                node_match = self.df_nodes[self.df_nodes[node_id_col] == node_id]
                if len(node_match) > 0:
                    node_type = node_match.iloc[0]['node_type']
                    # Handle GO nodes properly - they have GO_XXXXXXX format in node_type
                    if node_type.startswith('GO_') or 'GO Term' in str(node_match.iloc[0].get('node_category', '')):
                        go_id = node_type.replace('_', ':') if '_' in node_type else node_type
                        return {'type': f'GO Term ({go_id})', 'layer': 'GO_Terms_Layer', 'level': 2}
                    elif node_type == 'PatientSample':
                        return {'type': 'Patient Sample', 'layer': 'PatientSample_Layer', 'level': 1}
                    elif node_type == 'Protein':
                        return {'type': 'Protein', 'layer': 'Protein_Layer', 'level': 2}
                    elif node_type == 'Pathway':
                        return {'type': 'Pathway', 'layer': 'Pathway_Layer', 'level': 3}
                    elif node_type == 'Reaction':
                        return {'type': 'Reaction', 'layer': 'Reaction_Layer', 'level': 4}
        
        # Fallback classification based on node ID patterns
        # Patient Samples
        if node_str.startswith('Sample_'):
            return {'type': 'Patient Sample', 'layer': 'PatientSample_Layer', 'level': 1}
        
        # Proteins
        elif node_str.startswith('Protein_'):
            return {'type': 'Protein', 'layer': 'Protein_Layer', 'level': 2}
        
        # Pathways
        elif node_str.startswith('Pathway_'):
            return {'type': 'Pathway', 'layer': 'Pathway_Layer', 'level': 3}
        
        # Reactions
        elif node_str.startswith('Reaction_'):
            return {'type': 'Reaction', 'layer': 'Reaction_Layer', 'level': 4}
        
        # GO Terms (handle various formats: GO_, GO:, and _instance suffix)
        elif (node_str.startswith('GO_') or node_str.startswith('GO:') or 
              ('GO_' in node_str and '_instance' in node_str)):
            # Extract GO ID for better labeling
            if '_instance' in node_str:
                # For GO_0005737_instance format, extract GO_0005737
                go_part = node_str.split('_instance')[0]
                go_id = go_part.replace('_', ':') if go_part.startswith('GO_') else go_part
            elif node_str.startswith('GO_'):
                go_id = node_str.replace('_', ':')
            else:
                go_id = node_str
            return {'type': f'GO Term ({go_id})', 'layer': 'GO_Terms_Layer', 'level': 2}
        
        # Disease/Phenotype layer
        elif (node_str.startswith('MONDO:') or node_str.startswith('HP:') or
              node_str.startswith('DOID:') or node_str.startswith('OMIM:') or
              node_str.startswith('Disease_') or 'disease' in node_str.lower() or
              'phenotype' in node_str.lower()):
            return {'type': 'Disease/Phenotype', 'layer': 'Disease_Layer', 'level': 4}
        
        # Anatomy/Tissue layer
        elif (node_str.startswith('UBERON:') or node_str.startswith('CL:') or
              'tissue' in node_str.lower() or 'cell' in node_str.lower()):
            return {'type': 'Anatomy/Cell', 'layer': 'Anatomy_Layer', 'level': 4}
        
        # Drug/Chemical layer
        elif (node_str.startswith('DRUGBANK:') or node_str.startswith('PUBCHEM:') or
              'drug' in node_str.lower() or 'chemical' in node_str.lower()):
            return {'type': 'Drug/Chemical', 'layer': 'Chemical_Layer', 'level': 5}
        
        # Literature/Publication layer
        elif (node_str.startswith('PMID:') or node_str.startswith('DOI:') or
              'publication' in node_str.lower()):
            return {'type': 'Publication', 'layer': 'Literature_Layer', 'level': 6}
        
        # Default classification
        else:
            return {'type': 'Other', 'layer': 'Other_Layer', 'level': 7}
    
    def get_filtered_network(self, filters):
        """Generate network data based on current filters with optimizations."""
        import time
        t0 = time.time()
        # Apply filters without limits - ensure we always have some layers and edge types
        active_layers = filters.get('layers') or list(self.layer_definitions.keys())
        active_edge_types = filters.get('edge_types') or list(self.edge_types)
        layer_interactions = filters.get('layer_interactions', 'all')
        t1 = time.time()
        print(f"[PERF] Parsed filters in {t1-t0:.3f}s")
        # Ensure we have valid defaults
        if not active_layers:
            active_layers = list(self.layer_definitions.keys())
        if not active_edge_types:
            active_edge_types = list(self.edge_types)
        t2 = time.time()
        print(f"[PERF] Set defaults in {t2-t1:.3f}s")
        print(f"Filtering: {len(active_layers)} layers, {len(active_edge_types)} edge types")
        print(f"Active layers: {active_layers[:3]}...")
        print(f"Active edge types: {list(active_edge_types)[:3]}...")
        # Process full dataset without limits
        # Get all nodes from selected layers
        filtered_node_ids = set()
        for layer_id in active_layers:
            if layer_id in self.layer_definitions:
                layer_nodes = self.layer_definitions[layer_id]['nodes']
                filtered_node_ids.update(layer_nodes)
                print(f"  Layer {layer_id}: added {len(layer_nodes)} nodes")
            else:
                print(f"  Warning: Layer {layer_id} not found in layer_definitions")
        t3 = time.time()
        print(f"[PERF] Selected nodes in {t3-t2:.3f}s")
        if not filtered_node_ids:
            print("Warning: No nodes selected from any layer. Using all nodes from edges.")
            # Fallback: use all nodes from edges if no layer filtering worked
            edge_subject_col = 'subject' if 'subject' in self.df_edges.columns else ('source' if 'source' in self.df_edges.columns else 'from')
            edge_object_col = 'object' if 'object' in self.df_edges.columns else ('target' if 'target' in self.df_edges.columns else 'to')
            filtered_node_ids.update(self.df_edges[edge_subject_col].unique())
            filtered_node_ids.update(self.df_edges[edge_object_col].unique())
        t4 = time.time()
        print(f"[PERF] Fallback node selection in {t4-t3:.3f}s")
        print(f"Selected {len(filtered_node_ids)} nodes (fast mode)")
        # Find the correct column names for edges
        edge_subject_col = 'subject' if 'subject' in self.df_edges.columns else ('source' if 'source' in self.df_edges.columns else 'from')
        edge_object_col = 'object' if 'object' in self.df_edges.columns else ('target' if 'target' in self.df_edges.columns else 'to')
        edge_predicate_col = 'predicate' if 'predicate' in self.df_edges.columns else ('relation' if 'relation' in self.df_edges.columns else 'type')
        # Filter edges connecting our selected nodes
        print(f"Filtering edges...")
        t5 = time.time()
        connecting_edges = self.df_edges[
            (self.df_edges[edge_subject_col].isin(filtered_node_ids)) & 
            (self.df_edges[edge_object_col].isin(filtered_node_ids))
        ]
        t6 = time.time()
        print(f"[PERF] Filtered connecting edges in {t6-t5:.3f}s")
        # Apply edge type filter if needed
        if edge_predicate_col in connecting_edges.columns and len(active_edge_types) < len(self.edge_types):
            connecting_edges = connecting_edges[connecting_edges[edge_predicate_col].isin(active_edge_types)]
        t7 = time.time()
        print(f"[PERF] Applied edge type filter in {t7-t6:.3f}s")
        filtered_edges = connecting_edges
        print(f"Found {len(filtered_edges)} connecting edges")
        # Apply layer interaction filters
        if layer_interactions != 'all':
            edge_mask = []
            for _, edge in filtered_edges.iterrows():
                src_layer = self.node_layers.get(edge[edge_subject_col], {}).get('layer', 'Unknown')
                tgt_layer = self.node_layers.get(edge[edge_object_col], {}).get('layer', 'Unknown')
                if layer_interactions == 'intra':
                    edge_mask.append(src_layer == tgt_layer)
                elif layer_interactions == 'inter':
                    edge_mask.append(src_layer != tgt_layer)
                else:
                    edge_pair = f"{src_layer}-{tgt_layer}"
                    reverse_pair = f"{tgt_layer}-{src_layer}"
                    edge_mask.append(edge_pair in layer_interactions or reverse_pair in layer_interactions)
            filtered_edges = filtered_edges[edge_mask]
        t8 = time.time()
        print(f"[PERF] Applied layer interaction filter in {t8-t7:.3f}s")
        print(f"Final edge count: {len(filtered_edges)}")
        nodes_data = []
        edges_data = []
        print("Building final network data...")
        active_node_ids = set(filtered_edges[edge_subject_col].unique()) | set(filtered_edges[edge_object_col].unique())
        print(f"Processing {len(active_node_ids)} active nodes (singletons automatically removed)...")
        t9 = time.time()
        for node_id in active_node_ids:
            node_info = self.node_layers.get(node_id, {'type': 'Unknown', 'layer': 'Other_Layer'})
            layer_info = self.layer_definitions.get(node_info['layer'], {'color': '#8e8e93', 'name': 'Other'})
            label = str(node_id)
            if 'GO_' in label and '_instance' in label:
                go_part = label.split('_instance')[0]
                if go_part.startswith('GO_'):
                    label = go_part.replace('_', ':')
                else:
                    label = go_part
            elif '_instance' in label:
                label = label.replace('_instance', '')
            elif '_' in label and not label.startswith('Sample_'):
                parts = label.split('_')
                if len(parts) > 1 and parts[-1] != 'instance':
                    label = parts[-1][:15]
                else:
                    label = '_'.join(parts[:-1])[:15] if parts[-1] == 'instance' else label[:15]
            elif len(label) > 15:
                label = label[:15] + '...'
            nodes_data.append({
                'id': str(node_id),
                'label': label,
                'type': node_info['type'],
                'layer': node_info['layer'],
                'color': layer_info['color'],
                'size': 15,
                'title': f"{node_info['type']}: {node_id}"
            })
        t10 = time.time()
        print(f"[PERF] Built node data in {t10-t9:.3f}s")
        active_node_ids_str = set(str(nid) for nid in active_node_ids)
        chunk_size = 5000  # Larger chunks for better performance
        total_edges = len(filtered_edges)
        print(f"Processing {total_edges} filtered edges in chunks of {chunk_size}...")
        t11 = time.time()
        for chunk_start in range(0, total_edges, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_edges)
            chunk = filtered_edges.iloc[chunk_start:chunk_end]
            for _, edge in chunk.iterrows():
                from_node = str(edge[edge_subject_col])
                to_node = str(edge[edge_object_col])
                if from_node in active_node_ids_str and to_node in active_node_ids_str:
                    edges_data.append({
                        'from': from_node,
                        'to': to_node,
                        'type': edge.get(edge_predicate_col, 'connection'),
                        'title': edge.get(edge_predicate_col, 'connection')
                    })
        t12 = time.time()
        print(f"[PERF] Built edge data in {t12-t11:.3f}s")
        nodes_data = self.remove_singletons(nodes_data, edges_data)
        t13 = time.time()
        print(f"[PERF] Removed singletons in {t13-t12:.3f}s")
        layer_counts = {}
        for node in nodes_data:
            layer = node['layer']
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
        t14 = time.time()
        print(f"[PERF] Counted layers in {t14-t13:.3f}s")
        print(f"Final network: {len(nodes_data)} nodes, {len(edges_data)} edges (singletons removed)")
        print(f"[PERF] Total get_filtered_network time: {t14-t0:.3f}s")
        return {
            'nodes': nodes_data,
            'edges': edges_data,
            'stats': {
                'node_count': len(nodes_data),
                'edge_count': len(edges_data),
                'layer_counts': layer_counts
            }
        }
    
    def remove_singletons(self, nodes_data, edges_data):
        """Remove singleton nodes (nodes with no connections) from the network."""
        # Get all node IDs that appear in edges
        connected_node_ids = set()
        for edge in edges_data:
            connected_node_ids.add(edge['from'])
            connected_node_ids.add(edge['to'])
        
        # Filter nodes to only include those with connections
        connected_nodes = [node for node in nodes_data if node['id'] in connected_node_ids]
        
        removed_count = len(nodes_data) - len(connected_nodes)
        if removed_count > 0:
            print(f"Removed {removed_count} singleton nodes (no connections)")
        
        return connected_nodes
    
    def get_layer_summary(self):
        """Get summary information about all layers."""
        return {
            'layers': self.layer_definitions,
            'edge_types': list(self.edge_types),
            'total_nodes': len(self.node_layers),
            'total_edges': len(self.df_edges),
            'note': 'Singleton nodes (no connections) are automatically removed from network display'
        }

# Global network manager
network_manager = None

@app.route('/')
def index():
    """Main application page."""
    return render_template('multilayer_network.html')

@app.route('/api/network_data')
def get_network_data():
    """API endpoint to get filtered network data."""
    try:
        manager = initialize_manager()
        # Get filter parameters from request
        filters = {
            'layers': request.args.getlist('layers') or list(manager.layer_definitions.keys()),
            'edge_types': request.args.getlist('edge_types') or list(manager.edge_types),
            'layer_interactions': request.args.get('layer_interactions', 'all')
        }
        import time
        print(f"API request: {filters}")
        t0 = time.time()
        # Process the request
        network_data = manager.get_filtered_network(filters)
        t1 = time.time()
        print(f"[PERF] API network_data total time: {t1-t0:.3f}s")
        print(f"Returning {network_data['stats']['node_count']} nodes, {network_data['stats']['edge_count']} edges")
        return jsonify(network_data)
            
    except ValueError as e:
        return jsonify({'error': f'Invalid parameters: {str(e)}'}), 400
    except Exception as e:
        print(f"Error in network_data endpoint: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/layer_summary')
def get_layer_summary():
    """API endpoint to get layer and network summary."""
    try:
        manager = initialize_manager()
        summary = manager.get_layer_summary()
        print(f"Layer summary: {len(summary.get('layers', {}))} layers, {len(summary.get('edge_types', []))} edge types")
        return jsonify(summary)
    except Exception as e:
        print(f"Error in layer_summary endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/sample_network')
def get_sample_network():
    """API endpoint to get a pre-cached network for instant loading."""
    try:
        manager = initialize_manager()
        
        print("Returning pre-cached sample network (instant)...")
        
        # Return pre-computed cached data (instant response)
        result = {
            'nodes': manager.cached_nodes,
            'edges': manager.cached_edges,
            'stats': {
                'node_count': len(manager.cached_nodes),
                'edge_count': len(manager.cached_edges),
                'layer_counts': {layer: len([n for n in manager.cached_nodes if n['layer'] == layer]) 
                               for layer in manager.layer_definitions.keys()}
            }
        }
        
        print(f"Instant sample network: {len(manager.cached_nodes)} nodes, {len(manager.cached_edges)} edges")
        return jsonify(result)
        
    except Exception as e:
        print(f"Sample network error: {e}")
        # Return minimal fallback
        return jsonify({
            'nodes': [],
            'edges': [],
            'stats': {'node_count': 0, 'edge_count': 0, 'layer_counts': {}}
        })

@app.route('/api/patient_similarity')
def get_patient_similarity():
    """API endpoint to compute patient similarity network."""
    # This would compute and return patient similarity edges
    # Implementation similar to your existing patient similarity code
    return jsonify({'message': 'Patient similarity computation not yet implemented'})

def initialize_manager():
    """Initialize the global network manager."""
    global network_manager
    if network_manager is None:
        print("Initializing network manager...")
        network_manager = MultiLayerNetworkManager()
        print(f"Available layers: {list(network_manager.layer_definitions.keys())}")
        print(f"Available edge types: {list(network_manager.edge_types)}")
    return network_manager

if __name__ == '__main__':
    # Initialize network manager
    initialize_manager()
    
    print("Starting Multi-Layer Network Visualization Server...")
    print("\nOpen http://localhost:5002 in your browser to access the visualization")
    
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5002)