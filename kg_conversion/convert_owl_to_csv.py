"""
Convert OWL ontology file to CSV format
Extracts nodes, edges, and their attributes
"""
import xml.etree.ElementTree as ET
import pandas as pd
from collections import defaultdict
import re
from pathlib import Path


def clean_uri(uri):
    """Clean URI by removing namespace prefixes and extracting ID."""
    if not uri:
        return ''
    # Remove common prefixes
    uri = uri.replace('http://www.w3.org/2002/07/owl#', '')
    uri = uri.replace('http://www.w3.org/1999/02/22-rdf-syntax-ns#', '')
    uri = uri.replace('http://www.w3.org/2000/01/rdf-schema#', '')
    uri = uri.replace('http://purl.obolibrary.org/obo/', '')
    # Remove leading #
    uri = uri.lstrip('#')
    return uri


def classify_node_type(node_id, node_type):
    """Classify node into categories based on ID patterns."""
    if 'Sample_' in node_id or 'GSM' in node_id:
        return 'Patient'
    elif 'Protein_' in node_id or 'Gene_' in node_id:
        return 'Gene/Protein'
    elif 'GO:' in node_id or 'GO_' in node_id:
        return 'GO Term'
    elif 'Pathway' in node_id or 'REACT:' in node_id or 'R-HSA' in node_id:
        return 'Pathway'
    elif 'MONDO:' in node_id or 'HP:' in node_id:
        return 'Disease/Phenotype'
    else:
        return 'Other'


def parse_owl_to_csv(owl_file='output/GSE54514_enriched_ontology_degfilterv2.9.owl',
                     output_dir='owl_exports'):
    """
    Parse OWL file and export to CSV files.
    
    Creates:
    - nodes.csv: All entities with their types and attributes
    - edges.csv: All relationships between entities
    - classes.csv: Class hierarchy
    """
    
    print("="*80)
    print("CONVERTING OWL TO CSV")
    print("="*80)
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Data structures
    nodes_data = []
    edges_data = []
    classes_data = []
    
    # Parse OWL file
    try:
        print(f"\n📖 Parsing OWL file: {owl_file}")
        tree = ET.parse(owl_file)
        root = tree.getroot()
        
        # Common OWL/RDF namespaces
        ns = {
            'owl': 'http://www.w3.org/2002/07/owl#',
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
            'obo': 'http://purl.obolibrary.org/obo/',
        }
        
        # Extract all elements
        all_individuals = root.findall('.//owl:NamedIndividual', ns)
        all_classes = root.findall('.//owl:Class', ns)
        all_obj_properties = root.findall('.//owl:ObjectProperty', ns)
        
        print(f"Found {len(all_individuals)} individuals")
        print(f"Found {len(all_classes)} classes")
        print(f"Found {len(all_obj_properties)} object properties")
        
        # Process Classes
        print("\n🏷️  Processing classes...")
        for cls in all_classes:
            class_id = cls.get(f'{{{ns["rdf"]}}}about', '')
            if class_id:
                class_id = clean_uri(class_id)
                
                # Get label
                label_elem = cls.find('.//rdfs:label', ns)
                label = label_elem.text if label_elem is not None else class_id
                
                # Get parent class
                subclass_elem = cls.find('.//rdfs:subClassOf', ns)
                parent = ''
                if subclass_elem is not None:
                    parent = subclass_elem.get(f'{{{ns["rdf"]}}}resource', '')
                    parent = clean_uri(parent)
                
                classes_data.append({
                    'class_id': class_id,
                    'label': label,
                    'parent_class': parent
                })
        
        print(f"Extracted {len(classes_data)} classes")
        
        # Process Individuals (Nodes)
        print("\n👤 Processing individuals (nodes)...")
        node_count = 0
        
        for individual in all_individuals:
            ind_id = individual.get(f'{{{ns["rdf"]}}}about', '')
            if not ind_id:
                continue
                
            ind_id = clean_uri(ind_id)
            node_count += 1
            
            # Get type
            type_elem = individual.find('.//rdf:type', ns)
            node_type = 'Unknown'
            if type_elem is not None:
                type_uri = type_elem.get(f'{{{ns["rdf"]}}}resource', '')
                node_type = clean_uri(type_uri)
            
            # Classify node type
            node_category = classify_node_type(ind_id, node_type)
            
            # Collect all properties as attributes
            attributes = {}
            for child in individual:
                tag = child.tag.split('}')[-1]  # Remove namespace
                
                # Skip rdf:type (already captured)
                if tag == 'type':
                    continue
                
                # Get value
                if child.text:
                    attributes[tag] = child.text
                elif f'{{{ns["rdf"]}}}resource' in child.attrib:
                    # This is an object property (edge)
                    target = clean_uri(child.get(f'{{{ns["rdf"]}}}resource'))
                    
                    # Add as edge
                    edges_data.append({
                        'source': ind_id,
                        'relation': tag,
                        'target': target,
                        'weight': 1.0
                    })
            
            # Add node
            node_entry = {
                'node_id': ind_id,
                'node_type': node_type,
                'node_category': node_category,
            }
            node_entry.update(attributes)
            nodes_data.append(node_entry)
            
            if node_count % 1000 == 0:
                print(f"  Processed {node_count} nodes, {len(edges_data)} edges...")
        
        print(f"✅ Extracted {len(nodes_data)} nodes")
        print(f"✅ Extracted {len(edges_data)} edges")
        
        # Convert to DataFrames
        print("\n💾 Converting to DataFrames...")
        df_nodes = pd.DataFrame(nodes_data)
        df_edges = pd.DataFrame(edges_data)
        df_classes = pd.DataFrame(classes_data)
        
        # Save to CSV
        nodes_file = f"{output_dir}/nodes.csv"
        edges_file = f"{output_dir}/edges.csv"
        classes_file = f"{output_dir}/classes.csv"
        
        print("\n📁 Saving to CSV files...")
        df_nodes.to_csv(nodes_file, index=False)
        print(f"  ✓ Saved nodes to: {nodes_file}")
        print(f"    Shape: {df_nodes.shape}")
        print(f"    Columns: {', '.join(df_nodes.columns.tolist())}")
        
        df_edges.to_csv(edges_file, index=False)
        print(f"\n  ✓ Saved edges to: {edges_file}")
        print(f"    Shape: {df_edges.shape}")
        print(f"    Columns: {', '.join(df_edges.columns.tolist())}")
        
        if len(df_classes) > 0:
            df_classes.to_csv(classes_file, index=False)
            print(f"\n  ✓ Saved classes to: {classes_file}")
            print(f"    Shape: {df_classes.shape}")
        
        # Print summary statistics
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        
        print("\nNode Categories:")
        if 'node_category' in df_nodes.columns:
            print(df_nodes['node_category'].value_counts())
        
        print("\nEdge Relations:")
        if 'relation' in df_edges.columns:
            print(df_edges['relation'].value_counts())
        
        print("\n" + "="*80)
        print("✅ CONVERSION COMPLETE")
        print("="*80)
        
        return df_nodes, df_edges, df_classes
        
    except FileNotFoundError:
        print(f"❌ Error: OWL file not found at {owl_file}")
        return None, None, None
    except ET.ParseError as e:
        print(f"❌ Error parsing XML: {e}")
        print("\nTrying alternative parsing method...")
        
        # Alternative: treat as text and extract patterns
        return parse_owl_as_text(owl_file, output_dir)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def parse_owl_as_text(owl_file, output_dir):
    """Fallback parser: extract data using regex patterns."""
    print("\n📖 Parsing OWL as text file...")
    
    nodes = set()
    edges = []
    
    with open(owl_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find individuals
    individual_pattern = r'<owl:NamedIndividual\s+rdf:about="([^"]+)"'
    individuals = re.findall(individual_pattern, content)
    
    # Find relationships
    relation_pattern = r'<([^>\s]+)\s+rdf:resource="([^"]+)"'
    relations = re.findall(relation_pattern, content)
    
    print(f"Found {len(individuals)} individuals")
    print(f"Found {len(relations)} relations")
    
    # Create DataFrames
    nodes_data = [{'node_id': clean_uri(ind), 
                   'node_category': classify_node_type(clean_uri(ind), '')} 
                  for ind in individuals]
    
    edges_data = [{'source': 'Unknown', 'relation': clean_uri(rel), 
                   'target': clean_uri(target), 'weight': 1.0}
                  for rel, target in relations]
    
    df_nodes = pd.DataFrame(nodes_data)
    df_edges = pd.DataFrame(edges_data)
    
    # Save
    df_nodes.to_csv(f"{output_dir}/nodes.csv", index=False)
    df_edges.to_csv(f"{output_dir}/edges.csv", index=False)
    
    print(f"\n✅ Saved {len(df_nodes)} nodes and {len(df_edges)} edges")
    
    return df_nodes, df_edges, None


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert OWL ontology to CSV files')
    parser.add_argument('--owl-file', type=str, 
                       default='output/GSE54514_enriched_ontology_degfilterv2.9.owl',
                       help='Path to OWL file')
    parser.add_argument('--output-dir', type=str, default='kg_conversion',
                       help='Output directory for CSV files')
    
    args = parser.parse_args()
    
    df_nodes, df_edges, df_classes = parse_owl_to_csv(args.owl_file, args.output_dir)
    
    if df_nodes is not None:
        print("\n🎉 Success! CSV files are ready in:", args.output_dir)
        print("\nYou can now use these files with:")
        print(f"  - pandas: pd.read_csv('{args.output_dir}/nodes.csv')")
        print(f"  - NetworkX: nx.read_edgelist('{args.output_dir}/edges.csv')")
        print(f"  - PyVis visualization")
