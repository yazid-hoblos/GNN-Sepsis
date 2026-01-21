#!/usr/bin/env python3
"""
Enrich GO term IDs with their descriptions from go.obo ontology file.

This script loads common neighbors CSV and enriches GO terms with their definitions
from the Gene Ontology files used in KG construction.
"""

import pandas as pd
import re
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_go_file():
    """Search for go.obo file in common locations."""
    search_paths = [
        Path('/usr/share/ontologies/go.obo'),
        Path('/usr/local/share/ontologies/go.obo'),
        Path.home() / '.local/share/ontologies/go.obo',
        Path('data/go.obo'),
        Path('go.obo'),
    ]
    
    # Also search in OntoKGCreation locations
    search_paths.extend([
        Path('src/utils/OntoKGCreation'),
        Path('lib/OntoKGCreation'),
    ])
    
    for search_dir in search_paths:
        if search_dir.is_dir():
            go_file = search_dir / 'go.obo'
            if go_file.exists():
                logger.info(f"Found go.obo at: {go_file}")
                return go_file
        elif search_dir.is_file():
            logger.info(f"Found go.obo at: {search_dir}")
            return search_dir
    
    logger.warning("go.obo file not found in standard locations")
    return None


def parse_go_obo(go_file_path):
    """Parse go.obo file and extract GO term definitions."""
    go_defs = {}
    
    logger.info(f"Parsing go.obo file: {go_file_path}")
    
    try:
        current_id = None
        current_name = None
        current_def = None
        in_term = False
        
        with open(go_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                
                # Start of a term definition
                if line == '[Term]':
                    # Save previous term if exists
                    if current_id and (current_name or current_def):
                        # Store with multiple format variants
                        go_defs[current_id] = {
                            'name': current_name or 'Unknown',
                            'definition': current_def or 'No definition'
                        }
                        # Also store with _instance suffix
                        go_defs[current_id + '_instance'] = {
                            'name': current_name or 'Unknown',
                            'definition': current_def or 'No definition'
                        }
                    
                    current_id = None
                    current_name = None
                    current_def = None
                    in_term = True
                    continue
                
                if not in_term or not line:
                    continue
                
                # Parse ID line (e.g., "id: GO:0008150")
                if line.startswith('id: GO:'):
                    go_num = line.split('GO:')[1].strip()
                    # Store as both GO:XXXX and GO_XXXX formats
                    current_id = f'GO:{go_num}'
                
                # Parse name line
                elif line.startswith('name: '):
                    current_name = line.replace('name: ', '')
                
                # Parse definition line (may be multiline)
                elif line.startswith('def: "'):
                    # Extract definition (remove quotes and citation info)
                    def_text = line.split('def: "')[1].split('" [')[0]
                    current_def = def_text
            
            # Don't forget the last term
            if current_id and (current_name or current_def):
                go_defs[current_id] = {
                    'name': current_name or 'Unknown',
                    'definition': current_def or 'No definition'
                }
                # Also store with _instance suffix
                go_defs[current_id + '_instance'] = {
                    'name': current_name or 'Unknown',
                    'definition': current_def or 'No definition'
                }
        
        logger.info(f"Loaded {len(go_defs)} GO term variants from {go_file_path}")
        logger.info(f"Sample: {list(go_defs.items())[:3]}")
        return go_defs
    
    except Exception as e:
        logger.error(f"Error parsing go.obo: {e}")
        return {}


def enrich_common_neighbors(common_neighbors_csv, go_defs, output_csv):
    """Enrich common neighbors CSV with GO term definitions."""
    
    # Load common neighbors
    logger.info(f"Loading common neighbors from: {common_neighbors_csv}")
    neighbors_df = pd.read_csv(common_neighbors_csv)
    
    # Enrich GO terms
    enriched_rows = []
    for idx, row in neighbors_df.iterrows():
        entity = row['entity']
        
        # Check if it's a GO term
        if str(entity).startswith('GO_'):
            # Try exact match first
            go_info = go_defs.get(entity)
            
            # If not found, try removing _instance suffix
            if not go_info and entity.endswith('_instance'):
                base_id = entity.replace('_instance', '')
                go_info = go_defs.get(base_id)
            
            # Try alternative format
            if not go_info:
                # Convert GO_0008150 to GO:0008150
                alt_id = entity.replace('GO_', 'GO:').replace('_instance', '')
                go_info = go_defs.get(alt_id)
            
            if go_info:
                entity_name = go_info['name']
                definition = go_info['definition']
            else:
                # Extract just the GO number for display
                entity_name = entity.replace('GO_', '').replace('_instance', '')
                definition = "Definition not found in go.obo"
        else:
            entity_name = entity.replace('Protein_', '').replace('Pathway_', '').replace('GO_', '')
            definition = ""
        
        enriched_rows.append({
            'entity': entity,
            'entity_name': entity_name,
            'num_biomarkers': row['num_biomarkers'],
            'biomarkers': row['biomarkers'],
            'definition': definition
        })
    
    enriched_df = pd.DataFrame(enriched_rows)
    
    # Sort by number of biomarkers
    enriched_df = enriched_df.sort_values('num_biomarkers', ascending=False)
    
    # Save enriched file
    enriched_df.to_csv(output_csv, index=False)
    logger.info(f"Saved enriched common neighbors to: {output_csv}")
    
    # Print GO terms with definitions
    go_enriched = enriched_df[enriched_df['entity'].str.startswith('GO_')]
    if len(go_enriched) > 0:
        logger.info("\nGO Terms found as common neighbors:")
        print("\n" + "=" * 120)
        print("GO TERMS (Common Neighbors - Hub Connectors)")
        print("=" * 120)
        for idx, row in go_enriched.iterrows():
            print(f"\n{row['entity_name']} ({row['num_biomarkers']} biomarkers)")
            if row['definition'] and row['definition'] != 'Definition not found':
                def_text = row['definition'][:100]
                if len(row['definition']) > 100:
                    def_text += "..."
                print(f"  Definition: {def_text}")
            print(f"  Entity ID: {row['entity']}")
            print(f"  Connected biomarkers: {row['biomarkers']}")
        print("\n" + "=" * 120)
    
    return enriched_df


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Enrich GO terms with definitions from go.obo')
    parser.add_argument('--common-neighbors-csv', type=str,
                       default='results/interpretability/complex_svm_mapped/kg_subgraph_analysis/common_neighbors.csv',
                       help='Common neighbors CSV file')
    parser.add_argument('--go-obo', type=str, default=None,
                       help='Path to go.obo file (auto-detected if not provided)')
    parser.add_argument('--output-csv', type=str,
                       default='results/interpretability/complex_svm_mapped/kg_subgraph_analysis/common_neighbors_enriched.csv',
                       help='Output CSV file')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    
    # Find go.obo if not provided
    if args.go_obo:
        go_file = Path(args.go_obo)
    else:
        go_file = find_go_file()
    
    if not go_file or not go_file.exists():
        logger.error("Could not find go.obo file. Please provide --go-obo argument.")
        return
    
    # Parse GO definitions
    go_defs = parse_go_obo(go_file)
    
    if not go_defs:
        logger.error("No GO definitions loaded. Check the go.obo file.")
        return
    
    # Enrich
    enrich_common_neighbors(args.common_neighbors_csv, go_defs, args.output_csv)


if __name__ == '__main__':
    main()
