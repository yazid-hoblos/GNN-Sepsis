#!/usr/bin/env python3
"""
Enrich GO terms and reactions with biological descriptions.

This script:
1. Enriches GO terms with their Gene Ontology definitions
2. Provides context for reactions (Reactome)
3. Creates interpretable summaries
"""

import pandas as pd
import re
from pathlib import Path
import logging
import subprocess
import json
import requests

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


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
                    current_id = f'GO:{go_num}'
                
                # Parse name line
                elif line.startswith('name: '):
                    current_name = line.replace('name: ', '')
                
                # Parse definition line
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
                go_defs[current_id + '_instance'] = {
                    'name': current_name or 'Unknown',
                    'definition': current_def or 'No definition'
                }
        
        logger.info(f"Loaded {len(go_defs)} GO term variants")
        return go_defs
    
    except Exception as e:
        logger.error(f"Error parsing go.obo: {e}")
        return {}


def find_go_file():
    """Search for go.obo file."""
    search_paths = [
        Path('F:/yaz/OntoKGCreation/go.obo'),
        Path('/usr/share/ontologies/go.obo'),
        Path('data/go.obo'),
    ]
    
    for go_file in search_paths:
        if go_file.exists():
            logger.info(f"Found go.obo at: {go_file}")
            return go_file
    
    logger.warning("go.obo file not found")
    return None


def enrich_go_terms(go_df: pd.DataFrame, go_defs: dict) -> pd.DataFrame:
    """Enrich GO terms with definitions."""
    results = []
    
    for _, row in go_df.iterrows():
        go_term = row['go_term']
        go_info = go_defs.get(go_term)
        
        if not go_info:
            # Try without _instance
            base_id = go_term.replace('_instance', '')
            go_info = go_defs.get(base_id)
        
        if not go_info:
            # Try format conversion
            go_num = go_term.replace('GO_', '').replace('_instance', '')
            go_info = go_defs.get(f'GO:{go_num}')
        
        name = go_info['name'] if go_info else 'Unknown'
        definition = go_info['definition'] if go_info else 'Definition not available'
        
        results.append({
            'go_id': row['go_id'],
            'go_name': name,
            'go_definition': definition,
            'num_biomarkers': row['num_biomarkers'],
            'biomarkers': row['biomarkers']
        })
    
    df = pd.DataFrame(results)
    return df.sort_values('num_biomarkers', ascending=False)


def get_reactome_info(reaction_id: str) -> dict:
    """Get information about Reactome reaction (fetch display name if possible)."""
    base = f'https://reactome.org/ContentService/data/query/{reaction_id}'
    url = f'https://reactome.org/content/detail/{reaction_id}'
    try:
        resp = requests.get(base, timeout=5)
        if resp.ok:
            data = resp.json()
            name = data.get('displayName', reaction_id)
        else:
            name = reaction_id
    except Exception:
        name = reaction_id
    return {
        'name': name,
        'type': 'Reactome Reaction',
        'url': url
    }


def enrich_reactions(reaction_df: pd.DataFrame) -> pd.DataFrame:
    """Enrich reactions with biological context."""
    results = []
    
    for _, row in reaction_df.iterrows():
        reaction_id = row['reaction_name']
        info = get_reactome_info(reaction_id)
        
        results.append({
            'reaction_id': row['reaction_name'],
            'reaction_name': info['name'],
            'num_biomarkers': row['num_biomarkers'],
            'biomarkers': row['biomarkers'],
            'reactome_url': info['url']
        })
    
    df = pd.DataFrame(results)
    # Group by number of biomarkers
    return df.sort_values('num_biomarkers', ascending=False)


def create_summary_report(go_enriched: pd.DataFrame, reaction_enriched: pd.DataFrame, 
                         output_path: str):
    """Create a summary report of GO terms and reactions."""
    with open(output_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write("BIOMARKER CONSENSUS: GO TERMS AND REACTIONS ENRICHMENT\n")
        f.write("="*100 + "\n\n")
        
        # GO Terms Summary
        f.write("-"*100 + "\n")
        f.write("TOP 20 GO TERMS CONNECTED TO BIOMARKER PROTEINS\n")
        f.write("-"*100 + "\n\n")
        
        for idx, (_, row) in enumerate(go_enriched.head(20).iterrows(), 1):
            f.write(f"{idx:2}. {row['go_name']}\n")
            f.write(f"    ID: {row['go_id']}\n")
            f.write(f"    Biomarkers: {row['num_biomarkers']}/28\n")
            f.write(f"    Definition: {row['go_definition']}\n")
            f.write(f"    Proteins: {row['biomarkers'][:100]}...\n\n")
        
        # Reactions Summary
        f.write("-"*100 + "\n")
        f.write("TOP 20 REACTOME REACTIONS CONNECTED TO BIOMARKER PROTEINS\n")
        f.write("-"*100 + "\n\n")
        
        if len(reaction_enriched) > 0:
            # Group by number of biomarkers
            by_count = reaction_enriched.groupby('num_biomarkers').size()
            f.write("Reactions by biomarker connectivity:\n")
            for num_bm, count in by_count.sort_index(ascending=False).items():
                f.write(f"  {num_bm} biomarker{'s' if num_bm > 1 else ''}: {count} reactions\n")
            
            f.write("\n\nTop reactions (by biomarker count):\n")
            for idx, (_, row) in enumerate(reaction_enriched.head(20).iterrows(), 1):
                f.write(f"{idx:2}. {row.get('reaction_name', row['reaction_id'])}\n")
                f.write(f"    ID: {row['reaction_id']}\n")
                f.write(f"    Biomarkers: {row['num_biomarkers']}\n")
                f.write(f"    URL: {row['reactome_url']}\n")
                f.write(f"    Proteins: {row['biomarkers']}\n\n")
        else:
            f.write("No reactions found.\n")
        
        # Key Findings
        f.write("\n" + "="*100 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"Total unique GO terms: {len(go_enriched)}\n")
        f.write(f"Total unique reactions: {len(reaction_enriched)}\n\n")
        
        # Top biological processes
        top_go = go_enriched.head(5)
        f.write("Top biological processes represented:\n")
        for _, row in top_go.iterrows():
            f.write(f"  - {row['go_name']} ({row['num_biomarkers']} biomarkers)\n")
        
        f.write("\nInterpretation:\n")
        f.write("The consensus biomarker proteins are predominantly involved in:\n")
        if any('binding' in name.lower() for name in go_enriched['go_name'].head(10)):
            f.write("  * Protein binding and molecular interactions\n")
        if any('process' in name.lower() or 'cellular' in name.lower() 
               for name in go_enriched['go_name'].head(10)):
            f.write("  * Cellular processes and organization\n")
        if any('metabolic' in name.lower() or 'metabolism' in name.lower() 
               for name in go_enriched['go_name'].head(20)):
            f.write("  * Metabolic processes\n")
    
    logger.info(f"Saved summary report: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Enrich GO terms and reactions with biological descriptions'
    )
    parser.add_argument('--go-csv', type=str,
                       default='results/interpretability/complex_svm_mapped/kg_subgraph_analysis_consensus/biomarker_go_terms.csv',
                       help='Path to GO terms CSV')
    parser.add_argument('--reaction-csv', type=str,
                       default='results/interpretability/complex_svm_mapped/kg_subgraph_analysis_consensus/biomarker_reactions.csv',
                       help='Path to reactions CSV')
    parser.add_argument('--go-obo', type=str, default=None,
                       help='Path to go.obo file')
    parser.add_argument('--output-dir', type=str,
                       default='results/interpretability/complex_svm_mapped/kg_subgraph_analysis_consensus',
                       help='Output directory')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    
    # Load GO terms and reactions
    logger.info("Loading GO terms and reactions...")
    go_df = pd.read_csv(args.go_csv)
    reaction_df = pd.read_csv(args.reaction_csv)
    
    logger.info(f"Loaded {len(go_df)} GO terms")
    logger.info(f"Loaded {len(reaction_df)} reactions")
    
    # Find and parse go.obo
    go_file = args.go_obo or find_go_file()
    if go_file:
        go_defs = parse_go_obo(go_file)
        go_enriched = enrich_go_terms(go_df, go_defs)
    else:
        logger.warning("go.obo not found, using basic GO term IDs")
        go_enriched = go_df.copy()
        go_enriched['go_name'] = go_enriched['go_id']
        go_enriched['go_definition'] = 'Definition not available'
    
    # Enrich reactions
    reaction_enriched = enrich_reactions(reaction_df)
    
    # Save enriched files
    go_enriched_csv = output_dir / 'biomarker_go_terms_enriched.csv'
    go_enriched.to_csv(go_enriched_csv, index=False)
    logger.info(f"Saved enriched GO terms: {go_enriched_csv}")
    
    reaction_enriched_csv = output_dir / 'biomarker_reactions_enriched.csv'
    reaction_enriched.to_csv(reaction_enriched_csv, index=False)
    logger.info(f"Saved enriched reactions: {reaction_enriched_csv}")
    
    # Create summary report
    report_path = output_dir / 'entity_enrichment_report.txt'
    create_summary_report(go_enriched, reaction_enriched, str(report_path))
    
    # Print summary
    logger.info("\n" + "="*100)
    logger.info("GO TERMS & REACTIONS SUMMARY")
    logger.info("="*100 + "\n")
    
    logger.info("Top 10 GO Terms:")
    for idx, (_, row) in enumerate(go_enriched.head(10).iterrows(), 1):
        logger.info(f"  {idx:2}. ({row['num_biomarkers']:2} biomarkers) {row.get('go_name', row.get('go_id', 'Unknown'))}")
    
    logger.info(f"\nTop 10 Reactions (by biomarker count):")
    for idx, (_, row) in enumerate(reaction_enriched.head(10).iterrows(), 1):
        display = row.get('reaction_name', row['reaction_id'])
        logger.info(f"  {idx:2}. ({row['num_biomarkers']} biomarker{'s' if row['num_biomarkers'] > 1 else ''}) {display}")
    
    logger.info(f"\nReaction connectivity summary:")
    reaction_counts = reaction_enriched['num_biomarkers'].value_counts().sort_index(ascending=False)
    for num_bm, count in reaction_counts.items():
        logger.info(f"  {num_bm} biomarker{'s' if num_bm > 1 else ''}: {count} reactions")


if __name__ == '__main__':
    main()
