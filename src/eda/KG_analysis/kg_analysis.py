# =============================================================================
# kg_analysis.py
# =============================================================================
# This script performs descriptive and enrichment analysis of proteins based
# on knowledge graph annotations. It integrates transcriptomic regulation data
# with biological context (GO terms and Reactome pathways) to provide:
# 1. Descriptive reports for upregulated and downregulated proteins.
# 2. Statistical enrichment analysis using Fisher's Exact Test.
# =============================================================================

import pandas as pd
import numpy as np
import requests
from collections import Counter
from tqdm import tqdm
from tabulate import tabulate
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

# =============================================================================
# Section 1: Extract protein data from ontology
# =============================================================================

def extract_protein_data(onto):
    """
    Extracts relevant protein attributes from an ontology instance.

    Args:
        onto: An ontology object containing protein instances.

    Returns:
        pd.DataFrame: A dataframe containing protein names, regulation
                      status, log2 fold-change, p-values, and associated
                      biological pathways and GO terms.
    """
    protein_analysis_data = []

    for protein in onto.Protein.instances():
        # Transcriptomic regulation info
        regulation_status = getattr(protein, "hasTypeRegulation", ["Unknown"])[0]
        log2_fc_value = getattr(protein, "hasLog2_FC", [0.0])[0]

        # Statistical significance
        p_value = getattr(protein, "hasPValue", [1.0])[0]

        # Biological context
        pathways = [p.name for p in getattr(protein, "isAssociatedWithProteinPathway", [])]
        go_terms = [go.name for go in getattr(protein, "isAssociatedWithGO", [])]

        # Append to analysis dataset
        protein_analysis_data.append({
            "protein": protein.name,
            "regulation": regulation_status,
            "log2_fc": log2_fc_value,
            "p_value": p_value,
            "pathways": list(set(pathways)),
            "go_terms": list(set(go_terms))
        })

    return pd.DataFrame(protein_analysis_data)

# =============================================================================
# Section 2: Resolve biological term names (GO terms and Reactome pathways)
# =============================================================================

# Cache to avoid repeated API calls
name_cache = {}

def get_biological_name(term_id):
    """
    Resolve names for GO terms (QuickGO API) and Reactome pathways
    (Reactome Content Service).

    Args:
        term_id (str): Identifier of the GO term or Reactome pathway.

    Returns:
        str: Human-readable name of the biological term.
    """
    clean_id = term_id.replace('_instance', '').replace('GO_', 'GO:').replace('Pathway_', '')

    if clean_id in name_cache:
        return name_cache[clean_id]

    try:
        # Reactome pathway lookup
        if "R-HSA" in clean_id and "GO:" not in clean_id:
            url = f"https://reactome.org/ContentService/data/query/{clean_id}"
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                name = r.json().get('displayName', term_id)
                name_cache[clean_id] = name
                return name

        # GO term lookup via QuickGO
        if "GO:" in clean_id:
            url = f"https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/{clean_id}"
            r = requests.get(url, headers={"Accept": "application/json"}, timeout=5)
            if r.status_code == 200:
                name = r.json()['results'][0]['name']
                name_cache[clean_id] = name
                return name

    except Exception:
        pass

    # Return original term ID if resolution fails
    return term_id

# =============================================================================
# Section 3: Descriptive biological report (pre-enrichment)
# =============================================================================

def descriptive_biological_report(df):
    """
    Generate descriptive summaries of GO terms and pathways associated
    with upregulated and downregulated proteins.

    Args:
        df (pd.DataFrame): Protein data with regulation, pathways, and GO terms.

    Returns:
        dict: Dictionary containing term frequencies for each regulation category.
    """
    print(f"\n{'='*90}")
    print("SEPSIS BIOLOGICAL INSIGHTS")
    print("Descriptive analysis of differentially regulated proteins")
    print(f"{'='*90}\n")

    df_clean = df.copy()
    df_clean['regulation'] = df_clean['regulation'].astype(str).str.strip()

    results = {
        'Upregulated': {'GO': Counter(), 'Pathway': Counter()},
        'Downregulated': {'GO': Counter(), 'Pathway': Counter()}
    }

    for status in ['Upregulated', 'Downregulated']:
        subset = df_clean[df_clean['regulation'] == status].copy()

        if subset.empty:
            print(f"No proteins classified as {status}.\n")
            continue

        # Sort proteins by absolute fold-change to highlight impact
        subset['abs_fc'] = subset['log2_fc'].abs()
        subset = subset.sort_values(by='abs_fc', ascending=False)

        print(f"{status.upper()} ({len(subset)} proteins)")

        # Display top proteins by impact
        top_proteins = [p.replace('Protein_', '') for p in subset['protein'].head(15).tolist()]
        print(f"Top impact proteins: {', '.join(top_proteins)}\n")

        print("Note: Term frequencies indicate annotation prevalence in the "
              "knowledge graph and do not represent statistical enrichment.\n")

        # Analyze both pathways and GO terms
        for col, label in [('pathways', 'Pathway'), ('go_terms', 'GO')]:
            flat_items = [item for sublist in subset[col] for item in sublist]
            results[status][label].update(flat_items)

            if not flat_items:
                print(f"No {label} associations found.\n")
                continue

            top_items = Counter(flat_items).most_common(10)
            table_data = []

            for item_id, count in tqdm(top_items, desc=f"Resolving {status} {label} terms", leave=False):
                name = get_biological_name(item_id)
                table_data.append([name, item_id.replace('_instance', ''), count])

            print(f"Top {len(top_items)} {label} associations")
            print(tabulate(table_data, headers=["Name / Description", "ID", "Frequency"], tablefmt="simple"))
            print()

    return results

# =============================================================================
# Section 4: Enrichment analysis of biological terms
# =============================================================================

def perform_enrichment(term_type, category_name, df, results_mapping):
    """
    Perform Fisher's Exact Test for enrichment of GO terms or pathways.

    Args:
        term_type (str): "GO" or "Pathway".
        category_name (str): Regulation category ("Upregulated" / "Downregulated").
        df (pd.DataFrame): Protein data with annotations.
        results_mapping (dict): Term frequency mapping from descriptive report.

    Returns:
        pd.DataFrame: Enrichment results with counts, gene ratios, p-values, FDR, and -log10(FDR).
    """
    results = []
    background_counts = Counter()
    col_name = 'go_terms' if term_type == "GO" else 'pathways'

    # Define sets
    all_proteins = set(df['protein'])
    current_gene_set = set(df[df['regulation'] == category_name]['protein'])
    set_size = len(current_gene_set)

    # Count term occurrences in background
    for entry in df[col_name].dropna():
        terms = entry if isinstance(entry, list) else [t.strip() for t in str(entry).split(',')]
        for t in terms:
            background_counts[t] += 1

    # Perform Fisher's Exact Test for each term
    if category_name in results_mapping and term_type in results_mapping[category_name]:
        for term, a in results_mapping[category_name][term_type].items():
            if a == 0:
                continue

            term_bg_total = max(background_counts.get(term, 0), a)
            gene_ratio = a / set_size if set_size > 0 else 0

            # Contingency table
            b = set_size - a
            c = term_bg_total - a
            d = len(all_proteins) - set_size - c

            _, p_value = fisher_exact([[a, b], [c, d]], alternative="greater")
            results.append((term, a, term_bg_total, gene_ratio, p_value))

    res_df = pd.DataFrame(results, columns=["Term", "Count_in_set", "Count_in_background", "Gene_Ratio", "p_value"])

    if not res_df.empty:
        res_df['FDR'] = multipletests(res_df['p_value'], method="fdr_bh")[1]
        res_df['Category'] = category_name
        res_df['-log10_FDR'] = -np.log10(res_df['FDR'] + 1e-300)

    return res_df
