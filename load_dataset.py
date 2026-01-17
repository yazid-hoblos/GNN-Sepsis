import argparse
import warnings
import pandas as pd
import numpy as np
import GEOparse
import gc
warnings.filterwarnings("ignore")


# ===============================
# Utility functions
# ===============================

def map_probes_to_genes(gpl_annotations: pd.DataFrame) -> pd.DataFrame:
    """
    Maps probes to their corresponding Entrez gene IDs.

    Parameters
    ----------
    gpl_annotations : pd.DataFrame
        The GPL annotations dataframe containing 'ID' and 'Entrez_Gene_ID' columns.

    Returns
    -------
    pd.DataFrame
        DataFrame mapping probes to gene IDs.
    """
    probe_to_gene = gpl_annotations[['ID', 'Entrez_Gene_ID']].dropna(subset=['Entrez_Gene_ID'])
    probe_to_gene['Entrez_Gene_ID'] = pd.to_numeric(probe_to_gene['Entrez_Gene_ID'], errors='coerce')
    probe_to_gene = probe_to_gene.dropna(subset=['Entrez_Gene_ID'])
    probe_to_gene['Entrez_Gene_ID'] = probe_to_gene['Entrez_Gene_ID'].astype(int)
    return probe_to_gene



def create_expression_data(gse) -> pd.DataFrame:
    """
    Extracts expression data from GEO dataset and converts to DataFrame.

    Parameters
    ----------
    gse : GEOparse.GEO
        GEO dataset object.

    Returns
    -------
    pd.DataFrame
        Expression matrix (samples x probes).
    """
    expression_data = pd.DataFrame({
        gsm_name: gsm.table.set_index("ID_REF")["VALUE"] 
        for gsm_name, gsm in gse.gsms.items()
    }).T

    expression_data = expression_data.apply(pd.to_numeric, errors='coerce')
    expression_data.dropna(axis=1, inplace=True)
    return expression_data


def load_geo_data(filepath: str, gpl_filepath: str, dataset_name: str):
    """
    Loads GEO dataset and corresponding GPL annotations.

    Parameters
    ----------
    filepath : str
        Path to the GEO dataset file.
    gpl_filepath : str
        Path to the GPL annotation file.
    dataset_name : str
        Name of the dataset.

    Returns
    -------
    gse : GEOparse.GEO
        Loaded GEO dataset.
    gpl_annotations : pd.DataFrame
        Processed GPL annotations mapping probes to genes.
    """
    gse = GEOparse.get_GEO(filepath=filepath, silent=True)
    gpl_annotations = pd.read_csv(gpl_filepath, sep="\t", comment='#')

    gpl_annotations = gpl_annotations[['ID', 'Entrez_Gene_ID']].dropna(subset=['Entrez_Gene_ID'])

    # Handle dataset-specific annotation formats (keep only first gene ID if multiple)
    gpl_annotations['Entrez_Gene_ID'] = gpl_annotations['Entrez_Gene_ID'].apply(lambda x: str(x).split('///')[0].strip())
        
    return gse, gpl_annotations


def extract_sample_status(gse, dataset_name: str) -> list:
    """
    Extracts the sample status (disease/control) from GEO dataset.

    Parameters
    ----------
    gse : GEOparse.GEO
        GEO dataset object.
    dataset_name : str
        Dataset name to determine extraction method.

    Returns
    -------
    list
        Sample statuses corresponding to each sample.
    """
    def search_metadata(key, lst):
        for item in lst:
            if item.split(": ")[0] == key:
                return item.split(": ")[1]
        return ""

    sample_status = []
    if dataset_name == "GSE54514":
        sample_status = [gsm.metadata["characteristics_ch1"][1].split(": ")[1] for gsm in gse.gsms.values()]
    else:
        raise ValueError("Dataset not recognized for sample status extraction.")
    
    return sample_status

def calculate_differential_expression(expression_data, sample_status, dataset_name):
    """
    Calculate differentially expressed genes and create patient-gene links
    based on the average gene expression value per patient.

    @param expression_data: DataFrame with rows as patients and columns as genes
    @param sample_status: List with status of each patient
    @param dataset_name: Dataset name (used for control labeling and log2FC thresholds)
    @param filter: "deg" to consider only significant genes
    @return: degs DataFrame, patient_gene_links DataFrame
    """

    import pandas as pd
    import numpy as np
    from scipy.stats import ttest_ind
    from statsmodels.stats.multitest import multipletests

    # --- Step 1: Define control group ---
    control_labels = {
        "GSE54514": "healthy"
    }
    if dataset_name not in control_labels:
        raise ValueError("Dataset not recognized for control group labeling.")
    control_label = control_labels[dataset_name]

    # --- Step 2: Split patients into control vs others ---
    sample_info = pd.DataFrame({"Sample": expression_data.index, "Status": sample_status})
    group1 = sample_info[sample_info["Status"].str.contains(control_label, case=False, na=False)]["Sample"]
    group2 = sample_info[~sample_info["Status"].str.contains(control_label, case=False, na=False)]["Sample"]

    # --- Step 3: Perform t-test and adjust p-values ---
    p_values = np.array([
        ttest_ind(expression_data.loc[group1, gene],
                  expression_data.loc[group2, gene])[1]
        for gene in expression_data.columns
    ])
    _, p_adjusted, _, _ = multipletests(p_values, method='fdr_bh')

    results = pd.DataFrame({
        "Gene": expression_data.columns,
        "P_Value": p_values,
        "Adjusted_P_Value": p_adjusted
    })

    
    # Filtrar a DEGs si se desea
    degs = results[results["Adjusted_P_Value"] < 0.01]
    significant_genes = degs['Gene'].tolist()

    return expression_data[significant_genes]



def process_dataset(dataset: dict):
    """
    Load and process a dataset (GEO + GPL).

    Parameters
    ----------
    dataset : dict
        Dictionary with keys: 'name', 'filepath', 'gpl'
    """
    gse, gpl_annotations = load_geo_data(dataset['filepath'], dataset['gpl'], dataset['name'])
    probe_to_gene = map_probes_to_genes(gpl_annotations)
    sample_status = extract_sample_status(gse, dataset['name'])
    expression_data = create_expression_data(gse)

    expression_data = calculate_differential_expression(expression_data, sample_status, dataset['name'])

    sample_info = pd.DataFrame({"Sample": expression_data.index, "Status": sample_status})
    # Further processing can be added here
    return expression_data, sample_info


def main():
    parser = argparse.ArgumentParser(description="Process GEO datasets")
    parser.add_argument('--datasets', nargs='+', required=True, help="Dataset names to process")
    args = parser.parse_args()

    possible_datasets = [
        {"name": "GSE54514", "filepath": "data/GSE54514_family.soft.gz", "gpl": "data/GPL6947-13512.txt"},
    ]

    datasets_to_process = [d for d in possible_datasets if d['name'] in args.datasets]
    if not datasets_to_process:
        print(f"No matching datasets found for: {args.datasets}")
        return

    for dataset in datasets_to_process:
        expression_data, sample_info = process_dataset(dataset)
        print(f"Processed {dataset['name']} - Expression data shape: {expression_data.shape}")


if __name__ == "__main__":
    main()
