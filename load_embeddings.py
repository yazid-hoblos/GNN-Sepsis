import os
import random
import numpy as np
import pandas as pd


def load_patient_data(emb_path, mapping_csv, node_features_csv, control_label, patient_prefix="Sample"):
    # 1. Load entity embeddings
    emb = np.load(emb_path, allow_pickle=True)
    if np.iscomplexobj(emb):  # ComplEx produces complex vectors
        print(f" Complex embeddings detected in {emb_path}, splitting into real+imag.")
        emb = np.concatenate([emb.real, emb.imag], axis=1)

    # 2. Load PyKEEN mapping (entity_id ↔ label)
    df_map = pd.read_csv(mapping_csv)  # columns: entity_id, label

    # 3. Load node features (long format)
    df_csv = pd.read_csv(node_features_csv)  # columns: node_id, name_feature, value_feature

    # Pivot to wide format (node_id → feature_name: feature_value)
    df_wide = df_csv.pivot_table(
        index="node_id",
        columns="name_feature",
        values="value_feature",
        aggfunc="first"
    ).reset_index()

    # 4. Merge embeddings mapping (entity_id ↔ label ↔ node_id)
    df_merged = pd.merge(
        df_map, df_wide,
        left_on="label", right_on="node_id",
        how="inner"
    )

    # 5. Attach embeddings
    df_merged["embedding"] = df_merged["entity_id"].apply(lambda i: emb[i])

    # 6. Keep only patients (node_id starts with prefix)
    df_patients = df_merged[df_merged["node_id"].str.startswith(patient_prefix)].reset_index(drop=True)

    # 7. Extract X (embeddings), y (disease/control), and IDs
    X_patients = np.stack(df_patients["embedding"].values)
    y_patients = df_patients["hasDiseaseStatus"].apply(
        lambda x: 0 if str(x).lower() == control_label.lower() else 1
    ).values
    patient_ids = df_patients["entity_id"].tolist()

    return patient_ids, X_patients, y_patients, df_patients


# -------------------------------
# Load ALL entities (not only patients)
# -------------------------------
def load_all_entities(emb_path, mapping_csv, node_features_csv):
    # 1. Load embeddings
    emb = np.load(emb_path, allow_pickle=True)
    if np.iscomplexobj(emb):
        print(f" Complex embeddings detected in {emb_path}, splitting real+imag.")
        emb = np.concatenate([emb.real, emb.imag], axis=1)

    # 2. Load mapping + features
    df_map = pd.read_csv(mapping_csv)
    df_csv = pd.read_csv(node_features_csv)

    # Use LEFT join to keep all entities from entity_mapping, even without features
    df_merged = pd.merge(
        df_map, df_csv,
        left_on="label", right_on=df_csv.columns[0],
        how="left"
    )

    # 3. Attach embeddings
    df_merged["embedding"] = df_merged["entity_id"].apply(lambda i: emb[i])

    # For entities without features (like Proteins), fill node_id with label
    if 'node_id' not in df_merged.columns:
        df_merged['node_id'] = df_merged['label']
    else:
        df_merged['node_id'] = df_merged['node_id'].fillna(df_merged['label'])

    X = np.stack(df_merged["embedding"].values)
    entity_ids = df_merged["entity_id"].tolist()
    entity_labels = df_merged["label"].tolist()

    return entity_ids, X, entity_labels, df_merged


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":

    kgs = ['GSE54514']
    models = ['ComplEx', 'RGCN']
    
    for kg in kgs:
        for model in models:

            if kg == "GSE54514":
                control_label = "healthy"
            elif kg == "GSE76293":
                control_label = "HVT"
            elif kg == "GSE40012":
                control_label = "healthy control"
            else:
                raise ValueError("Dataset not recognized for control group labeling.")
          
            emb_path = f"./models/executions/{kg}_enriched_ontology_degfilterv2.9/outputmodel_{model}_entity_embeddings.npy"
            node_features_csv = f"./models/executions/{kg}_enriched_ontology_degfilterv2.9/node_features.csv"
            map_csv = f"./models/executions/{kg}_enriched_ontology_degfilterv2.9/outputmodel_{model}_entity_mapping.csv"
            
            # Load patients
            patient_ids, X_patients, y_patients, df_patients = load_patient_data(emb_path, map_csv, node_features_csv, control_label)
            print(f"Loaded {len(patient_ids)} patients, embedding dim={X_patients.shape[1]} for {kg}-{model}")
            
            # --- Load all entities ---
            entity_ids, X_all, entity_labels, df_all = load_all_entities(emb_path, map_csv, node_features_csv)
            print(f"Loaded {len(entity_ids)} entities for {kg}-{model}")
            # ecris ligne pour afficher les lignes ou la prmeiere colonne commence par "Protein_" de df_all
            print(df_all[df_all['label'].str.startswith('Sample_')].head(500))