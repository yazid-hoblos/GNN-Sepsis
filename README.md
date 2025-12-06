# 2526-m2geniomhe-GNN-sepsis

# Comparing Traditional and Graph-Enhanced Pipelines for Sepsis Prediction Using Omics Data

## Project Overview

This project explores **traditional versus graph-enhanced pipelines** for predicting sepsis outcomes using omics datasets.
We leverage both **preprocessed datasets** and **graph-based embeddings** (e.g., ComplEx and RGCN) to compare performance and analyze patient-specific biological networks.

---

## Project Structure

```
M2_GENIOMHE-embeddingsGraphs/
│
├── data/                 # Example directory containing preprocessed datasets
├── load_dataset.py       # Scripts to load and process GEO datasets
├── load_embeddings.py    # Scripts to load pretrained graph embeddings
├── models/               # Pretrained embeddings for ComplEx and RGCN algorithms 
                          # (or retrain with OWL-based graphs)
├── output/               # Generated graphs and statistics from embeddings
```

---

## Data Overview

| Number | Study Title                                                       | Series Accession | Series ID | Disease | Samples | Preprocessing Summary                                                                                                                                                                                              |
| ------ | ----------------------------------------------------------------- | ---------------- | --------- | ------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1      | Whole blood transcriptome of survivors and nonsurvivors of sepsis | GSE54514         | 200054514 | Sepsis  | 163     | Raw data scanned, pre-processed in GenomeStudio. Probes filtered by detection p<0.005 in ≥1 sample. Quantile normalized and log-transformed in BRB-ArrayTools. Low-variance genes removed; validated with qRT-PCR. |

---

## Usage

### 1. Load Dataset

Use the `data/` folder as an example directory containing preprocessed datasets:

```bash
python load_dataset.py --datasets GSE54514
```

### 2. Load Pretrained Embeddings

The `models/` directory contains embeddings generated with **ComplEx** and **RGCN**.
You can either use these pretrained embeddings or retrain using OWL-based graphs:

```bash
python load_embeddings.py
```

---

## Notes

* This repository allows for **both traditional omics pipelines** and **graph-enhanced pipelines** for sepsis prediction.
* Graph embeddings capture **network relationships between genes, pathways, and patients**, which can be used for downstream classification or clustering.
* Students can retrain embeddings using their own ontologies or datasets if desired.

## Branches

* `dev/ml-test`: ML model training and evaluation branch (details on training in readme there)
* `dev/KGE`: KG exploration branch