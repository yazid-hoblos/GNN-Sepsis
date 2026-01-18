# Healthcare Context

# Proposed Subject

This project compares traditional and graph-enhanced pipelines for sepsis prediction using omics data. The goal is to evaluate whether graph embeddings encoding biological relationships (genes, proteins, pathways) improve classification performance compared to raw gene expression.

We were provided with pretrained KG embeddings (ComplEx, RGCN) and extended the work by training our own GNN models (GraphSAGE, RGCN, GAT) on the heterogeneous graph. We also explored HAN (Heterogeneous Attention Network) for node prediction and interpretability through attention weights.

# Dataset

The dataset comes from GEO (GSE54514), a whole blood transcriptome study of sepsis survivors and non-survivors measured on Illumina HumanHT-12 V4.0 (GPL6947) with 24,840 probes. 

It includes 54 unique patients (18 healthy, 36 sepsis), where sepsis patients were measured at multiple time points, giving 163 total samples (36 healthy, 127 sepsis). The raw data was already log2-transformed and quantile-normalized by the original authors.

For the Knowledge Graph, DEG filtering (adjusted p-value < 0.01, t-test with FDR correction) was applied, keeping ~1,295 proteins.

## Knowledge Graph Construction

The KG was built by integrating the gene expression data with external biological databases (Reactome, KEGG, Gene Ontology). Each sample is connected to the proteins it expresses, and proteins are linked to their pathways and functional annotations. The KG is stored as an OWL ontology file.

**Node types (9,212 total):**
- Sample: 163 patients with metadata (age, gender, disease status, severity, site of infection)
- Protein: 1,288 proteins encoded by DEGs
- Pathway: 1,556 biological pathways
- Reaction: 5,701 reactions
- GO Term: ~3,100 functional annotations (biological process, molecular function, cellular component)

**Knowledge Graph structure:**

![KG representation](docs/assets/KG_rep.png)


Sample metadata is stored as annotation properties: `hasDiseaseStatus`, `hasAge`, `hasGender`, `hasSeverity`, `hasSiteOfInfection`, `hasNeutrophilProportion`.

### Graph Versions

Two versions of the KG were generated to handle expression values differently:

- **v2.10 (Binning)**: Expression values are discretized into 3 bins with 0.2 overlapping. This creates categorical relationships between samples and proteins.
- **v2.11 (Average)**: Expression values are kept as continuous weights (averaged across replicates). This preserves the original expression intensity.

## Pretrained KG Embeddings 

The project was provided with pretrained KG embeddings using 2 models :

- **RGCN**: Relational Graph Convolutional Network, learns embeddings by aggregating neighbors with relation-specific weights. Produces 100-dim embeddings.
- **ComplEx**: Complex-valued embeddings that capture asymmetric relations. Stored as complex numbers, split into real+imaginary parts : 200-dim embeddings.

Embeddings for all 9212 entities (samples, proteins, pathways, GO terms) are stored in `models/executions/` as `.npy` files.

## Train our GNN Embeddings

We trained our own GNN models to extract embeddings comparable to the pretrained ones. To do this, we first needed to convert the OWL ontology into a format compatible with PyTorch Geometric.

### OWL to HeteroData Parsing 

The OWL file is parsed with RDFlib as RDF triplets `(subject, predicate, object)`. For example, `(Sample_GSM1234, hasExpression, Protein_ACTB)` represents a sample expressing a protein. The node type is inferred from the URI prefix: `Sample_` → sample, `Protein_` → protein, `GO:` → goterm, etc.

For sample nodes, the `hasDiseaseStatus` predicate is used to extract labels (0=healthy, 1=sepsis) for node classification. Edge weights are extracted from `hasExpressionValue` annotations in OWL Axioms and averaged across replicates.

The output is a PyTorch Geometric `HeteroData` with random normalized node features (dim=128), edge indices per relation type, and edge weights for hasExpression edges. To run the conversion:

```bash
python src/gnn/load_heterodata.py --version v2.11
```

### GNN Models

We trained 3 heterogeneous GNN models on this HeteroData graph. All models are adapted for heterogeneous graphs using `HeteroConv`, which applies a separate convolution per edge type and aggregates the results.

**GraphSAGE** (`HeteroGraphSAGE`): Aggregates neighbor features using weighted mean pooling. Messages are weighted by edge weights (expression values) before aggregation. Separate linear transformations are applied to neighbor features and self features.

**Weighted RGCN** (`HeteroRGCN`): Relational GCN with one weight matrix per edge type. Each relation type learns its own transformation. Messages are weighted by edge weights and normalized by weighted degree.

**GAT** (`HeteroGAT`): Uses attention to learn importance of each neighbor. Edge weights are incorporated as edge attributes (edge_dim=1) in the attention computation. First layer uses 4 attention heads (output dim = hidden/4 per head), second layer uses 1 head.

All models use 2 layers, 100 hidden dimensions (to match pretrained embeddings), and dropout 0.5. GraphSAGE and RGCN use ReLU activation, GAT uses ELU (default for attention networks). GAT uses 4 heads so that 100 is divisible (25 dims per head).

### Training 

Models are trained on a link prediction task: predict whether an edge exists between two nodes. We use BCE loss with negative sampling (ratio 1:1), Adam optimizer (lr=0.001), and early stopping (patience=20, max 200 epochs).

```bash
python src/gnn/extract_embeddings.py --model all --version v2.11
```

### Training Results

| Model | Epochs | Best Val Loss |
|-------|--------|---------------|
| GAT | 132 (early stop) | 0.8966 |
| RGCN | 200 | 1.0819 |
| GraphSAGE | 200 | 5.7926 |

<img src="docs/assets/val_loss_function_gnn.png" >

GAT achieves the lowest validation loss and converges fastest (early stop at epoch 132). RGCN and GraphSAGE train for the full 200 epochs. GraphSAGE has higher loss values but continues to improve, suggesting it may benefit from more epochs or different hyperparameters.

After training, only sample and protein embeddings are extracted and saved (the node types needed for sepsis prediction). Output: `results/embeddings/{version}/{model}/sample_embeddings.csv` and `protein_embeddings.csv`.

## Sepsis prediction - ML Classifiers

With the embeddings extracted, we can now train classifiers to predict sepsis and compare performance between raw gene expression and graph-based embeddings.

#### Feature Types for ML Classifiers

We use several feature types:

| Type | Description | Dim |
|------|-------------|-----|
| `gene_expression` | Full microarray expression values | 24,840 |
| `*_sample_embeddings` | Sample embeddings | 100 |
| `*_protein_embeddings` | Protein embeddings weighted by expression | 100 or 200 _(for Complex)_ |
| `*_concatenated_protein_embeddings` | Concatenated pretrained protein embeddings from Complex and RGCN | 300 |

###  Preprocessing 

Gene expression and sample embeddings are used without transformation. Gene expression was already normalized at source, and sample embeddings are taken directly from GNN output.

For protein embeddings, several preprocessing steps are applied:

#### Probe-to-Gene Mapping

Probes are mapped to Entrez Gene IDs using GPL6947 annotations. If multiple probes target the same gene, their expression values are averaged.

#### Protein Embeddings Weighting

Protein embeddings are weighted by gene expression. 

Formula: `sample_feature_i = Σ(expression_gene × embedding_protein_i) / Σ(expression_gene)`

The gene expression matrix is normalized before weighting. Available normalization methods: `robust`, `standard`, `minmax`, `log1p`, `none`.


The script `load_matrix.py` loads these features with configurable parameters (version, model, feature type, normalization for protein embeddings). Example:

```bash
python src/ml/load_matrix.py Complex_protein_embeddings --version v2.11 --normalization minmax
```



