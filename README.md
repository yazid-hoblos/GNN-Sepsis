# 2526-m2geniomhe-GNN-sepsis

# Comparing Traditional and Graph-Enhanced Pipelines for Sepsis Prediction Using Omics Data

## Table of Contents
- [Abstract](#abstract)
- [Approach diagram](#approach-diagram)
- [Reproducibility instructions](#reproducibility-instructions)
- [Datasets](#datasets)
- [Results](#results)


 
## Abstract

description of the problem addressed, the proposed approach, and the main results

## Proposed Subject??

This project compares traditional and graph-enhanced pipelines for sepsis prediction using omics data. The goal is to evaluate whether graph embeddings encoding biological relationships (genes, proteins, pathways) improve classification performance compared to raw gene expression.

We were provided with pretrained KG embeddings (ComplEx, RGCN) and extended the work by training our own GNN models (GraphSAGE, RGCN, GAT) on the heterogeneous graph. We also explored HAN (Heterogeneous Attention Network) for node prediction and interpretability through attention weights.




## Approach diagram

<img src="Visual_Diagram.png" alt="Approach Diagram" >

## Reproducibility Instructions

To reproduce training over different seeds, for any dataset, model, normalization, version, seed you can use the following command structure:

```bash
python -m src.ml.train_all -h
```
```
usage: train_all.py [-h] [--versions VERSIONS [VERSIONS ...]] [--normalization NORMALIZATION] [--logging] [--cache-dir CACHE_DIR] [--threads THREADS]
                    [--model-types MODEL_TYPES [MODEL_TYPES ...]] [--datasets DATASETS [DATASETS ...]] [--split-ratio SPLIT_RATIO] [--random-state RANDOM_STATE]
                    [--hyperparameter-file HYPERPARAMETER_FILE] [--kfold KFOLD]

-- Training combinations of ML models on combinations of datasets for all specified versions --

by default, trains all available models on all available datasets for versions v2.10 and v2.11

AVAILABLE OPTIONS:  
    model-types: svm, random_forest, xgboost, pytorch_mlp, sklearn_mlp
    datasets: gene_expression, RGCN_sample_embeddings, Complex_sample_embeddings, concatenated_sample_embeddings, RGCN_protein_embeddings, Complex_protein_embeddings, concatenated_protein_embeddings, GAT_sample_embeddings, GAT_protein_embeddings, GraphSAGE_sample_embeddings, GraphSAGE_protein_embeddings, weighted_RGCN_sample_embeddings, weighted_RGCN_protein_embeddings
    versions: v2.10, v2.11, v2.9
    normalization: robust, standard, minmax, log1p, none

run this script to train all ML models on all datasets for all specified versions
saves trained models in joblib files in dump/{version}/ folder

> [!TIP]
> run it from the project root as:

    python -m src.ml.train_all -h  
    python -m src.ml.train_all      # -- which is equivalent to:  
    python -m src.ml.train_all --versions v2.10 v2.11 --cache-dir ./dump/ --threads 1 --model-types svm xgboost random_forest sklearn_mlp  

    python -m src.ml.train_all --versions v2.10 --model-types svm
    python -m src.ml.train_all --versions v2.10 --datasets gene_expression RGCN_sample_embeddings --model-types random_forest

options:
  -h, --help            show this help message and exit
  --versions VERSIONS [VERSIONS ...]
  --normalization NORMALIZATION
  --logging             Whether to enable logging to file
  --cache-dir CACHE_DIR
  --threads THREADS
  --model-types MODEL_TYPES [MODEL_TYPES ...]
  --datasets DATASETS [DATASETS ...]
  --split-ratio SPLIT_RATIO
  --random-state RANDOM_STATE
  --hyperparameter-file HYPERPARAMETER_FILE
                        Path to a JSON file containing hyperparameter grids for models (might be hard to deal with, stick to defined hyperparams in
                        src/ml/model_trainer.py MLModel class)
  --kfold KFOLD         Number of folds for k-fold cross-validation
```

You can find example runs in [`src/ml/retrain_helper.sh`](./src/ml/retrain_helper.sh). To run 10 seeds on v2.11, min-max normalization for all datasets and models, you can use:

```bash
for seed in {0..9}; do
    python -m src.ml.train_all --versions v2.11 --normalization minmax --random-state $seed --model-types svm random_forest xgboost sklearn_mlp --datasets gene_expression RGCN_sample_embeddings Complex_sample_embeddings GAT_sample_embeddings GraphSAGE_sample_embeddings weighted_RGCN_sample_embeddings RGCN_protein_embeddings Complex_protein_embeddings GAT_protein_embeddings GraphSAGE_protein_embeddings weighted_RGCN_protein_embeddings
done
```

If you want to try and run individual models and datasets and check results without caching, check the [`HOW TO TRAIN `](./notebooks/ml/HOW_TO_TRAIN.ipynb) notebook.

For metrics evaluation and results of protein and sample embeddings vs gene expression, the notebooks are in [metrics](./notebooks/ml/metrics.ipynb) and [results_sample](./notebooks/ml/results_sample.ipynb), and [results](./notebooks/ml/results.ipynb) respectively.




## Datasets

The dataset comes from GEO (GSE54514), a whole blood transcriptome study of sepsis survivors and non-survivors measured on Illumina HumanHT-12 V4.0 (GPL6947) with 24,840 probes. 

It includes 54 unique patients (18 healthy, 36 sepsis), where sepsis patients were measured at multiple time points, giving 163 total samples (36 healthy, 127 sepsis). The raw data was already log2-transformed and quantile-normalized by the original authors.

For the Knowledge Graph, DEG filtering (adjusted p-value < 0.01, t-test with FDR correction) was applied, keeping ~1,295 proteins.

#### Knowledge Graph Construction

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

#### Graph Versions

Two versions of the KG were generated to handle expression values differently:

- **v2.10 (Binning)**: Expression values are discretized into 3 bins with 0.2 overlapping. This creates categorical relationships between samples and proteins.
- **v2.11 (Average)**: Expression values are kept as continuous weights (averaged across replicates). This preserves the original expression intensity.

### Pretrained KG Embeddings 

The project was provided with pretrained KG embeddings using 2 models :

- **RGCN**: Relational Graph Convolutional Network, learns embeddings by aggregating neighbors with relation-specific weights. Produces 100-dim embeddings.
- **ComplEx**: Complex-valued embeddings that capture asymmetric relations. Stored as complex numbers, split into real+imaginary parts : 200-dim embeddings.

Embeddings for all 9212 entities (samples, proteins, pathways, GO terms) are stored in `models/executions/` as `.npy` files.

### Train our GNN Embeddings

We trained our own GNN models to extract embeddings comparable to the pretrained ones. To do this, we first needed to convert the OWL ontology into a format compatible with PyTorch Geometric.

#### OWL to HeteroData Parsing 

The OWL file is parsed with RDFlib as RDF triplets `(subject, predicate, object)`. For example, `(Sample_GSM1234, hasExpression, Protein_ACTB)` represents a sample expressing a protein. The node type is inferred from the URI prefix: `Sample_` → sample, `Protein_` → protein, `GO:` → goterm, etc.

For sample nodes, the `hasDiseaseStatus` predicate is used to extract labels (0=healthy, 1=sepsis) for node classification. Edge weights are extracted from `hasExpressionValue` annotations in OWL Axioms and averaged across replicates.

The output is a PyTorch Geometric `HeteroData` with random normalized node features (dim=128), edge indices per relation type, and edge weights for hasExpression edges. To run the conversion:

```bash
python src/gnn/load_heterodata.py --version v2.11
```

#### GNN Models

We trained 3 heterogeneous GNN models on this HeteroData graph. All models are adapted for heterogeneous graphs using `HeteroConv`, which applies a separate convolution per edge type and aggregates the results.

**GraphSAGE** (`HeteroGraphSAGE`): Aggregates neighbor features using weighted mean pooling. Messages are weighted by edge weights (expression values) before aggregation. Separate linear transformations are applied to neighbor features and self features.

**Weighted RGCN** (`HeteroRGCN`): Relational GCN with one weight matrix per edge type. Each relation type learns its own transformation. Messages are weighted by edge weights and normalized by weighted degree.

**GAT** (`HeteroGAT`): Uses attention to learn importance of each neighbor. Edge weights are incorporated as edge attributes (edge_dim=1) in the attention computation. First layer uses 4 attention heads (output dim = hidden/4 per head), second layer uses 1 head.

All models use 2 layers, 100 hidden dimensions (to match pretrained embeddings), and dropout 0.5. GraphSAGE and RGCN use ReLU activation, GAT uses ELU (default for attention networks). GAT uses 4 heads so that 100 is divisible (25 dims per head).

#### Training 

Models are trained on a link prediction task: predict whether an edge exists between two nodes. We use BCE loss with negative sampling (ratio 1:1), Adam optimizer (lr=0.001), and early stopping (patience=20, max 200 epochs).

```bash
python src/gnn/extract_embeddings.py --model all --version v2.11
```

**Training Results**

| Model | Epochs | Best Val Loss |
|-------|--------|---------------|
| GAT | 132 (early stop) | 0.8966 |
| RGCN | 200 | 1.0819 |
| GraphSAGE | 200 | 5.7926 |

<img src="docs/assets/val_loss_function_gnn.png" >

GAT achieves the lowest validation loss and converges fastest (early stop at epoch 132). RGCN and GraphSAGE train for the full 200 epochs. GraphSAGE has higher loss values but continues to improve, suggesting it may benefit from more epochs or different hyperparameters.

After training, only sample and protein embeddings are extracted and saved (the node types needed for sepsis prediction). Output: `results/embeddings/{version}/{model}/sample_embeddings.csv` and `protein_embeddings.csv`.

### Sepsis prediction - ML Classifiers

With the embeddings extracted, we can now train classifiers to predict sepsis and compare performance between raw gene expression and graph-based embeddings.

##### Feature Types for ML Classifiers

We use several feature types:

| Type | Description | Dim |
|------|-------------|-----|
| `gene_expression` | Full microarray expression values | 24,840 |
| `*_sample_embeddings` | Sample embeddings | 100 |
| `*_protein_embeddings` | Protein embeddings weighted by expression | 100 or 200 _(for Complex)_ |
| `*_concatenated_protein_embeddings` | Concatenated pretrained protein embeddings from Complex and RGCN | 300 |

####  Preprocessing 

Gene expression and sample embeddings are used without transformation. Gene expression was already normalized at source, and sample embeddings are taken directly from GNN output.

For protein embeddings, several preprocessing steps are applied:

##### Probe-to-Gene Mapping

Probes are mapped to Entrez Gene IDs using GPL6947 annotations. If multiple probes target the same gene, their expression values are averaged.

##### Protein Embeddings Weighting

Protein embeddings are weighted by gene expression. 

Formula: `sample_feature_i = Σ(expression_gene × embedding_protein_i) / Σ(expression_gene)`

The gene expression matrix is normalized before weighting. Available normalization methods: `robust`, `standard`, `minmax`, `log1p`, `none`.


The script `load_matrix.py` loads these features with configurable parameters (version, model, feature type, normalization for protein embeddings). Example:

```bash
python src/ml/load_matrix.py Complex_protein_embeddings --version v2.11 --normalization minmax
```


## Methods

### Training ML Classifiers

As each GNN was trained from embeddings obtained from two different knowledge graph versions (based on binning v2.10, and averaging v2.11). Moreover, as the protein embeddings were multiplied with gene expression values, we considered different normalization techniques to identify the best performing one. We experimented with standard, robust, min-max, log1p normalizations, as well as the option with no normalization applied, resulting in a total of five normalization types.  
Each training also involved a grid search to find the optimal hyperparameters for the specific data space being trained on. To ensure a fair-er comparison across different datasets and models, we trained each model for every (dataset, model type, version, normalization) combination using 10 different random seeds, averaging the results to obtain the final performance metrics, which we found to be necessary in preliminary robustness analyses of our results that showed fluctuations of performance with different starting points (e.g. initialization, data splits, etc.). The results we will be showing correspond to the average over these 10 seeds.
Overall, this led to a comprehensive evaluation framework, allowing us to systematically assess the impact of different graph embeddings, normalization techniques, and model architectures on the prediction of sepsis from gene expression data.

### Evaluation Methodology

To have a performance comparison across the different models and datasets used, the evaluation code consists of three main stages: **collecting predictions**, **computing metrics**, and **visualizing results**.

#### 1. Collecting Results

All model predictions, along with the corresponding true labels, are stored in a structured way using the **`ResultsCollector`** class (in `src/ml/collector.py`). This class keeps track of:  

- `y_test`: the ground-truth labels for the test set  
- `y_pred`: the predicted class labels  
- `y_proba`: the predicted probabilities for the positive class  

Data is stored separately for each combination of model and input type, which allows us to easily compare models and see how the different inputs affect performance.

#### 2. Computing Metrics

Once the predictions are collected, the **`Evaluator`** class (in `src/ml/evaluator.py`) calculates a set of standard metrics for binary classification. These metrics capture different aspects of model performance, from general accuracy to how well the model predicts probabilities. The metrics computed include:

- **Balanced Accuracy**: the average recall across the positive and negative classes. It is especially useful for imbalanced datasets, because it treats both classes equally:  
  \[
  \text{Balanced Accuracy} = \frac{\text{Recall}_{\text{positive}} + \text{Recall}_{\text{negative}}}{2}
  \]

- **Precision**: the proportion of predicted positives that are actually correct:  
  \[
  \text{Precision} = \frac{TP}{TP + FP}
  \]  
  where **TP** = true positives and **FP** = false positives.

- **Recall (Sensitivity)**: the proportion of actual positives that the model correctly identifies:  
  \[
  \text{Recall} = \frac{TP}{TP + FN}
  \]  
  where **FN** = false negatives.

- **F1 Score**: the harmonic mean of precision and recall. It provides a single score that balances the two metrics:  
  \[
  F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
  \]

- **Matthews Correlation Coefficient (MCC)**: a correlation coefficient between predicted and true labels. It is robust to class imbalance and provides a single measure of overall prediction quality (Chicco & Jurman, 2020):  
  \[
  MCC = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
  \]  
  where **TN** = true negatives.

- **AUROC (Area Under the ROC Curve)**: measures how well the model separates positive and negative classes across all classification thresholds. Values closer to 1 indicate better discrimination.

- **AUPRC (Area Under the Precision-Recall Curve)**: summarizes the trade-off between precision and recall across thresholds. Particularly useful for imbalanced datasets where the positive class is rare (Saito & Rehmsmeier, 2015).

- **Brier Score**: measures the mean squared error of predicted probabilities, evaluating both accuracy and calibration of the model (Rufibach, 2010):  
  \[
  \text{Brier Score} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{p}_i)^2
  \]  
  where \(y_i\) is the true label (0 or 1) and \(\hat{p}_i\) is the predicted probability for the positive class.

> **Note:** For all metrics, higher values indicate better performance, except for the Brier Score, where lower values are better.

#### 3. Visualizing Performance

To make evaluation results easier to understand, the **`DashboardVisualizer`** class (in `src/ml/visualizer.py`) provides several plotting options. The visualizations can show:  

- **Metric grids**: bar plots for each metric across models and inputs, useful for side-by-side comparisons.  
- **Radar plots**: combine multiple metrics in a single plot for each model-input combination.  
- **ROC and Precision-Recall curves**: help inspect model performance for different thresholds.  

### SHAP Explanation

Shapley Additive exPlanations (SHAP) is a method to explain predictions connecting optimal credit allocation with local explanations using the classic Shapley values from cooperative game theory and their related extensions.  
The idea of this interpretation is mainly addressing the protein embeddings data in the following context:

Protein embeddings used for training are built using a "linear" approach, if can be seen as a LF or matrix multiplication in a sense:
Let $G$ be the `samples x genes` gene expression matrix (genes and proteins here are used interchangeably), and $E_protein$ be the `proteins x embedding_dimension` protein embeddings matrix, then the protein embeddings for samples $E_p$ is computed as:
$$E_p = \frac{G  E_{protein}}{\sum_i G_i}$$

Where $E_p$ is the `samples x proteins` protein embeddings matrix, input of the ML models (its the matrix multiplication + row normalization)

To make the training process more interpretable we aim to use SHAP values to link the importance of each protein embedding dimension back to the genes/proteins space, so a plan of 2 steps: find the most important embedding dimensions using SHAP, then link them back to genes/proteins by computing "loadings" (similar concept to pca loading and MOFA) that are the contributing the most to these embedding dimensions (they behave like highly weighted features in a linear model) - this idea is captured in the diagram.


### Knowledge Graph Optimization (`OntoKGCreation/`)

We started considering optimizing the knowledge graph to balance comprehensiveness with computational efficiency and biological relevance. We modified the scripts provided in `OntoKGCreation/` for KG refinement.

#### Motivation for KG Optimization

Our initial knowledge graph, constructed from multiple biomedical ontologies and databases, contained:
- Thousands of entities (proteins, pathways, GO terms, reactions) - we particularly notice that > 60% of nodes are pathways/reactions with many embedded connections between them 
- Hundreds of thousands of unfiltered relationships with varying confidence levels (proteins are very densely connected since all PPI data is used without filtering)
- Redundant and low-information connections that add noise

This scale posed challenges for:
- **Computational Efficiency**: GNN training time and memory requirements
- **GNN Performance**: Risk of confusing models with irrelevant or noisy information
- **Interpretability**: Overwhelming number of potential features to analyze
- **Biological Focus**: Dilution of signal by weakly relevant entities

#### Filtering and Optimization Strategies

##### Ontology-Based Refinement
Leveraging the ontological structure:
- **Term Specificity**: Filtering overly general GO terms (e.g., "biological process") in favor of specific terms
- **Hierarchy Pruning**: Removing redundant parent-child chains where leaf terms suffice

Through iterative refinement, we aim to achieved:
- **Size Reduction**: significant reduction in node count/edge density while preserving (or improving if the GNNs could learn better) performance and interpretability signals
- **Performance Improvement**: Faster training (reduced embedding dimensions) with comparable or improved accuracy
- **Enhanced Interpretability**: Cleaner, more focused feature importance rankings
- **Biological Coherence**: Increased functional enrichment significance for top features

OntoKGCreation/converted/optimized/ -> retrain embeddings -> ML approaches -> comparison to original KG results.


## Results

### Exploratory Data Analysis (EDA)

#### Phenotypic Variables Analysis 

- **Age Distribution par Groupe** -  Kruskal-Wallis p < 0.001 ***

- **Gender Distribution par Groupe** - Chi² test p = 0.7053 ns 

- **APACHE II Severity Score (Sepsis patients seulement)** - Mann-Whitney p < 0.001 ***

<img src="docs/assets/01_phenotypic_analysis.png" alt="Knowledge Graph Schema" width="600"/>


#### Genes - Probes mapping 

<img src="docs/assets/02_probes_per_gene.png" alt="Knowledge Graph Schema" width="600"/>

Microarrays technology can use multiple probes to measure the expression of a single gene.
Must work on gene level (not probe level) → aggregation step needed.


#### Gene expression profiles 

<img src="docs/assets/05_pca_analysis.png" alt="Knowledge Graph Schema" width="600"/>

*Based on gene expression matrix (we mapped probes to genes by aggregating multiple probes per gene with mean weighted value)*

**No clear clustering** in the first 2 principal components between our 3 sub-groups, the first components don't explain much variance (C1 : ~12%, C1 + C2 : ~22%)

<img src="docs/assets/06_clustering_heatmap.png" alt="Knowledge Graph Schema" width="600"/>

#### Timeline Analysis


#### Knowledge Graph Visualization

We developed a comprehensive suite of visualization tools to enable intuitive exploration complex multi-layer knowledge graphs. The `src/visualize/` directory contains scripts for static plotting, interactive visualization, and web-based network exploration.

##### Interactive Multi-Layer Network Web Application (`multilayer_network_app.py`)

We developed a Flask-based web application that provides dynamic, real-time exploration of the knowledge graph:

###### Core Features

####### Multi-Layer Graph Management (`MultiLayerNetworkManager` class)
- **Flexible Data Loading**: Loads nodes, edges, and entity classes from KG conversion and model execution outputs
- **Layer Definitions**: Automatically classifies nodes into semantic layers (Patients, Proteins, Pathways, GO Terms, Diseases, etc.)
- **Edge Type Tracking**: Catalogs all relationship types in the graph for selective filtering
- **Patient Embedding Integration**: Links graph structure with learned model embeddings

####### Real-Time Filtering and Exploration
The web interface enables users to:
- **Layer Selection**: Toggle individual node type layers on/off dynamically
- **Edge Type Filtering**: Show/hide specific relationship types (physical, genetic, regulatory)
- **Statistics Dashboard**: Display real-time graph statistics (node counts, edge counts...)
- **Force-Directed Layout**: Physics-based layouts for intuitive spatial organization

![Interactive Web Application Interface](figures/visualization_app.png)
*Figure: Screenshot of the interactive multi-layer network visualization web application, showing real-time filtering controls, layer management, and dynamic network rendering.*

This is relevant both for initial exploratory analysis and examination of specific neighborhood to help guide analysis.

##### Static Graph Visualization (`visualize_multilayer_graph.py`)

For batch generation of network plots and non-interactive figures, `visualize_multilayer_graph.py` provides:

###### PyVis Interactive HTML Exports
- **Standalone HTML Files**: Self-contained interactive visualizations
- **Custom Physics**: Configurable force-directed algorithms (Barnes-Hut, repulsion strength)
- **Legend Integration**: Automatic generation of interactive legends

###### NetworkX-Based Static Plots
- **Multiple Layout Algorithms**: Spring, Kamada-Kawai, hierarchical, circular
- **High-Resolution Output**: PNG at 300+ DPI for publication quality
- **Vector Graphics**: SVG/PDF formats for scalable figures
- **Matplotlib Integration**: Full access to matplotlib styling and customization

##### Gephi Export Pipeline (`gephi_exports/`)

###### Graph Export Formats
- **GEXF Files**: Native Gephi format with full metadata preservation
- **GraphML**: Alternative format supporting complex attribute schemas
- **CSV Edge/Node Lists**: Simple tabular format for custom processing


#### Knowledge Graph Analysis 

As an exploratory data analysis step, we investigated the biological significance of the knowledge graph (KG) in the context of sepsis by identifying upregulated and downregulated genes using the attributes `hasTypeRegulation` and `hasLog2_FC`, and examining the associated pathways and Gene Ontology (GO) terms. The analysis code is available in `src/eda/KG_analysis/`.

Knowing that the knowledge graph construction was based on differentially expressed genes (DEGs) identified from the gene expression data, which were subsequently mapped to their corresponding proteins, we explored the graph and observed that all proteins are associated with statistically significant p-values (< 0.05) through the `hasPValue` attribute. However, among the 1,288 proteins in the graph, only 7 proteins are labeled as *Upregulated* and 3 as *Downregulated*, with the remaining 1,278 proteins marked as *No change*.

To understand this pattern, we examined the log2FC (fold‑change) magnitude statistics:

| Regulation      | Min     | Median  | Mean    | Max     |
|-----------------|---------|---------|---------|---------|
| Downregulated   | 0.7563  | 0.7581  | 0.7629  | 0.7742  |
| No change       | 0.0365  | 0.2067  | 0.2307  | 0.7431  |
| Upregulated     | 0.7961  | 0.8929  | 0.8873  | 0.9942  |

There is a clear boundary at a log2FC magnitude of approximately 0.75. Even though the *No change* proteins are statistically significant, their fold changes were too small to be considered biologically relevant for this regulation label. Upregulated proteins show stronger overall changes, with higher mean (0.8873) and maximum (0.9942) values, while downregulated proteins cluster closer to the cutoff.

A volcano plot illustrating log2FC versus −log10(p‑value) further highlights the separation between upregulated, downregulated, and non‑regulated proteins, with the names of regulated proteins explicitly labeled:

<img src="results/EDA/KG_analysis/volcano_plot_v211.png" width="700"/>


##### Biological Enrichment Analysis

To explore the functional roles of the differentially regulated proteins, we performed a statistical enrichment analysis for both Gene Ontology (GO) terms and Reactome pathways. For each set of proteins (upregulated and downregulated), we used the knowledge graph to link proteins to their associated GO terms and pathways, and tested for enrichment against all proteins in the KG using Fisher’s Exact Test, followed by Benjamini-Hochberg (FDR) correction.

For visualization, we selected the top 10 terms from each set. If a term was top 10 in one set but also significant in the other set (according to the respective FDR threshold), it is plotted for both sets to allow a direct comparison. The only difference between GO and pathway analysis is the FDR threshold used to filter significant terms: 0.5 for GO to highlight trends in the small downregulated set (3 proteins), and 0.1 for pathways. 

<div style="display: flex; justify-content: space-around; align-items: flex-start;">

  <div style="flex: 1; text-align: center;">
    <img src="results/EDA/KG_analysis/go_enrichment_plot.png" style="height:380px;"/>
  </div>

  <div style="flex: 1; text-align: center;">
    <img src="results/EDA/KG_analysis/pathway_enrichment_plot.png" style="height:380px;"/>
  </div>

</div>

By looking at both the Gene Ontology and Pathway enrichment results, it’s clear that during sepsis the body focuses on immediate immune defense while reducing the activity in non-essential cellular functions.

- **Strong Immune Activation:**  
The upregulated proteins are linked to immune defense and innate immune pathways. This shows that the body ramps up its early immune response, quickly activating antimicrobial and inflammatory mechanisms to fight infection (Brandes-Leibovitz et al., 2024). The enriched pathways, including interferon signaling, also highlight this heightened immune activity (Brandes-Leibovitz et al., 2024).

- **Slower Cellular Maintenance:**  
Downregulated proteins are connected to processes like organelle organization, cytoskeleton, and other core cell functions. This suggests that during the acute phase of sepsis, cells temporarily reduce routine maintenance and structural activities to focus energy on immune defense. Similar patterns have been reported in sepsis studies, where metabolic and biosynthetic pathways are reprogrammed to prioritize immune responses (Liu et al., 2023; Willmann & Moita, 2024).


#### Graph Embeddings Visualization

To visualize the learned graph **protein** embeddings, we employed dimensionality reduction techniques such as Principal Component Analysis (PCA) and Uniform Manifold Approximation and Projection (UMAP) and t-SNE. These methods help in projecting high-dimensional embeddings into a 2D space, allowing us to observe potential clustering patterns among septic and non-septic patients. These were tested for all GNN models, on different versions (v2.10, v2.11) and normalizations (none, min-max, standard, robust, log1p). As our analysis will be later based on v2.11 min-max, these visualizations correspond to this version and norm. (see [`results/figures/projections/`](./results/figures/projections/) for other versions and normalizations).

| Dataset| PCA | UMAP | t-SNE |
|-----|------|-------|-------|
| ComplEX|  ![pca](./results/figures/projections/pca_v2.11_minmax/complex_protein_embeddings.png) | ![umap](./results/figures/projections/umap_v2.11_minmax/complex_protein_embeddings.png) | ![tsne](./results/figures/projections/tsne_v2.11_minmax/complex_protein_embeddings.png) |
| RGCN | ![pca](./results/figures/projections/pca_v2.11_minmax/rgcn_protein_embeddings.png) | ![umap](./results/figures/projections/umap_v2.11_minmax/rgcn_protein_embeddings.png) | ![tsne](./results/figures/projections/tsne_v2.11_minmax/rgcn_protein_embeddings.png) |
| weighted RGCN | ![pca](./results/figures/projections/pca_v2.11_minmax/weighted_rgcn_protein_embeddings.png) | ![umap](./results/figures/projections/umap_v2.11_minmax/weighted_rgcn_protein_embeddings.png) | ![tsne](./results/figures/projections/tsne_v2.11_minmax/weighted_rgcn_protein_embeddings.png) |
| GraphSAGE | ![pca](./results/figures/projections/pca_v2.11_minmax/graphsage_protein_embeddings.png) | ![umap](./results/figures/projections/umap_v2.11_minmax/graphsage_protein_embeddings.png) | ![tsne](./results/figures/projections/tsne_v2.11_minmax/graphsage_protein_embeddings.png) |
| GAT | ![pca](./results/figures/projections/pca_v2.11_minmax/gat_protein_embeddings.png) | ![umap](./results/figures/projections/umap_v2.11_minmax/gat_protein_embeddings.png) | ![tsne](./results/figures/projections/tsne_v2.11_minmax/gat_protein_embeddings.png) |
| Gene Expression (reference) | ![pca](./results/figures/projections/pca_v2.10_none/gene_expression.png) | ![umap](./results/figures/projections/umap_v2.10_none/gene_expression.png) | ![tsne](./results/figures/projections/tsne_v2.10_none/gene_expression.png) |

As can be seen that gene expression data have no structure or seperation between septic and non-septic patients in all projections, some GNN protein embeddings have show a clear seperation, particualrly if we look at ComplEx and GAT, hence already showing potential in the classification task.  
PCA is a linear dimensionality reduction technique, which may not capture complex relationships in the data, while UMAP and t-SNE are non-linear techniques - that's why we can have a better view on class distribution in these two methods. Weighted RGCN for instance doesn't have a high variance explained in PCA, yet in UMAP and t-SNE we can see colors aggregate, which might reflect in this dataset's performance in some models compared to others.


### Evaluation Results

#### Versions and Normalizations

| different normalizations  | different versions |
|--------------------------|--------------------|
| ![normalizations](./results/figures/comparative_normalization_version/avg_seeds_normalizations/boxplot_comparative_normalization_version_balanced_accuracy.png) | ![versions](./results/figures/comparative_normalization_version/avg_seeds_versions/boxplot_comparative_normalization_version_balanced_accuracy.png) |

Between different normalizations, MinMax show the best performance overall, where median balanced accuracy of v2.10 is highest, and spread is the lowest in v2.11. As for versions, v2.11 tends to have better performance than v2.10, while still pretty much model/dataset dependent. Thus for the rest of the analysis, we will be focusing on MinMax normalization and version v2.11.

#### Overall Performance Comparison

The traditional machine learning pipeline generally consists of training models directly on gene expression data (~20k genes).
The graph augmented pipeline can be split into **sample embeddings** and **protein embeddings** datasets for training, these were 100-200 features datasets.  
For a better representation of the results due to high number of models and datasets, we will be comparing **sample embeddings vs gene expression** and **protein embeddings vs gene expression** separately, to investigate the performance gain brought by graph augmented data in both cases.

##### Sample Embeddings vs Gene Expression

These embeddings are obtained directly from the GNN models trained on the knowledge graph, representing each patient/sample as a vector in the embedding space. We compared the performance of models trained on these sample embeddings against those trained on raw gene expression data:


![_heatmap of sample embeddings vs gene expression (all models and datasets)_](./results/figures/metrics/sample_heatmap.png)

We can notice while accuracies are generally low, we have a precision constantly higher than 70% and at times a recall of 100%. This in fact is due to the models being very conservative, predicting most samples as non-septic, which is also reflected in the low balanced accuracies. Because of the samll sample size (163 with imbalance => splitted to train and test would get 30% only for testing, startified splitting with ratio ~3:7), leading to variance in results, where a small number of samples being misclassified can significantly impact metrics like recall and precision.  The highest accuracy models actually belong to gene expression dataset, reaching up to 93% accuracy with SVM, meaning that sample embeddings were not able to enhance the performance in this case.

![_grouped barplots of sample embeddings vs gene expression for different models (rf, xgb, svm, mlp) and metrics (balanced accuracy, precision, recall, f1...)_](./results/figures/metrics/sample_grouped_barplots_horizontal.png)

Overall, sample embeddings did not produce an good performance, accross the different models. In fact, it's worth noting that the only dataset with a slightly good performance (accuracy of 0.675, 0.64, 0.692 for random forest, xgboost and svm respectively) is the one obtained from **RGCN** model, which is by far worse than the gene expression based model ranging between 70.2% for random forest up to 93% for SVM. This suggests that sample embeddings alone may not capture sufficient information for accurate sepsis prediction, as embeddings retrieved from training the gene expression dataset on the knowledge graph (possibly due to the complexity of the underlying biological processes that are not fully represented in the embeddings).

| XGB vs RF | SVM vs MLP |
|-----------|-------------|
| ![_scatterplot of rf vs xgboost_](./results/figures/metrics/sample_rf_vs_xgb.png) | ![_heatmap of sample embeddings vs gene expression (all models and datasets)_](./results/figures/metrics/sample_svm_vs_mlp.png) |

The only case where GNN performs better than gene expression is with MLP model on GraphSAGE sample embeddings with 0.59 balanced accuracy, still not a good performance overall.

![_mlp radar plort of sample embeddings vs gene expression_](./results/figures/metrics/sample_radar_mlp.png)


##### Protein Embeddings vs Gene Expression

The protein section yielded more promising results. As this approach is derived from a matrix multiplication weighting on gene expression values, it effectively integrates both the graph structure and the gene expression data beyound just the GNN training, potentially capturing more relevant features for sepsis prediction.

![_grouped barplots of protein embeddings vs gene expression for different models (rf, xgb, svm, mlp) and metrics (balanced accuracy, precision, recall, f1...)_](./results/figures/metrics/protein_grouped_barplots_horizontal.png)

![_heatmap of protein embeddings vs gene expression (all models and datasets)_](./results/figures/metrics/protein_heatmap.png)

At a first glance, we can see that protein embeddings outperform gene expression in most models and metrics, with significant improvements in balanced accuracy, precision, recall, and F1-score. The most notable performance is observed with the MLP model where gene expression based MLP achieves only 50% average balanced accuracy (over 10 seeds), while protein embeddings based MLP reaches up to 91.35% balanced accuracy (a huge improvement of more than 40 percentage points). We remark higher performance with tree based models (random forest and xgboost), while SVM for gene expression was already performing very well (93% balanced accuracy), thus not much room for improvement there, yet protein embeddings were able to reach it and a 95.3% balanced accuracy with SVM (ComplEx).



![_robustness analysis plots (maybe put in appendix)_](./results/figures/metrics/protein_robustness_boxplot.png)

Some models portray a high variance in performance across different seeds, especially the MLP models, which can be attributed to their sensitivity to weight initialization and hyperparameter settings. This variance seem to be autonomous of the dataset used, whether it's gene expression or protein embeddings, indicating that the model architecture itself may be a significant factor in performance stability. Another interesting observation is the low variance of ComplEx protein embeddings based models, which could be linked to the nature of the embeddings produced by ComplEx, potentially being more robust or informative for the sepsis prediction task, which always shows high performance accross different models and metrics (lowest variance in SVM with >90% balanced accuracy across all seeds).  
The most unstable behaviior is seen in weighted RGCN protein embeddings MLP models, ranging between 50-90% balanced accuracy (not conclusive), while others seem to have a better well defined range (at max is range of 30% for GraphSAGE in random forest).  
Side note, `concatenated pretein embeddings` is a concatenation of ComplEx and RGCN (choice was defined on what was available at earlier stages of the project, results kept to explore), which shows a good performance, but not better than ComplEx, which is expected since RGCN is one of the lowest performing datasets. So based on this, we will not be considering this dataset further in the analysis.


| ![_radar plot for random_forest_](./results/figures/metrics/protein_radar_random_forest.png) | ![_radar plot for xgboost_](./results/figures/metrics/protein_radar_xgboost.png) |
|-----------------------------------------------|-------------------------------------|
| ![_radar plot for svm_](./results/figures/metrics/protein_radar_svm.png) | ![_radar plot for mlp_](./results/figures/metrics/protein_radar_sklearn_mlp.png) |

The radar plot shows gene expression in the outermost layer for XGBoost and SVM (overlapped with some other protein embeddings), indicating its already strong capability in sepsis prediction for these models. In Random forests, while still of good performance, it seem to be outperforemd by almost all GNN based protein embeddings. The most significant improvement is super highlighted in MLP models, where gene expression lags far behind all protein embeddings (except for RGCN in purple), showing the potential of graph augmented data in enhancing the performance of neural networks for this task. Weighted RGCN protein embeddings performance in SVM is particulalry flashy, with a balanced accuracy of 50% in average - way less than the unweighted RGCN approach's embeddings - and also a bit lower than the other performing GNNs in  MLP (however, as concluded before not conclusive), while it has a high standing in tree based models.


| RF vs XGB | SVM vs MLP |
|-----------|-------------|
|![_scatter plot for tree based model and svm/mlp (maybe put in table)_](./results/figures/metrics/protein_rf_vs_xgb.png) | ![_scatter plot for svm vs mlp_](./results/figures/metrics/protein_svm_vs_mlp.png) |

These plots were grouped XGBoost/Random forest and SVM/MLP to better visualize the performance differences between similar model types (in terms of learning approach and previous performance evaluation).
It's worth noting the lowest performance is actually not for gene expression based models but rather for RGCN protein embeddings based models - particularly those with no edge weight used during training and weighted for SVM/MLP.  
The dashed diagonal line represents equal performance between the two models being compared. Points above the line indicate better performance by the model on the y-axis, while points below the line indicate better performance by the model on the x-axis, we generally aim to define the "best" dataset by the one in upper left quadrant and nearest to this diagonal line.

Starting with tree based models, all datasets have highesr accuracies with XGBoost compared to Random Forest, with no excpetion. GraphSAGE, Complex and GAT show lowest xgboost-rf differences, while RGXN, weighted RGCN and gene expression have the highest difference. GAT protein embeddings based models show the best performance overall in this category.  
In SVM vs MLP, we can see a more mixed behavior, with some 3 main clusters: those performing very well in MLP and poorly in SCM (top left quadrat: GAT, GraphSAGE, weighted RGCN), those performing very well in SVM and poorly in MLP (bottom right quadrant: gene expression, RGCN) and those performing well in both models (top right quadrant: cluster of one - ComplEx). ComplEx protein embeddings based models show the best performance overall in this category.


![_rank distribution plot_](./results/figures/metrics/protein_rank_distribution.png)

This final plot summarizes the rankings accross models/datasets based on balanced accuracy, where each dataset is ranked by order. XGBoost show good performance curve for all datasets, toppest being weighted RGCN/GAT, gene expression being 3rd. Similarly for random forest where GAT/Complex/weighted RGCN are on top.  
SVM and MLP show a sudden drop, as expected from previous scatterplot analysis, where in here the top protein embedding is ComplEx, gene expression 2nd in SVM.


### Interpretability 















## Discussion 

### Performance 

As the analysis was performed on v2.11 with Minmax normalization with most robust and best performance overall, we will discuss some findings regarding traditional vs graph augmented ML pipelines for sepsis prediction from gene expression data.

Gene expression data perform greatly using SVM models. This might be due to the high dimensionality of the data (20k genes) where SVMs are known to handle high-dimensional spaces well, especially with appropriate kernel functions, it might be aslo overfitting to some extent as dataset is small. ComplEx ourperformed this high achieving model, with and svm linear kernel (from grid). Previosuly we noticed ComplEx protein embeddigns being the best dataset showing clear seperation in different DR approaches, particularly siting PCA as the only dataset with a decent variance explained in first 2 components - a linear method - which might explain why SVM with linear kernel works well here.
Other models like random forest and xgboost also show good performance with gene expression data, indicating that tree based models can effectively capture relevant patterns in the data without the need for graph augmentation. MLP models however perform poorly, likely due to the small sample size and high dimensionality leading to overfitting (constant bad performance around 50% balanced accuracym from robustness analysis). And this is a known issue with neural networks in general when working with tabular data, especially with limited samples, thus finding ways to enhance it using graph embeddings would be particularly useful.

Each model type has its own set of performing datasets, while in all of them there is a seen improvement using graph augmented data (protein embeddings), justifying the benefit of integrating knowledge graphs in the analysis AND the gene expression weighting transformation (since sample embeddings that were used without any transformation did not perform well). Overall, even though some GNNs embeddings like GAT have slight higher performance in some models, ComplEx protein embeddings based models show the most consistent high performance accross different models and metrics, making it the best performing dataset overall for sepsis prediction in this study. This conclusion is not just based on its top rankings across models, but also from robustness analysis showing lowest spread in performance across seeds and the grounbraking performance in MLP models as rank 1 with 40%+ balanced accuracy improvement over gene expression based MLP, SVM's rank 1 and direct competitor to gene expression, and strong showings in tree based models as well.

An interesting aspect we noticed regarding embeddings, ComplEx is the only model that has 200 embeddings dimensions, while all the other are 100 only. Due to its architecture, ComplEx (as name suggests) is actually operating in complex number space, meaning each dimension can be seen as 2 dimensions (real and imaginary parts). Thus effectively, to have a fairer comparison, would conisdering raising the other GNNs embedding dimensions to 200 as well, or lower this one to 50 (which will be multiplied by 2) to test if this is the reason behind its good performance. However, due to time constraints we could not perform this analysis - a lot of parameters and hyperparameters are in fact worth exploring as we have a lot of variables in this study (different GNNs, different ML models, different normalizations, different versions, different seeds, different parameters).

### Pytorch 

Concerning Multiple Layer Perceptron, we fixed at the end to studying them using scikit learn's implementation on 500 epochs. Our machine learning pipeline relied on `MLModel` class definition that can take different skleran model types, datasets, normalizations, versions, seeds, parameters and hyperparameters for grid search, We also implemented a cutom PytorchMLP model inheriting ClassiferMixin and BaseEstimator from sklearn to be able to use it in the same pipeline and even trained on a large set of models (not all though), however as there were more available options of optimizer, activation functions etc. from sklearn, we started exploring them more and createed a large grid search of hyperparameters, leading to long training times and instabilities (high variance of results) for only 50 epochs (also, early stopping was a hastle to implement with cross validation, ensuring caching and returning the best model across the last  - which have not been successful at the end). We diverted to sklearn's MLP implementation for a more stable and faster training. More complex architectures and hyperparameter tuning can be explored in future work to fully leverage the potential of neural networks in this context, especially with the promising results seen with Complex protein embeddings, and the failure of gene expression based MLP models. It's sensitive to work with as dataset is small, split and imbalanced.


### Limitations and Challenges

#### Current Limitations
1. **KG Completeness**: Despite optimization efforts, the knowledge graph may still miss important but poorly annotated entities
2. **Interaction Confidence**: Variable quality of edge annotations requires cautious interpretation of network results
3. **Sample Size**: Limited patient cohort size constrains statistical power for rare outcome prediction
4. **Temporal Dynamics**: Current models do not capture time-dependent disease progression

#### Technical Challenges
- Balancing KG size with computational feasibility and interpretability
- Handling heterogeneous data types and scales across node features
- Ensuring gradient stability in complex heterogeneous graph architectures
- Managing visualization complexity for large-scale networks

### Future Research Directions

#### 1. Enhanced Knowledge Graph Optimization

**Reduce/Optimize KG**: Continued refinement of the knowledge graph through:
- **Active Learning Approaches**: Using model uncertainty to guide selective KG expansion in informative regions
- **Expert-in-the-Loop Curation**: Incorporating domain expert feedback to validate and refine entity/relationship selections
- **Multi-Task Optimization**: Creating task-specific KG variants optimized for different prediction goals (severity, outcome, treatment response, trajectory)
- **Temporal Integration**: Incorporating time-dependent relationships and dynamic processes
- **Confidence Modeling**: Learning edge weights and uncertainty estimates rather than binary inclusion/exclusion

#### 2. Interpretability-Visualization Integration

**Connect Interpretability Findings to KG Visualization**: Tighter integration of model explanations with network exploration:
- **SHAP-Driven Interactive Highlighting**: Real-time updates of node/edge importance in the web application based on SHAP scores
- **Attention Heatmap Overlay**: Visualizing HAN attention distributions directly on the interactive graph
- **Gradient Flow Visualization**: Animating gradient propagation through patient subgraphs to show causal pathways
- **Perturbation Explorer**: Interactive tools to simulate node/edge removal and observe prediction changes
- **Explanation Provenance**: Linking each biomarker back to the specific analyses and models that identified it

#### 3. Advanced Visualization Enhancements

**Visualization Enhancement**: Ongoing improvements to support deeper exploration:
- **Dynamic Filtering Interface**: More intuitive controls for multi-dimensional filtering (importance, confidence, node type, functional category)
- **Annotation Layer System**: User-added notes, hypotheses, and literature references directly on network views
- **Comparative Visualization**: Side-by-side comparison of patient subgraphs or model predictions
- **3D Network Rendering**: Spatial layouts leveraging additional dimensions for complex hierarchies
- **Automated Layout Optimization**: Machine learning-based layout algorithms that optimize for biological interpretability
- **Export and Reproducibility**: One-click export of full analysis provenance (data, filters, settings) for reproducibility
- **Biomarker Highlighting**: Emphasize identified biomarker nodes with custom styling
- **Search Functionality**: Find specific entities by ID or name

#### 4. Model Architecture Advances

- **Temporal Graph Networks**: Incorporating time-series patient data and dynamic KG relationships
- **Explainable-by-Design Architectures**: Models with built-in interpretability mechanisms rather than post-hoc analysis
- **Causal Graph Discovery**: Moving beyond correlation to identify causal relationships in biological networks
- **Multi-Modal Integration**: Combining KG-based models with imaging, clinical notes, and other data modalities

#### 5. Clinical Translation

- **Prospective Validation**: Testing identified biomarkers in independent patient cohorts
- **Experimental Validation**: Wet-lab experiments to validate predicted protein interactions and mechanisms
- **Clinical Decision Support**: Adapting models and visualizations for real-time clinical use
- **Treatment Response Prediction**: Extending models to predict which patients will respond to specific therapies

#### 6. Scalability and Efficiency

- **Distributed Training**: Scaling to larger KGs and datasets through distributed GNN training
- **Incremental Learning**: Updating models with new data without full retraining
- **Real-Time Inference**: Optimizing model deployment for low-latency predictions
- **Federated Learning**: Training across multiple institutions while preserving data privacy

### Conclusion

### Methodological Contributions

1. **Multi-Model Interpretability Framework**: Successfully implemented SHAP, attention, and gradient-based analyses across diverse architectures (Random Forests, SVMs, ComplEx, RGCN, HAN), providing multiple complementary perspectives on model decisions.

2. **Consensus Biomarker Discovery**: Developed robust pipelines to aggregate evidence across models, increasing confidence in identified biomarkers and reducing model-specific biases.

3. **Heterogeneous Graph Analysis**: Advanced HAN architecture with gradient-based patient-level interpretability, enabling personalized explanations of predictions.

4. **Advanced Visualization Suite**: Created publication-ready static plots and interactive web applications for exploring complex multi-layer knowledge graphs.

### Technical Infrastructure
- Established reproducible pipelines for model training, interpretability analysis, and visualization
- Developed modular, reusable code for knowledge graph construction, optimization, and analysis
- Created interactive tools for collaborative exploration and hypothesis generation




## References
- Chicco, D., & Jurman, G. (2020). The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation. BMC genomics, 21(1), 6.
- Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PloS one, 10(3), e0118432.
- Rufibach, K. (2010). Use of Brier score to assess binary predictions. Journal of clinical epidemiology, 63(8), 938-939.
- Brandes-Leibovitz, R., Riza, A., Yankovitz, G., Pirvu, A., Dorobantu, S., Dragos, A., ... & Netea, M. G. (2024). Sepsis pathogenesis and outcome are shaped by the balance between the transcriptional states of systemic inflammation and antimicrobial response. Cell Reports Medicine, 5(11).
- Liu, W., Liu, T., Zheng, Y., & Xia, Z. (2023). Metabolic reprogramming and its regulatory mechanism in sepsis-mediated inflammation. Journal of inflammation research, 1195-1207.
- Willmann, K., & Moita, L. F. (2024). Physiologic disruption and metabolic reprogramming in infection and sepsis. Cell metabolism, 36(5), 927-946.
- [1] Jiang Y, Miao Q, Hu L, Zhou T, Hu Y, Tian Y. FYN and CD247: Key Genes for Septic Shock Based on Bioinformatics and Meta-Analysis. Comb Chem High Throughput Screen. 2022;25(10):1722-1730. doi: 10.2174/1386207324666210816123508. PMID: 34397323.