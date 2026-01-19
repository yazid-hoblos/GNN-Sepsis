####### Healthcare Context

######## Data Overview

| Number | Study Title                                                       | Series Accession | Series ID | Disease | Samples | Preprocessing Summary                                                                                                                                                                                              |
| ------ | ----------------------------------------------------------------- | ---------------- | --------- | ------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1      | Whole blood transcriptome of survivors and nonsurvivors of sepsis | GSE54514         | 200054514 | Sepsis  | 163     | Raw data scanned, pre-processed in GenomeStudio. Probes filtered by detection p<0.005 in ≥1 sample. Quantile normalized and log-transformed in BRB-ArrayTools. Low-variance genes removed; validated with qRT-PCR. |







## HAN (Heterogeneous Attention Network) Analysis (`src/han/`)

We implemented a comprehensive Heterogeneous Attention Network framework for patient-level graph analysis, enabling interpretable predictions through multiple complementary mechanisms. The `src/han/` directory contains scripts for model training, attention extraction, and gradient-based interpretability.

### Model Architecture and Training (`model.py`, `run_han.py`)

The `SepsisHANClassifier` implements a multi-layer heterogeneous attention network that:
- Processes multiple node types (Patients, Proteins, Pathways, GO Terms, Reactions) simultaneously
- Learns type-specific transformations and cross-type attention weights
- Aggregates multi-relational information through semantic-level attention
- Enables end-to-end training for patient outcome prediction

The `run_han.py` script orchestrates the full training pipeline with proper data loading, hyperparameter tuning, and model checkpointing.

### Attention Mechanism Analysis

#### Node-Level Attention Extraction (`extract_han_attention.py`, `attention_analysis.py`)

These scripts extract and analyze attention weights learned by the HAN model:
- **Edge Type Importance**: Quantifies which relationship types (Patient-Protein, Protein-Pathway, etc.) contribute most to predictions
- **Node-Level Attention Scores**: Identifies which specific entities receive high attention for each patient
- **Attention Distribution Analysis**: Visualizes how attention is distributed across different node types and edge types

The attention extraction reveals which parts of the knowledge graph the model focuses on when making predictions, providing a first layer of interpretability.

#### Semantic-Level Attention Analysis (`han_attention_enhanced.py`, `han_attention_extraction.py`)

Beyond individual edge attention, semantic-level attention aggregates information across entire metapaths (multi-hop paths through different node types). The enhanced attention analysis:
- **Metapath Importance**: Identifies which semantic pathways (e.g., Patient→Protein→GO Term→Pathway) are most predictive
- **Cross-Type Information Flow**: Tracks how information propagates through heterogeneous graph structure
- **Attention Heatmaps**: Generates visualizations showing attention patterns across semantic spaces

![HAN Semantic-Level Attention Analysis](figures/12_han_semantic_attention.png)

### Gradient-Based Interpretability and Subgraph Analysis

#### Patient-Level Gradient Attribution (`patient_gradient_analysis.py`)

The `GradientBasedAttentionAnalyzer` class implements sophisticated gradient-based methods to identify influential nodes:

1. **Gradient-Based Attribution**: Computes gradients of predictions with respect to node features, identifying which input features matter most for each patient
2. **Neighbor Perturbation**: Systematically perturbs different neighbors to measure their impact on predictions, revealing causal relationships
3. **Embedding Similarity Analysis**: Identifies neighbors with similar learned representations, highlighting functionally related entities

This goes beyond simple attention weights by assessing actual influence on model decisions through gradient flow.

#### Heterogeneous Patient Subgraph Visualization (`han_heterogeneous_analysis_initial.py`)

- **Patient-Specific Neighborhoods**: Extracts all heterogeneous neighbors (Proteins, Pathways, GO Terms, Reactions) connected to each patient
- **GO Term Annotation**: Integrates GO ontology definitions and semantic information using `parse_go_obo()`, providing rich contextual information about biological processes
- **Gradient-Based Node Importance**: Ranks neighborhood nodes by their gradient-attributed importance to patient predictions, measuring which neighbors would most impact the prediction if perturbed

![Patient-Specific Subgraph Visualization](figures/14_patient_subgraph_visualization.png)
*Figure 14: Heterogeneous patient subgraph showing gradient-attributed importance of proteins, pathways, and GO terms. Node size indicates gradient-based importance (larger = more causal influence on prediction), color indicates node type (green = protein, blue = GO term, orange = pathway). Edge color is proportional to edge importance. This represents a single patient's personalized mechanistic explanation of their predicted outcome.*

##### Gradient-Based Protein Ranking
For each patient, the script:
- Computes gradient attributions for all connected proteins
- Ranks proteins by their contribution to the prediction
- Identifies key driver proteins whose perturbation would most impact the outcome
- Exports ranked lists for downstream experimental validation

##### GO & Reactome Term Integration
The script enriches protein nodes with functional annotations by:
- Parsing GO ontology files (`go.obo`) for term definitions
- Using Reactome API to retrieve pathway id details
- Mapping proteins to their associated GO terms through the knowledge graph
- Visualizing functional themes (immune response, cell signaling) in patient subgraphs

#### Complex Overlap Analysis (`han_complex_overlap.py`)

This script integrates HAN attention with ComplEx embedding-based biomarkers to identify:
- Proteins that are both highly attended by HAN and important in ComplEx embeddings
- Consensus entities supported by both graph structure (HAN) and semantic similarity (ComplEx)
- Robust biomarker candidates with multiple lines of computational evidence

### HAN Data Loading Pipelines

#### OWL-Based Data Loading (`owl_data_loader.py`, `owl_data_loader_with_features.py`)

Custom data loaders for heterogeneous graphs stored in OWL ontology format:
- **Multi-Type Node Features**: Loads feature matrices for each entity type
- **Heterogeneous Edge Indices**: Constructs edge indices for all relationship types
- **Patient Label Integration**: Merges clinical outcome labels with graph structure
- **Feature Normalization**: Applies appropriate preprocessing for different node types

The `load_hetero_graph_from_owl()` function creates PyTorch Geometric `HeteroData` objects ready for HAN training.