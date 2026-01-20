# Interpretability

## Overview: Model Interpretation Pipeline (`src/interpretability/`)

We developed a comprehensive interpretability framework to bridge our GNN-based ML predictions with biological interpretability. The `src/interpretability/` directory contains multiple scripts that systematically extract, analyze, and visualize feature importance across different model architectures. Key achievements include:

1. **Multi-Model SHAP Analysis**: we employed SHAP (SHapley Additive exPlanations) analysis across all major architectures. SHAP values provide a unified framework to quantify the contribution of each feature to model outputs. This allows direct comparison of feature importance across models, revealing both consensus and model-specific biomarkers. The `run_interpretability.py` script (based on `interpretability.py`) computes SHAP values across Random Forest, SVM, XGBoost, and MLP, providing a unified framework to quantify feature contributions to predictions.

As highlighted in Figure 1 for ComplEx with SVM (highest performing model), we computed the aggregate SHAP values distributions, identifying the most consistently predictive embedding dimensions. Figure 2 highlights class-specific distributions. Figure 3 demonstrates individual patient-level explanations, where we can see how combinations of features contribute to specific predictions (relevant for clinical interpretability).

![SHAP Feature Importance Analysis](figures/01_shap_feature_importance.png) | ![SHAP Model Comparison](figures/02_shap_model_comparison.png)
:---:|:---:
*Figure 1: Aggregate SHAP value distribution for ComplEx with SVM (best model), highlighting the most consistently predictive biomarkers across all samples.* | *Figure 2: SHAP value distributions stratified by class, illustrating how feature importance patterns differ between outcome groups.*

![SHAP Individual Prediction Explanations](figures/03_shap_individual_predictions.png)
*Figure 3: Individual patient-level SHAP explanations showing feature contributions to specific predictions. Red bars indicate increased prediction of poor outcome, blue bars decreased prediction.*

2. **Feature Consolidation**: The `consolidate_interpretability.py` script aggregates feature importance rankings from all model-dataset combinations, applying normalization strategies (minmax, z-score) to identify consensus biomarkers that are robust across architectures. This aims to identify high-confidence biomarkers that consistently emerge across diverse models and normalization schemes.

![Feature Consolidation and Consensus Ranking](figures/04_feature_consolidation.png)
*Figure 4: Aggregated feature importance rankings across all 4 ML models with ComplEx and RGCN, identifying robust consensus biomarkers.*

![GNN Embedding Dimension Importance](figures/05_gnn_interpretability.png)
*Figure 5: Heatmap of the correlation between the top embedding dimensions.*

3. **Automated Feature Mapping**: `auto_feature_mapping.py` and `map_features_per_dataset.py` automatically trace abstract feature indices back to their biological entities (proteins, pathways, GO terms), enabling interpretable biomarker identification. Primary focus was given to protein mapping considering the fact that they were exclusively used in all models training. Nonetheless, we could still examine the neighborhoods of our identified proteins of interest in the knowledge graph to retrieve associated pathways and GO terms.

4. **GNN-Specific Interpretability**: The `gnn_interpretability.py` module provides specialized analysis for GNNs, extracting embedding dimension importance and linking them to graph structure. Unlike traditional feature importance, GNN interpretability must account for the learned representations in the embedding space. By analyzing the relationship between dimensions and entity embeddings, we can understand what high-level concepts the network has learned.

Overall, this pipeline ensures that model predictions are not only accurate but also biologically meaningful and actionable.

## RGCN-Focused Analysis (`src/interpretability/analysis/`)

Initially, we started with RGCN (Relational Graph Convolutional Network) analysis focusing on understanding how graph structure and learned embeddings contribute to model predictions.

### Mapping Features Back to Biological Entities (`analyze_rgcn_dimensions.py`)

We map top RGCN embedding dimensions (identified from consolidated interpretability results) back to the biological entities they represent (uses `GNNInterpreter` class). By analyzing the contribution of each dimension to entity embeddings, we can understand what biological signals each learned feature captures.  

### Graph Structure Importance Analysis (`analyze_graph_structure.py`)

The `GraphStructureAnalyzer` class systematically evaluates which elements of the knowledge graph contribute most to RGCN's predictive power by computing multiple centrality measures and structural properties:

- **Node Importance**: Identifies central entities using degree, betweenness, and closeness centrality metrics. Degree centrality reveals the most connected proteins (hub proteins), while betweenness identifies nodes that serve as bridges between different network regions.
- **Edge Type Importance**: Quantifies which relationship types (physical, genetic, regulatory) are most predictive by analyzing their frequency and their correlation with model performance.

![Graph Structure and Node Centrality Analysis](figures/06_graph_structure_centrality.png)
*Figure 6: Edge types distribution in the knowledge graph.*

### Hub-Dimension Linkage (`link_hub_to_dimensions.py`)

This script links graph hub proteins (high-degree nodes in patient-connected subgraphs) to statistically significant RGCN embedding dimensions. By identifying which hubs drive which dimensions, we can:
- Validate that biologically important entities are captured in learned representations
- Prioritize dimensions for further investigation based on their association with known regulatory hubs
- Understand the mechanistic basis of dimension-level predictions

This linkage proved valuable for validating the RGCN learning process, identifying FYN as a primary driver of RGCN's sepsis prediction power, in accordance with previous literature [1].

**Dimensions showing significant septic vs. healthy differences (p<0.05):**
  - Dimension 45: p=0.0024 ✓ FYN (rank #14)
  - Dimension 63: p=0.0063 ✓ FYN (rank #4) **← Most significant**
  - Dimension 33: p=0.0497 ✓ FYN (rank #14)
  - Dimension 67: p=0.0415 ✓ FYN (rank #8)
  - Dimension 38: p=0.0322 ✓ FYN (rank #6)

More details on biological findings could be found in the following section.

### Normalization Comparison (`analyze_interpretability.py`)

We considered comparing different normalization strategies for feature importance aggregation to ensure biomarker rankings are not artifacts of preprocessing. This analysis established normalization had a big impact on the top features identified.

![Normalization Strategy Comparison](figures/07_normalization_comparison.png)
*Figure 7: Comparison of feature importance rankings with and without normalization (none vs minmax).*

## ComplEx-Focused Analysis

### SVM with ComplEx Embeddings - Top Model Analysis (`src/interpretability/model_specific/`)

The SVM trained on ComplEx knowledge graph embeddings achieved the best overall performance. The `src/interpretability/model_specific/` directory contains scripts dedicated to analyzing this top model in depth:

#### Feature Mapping (`map_complex_svm_features.py`)
Maps abstract ComplEx embedding dimensions to interpretable biological entities by:
- Loading ComplEx entity embeddings and mappings from the pre-trained knowledge graph embedding model
- Identifying which entities (proteins, pathways, GO terms) contribute most to each top-ranked feature dimension using nearest neighbor analysis in embedding space

The feature mapping process revealed that top-ranked SVM features correspond to highly semantically coherent clusters in the ComplEx embedding space. 

![ComplEx Embedding Dimension Mapping](figures/08_complex_embedding_mapping.png)
*Figure 8: The top identified biomarkers.*

#### Biomarker Extraction (`extract_biomarker_subgraph.py` & `extract_biomarker_subgraph_proteins_only.py`)
Extracts PPI and heterogeneous biomarker subgraphs, enabling:
- Identification of direct and indirect protein interactions among biomarkers
- Assessment of network connectivity and module structure
- Connectivity of biomarkers of all types 

| ![SVM Network Module Structure](figures/11_svm_network_visualization.png)! | ![ComplEx SVM Top Model Network Visualization](figures/10_complex_network_visualization.png) |
|:---:|:---:|
| *Figure 11: Biomarkers sub-network for ComplEx+SVM top model biomarkers.* | *Figure 12: bi-partite graph of protein-to-pathways connections.* |

#### GO Enrichment Analysis (`enrich_go_terms.py`)
Maps GO terms to their corresponding names and details using `go.obo` ontology file from `OntoKGCreation/`. 

### Consensus Analysis Across All Models (`src/interpretability/consensus/`)

To ensure robustness, we developed a consensus framework aggregating results from all four model architectures (Random Forest, SVM, ComplEx, RGCN):

#### Consensus Protein Extraction (`extract_complex_consensus_proteins.py`)
Identifies proteins that rank highly across multiple models, reducing model-specific biases and increasing confidence in biomarker selection.

#### Consensus Biomarker Network Extraction (`extract_biomarker_subgraph_proteins_only.py`)
Builds PPI networks using only consensus proteins, revealing core regulatory modules that are consistently important regardless of modeling approach.

#### GO Enrichment of Consensus Entities (`enrich_consensus_entities.py`)
Performs functional enrichment specifically on consensus biomarkers to identify robust functional themes supported by multiple lines of evidence (offers option to consider hub proteins as well). 

| ![Consensus GO Enrichment Analysis](figures/09_consensus_go_enrichment.png) | ![Consensus Biomarker Summary Table](figures/13_consensus_biomarker_table.png) |
|:---:|:---:|
| *Figure 9: KG-based GO term enrichment results for consensus biomarkers showing significantly over-represented biological processes.* | *Figure 10: GO and GSEA enrichment results.* |


## Biological Insights 

The interpretability analyses successfully identified several key genes and proteins with strong literature support for their roles in sepsis and immune response. 

Across 35 trained models (24 v2.11_none, 11 v2.11_minmax), we identified a **robust 16-gene signature** driving sepsis classification:

| **Pathway** | **Genes** | **Role in Sepsis** |
|---|---|---|
| **RAS/MAPK** | KRAS, GRB2, SHC1, MAPK1 | Growth signal transduction; inflammatory amplification |
| **PI3K/AKT** | PIK3R1 | Cell survival; immune cell activation |
| **JAK/STAT** | STAT1, FYN | Interferon signaling; immune response |
| **Transcription** | MYC | Cellular proliferation; metabolic reprogramming |
| **DNA/Repair** | BARD1, RPA2 | DNA damage response; stress response |
| **Structural** | CTNNB1, ITM2B | Cell adhesion; organellar integrity |
| **Ubiquitination** | SPSB1, RNF4, STXBP3, NXF1 | Protein degradation; signaling regulation |

**Key Insight**: Top consensus features (Feature_4, 63, 73) consistently map to **STAT1/FYN** (JAK/STAT inflammatory axis) and **KRAS/GRB2/PIK3R1** (PI3K/MAPK growth/survival axis)—classic sepsis dysfunction nodes linking immune dysregulation to cellular stress.

---

For RGCN, The two most significant dimensions revealed opposing aspects of sepsis:

**Dimension 45 (p=0.0024 ← MOST SIGNIFICANT)**
**"Loss of Transcriptional Gene Expression"**

**Top Entities (Positive signal in sepsis):**
- GO:0035556 - Intracellular signal transduction
- GO:0007165 - Signal transduction  
- GO:0006351 - Transcription, DNA-templated
- GO:0050794 - Regulation of cellular process
- GO:0065007 - Biological regulation

**Negative signal (Healthy controls have high values):**
- Protein_FYN (TCR kinase)
- Protein_GRB2 (TCR adaptor)
- Protein_CTNNB1 (Signaling hub)
- Protein_MYC (Transcription factor)
- Protein_CDC42 (Rho GTPase, immune response)
- Protein_SMAD4 (TGF-β signaling)

**Interpretation:**
- Septic patients show **reduced transcriptional processes**
- Loss of signal transduction capacity
- Suggests "immune paralysis" phase of sepsis
- TCR components are actively suppressed in sepsis

**Dimension 63 (p=0.0063)**
**"T-Cell Receptor Signaling Activation"**

**Top Entities (Elevated in sepsis):**
1. Protein_GRB2 - TCR adaptor
2. Protein_CTNNB1 - β-catenin (signaling)
3. Protein_PIK3R1 - PI3K pathway
4. Protein_FYN - TCR initiator kinase
5. Protein_SHC1 - TCR adaptor
6. Protein_SUMO1 - Signal modification
7. Protein_MAPK1 - ERK/MAPK cascade
8. Protein_KRAS - RAS signaling
9. Protein_ITGB1 - Integrin (trafficking)
10. Protein_SMAD4 - TGF-β signaling

**Interpretation:**
- Septic patients show **elevated TCR/signaling proteins**
- Classical cascade: FYN → GRB2 → RAS/MAPK → response
- Represents inflammatory/hyperinflammatory phase
- Coordinated upregulation of entire signaling pathway

As sepsis involves **two competing phases**:

```
Timeline in Sepsis:
├─ Early: HYPERINFLAMMATION
│  ├─ Excessive TCR activation (Dimension 63 HIGH)
│  ├─ Cytokine storm
│  └─ Tissue damage
│
└─ Late: IMMUNE PARALYSIS
   ├─ Loss of immune function (Dimension 45 HIGH)
   ├─ Transcription shutdown
   └─ Increased mortality risk
```

It appears that RGCN could have learnt both patterns and their prognostic value:
- Dimension 63: Active immune response (separates outcomes)
- Dimension 45: Loss of immune capacity (separates outcomes)

# HAN (Heterogeneous Attention Network) Analysis (`src/han/`)

We implemented a comprehensive Heterogeneous Attention Network framework for patient-level graph analysis, enabling interpretable predictions through multiple complementary mechanisms. The `src/han/` directory contains scripts for model training, attention extraction, and gradient-based interpretability.

## Model Architecture and Training (`model.py`, `run_han.py`)

The `SepsisHANClassifier` implements a multi-layer heterogeneous attention network that:
- Processes multiple node types (Patients, Proteins, Pathways, GO Terms, Reactions) simultaneously
- Learns type-specific transformations and cross-type attention weights
- Aggregates multi-relational information through semantic-level attention
- Enables end-to-end training for patient outcome prediction

The `run_han.py` script orchestrates the full training pipeline with proper data loading, hyperparameter tuning, and model checkpointing.

## Attention Mechanism Analysis

### Node-Level Attention Extraction (`extract_han_attention.py`, `attention_analysis.py`)

These scripts extract and analyze attention weights learned by the HAN model:
- **Edge Type Importance**: Quantifies which relationship types (Patient-Protein, Protein-Pathway, etc.) contribute most to predictions
- **Node-Level Attention Scores**: Identifies which specific entities receive high attention for each patient
- **Attention Distribution Analysis**: Visualizes how attention is distributed across different node types and edge types

The attention extraction reveals which parts of the knowledge graph the model focuses on when making predictions, providing a first layer of interpretability.

### Semantic-Level Attention Analysis (`han_attention_enhanced.py`, `han_attention_extraction.py`)

Beyond individual edge attention, semantic-level attention aggregates information across entire metapaths (multi-hop paths through different node types). The enhanced attention analysis:
- **Metapath Importance**: Identifies which semantic pathways (e.g., Patient→Protein→GO Term→Pathway) are most predictive
- **Cross-Type Information Flow**: Tracks how information propagates through heterogeneous graph structure
- **Attention Heatmaps**: Generates visualizations showing attention patterns across semantic spaces

![HAN Semantic-Level Attention Analysis](figures/12_han_semantic_attention.png)

## Gradient-Based Interpretability and Subgraph Analysis

### Patient-Level Gradient Attribution (`patient_gradient_analysis.py`)

The `GradientBasedAttentionAnalyzer` class implements sophisticated gradient-based methods to identify influential nodes:

1. **Gradient-Based Attribution**: Computes gradients of predictions with respect to node features, identifying which input features matter most for each patient
2. **Neighbor Perturbation**: Systematically perturbs different neighbors to measure their impact on predictions, revealing causal relationships
3. **Embedding Similarity Analysis**: Identifies neighbors with similar learned representations, highlighting functionally related entities

This goes beyond simple attention weights by assessing actual influence on model decisions through gradient flow.

### Heterogeneous Patient Subgraph Visualization (`han_heterogeneous_analysis_initial.py`)

- **Patient-Specific Neighborhoods**: Extracts all heterogeneous neighbors (Proteins, Pathways, GO Terms, Reactions) connected to each patient
- **GO Term Annotation**: Integrates GO ontology definitions and semantic information using `parse_go_obo()`, providing rich contextual information about biological processes
- **Gradient-Based Node Importance**: Ranks neighborhood nodes by their gradient-attributed importance to patient predictions, measuring which neighbors would most impact the prediction if perturbed

![Patient-Specific Subgraph Visualization](figures/14_patient_subgraph_visualization.png)
*Figure 14: Heterogeneous patient subgraph showing gradient-attributed importance of proteins, pathways, and GO terms. Node size indicates gradient-based importance (larger = more causal influence on prediction), color indicates node type (green = protein, blue = GO term, orange = pathway). Edge color is proportional to edge importance. This represents a single patient's personalized mechanistic explanation of their predicted outcome.*

#### Gradient-Based Protein Ranking
For each patient, the script:
- Computes gradient attributions for all connected proteins
- Ranks proteins by their contribution to the prediction
- Identifies key driver proteins whose perturbation would most impact the outcome
- Exports ranked lists for downstream experimental validation

#### GO & Reactome Term Integration
The script enriches protein nodes with functional annotations by:
- Parsing GO ontology files (`go.obo`) for term definitions
- Using Reactome API to retrieve pathway id details
- Mapping proteins to their associated GO terms through the knowledge graph
- Visualizing functional themes (immune response, cell signaling) in patient subgraphs

### Complex Overlap Analysis (`han_complex_overlap.py`)

This script integrates HAN attention with ComplEx embedding-based biomarkers to identify:
- Proteins that are both highly attended by HAN and important in ComplEx embeddings
- Consensus entities supported by both graph structure (HAN) and semantic similarity (ComplEx)
- Robust biomarker candidates with multiple lines of computational evidence

## HAN Data Loading Pipelines

### OWL-Based Data Loading (`owl_data_loader.py`, `owl_data_loader_with_features.py`)

Custom data loaders for heterogeneous graphs stored in OWL ontology format:
- **Multi-Type Node Features**: Loads feature matrices for each entity type
- **Heterogeneous Edge Indices**: Constructs edge indices for all relationship types
- **Patient Label Integration**: Merges clinical outcome labels with graph structure
- **Feature Normalization**: Applies appropriate preprocessing for different node types

The `load_hetero_graph_from_owl()` function creates PyTorch Geometric `HeteroData` objects ready for HAN training.

# Visualization Infrastructure (`src/visualize/`)

We developed a comprehensive suite of visualization tools to enable intuitive exploration complex multi-layer knowledge graphs. The `src/visualize/` directory contains scripts for static plotting, interactive visualization, and web-based network exploration.

## Interactive Multi-Layer Network Web Application (`multilayer_network_app.py`)

We developed a Flask-based web application that provides dynamic, real-time exploration of the knowledge graph:

### Core Features

#### Multi-Layer Graph Management (`MultiLayerNetworkManager` class)
- **Flexible Data Loading**: Loads nodes, edges, and entity classes from KG conversion and model execution outputs
- **Layer Definitions**: Automatically classifies nodes into semantic layers (Patients, Proteins, Pathways, GO Terms, Diseases, etc.)
- **Edge Type Tracking**: Catalogs all relationship types in the graph for selective filtering
- **Patient Embedding Integration**: Links graph structure with learned model embeddings

#### Real-Time Filtering and Exploration
The web interface enables users to:
- **Layer Selection**: Toggle individual node type layers on/off dynamically
- **Edge Type Filtering**: Show/hide specific relationship types (physical, genetic, regulatory)
- **Statistics Dashboard**: Display real-time graph statistics (node counts, edge counts...)
- **Force-Directed Layout**: Physics-based layouts for intuitive spatial organization

![Interactive Web Application Interface](figures/visualization_app.png)
*Figure: Screenshot of the interactive multi-layer network visualization web application, showing real-time filtering controls, layer management, and dynamic network rendering.*

This is relevant both for initial exploratory analysis and examination of specific neighborhood to help guide analysis.

## Static Graph Visualization (`visualize_multilayer_graph.py`)

For batch generation of network plots and non-interactive figures, `visualize_multilayer_graph.py` provides:

### PyVis Interactive HTML Exports
- **Standalone HTML Files**: Self-contained interactive visualizations
- **Custom Physics**: Configurable force-directed algorithms (Barnes-Hut, repulsion strength)
- **Legend Integration**: Automatic generation of interactive legends

### NetworkX-Based Static Plots
- **Multiple Layout Algorithms**: Spring, Kamada-Kawai, hierarchical, circular
- **High-Resolution Output**: PNG at 300+ DPI for publication quality
- **Vector Graphics**: SVG/PDF formats for scalable figures
- **Matplotlib Integration**: Full access to matplotlib styling and customization

## Gephi Export Pipeline (`gephi_exports/`)

### Graph Export Formats
- **GEXF Files**: Native Gephi format with full metadata preservation
- **GraphML**: Alternative format supporting complex attribute schemas
- **CSV Edge/Node Lists**: Simple tabular format for custom processing


# Knowledge Graph Optimization (`OntoKGCreation/`)

We started considering optimizing the knowledge graph to balance comprehensiveness with computational efficiency and biological relevance. We modified the scripts provided in `OntoKGCreation/` for KG refinement.

## Motivation for KG Optimization

Our initial knowledge graph, constructed from multiple biomedical ontologies and databases, contained:
- Thousands of entities (proteins, pathways, GO terms, reactions) - we particularly notice that > 60% of nodes are pathways/reactions with many embedded connections between them 
- Hundreds of thousands of unfiltered relationships with varying confidence levels (proteins are very densely connected since all PPI data is used without filtering)
- Redundant and low-information connections that add noise

This scale posed challenges for:
- **Computational Efficiency**: GNN training time and memory requirements
- **GNN Performance**: Risk of confusing models with irrelevant or noisy information
- **Interpretability**: Overwhelming number of potential features to analyze
- **Biological Focus**: Dilution of signal by weakly relevant entities

## Filtering and Optimization Strategies

### Ontology-Based Refinement
Leveraging the ontological structure:
- **Term Specificity**: Filtering overly general GO terms (e.g., "biological process") in favor of specific terms
- **Hierarchy Pruning**: Removing redundant parent-child chains where leaf terms suffice

Through iterative refinement, we aim to achieved:
- **Size Reduction**: significant reduction in node count/edge density while preserving (or improving if the GNNs could learn better) performance and interpretability signals
- **Performance Improvement**: Faster training (reduced embedding dimensions) with comparable or improved accuracy
- **Enhanced Interpretability**: Cleaner, more focused feature importance rankings
- **Biological Coherence**: Increased functional enrichment significance for top features

OntoKGCreation/converted/optimized/ -> retrain embeddings -> ML approaches -> comparison to original KG results.

# Discussion & Future Work

## Limitations and Challenges

### Current Limitations
1. **KG Completeness**: Despite optimization efforts, the knowledge graph may still miss important but poorly annotated entities
2. **Interaction Confidence**: Variable quality of edge annotations requires cautious interpretation of network results
3. **Sample Size**: Limited patient cohort size constrains statistical power for rare outcome prediction
4. **Temporal Dynamics**: Current models do not capture time-dependent disease progression

### Technical Challenges
- Balancing KG size with computational feasibility and interpretability
- Handling heterogeneous data types and scales across node features
- Ensuring gradient stability in complex heterogeneous graph architectures
- Managing visualization complexity for large-scale networks

## Future Research Directions

### 1. Enhanced Knowledge Graph Optimization

**Reduce/Optimize KG**: Continued refinement of the knowledge graph through:
- **Active Learning Approaches**: Using model uncertainty to guide selective KG expansion in informative regions
- **Expert-in-the-Loop Curation**: Incorporating domain expert feedback to validate and refine entity/relationship selections
- **Multi-Task Optimization**: Creating task-specific KG variants optimized for different prediction goals (severity, outcome, treatment response, trajectory)
- **Temporal Integration**: Incorporating time-dependent relationships and dynamic processes
- **Confidence Modeling**: Learning edge weights and uncertainty estimates rather than binary inclusion/exclusion

### 2. Interpretability-Visualization Integration

**Connect Interpretability Findings to KG Visualization**: Tighter integration of model explanations with network exploration:
- **SHAP-Driven Interactive Highlighting**: Real-time updates of node/edge importance in the web application based on SHAP scores
- **Attention Heatmap Overlay**: Visualizing HAN attention distributions directly on the interactive graph
- **Gradient Flow Visualization**: Animating gradient propagation through patient subgraphs to show causal pathways
- **Perturbation Explorer**: Interactive tools to simulate node/edge removal and observe prediction changes
- **Explanation Provenance**: Linking each biomarker back to the specific analyses and models that identified it

### 3. Advanced Visualization Enhancements

**Visualization Enhancement**: Ongoing improvements to support deeper exploration:
- **Dynamic Filtering Interface**: More intuitive controls for multi-dimensional filtering (importance, confidence, node type, functional category)
- **Annotation Layer System**: User-added notes, hypotheses, and literature references directly on network views
- **Comparative Visualization**: Side-by-side comparison of patient subgraphs or model predictions
- **3D Network Rendering**: Spatial layouts leveraging additional dimensions for complex hierarchies
- **Automated Layout Optimization**: Machine learning-based layout algorithms that optimize for biological interpretability
- **Export and Reproducibility**: One-click export of full analysis provenance (data, filters, settings) for reproducibility
- **Biomarker Highlighting**: Emphasize identified biomarker nodes with custom styling
- **Search Functionality**: Find specific entities by ID or name

### 4. Model Architecture Advances

- **Temporal Graph Networks**: Incorporating time-series patient data and dynamic KG relationships
- **Explainable-by-Design Architectures**: Models with built-in interpretability mechanisms rather than post-hoc analysis
- **Causal Graph Discovery**: Moving beyond correlation to identify causal relationships in biological networks
- **Multi-Modal Integration**: Combining KG-based models with imaging, clinical notes, and other data modalities

### 5. Clinical Translation

- **Prospective Validation**: Testing identified biomarkers in independent patient cohorts
- **Experimental Validation**: Wet-lab experiments to validate predicted protein interactions and mechanisms
- **Clinical Decision Support**: Adapting models and visualizations for real-time clinical use
- **Treatment Response Prediction**: Extending models to predict which patients will respond to specific therapies

### 6. Scalability and Efficiency

- **Distributed Training**: Scaling to larger KGs and datasets through distributed GNN training
- **Incremental Learning**: Updating models with new data without full retraining
- **Real-Time Inference**: Optimizing model deployment for low-latency predictions
- **Federated Learning**: Training across multiple institutions while preserving data privacy

## Conclusion

## Methodological Contributions

1. **Multi-Model Interpretability Framework**: Successfully implemented SHAP, attention, and gradient-based analyses across diverse architectures (Random Forests, SVMs, ComplEx, RGCN, HAN), providing multiple complementary perspectives on model decisions.

2. **Consensus Biomarker Discovery**: Developed robust pipelines to aggregate evidence across models, increasing confidence in identified biomarkers and reducing model-specific biases.

3. **Heterogeneous Graph Analysis**: Advanced HAN architecture with gradient-based patient-level interpretability, enabling personalized explanations of predictions.

4. **Advanced Visualization Suite**: Created publication-ready static plots and interactive web applications for exploring complex multi-layer knowledge graphs.

## Technical Infrastructure
- Established reproducible pipelines for model training, interpretability analysis, and visualization
- Developed modular, reusable code for knowledge graph construction, optimization, and analysis
- Created interactive tools for collaborative exploration and hypothesis generation


# References

[1] Jiang Y, Miao Q, Hu L, Zhou T, Hu Y, Tian Y. FYN and CD247: Key Genes for Septic Shock Based on Bioinformatics and Meta-Analysis. Comb Chem High Throughput Screen. 2022;25(10):1722-1730. doi: 10.2174/1386207324666210816123508. PMID: 34397323.