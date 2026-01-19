# Abstract

# Evaluation

To have a performance comparison across the different models and datasets used, the evaluation code consists of three main stages: **collecting predictions**, **computing metrics**, and **visualizing results**.

### 1. Collecting Results

All model predictions, along with the corresponding true labels, are stored in a structured way using the **`ResultsCollector`** class (in `src/ml/collector.py`). This class keeps track of:  

- `y_test`: the ground-truth labels for the test set  
- `y_pred`: the predicted class labels  
- `y_proba`: the predicted probabilities for the positive class  

Data is stored separately for each combination of model and input type, which allows us to easily compare models and see how the different inputs affect performance.

### 2. Computing Metrics

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

### 3. Visualizing Performance

To make evaluation results easier to understand, the **`DashboardVisualizer`** class (in `src/ml/visualizer.py`) provides several plotting options. The visualizations can show:  

- **Metric grids**: bar plots for each metric across models and inputs, useful for side-by-side comparisons.  
- **Radar plots**: combine multiple metrics in a single plot for each model-input combination.  
- **ROC and Precision-Recall curves**: help inspect model performance for different thresholds.  


# Results

## EDA

### Knowledge Graph Analysis

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


#### Biological Enrichment Analysis

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


# References
- Chicco, D., & Jurman, G. (2020). The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation. BMC genomics, 21(1), 6.
- Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PloS one, 10(3), e0118432.
- Rufibach, K. (2010). Use of Brier score to assess binary predictions. Journal of clinical epidemiology, 63(8), 938-939.
- Brandes-Leibovitz, R., Riza, A., Yankovitz, G., Pirvu, A., Dorobantu, S., Dragos, A., ... & Netea, M. G. (2024). Sepsis pathogenesis and outcome are shaped by the balance between the transcriptional states of systemic inflammation and antimicrobial response. Cell Reports Medicine, 5(11).
- Liu, W., Liu, T., Zheng, Y., & Xia, Z. (2023). Metabolic reprogramming and its regulatory mechanism in sepsis-mediated inflammation. Journal of inflammation research, 1195-1207.
- Willmann, K., & Moita, L. F. (2024). Physiologic disruption and metabolic reprogramming in infection and sepsis. Cell metabolism, 36(5), 927-946.