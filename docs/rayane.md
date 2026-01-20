<!-- ⦁	Exploratory data analysis
-- gene expression pca/(im not sure if differential analysis here or later is better), distribution of patients.. (plots in silvia.md)
-- PCA/UMAP (dr) of embeddings we have, showing split in complex for example
⦁	GNN training
-- what we have been provided with (complex and rgcn, versions)
-- new trainings and details of them (rgcn weights, gat, graphsage)
-- output: embeddings and shape?
⦁	Model training
-- preprocessing and normalizations, datasets
-- models and trainings details,
-- evaluation 
⦁	Interpretability
## KG interpretation/dow up regulated genes
## shap -> graph
## HAN

You can maybe highlight the most important results/parameters in the main readme and put a link(s) to a sub-readme(s) (appendix) that includes all results/parameters.
You are free to choose the most appropriate way to show your results.

Results
Analysis: discussion, interpretation, limitations on the method
 -->

## reproducibility

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

## Exploratory Data Analysis

### Graph Embeddings Visualization

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

## Machine Learning Model Training and Evaluation


### Preprocessing and Normalizations


<!-- ... datasets description ... -->

<!-- * sample embeddings $E_s$ obtained directly from GNNs embeddings of samples
* protein embeddings $E_p$ obtained from GNNs embeddings of proteins, multiplied with gene expression data matrix $G$ as follows: $$E_p = \frac{G \times E_{prot}}{\sum G_i}$$ where $E_{prot}$ are the protein embeddings from the GNNs and the division represents row normalization
(matrix multiplication of dimensions: `[num_samples, num_proteins] x  [num_proteins, embedding_dim] => [num_samples, embedding_dim]`, followed by row normalization, so at the end `[num_samples, embedding_dim]`).

For each trained model (ComplEx, RGCN, weighted RGCN, GraphSAGE, GAT), we obtained both sample embeddings and protein embeddings datasets for training machine learning models, leaving us with a total of 10 datasets (5 sample embeddings + 5 protein embeddings), in addition to the original gene expression dataset, making it 11 datasets in total: -->

| Pipeline  | Dataset Type            | Models Used                | Number of Datasets |
|-------------------------|----------------------------|--|------------------|
| Traditional ML Pipeline | Gene Expression Data       | N/A                        | 1                  |
| Graph augmented ML Pipeline | Sample Embeddings       | ComplEx, RGCN, weighted RGCN, GraphSAGE, GAT | 5                  |
| Graph augmented ML Pipeline | Protein Embeddings      | ComplEx, RGCN, weighted RGCN, GraphSAGE, GAT | 5                  |


As each GNN was trained from embeddings obtained from two different knowledge graph versions (based on binning v2.10, and averaging v2.11). Moreover, as the protein embeddings were multiplied with gene expression values, we considered different normalization techniques to identify the best performing one. We experimented with standard, robust, min-max, log1p normalizations, as well as the option with no normalization applied, resulting in a total of five normalization types.  
Each training also involved a grid search to find the optimal hyperparameters for the specific data space being trained on. To ensure a fair-er comparison accross different datasets and models, we trained each model for every (dataset, model type, version, normalization) combination using 10 different random seeds, averaging the results to obtain the final performance metrics, which we found to be necessary in preliminary robustness analyses of our results that showed fluctuations of performance with different starting points (e.g. initialization, data splits, etc.). The results we will be showing correspond to the average over these 10 seeds.
Overall, this led to a comprehensive evaluation framework, allowing us to systematically assess the impact of different graph embeddings, normalization techniques, and model architectures on the prediction of sepsis from gene expression data.


| different normalizations  | different versions |
|--------------------------|--------------------|
| ![normalizations](./results/figures/comparative_normalization_version/avg_seeds_normalizations/boxplot_comparative_normalization_version_balanced_accuracy.png) | ![versions](./results/figures/comparative_normalization_version/avg_seeds_versions/boxplot_comparative_normalization_version_balanced_accuracy.png) |

Between different normalizations, MinMax show the best performance overall, where median balanced accuracy of v2.10 is highest, and spread is the lowest in v2.11. As for versions, v2.11 tends to have better performance than v2.10, while still pretty much model/dataset dependent. Thus for the rest of the analysis, we will be focusing on MinMax normalization and version v2.11.

### Evaluation Metrics and Results Visualization

To evaluate the performance of our machine learning models, we employed several key metrics: balanced accuracy, precision, recall, and F1-score, AUROC, AUPRC, Matthews correlation coefficient (MCC), and Brier score. These metrics provide a comprehensive view of model performance, particularly in the context of imbalanced datasets like ours (0.28:0.72 sepsis:non-sepsis ratio). Balanced accuracy is especially crucial as it accounts 

The traditional machine learning pipeline generally consists of training models directly on gene expression data (~20k genes).
The graph augmented pipeline can be split into **sample embeddings** and **protein embeddings** datasets for training, these were 100-200 features datasets.  
For a better representation of the results due to high number of models and datasets, we will be comparing **sample embeddings vs gene expression** and **protein embeddings vs gene expression** separately, to investigate the performance gain brought by graph augmented data in both cases.

#### Sample Embeddings

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


#### Protein Embeddings

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

### Overall Comparison and Conclusion
As the analysis was performed on v2.11 with Minmax normalization with most robust and best performance overall, we will discuss some findings regarding traditional vs graph augmented ML pipelines for sepsis prediction from gene expression data.

Gene expression data perform greatly using SVM models. This might be due to the high dimensionality of the data (20k genes) where SVMs are known to handle high-dimensional spaces well, especially with appropriate kernel functions, it might be aslo overfitting to some extent as dataset is small. ComplEx ourperformed this high achieving model, with and svm linear kernel (from grid). Previosuly we noticed ComplEx protein embeddigns being the best dataset showing clear seperation in different DR approaches, particularly siting PCA as the only dataset with a decent variance explained in first 2 components - a linear method - which might explain why SVM with linear kernel works well here.
Other models like random forest and xgboost also show good performance with gene expression data, indicating that tree based models can effectively capture relevant patterns in the data without the need for graph augmentation. MLP models however perform poorly, likely due to the small sample size and high dimensionality leading to overfitting (constant bad performance around 50% balanced accuracym from robustness analysis). And this is a known issue with neural networks in general when working with tabular data, especially with limited samples, thus finding ways to enhance it using graph embeddings would be particularly useful.

Each model type has its own set of performing datasets, while in all of them there is a seen improvement using graph augmented data (protein embeddings), justifying the benefit of integrating knowledge graphs in the analysis AND the gene expression weighting transformation (since sample embeddings that were used without any transformation did not perform well). Overall, even though some GNNs embeddings like GAT have slight higher performance in some models, ComplEx protein embeddings based models show the most consistent high performance accross different models and metrics, making it the best performing dataset overall for sepsis prediction in this study. This conclusion is not just based on its top rankings across models, but also from robustness analysis showing lowest spread in performance across seeds and the grounbraking performance in MLP models as rank 1 with 40%+ balanced accuracy improvement over gene expression based MLP, SVM's rank 1 and direct competitor to gene expression, and strong showings in tree based models as well.

An interesting aspect we noticed regarding embeddings, ComplEx is the only model that has 200 embeddings dimensions, while all the other are 100 only. Due to its architecture, ComplEx (as name suggests) is actually operating in complex number space, meaning each dimension can be seen as 2 dimensions (real and imaginary parts). Thus effectively, to have a fairer comparison, would conisdering raising the other GNNs embedding dimensions to 200 as well, or lower this one to 50 (which will be multiplied by 2) to test if this is the reason behind its good performance. However, due to time constraints we could not perform this analysis - a lot of parameters and hyperparameters are in fact worth exploring as we have a lot of variables in this study (different GNNs, different ML models, different normalizations, different versions, different seeds, different parameters).

Concerning Multiple Layer Perceptron, we fixed at the end to studying them using scikit learn's implementation on 500 epochs. Our machine learning pipeline relied on `MLModel` class definition that can take different skleran model types, datasets, normalizations, versions, seeds, parameters and hyperparameters for grid search, We also implemented a cutom PytorchMLP model inheriting ClassiferMixin and BaseEstimator from sklearn to be able to use it in the same pipeline and even trained on a large set of models (not all though), however as there were more available options of optimizer, activation functions etc. from sklearn, we started exploring them more and createed a large grid search of hyperparameters, leading to long training times and instabilities (high variance of results) for only 50 epochs (also, early stopping was a hastle to implement with cross validation, ensuring caching and returning the best model across the last  - which have not been successful at the end). We diverted to sklearn's MLP implementation for a more stable and faster training. More complex architectures and hyperparameter tuning can be explored in future work to fully leverage the potential of neural networks in this context, especially with the promising results seen with Complex protein embeddings, and the failure of gene expression based MLP models. It's sensitive to work with as dataset is small, split and imbalanced.

## Interpretability

Shapley Additive exPlanations (SHAP) is a method to explain predictions connecting optimal credit allocation with local explanations using the classic Shapley values from cooperative game theory and their related extensions.  
The idea of this interpretation is mainly addressing the protein embeddings data in the following context:

Protein embeddings used for training are built using a "linear" approach, if can be seen as a LF or matrix multiplication in a sense:
Let $G$ be the `samples x genes` gene expression matrix (genes and proteins here are used interchangeably), and $E_protein$ be the `proteins x embedding_dimension` protein embeddings matrix, then the protein embeddings for samples $E_p$ is computed as:
$$E_p = \frac{G  E_{protein}}{\sum_i G_i}$$

Where $E_p$ is the `samples x proteins` protein embeddings matrix, input of the ML models (its the matrix multiplication + row normalization)

To make the training process more interpretable we aim to use SHAP values to link the importance of each protein embedding dimension back to the genes/proteins space, so a plan of 2 steps: find the most important embedding dimensions using SHAP, then link them back to genes/proteins by computing "loadings" (similar concept to pca loading and MOFA) that are the contributing the most to these embedding dimensions (they behave like highly weighted features in a linear model) - this idea is captured in the diagram. Here, we would rank the proteins by their absolute loadings values or weights for the top important embedding dimensions, and select the top k proteins overall and study their biological significance in the context of sepsis, and furthermore perform pathway enrichment analysis on the ranked list of proteins to identify enriched biological pathways or functions associated with sepsis (a possible way to explore the protein embedding space and color by the top genes/proteins contributing to these important embeddings, to find structure. in fact, if we look at their distribution in umap space we clearly see some clusters, indicating possible grouping by functions/pathways or any common characteristic). 
However, as we retrieved due to time limits, this analysis was not fully completed or analyzed. In future work, this interpretability approach can provide valuable insights from a biological perspective.
