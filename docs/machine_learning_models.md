# Machine Learning

This section aims to compare traditional vs graph-augmented ML models predicting sepsis from gene expression data.

## Gene expression

Will be purely desrcibed during EDA
-> training

## Graph augmented data

Here need to talk about:
* The learning perfomred on the KG using ComplEx and RGCN 
* ComplEx and RGCN embeddings
* Training our GNN models: weighted RGCN, GraphSAGE, GAT (link prediction)
* weighted RGCN, GraphSAGE and GAT embeddings

Each model we take from it:  
* sample embeddings -> direct training
* protein embeddings -> matrix multiplication and row normalization with gene expression data -> training

## Complexity
We would end up with 10 datasets (1 is concatenation of RGCN and ComplEx => 11) from graph embeddings.

However, that's only for one version of the graph. The graph can also be built based on averaging rather than binning.  
Thus we have 2 versions (2.10 and 2.11), that will double the number of datasets.

Moreover, as we're multiplying protein embeddings with gene expression values, we were considering normalization, to identify what normalization works best, we tried standard, robust, minmax, log1p, and the option with no normalization applies (so a total of 5 types)

We considered 4 machine learning models: Random Forest, XGBoost, SVM, MLP (scikit learn and pytorch)

Another thing we noticed after performing some robustness analysis on different trainings of a model with same starting point is that results sometimes are fluctuating, making direct comparison between 2 datasets rather biased. That's why we conisdered training 10 models for each (dataset, model type, version, normalization) combination at 10 different seeds, and averaged them to get the end results. Note that each is performing a grid search to find the right hyperparameters that fit the data space we're training on.

In total we have 11 datasets * 2 versions * 5 normalizations * 5 models * 10 seeds (upper bound, since not all datasets will be subject to normalization). Thsi increased complexity wasn't expected at earlier stages of the project, where the topmost dimensions to consider were 3 models, 6 datasets (not counting teh genex expression one used for the traditional ML pipleine) and 2 version. Back then we found it imperative to parametrize this the most possible to ensure a proper automation, and a reliable way to store results, which made this easier to deal with and integrate with the code design.


## Results

### Comparing versions and normalizations

