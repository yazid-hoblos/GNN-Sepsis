# Plan

- [ ] KG extraction to train prep
- [ ] HAN training (+other graphsage and rgcn)
- [ ] kg results interpretability on go and pathway + down and upregulated gex
- [ ] ml on traditional vs embedding (results analysis + viz) - interpretability of results from ml part 
- [ ] kg exploration/viz (optional)

## ml part

- [x] add version parameter to MLModel
- [x] add weights to fix the imbalance (check overall hyperparameters)
- [x] stratification during split (maybe)
- [x] add random forest
- [x] embed a pytroch model in MLModel instead of sklearn MLPClassifier
- [x] fix non linearly seperable data issue with SVM on RGCN (figure an option to skip in gridsearch wo affecting other models)
- [x] fix directory structure!!!
- [x] ask if can add a parameter in `load_df()` to automatically load only real values - no metadat like age.. (also about indexing the labels) [could be smtg like `prepare_df = lambda df: df.set_index('label').iloc[:, 3:]` - would be interesting to add it as a decorator?]
- [x] standardize when performing protein embeddings generation from gene expression 
- [x] normalize differently (RobustScaler?)
- [x] consider stratifying the split?
- [x] fix caching
- [x] fix logging to work with printing messages in notebook
- [x] ~make attributes private and accessible strictly through setters (no change of global variables)~ bad idea for global class variables maybe consider object ones like dataset_name (not necessary tho - no restrctions on accessing them should be enforced except for y_test and these maybe)
- [x] ~method~ *global attribute DEFAULT_SAVE* to save to a file and load from a file the model (joblib)
- [x] rerun everything again and save in dump/ with versioning (just run `python -m src.ml.train_all`) *ongoing through `python -m src.ml.train_all`*
- [x] added scripts for umap and pca plots in eda/ folder and figures saved in figures/projections/{umap,pca}/
- [x] rerun for normalized version of prot embeddings (using load_matrix_normalized) and save figures in figures/projections_standardized/ and models in dump_standardized/
- [ ] migrate mlp FULLY to pytorch