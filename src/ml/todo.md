
- [ ] fix directory structure!!!
- [ ] ask if can add a parameter in `load_df()` to automatically load only real values - no metadat like age.. (also about indexing the labels) [could be smtg like `prepare_df = lambda df: df.set_index('label').iloc[:, 3:]` - would be interesting to add it as a decorator?]
- [ ] add random forest
- [ ] add weights to fix the imbalance (check overall hyperparameters)
- [ ] fix caching
- [ ] fix logging to work with printing messages in notebook
- [ ] make attributes private and accessible strictly through setters (no change of global variables)
- [ ] method to save to a file and load from a file the model (joblib)