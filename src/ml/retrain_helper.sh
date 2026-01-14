#!/bin/bash

if [ ! -d "./dump_seeds/" ]; then
  mkdir ./dump_seeds/
fi

# -- will be leaving threads = 1 bcs training is on n_jobs=-1 => bad oversubscription
# -- will be doing test ratio 30% of initial dataset (163 => 49 test/114 train)

# =========================================================== #
# -- training 3 model types on 1 seed



seed=6
# -- gex once
python -m src.ml.train_all --cache-dir ./dump_seeds/dump_$seed --model-types sklearn_mlp xgboost svm random_forest --normalization none --dataset gene_expression --random-state $seed --threads 1 --split-ratio 0.3 --versions v2.10

# -- all other datasets with 'none' normalization
python -m src.ml.train_all --cache-dir ./dump_seeds/dump_$seed --model-types sklearn_mlp xgboost svm random_forest --normalization none --versions v2.10 v2.11 --dataset RGCN_sample_embeddings Complex_sample_embeddings concatenated_sample_embeddings RGCN_protein_embeddings Complex_protein_embeddings concatenated_protein_embeddings --random-state $seed --threads 1 --split-ratio 0.3

# -- the 3 datasets: Complex_protein_embeddings, RGCN_protein_embeddings, concatenated_protein_embeddings with all norms
for norm in "minmax" "standard" "robust" "log1p";do
  python -m src.ml.train_all --cache-dir ./dump_seeds/dump_$seed --model-types sklearn_mlp xgboost svm random_forest --normalization $norm --versions v2.10 v2.11 --dataset RGCN_protein_embeddings Complex_protein_embeddings concatenated_protein_embeddings --random-state $seed --threads 1 --split-ratio 0.3
done

# =========================================================== #

# -- training 3 model types on 5 seeds
for seed in 0 2 3 4; do
  python -m src.ml.train_all --cache-dir ./dump_seeds/dump_$seed --threads 1 --model-types sklearn_mlp xgboost svm random_forest --normalization none --versions v2.10 --dataset gene_expression --random-state $seed --split-ratio 0.3

  python -m src.ml.train_all --cache-dir ./dump_seeds/dump_$seed --threads 1 --model-types sklearn_mlp xgboost svm random_forest --normalization none --versions v2.10 v2.11 --dataset RGCN_sample_embeddings Complex_sample_embeddings concatenated_sample_embeddings RGCN_protein_embeddings Complex_protein_embeddings concatenated_protein_embeddings --random-state $seed --split-ratio 0.3

  for norm in "minmax" "standard" "robust" "log1p"; do
    python -m src.ml.train_all --cache-dir ./dump_seeds/dump_$seed --threads 1 --model-types sklearn_mlp xgboost svm random_forest --normalization $norm --versions v2.10 v2.11 --dataset RGCN_protein_embeddings Complex_protein_embeddings concatenated_protein_embeddings --random-state $seed --split-ratio 0.3
  done
done

# ========================================================== #

# -- pytorch time!

# -- training 3 model types on 5 seeds
for seed in 1 2 3 4; do
  python -m src.ml.train_all --cache-dir ./dump_seeds/dump_$seed --threads 1 --model-types pytorch_mlp --normalization none --versions v2.10 --dataset gene_expression --random-state $seed --split-ratio 0.3

  python -m src.ml.train_all --cache-dir ./dump_seeds/dump_$seed --threads 1 --model-types pytorch_mlp --normalization none --versions v2.10 v2.11 --dataset RGCN_sample_embeddings Complex_sample_embeddings concatenated_sample_embeddings RGCN_protein_embeddings Complex_protein_embeddings concatenated_protein_embeddings --random-state $seed --split-ratio 0.3

  for norm in "minmax" "standard" "robust" "log1p"; do
    python -m src.ml.train_all --cache-dir ./dump_seeds/dump_$seed --threads 1 --model-types pytorch_mlp --normalization $norm --versions v2.10 v2.11 --dataset RGCN_protein_embeddings Complex_protein_embeddings concatenated_protein_embeddings --random-state $seed --split-ratio 0.3
  done
done