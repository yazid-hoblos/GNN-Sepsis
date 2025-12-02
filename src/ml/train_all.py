#!/usr/bin/env python

'''
run this script to train all ML models on all datasets for all specified versions
saves trained models in joblib files in dump/{version}/ folder

> [!TIP]
> run it from the project root as:

    python -m src.ml.train_all -h  
    python -m src.ml.train_all # -- which is equivalent to:  
    python -m src.ml.train_all --versions v2.10 v2.11 --dump-dir ./dump/ --threads 12  
'''

import os
import argparse
# -- package relative imports are essential when running as module --

from src.ml.load_matrix import load_df
from src.ml.model_trainer import MLModel
from src.ml.utils import train_all, set_num_threads

def main():
    parser = argparse.ArgumentParser(description="-- train all ML models on all datasets of all versions")
    parser.add_argument(
        "--versions",
        nargs="+",
        default=["v2.10", "v2.11"],
    )
    parser.add_argument(
        "--dump-dir",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../dump")),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=12,
    )

    args = parser.parse_args()
    set_num_threads(args.threads)
    os.makedirs(args.dump_dir, exist_ok=True)

    for version in args.versions:
        version_dir = os.path.join(args.dump_dir, version)
        os.makedirs(version_dir, exist_ok=True)
        print(f"--- Training all models for version: {version} ---")
        train_all(version=version, cache_dir=version_dir,model_types=['svm','random_forest','xgboost'])

if __name__ == "__main__":
    main()
