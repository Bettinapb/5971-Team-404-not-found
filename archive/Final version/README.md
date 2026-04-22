# Final Version

This folder contains a fresh, leakage-safe rebuild of the final Instacart pipeline.

## Files

- [DATA_PREPROCESSING_AND_FEATURE_ENGINEERING.md](/Users/bettinabopeng/Documents/GitHub/5971/Final%20version/DATA_PREPROCESSING_AND_FEATURE_ENGINEERING.md)
  Method document for raw data ingestion, preprocessing, candidate construction, and fold-wise feature engineering rules.
- [leakage_safe_preprocess.py](/Users/bettinabopeng/Documents/GitHub/5971/Final%20version/leakage_safe_preprocess.py)
  Downloads the raw dataset through `kagglehub` when needed, preprocesses the raw tables, and writes a reusable base artifact bundle under `Final version/artifacts/`.
- [leakage_safe_preprocess.ipynb](/Users/bettinabopeng/Documents/GitHub/5971/Final%20version/leakage_safe_preprocess.ipynb)
  Notebook version of the preprocessing step for cloud/Jupyter workflows.
- [final_experiments.ipynb](/Users/bettinabopeng/Documents/GitHub/5971/Final%20version/final_experiments.ipynb)
  Single notebook for nested tuning, final grouped evaluation, ablation, segment analysis, and top-k analysis.
- [make_final_experiments_notebook.py](/Users/bettinabopeng/Documents/GitHub/5971/Final%20version/make_final_experiments_notebook.py)
  Regenerates the notebook if you want to edit the template programmatically.
- [make_preprocess_notebook.py](/Users/bettinabopeng/Documents/GitHub/5971/Final%20version/make_preprocess_notebook.py)
  Regenerates the preprocessing notebook if you want to edit that template programmatically.

## Recommended run order

1. Run the preprocessing notebook:

- [leakage_safe_preprocess.ipynb](/Users/bettinabopeng/Documents/GitHub/5971/Final%20version/leakage_safe_preprocess.ipynb)

2. Open and run [final_experiments.ipynb](/Users/bettinabopeng/Documents/GitHub/5971/Final%20version/final_experiments.ipynb) on your cloud machine.

## Important design choice

The preprocessing script only writes base tables that are safe to materialize globally.

It does **not** precompute leakage-prone learned features such as:

- KMeans clusters
- Apriori rule hits
- ALS latent features
- cluster-product target rates
- fold-specific user/product target statistics

All of those are rebuilt inside the notebook within grouped train folds only.
