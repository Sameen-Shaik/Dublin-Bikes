# Fixed Dublin Bikes notebooks

Run in this order:

1. `01_data_cleaning.ipynb`
2. `02_exploratory_analysis.ipynb`
3. `03_forecasting_model.ipynb`
4. `04_feature_ablation.ipynb`

Expected existing input:

- `Data/dataset.csv`

Generated files include:

- `Data/dataset_cleaned.csv`
- `Data/dataset_forecasting_features.csv`
- `Outputs/test_predictions.csv`
- `Outputs/Models/dublin_bikes_xgboost_forecaster.joblib`
- analysis figures under `Outputs/Figures/`

Main corrections:

- chronological train/validation/test split
- real future target
- lag and rolling features
- no full-dataset scaling leakage
- metrics reported in bike units
- persistence and station-hour baselines
- portable paths
- station identity retained
- feature ablation instead of random-split RFE

The notebooks were generated without executing the full pipeline because the repository dataset is not mounted in this runtime. They include assertions and clear errors so they can be run against the repository data locally.
