from pathlib import Path                                                                                                   
                                                                                                                               
# Base Paths                                                                                                               
ROOT_DIR = Path(__file__).resolve().parent.parent                                                                          
DATA_DIR = ROOT_DIR / "data"                                                                                               
MODELS_DIR = ROOT_DIR / "outputs" / "models"                                                                               
                                                                                                                               
# File Paths                                                                                                               
RAW_DATA_PATH = DATA_DIR / "dataset.csv"                                                                                   
CLEAN_DATA_PATH = DATA_DIR / "dataset_cleaned.csv"                                                                         
PREDICTIONS_PATH = ROOT_DIR / "outputs" / "test_predictions.csv"                                                           
                                                                                                                               
# Model Parameters                                                                                                         
XGB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.04,
    "max_depth": 6,
    "min_child_weight": 3,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
}                     