import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


def build_pipeline(numeric_features: list[str], categorical_features: list[str], xgb_params: dict) -> Pipeline:            
    """                                                                                                                    
    Builds the complete scikit-learn pipeline including imputation, encoding, and the XGBoost model.                       
    """                                                                                                                    
    numeric_pipeline = Pipeline([                                                                                          
        ("imputer", SimpleImputer(strategy="median")),                                                                     
    ])                                                                                                                     
                                                                                                                            
    categorical_pipeline = Pipeline([                                                                                      
        ("imputer", SimpleImputer(strategy="most_frequent")),                                                              
        ("encoder", OneHotEncoder(handle_unknown="ignore")),                                                               
    ])                                                                                                                     
                                                                                                                            
    preprocessor = ColumnTransformer([                                                                                     
        ("numeric", numeric_pipeline, numeric_features),                                                                   
        ("categorical", categorical_pipeline, categorical_features),                                                       
    ])                                                                                                                     
                                                                                                                            
    model = XGBRegressor(**xgb_params)                                                                                     
                                                                                                                            
    pipeline = Pipeline([                                                                                                  
        ("preprocessor", preprocessor),                                                                                    
        ("model", model),                                                                                                  
    ])                                                                                                                     
                                                                                                                            
    return pipeline                                                                                                        
                                                                                                                            
def evaluate_predictions(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:                                       
    """Calculates standard regression metrics."""                                                                          
    return {                                                                                                               
        "model": name,                                                                                                     
        "MAE": mean_absolute_error(y_true, y_pred),                                                                        
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,                                                                 
        "MedianAE": median_absolute_error(y_true, y_pred),                                                                 
        "R2": r2_score(y_true, y_pred),                                                                                    
    }                                                                                                                      
                                                                                                                            
def save_model_bundle(pipeline: Pipeline, metadata: dict, filepath: str):                                                  
    """Saves the trained pipeline and associated metadata to disk."""                                                      
    bundle = {"pipeline": pipeline, **metadata}                                                                            
    joblib.dump(bundle, filepath) 