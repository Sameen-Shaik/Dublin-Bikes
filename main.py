                                                                                                                        
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, RAW_DATA_PATH, XGB_PARAMS
from src.data import chronological_split, load_raw_data
from src.features import clean_bikes_data, engineer_time_series_features
from src.models import build_pipeline, evaluate_predictions, save_model_bundle


def run_pipeline():                                                                                                        
    print("1. Loading Data...")                                                                                            
    raw_df = load_raw_data(RAW_DATA_PATH)                                                                                  
                                                                                                                            
    print("2. Cleaning and Engineering Features...")                                                                       
    clean_df = clean_bikes_data(raw_df)                                                                                    
    features_df = engineer_time_series_features(clean_df, forecast_steps=1)                                                
                                                                                                                            
    print("3. Splitting Data (Chronological)...")                                                                          
    train_df, val_df, test_df = chronological_split(features_df)                                                           
                                                                                                                            
    # Combine train and validation for final training (like you did in notebook 3!)                                        
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)                                                        
                                                                                                                            
    # Define features                                                                                                      
    numeric_features = [                                                                                                   
        "BIKE STANDS", "LATITUDE", "LONGITUDE", "AVAILABLE_BIKES_LAG_1",                                                   
        "AVAILABLE_BIKES_LAG_2", "AVAILABLE_BIKES_LAG_4", "AVAILABLE_BIKES_LAG_8",                                         
        "ROLLING_MEAN_4", "ROLLING_STD_4", "ROLLING_MEAN_8", "MINUTES_SINCE_PREVIOUS",                                     
        "HOUR_SIN", "HOUR_COS", "DOW_SIN", "DOW_COS", "IS_WEEKEND"                                                         
    ]                                                                                                                      
    categorical_features = ["STATION ID", "STATUS"]                                                                        
    target_column = "TARGET_AVAILABLE_BIKES"                                                                               
                                                                                                                            
    X_train_val = train_val_df[numeric_features + categorical_features]                                                    
    y_train_val = train_val_df[target_column]                                                                              
    X_test = test_df[numeric_features + categorical_features]                                                              
    y_test = test_df[target_column]                                                                                        
                                                                                                                            
    print("4. Training Pipeline...")                                                                                       
    pipeline = build_pipeline(numeric_features, categorical_features, XGB_PARAMS)                                          
    pipeline.fit(X_train_val, y_train_val)                                                                                 
                                                                                                                            
    print("5. Evaluating on Test Set...")                                                                                  
    test_pred = pipeline.predict(X_test)                                                                                   
                                                                                                                            
    # Clip predictions to valid bike stands capacity                                                                       
    test_pred = np.clip(test_pred, 0, test_df["BIKE STANDS"].to_numpy())                                                   
                                                                                                                            
    metrics = evaluate_predictions("XGBoost", y_test, test_pred)                                                           
    print(f"Test Results: {metrics}")                                                                                      
                                                                                                                            
    print("6. Saving Model Bundle...")                                                                                     
    MODELS_DIR.mkdir(parents=True, exist_ok=True)                                                                          
    bundle_metadata = {                                                                                                    
        "feature_columns": numeric_features + categorical_features,                                                        
        "numeric_features": numeric_features,                                                                              
        "categorical_features": categorical_features,                                                                      
        "test_metrics": [metrics]                                                                                          
    }
    save_model_bundle(pipeline, bundle_metadata, MODELS_DIR / "dublin_bikes_xgboost.joblib")
    print("Pipeline Complete!")

if __name__ == "__main__":
    run_pipeline()