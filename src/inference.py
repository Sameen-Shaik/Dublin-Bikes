
import joblib
import pandas as pd

from src.config import MODELS_DIR

MODEL_BUNDLE_PATH = MODELS_DIR / "dublin_bikes_xgbregressor.joblib"

def load_model():
    if not MODEL_BUNDLE_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_BUNDLE_PATH}. Run ")
    return joblib.load(MODEL_BUNDLE_PATH)


# "bundle" (or a Model Artifact) is a term used in production MLOps to describe a packaged model with all its dependencies and metadata.
# It contains the pipeline, feature columns, and metrics.

def _get_model():                                                                                                        
    """Lazy loader to prevent import-time crashes in CI"""                                                               
    global _bundle, _pipeline, _feature_cols                                                                             
    if _pipeline is None:                                                                                                
        _bundle = load_model()                                                                                           
        _pipeline = _bundle["pipeline"]                                                                                  
        _feature_cols = _bundle["feature_columns"]                                                                       
    return _pipeline, _feature_cols   

def predict_bikes(features: dict) -> float:
    """
    Takes a dictionary of features, converts to a dataframe, and predicts
    """
    pipeline, feature_cols = _get_model() 
    
    input_df = pd.DataFrame([features], columns=feature_cols)
    prediction = pipeline.predict(input_df)[0]
    return float(max(0.0, prediction))
    

if __name__ == "__main__":                                                                                                                                               
    # Incremental Testing: Proving it works before touching the API                                                                                                      
    print("Testing Inference Bridge...")                                                                                                                                 
                                                                                                                                                                            
    # A dummy dictionary with all 16 required features                                                                                                                   
    dummy_input = {                                                                                                                                                      
        "BIKE STANDS": 30, "LATITUDE": 53.33, "LONGITUDE": -6.24,                                                                                                        
        "AVAILABLE_BIKES_LAG_1": 15, "AVAILABLE_BIKES_LAG_2": 16,                                                                                                        
        "AVAILABLE_BIKES_LAG_4": 15, "AVAILABLE_BIKES_LAG_8": 14,                                                                                                        
        "ROLLING_MEAN_4": 15.5, "ROLLING_STD_4": 1.2, "ROLLING_MEAN_8": 15.1,                                                                                            
        "MINUTES_SINCE_PREVIOUS": 5, "HOUR_SIN": 0.5, "HOUR_COS": -0.86,                                                                                                 
        "DOW_SIN": 0.0, "DOW_COS": 1.0, "IS_WEEKEND": 0,                                                                                                                 
        "STATION ID": 42, "STATUS": "OPEN"                                                                                                                               
    }                                                                                                                                                                    
                                                                                                                                                                            
    result = predict_bikes(dummy_input)                                                                                                                                  
    print(f"Success! Predicted available bikes: {result:.2f}")     