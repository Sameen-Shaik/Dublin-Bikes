from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.inference import predict_bikes
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pathlib import Path
from src.config import DATA_DIR

app = FastAPI(title="Dublin Bikes ML API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading historical data into memory...")                                                                            
historical_data = pd.read_csv(DATA_DIR / "dataset_forecasting_features.csv")     
historical_data["TIME"] = pd.to_datetime(historical_data["TIME"])  

class BikePredictionRequest(BaseModel):
    STATION_ID: int
    TARGET_TIME: str

@app.get("/health")
def health_check():
    return {"status": "ok", "model_version": "0.1.0"}

                                                                                                                            
                                                                                                                             
@app.post("/predict")                                                                                                      
def predict(request: BikePredictionRequest):                                                                               
    target_dt = pd.to_datetime(request.TARGET_TIME)                                                                        
                                                                                                                            
    station_data = historical_data[historical_data["STATION ID"] == request.STATION_ID]                                    
                                                                                                                            
    past_data = station_data[station_data["TIME"] <= target_dt]                                                             
    past_data = station_data[station_data["TIME"] <= target_dt]                                                            
                                                                                                                            
    if past_data.empty:                                                                                                    
        raise HTTPException(status_code=404, detail="No data found before this time.")                                     
                                                                                                                            
    latest_row = past_data.iloc[-1].to_dict()                                                                              
                                                                                                                            
    features = {                                                                                                           
        "BIKE STANDS": latest_row["BIKE STANDS"],                                                                          
        "LATITUDE": latest_row["LATITUDE"],                                                                                
        "LONGITUDE": latest_row["LONGITUDE"],                                                                              
        "AVAILABLE_BIKES_LAG_1": latest_row["AVAILABLE_BIKES_LAG_1"],                                                      
        "AVAILABLE_BIKES_LAG_2": latest_row["AVAILABLE_BIKES_LAG_2"],                                                      
        "AVAILABLE_BIKES_LAG_4": latest_row["AVAILABLE_BIKES_LAG_4"],                                                      
        "AVAILABLE_BIKES_LAG_8": latest_row["AVAILABLE_BIKES_LAG_8"],                                                      
        "ROLLING_MEAN_4": latest_row["ROLLING_MEAN_4"],                                                                    
        "ROLLING_STD_4": latest_row["ROLLING_STD_4"],                                                                      
        "ROLLING_MEAN_8": latest_row["ROLLING_MEAN_8"],                                                                    
        "MINUTES_SINCE_PREVIOUS": latest_row["MINUTES_SINCE_PREVIOUS"],                                                    
        "HOUR_SIN": latest_row["HOUR_SIN"],                                                                                
        "HOUR_COS": latest_row["HOUR_COS"],                                                                                
        "DOW_SIN": latest_row["DOW_SIN"],                                                                                  
        "DOW_COS": latest_row["DOW_COS"],                                                                                  
        "IS_WEEKEND": latest_row["IS_WEEKEND"],                                                                            
        "STATION ID": latest_row["STATION ID"],                                                                            
        "STATUS": latest_row["STATUS"]                                                                                     
    }                                                                                                                                                                                                                                          
                                                                                                                            
    predicted_bikes = predict_bikes(features)                                                                              
                                                                                                                            
    return {                                                                                                               
        "station_id": request.STATION_ID,                                                                                  
        "selected_time": request.TARGET_TIME,                                                                              
        "bike_capacity": latest_row["BIKE STANDS"],                                                                        
        "historical_available_bikes": latest_row["AVAILABLE BIKES"],                                                       
        "predicted_available_bikes": round(predicted_bikes)                                                                 
    }                                  
