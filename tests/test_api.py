from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture                                                                                                          
def client():                                                                                                            
    with TestClient(app) as c:                                                                                           
        yield c                                                                                                          
                                                                                                                            
@patch("api.main.get_historical_data")                                                                                   
@patch("api.main.predict_bikes")                                                                                         
def test_predict_endpoint_valid_request(mock_predict, mock_get_data, client):                                            
    # 1. Fake the heavy CSV data                                                                                         
    fake_df = pd.DataFrame([{                                                                                            
        "STATION ID": 42,                                                                                                
        "TIME": pd.to_datetime("2021-11-15T10:00"),                                                                      
        "BIKE STANDS": 30, "LATITUDE": 53.33, "LONGITUDE": -6.24,                                                        
        "AVAILABLE_BIKES_LAG_1": 15, "AVAILABLE_BIKES_LAG_2": 16,                                                        
        "AVAILABLE_BIKES_LAG_4": 15, "AVAILABLE_BIKES_LAG_8": 14,                                                        
        "ROLLING_MEAN_4": 15.5, "ROLLING_STD_4": 1.2, "ROLLING_MEAN_8": 15.1,                                            
        "MINUTES_SINCE_PREVIOUS": 5, "HOUR_SIN": 0.5, "HOUR_COS": -0.86,                                                 
        "DOW_SIN": 0.0, "DOW_COS": 1.0, "IS_WEEKEND": 0, "STATUS": "OPEN",                                               
        "AVAILABLE BIKES": 15                                                                                            
    }])                                                                                                                  
    mock_get_data.return_value = fake_df                                                                                 
                                                                                                                            
    # 2. Fake the heavy ML Model                                                                                         
    mock_predict.return_value = 20.0                                                                                     
                                                                                                                            
    # 3. Test the API logic!                                                                                             
    payload = {                                                                                                          
        "STATION_ID": 42,
        "TARGET_TIME": "2021-11-15T12:00"
    }

    response = client.post('/predict', json=payload)

    assert response.status_code == 200
    assert response.json()["predicted_available_bikes"] == 20