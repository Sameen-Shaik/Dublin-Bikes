import pandas as pd
import pytest

from src.features import engineer_time_series_features


@pytest.fixture
def mock_bike_data():
    return pd.DataFrame({
        "STATION ID": [42, 42, 42, 42],                                                                
        "TIME": pd.to_datetime([                                                                       
            "2021-11-01 10:00:00",                                                                     
            "2021-11-01 10:05:00",                                                                     
            "2021-11-01 10:10:00",                                                                     
            "2021-11-01 10:15:00"                                                                      
        ]),                                                                                            
        "AVAILABLE BIKES": [10, 8, 5, 2],                                                              
        "BIKE STANDS": [30, 30, 30, 30]  
    })

def test_engineer_time_series_features(mock_bike_data):
    result = engineer_time_series_features(mock_bike_data, forecast_steps=1)
    assert 'HOUR' in result.columns
    assert 'DAY_OF_WEEK' in result.columns
    assert 'IS_WEEKEND' in result.columns

def test_engineer_time_series_features_lag_features(mock_bike_data):
    features_df = engineer_time_series_features(mock_bike_data, forecast_steps=1)
    
    assert "AVAILABLE_BIKES_LAG_1" in features_df.columns
    assert features_df.iloc[-1]["AVAILABLE_BIKES_LAG_1"] == 8.0