import numpy as np                                                                                                         
import pandas as pd                                                                                                        
                                                                                                                            
def clean_bikes_data(df_raw: pd.DataFrame) -> pd.DataFrame:                                                                
    """                                                                                                                    
    Standardizes schema, applies business rules, and cleans the raw Dublin Bikes dataframe.                                
    """                                                                                                                    
    df = df_raw.copy()                                                                                                     
                                                                                                                            
    # 1. Standardize Schema                                                                                                
    df.columns = [column.strip().upper().replace("_", " ") for column in df.columns]                                       
    rename_map = {                                                                                                         
        "STATIONID": "STATION ID",                                                                                         
        "STATION ID": "STATION ID",                                                                                        
        "LASTUPDATED": "LAST UPDATED",                                                                                     
        "LAST UPDATED": "LAST UPDATED",                                                                                    
        "BIKESTANDS": "BIKE STANDS",                                                                                       
        "BIKE STANDS": "BIKE STANDS",                                                                                      
        "AVAILABLEBIKESTANDS": "AVAILABLE BIKE STANDS",                                                                    
        "AVAILABLE BIKE STANDS": "AVAILABLE BIKE STANDS",                                                                  
        "AVAILABLEBIKES": "AVAILABLE BIKES",                                                                               
        "AVAILABLE BIKES": "AVAILABLE BIKES",                                                                              
        "LAT": "LATITUDE",                                                                                                 
        "LNG": "LONGITUDE",                                                                                                
        "LON": "LONGITUDE",                                                                                                
    }                                                                                                                      
    df = df.rename(columns={column: rename_map.get(column.replace(" ", ""), column) for column in df.columns})             
                                                                                                                            
    # 2. Parse types and remove unusable records                                                                           
    for column in ["TIME", "LAST UPDATED"]:                                                                                
        if column in df.columns:                                                                                           
            df[column] = pd.to_datetime(df[column], errors="coerce", utc=False)                                            
                                                                                                                            
    numeric_columns = [                                                                                                    
        "STATION ID", "BIKE STANDS", "AVAILABLE BIKE STANDS",                                                              
        "AVAILABLE BIKES", "LATITUDE", "LONGITUDE"                                                                         
    ]                                                                                                                      
    for column in numeric_columns:                                                                                         
        df[column] = pd.to_numeric(df[column], errors="coerce")                                                            
                                                                                                                            
    df = df.dropna(subset=[                                                                                                
        "STATION ID", "TIME", "BIKE STANDS",                                                                               
        "AVAILABLE BIKE STANDS", "AVAILABLE BIKES",                                                                        
        "LATITUDE", "LONGITUDE"                                                                                            
    ])                                                                                                                     
    df = df.drop_duplicates()                                                                                              
                                                                                                                            
    # 3. Apply business-rule validation                                                                                    
    rule_masks = {                                                                                                         
        "negative_capacity": df["BIKE STANDS"] < 0,                                                                        
        "negative_available_bikes": df["AVAILABLE BIKES"] < 0,                                                             
        "negative_available_stands": df["AVAILABLE BIKE STANDS"] < 0,                                                      
        "bikes_above_capacity": df["AVAILABLE BIKES"] > df["BIKE STANDS"],                                                 
        "stands_above_capacity": df["AVAILABLE BIKE STANDS"] > df["BIKE STANDS"],                                          
    }                                                                                                                      
    invalid_mask = np.logical_or.reduce(list(rule_masks.values()))                                                         
    df = df.loc[~invalid_mask].copy()                                                                                      
                                                                                                                            
    # 4. Sensor consistency feature                                                                                        
    df["CAPACITY_DIFFERENCE"] = (                                                                                          
        df["BIKE STANDS"] - (df["AVAILABLE BIKES"] + df["AVAILABLE BIKE STANDS"])                                          
    )                                                                                                                      
                                                                                                                            
    # 5. Final casting and sorting                                                                                         
    df["STATION ID"] = df["STATION ID"].astype("int64")                                                                    
    df["BIKE STANDS"] = df["BIKE STANDS"].astype("int64")                                                                  
    df["AVAILABLE BIKE STANDS"] = df["AVAILABLE BIKE STANDS"].astype("int64")                                              
    df["AVAILABLE BIKES"] = df["AVAILABLE BIKES"].astype("int64")                                                          
    df["STATUS"] = df["STATUS"].astype(str).str.strip().str.upper()                                                        
                                                                                                                            
    df = df.sort_values(["TIME", "STATION ID"]).reset_index(drop=True)                                                     
                                                                                                                            
    return df


def engineer_time_series_features(df_clean: pd.DataFrame, forecast_steps: int = 1) -> pd.DataFrame:                        
    """                                                                                                                    
    Creates forecasting targets, lag features, rolling windows, and cyclical time features.                                
    """                                                                                                                    
    df = df_clean.copy()                                                                                                   
    grouped = df.groupby("STATION ID", group_keys=False)                                                                   
                                                                                                                            
    # 1. Target generation                                                                                                 
    df["TARGET_AVAILABLE_BIKES"] = grouped["AVAILABLE BIKES"].shift(-forecast_steps)                                       
                                                                                                                            
    # 2. Lag features                                                                                                      
    for lag in [1, 2, 4, 8]:                                                                                               
        df[f"AVAILABLE_BIKES_LAG_{lag}"] = grouped["AVAILABLE BIKES"].shift(lag)                                           
                                                                                                                            
    # 3. Rolling windows (shifted by 1 to prevent leakage)                                                                 
    df["ROLLING_MEAN_4"] = grouped["AVAILABLE BIKES"].transform(                                                           
        lambda series: series.shift(1).rolling(4, min_periods=1).mean()                                                    
    )                                                                                                                      
    df["ROLLING_STD_4"] = grouped["AVAILABLE BIKES"].transform(                                                            
        lambda series: series.shift(1).rolling(4, min_periods=2).std()                                                     
    )                                                                                                                      
    df["ROLLING_MEAN_8"] = grouped["AVAILABLE BIKES"].transform(                                                           
        lambda series: series.shift(1).rolling(8, min_periods=1).mean()                                                    
    )                                                                                                                      
                                                                                                                            
    # 4. Temporal features                                                                                                 
    df["HOUR"] = df["TIME"].dt.hour                                                                                        
    df["DAY_OF_WEEK"] = df["TIME"].dt.dayofweek                                                                            
    df["IS_WEEKEND"] = df["DAY_OF_WEEK"].isin([5, 6]).astype(int)                                                          
                                                                                                                            
    # 5. Cyclical encoding                                                                                                 
    df["HOUR_SIN"] = np.sin(2 * np.pi * df["HOUR"] / 24)                                                                   
    df["HOUR_COS"] = np.cos(2 * np.pi * df["HOUR"] / 24)                                                                   
    df["DOW_SIN"] = np.sin(2 * np.pi * df["DAY_OF_WEEK"] / 7)                                                              
    df["DOW_COS"] = np.cos(2 * np.pi * df["DAY_OF_WEEK"] / 7)                                                              
                                                                                                                            
    # 6. Time delta feature                                                                                                
    df["MINUTES_SINCE_PREVIOUS"] = (                                                                                       
        grouped["TIME"].diff().dt.total_seconds().div(60)                                                                  
    )                                                                                                                      
                                                                                                                            
    # 7. Drop rows with NaN targets or core lags                                                                           
    df = df.dropna(subset=[                                                                                                
        "TARGET_AVAILABLE_BIKES",                                                                                          
        "AVAILABLE_BIKES_LAG_1",                                                                                           
        "AVAILABLE_BIKES_LAG_2",                                                                                           
    ]).copy()                                                                                                              
                                                                                                                            
    return df                     
    
