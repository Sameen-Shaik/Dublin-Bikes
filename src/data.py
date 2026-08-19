
import numpy as np
import pandas as pd

from src.config import RAW_DATA_PATH


def load_raw_data(filepath: str = RAW_DATA_PATH) -> pd.DataFrame:                                                          
    """Loads the raw Dublin Bikes dataset."""                                                                              
    df = pd.read_csv(filepath)                                                                                             
    return df                                                                                                              
                                                                                                                            
def chronological_split(df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15) -> tuple[pd.DataFrame, pd.   
DataFrame, pd.DataFrame]:                                                                                                    
    """                                                                                                                    
    Splits the dataset temporally into train, validation, and test sets.                                                   
    Ensures that validation strictly follows train, and test strictly follows validation.                                  
    """                                                                                                                    
    unique_times = np.array(sorted(df["TIME"].unique()))                                                                   
                                                                                                                            
    train_idx = int(len(unique_times) * train_ratio)                                                                       
    val_idx = int(len(unique_times) * (train_ratio + val_ratio))                                                           
                                                                                                                            
    train_cutoff = unique_times[train_idx]                                                                                 
    validation_cutoff = unique_times[val_idx]                                                                              
                                                                                                                            
    train_df = df[df["TIME"] < train_cutoff].copy()                                                                        
    validation_df = df[(df["TIME"] >= train_cutoff) & (df["TIME"] < validation_cutoff)].copy()                             
    test_df = df[df["TIME"] >= validation_cutoff].copy()                                                                   
                                                                                                                            
    return train_df, validation_df, test_df 