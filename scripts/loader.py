# src/loader.py
import pandas as pd
from typing import Optional

def load_data(filepath: str) -> Optional[pd.DataFrame]:
    """
    Loads the dataset from the specified filepath.
    Returns None if file is not found.
    """
    try:
        df = pd.read_csv(filepath)
        print("Data loaded successfully.")
        
        if 'TransactionStartTime' in df.columns:
            df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
            
        return df
    except FileNotFoundError:
        print(f"File not found at {filepath}. Please check the path.")
        return None