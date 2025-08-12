# utils.py

import pandas as pd

def load_csv(path):
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(path)

def save_csv(df, path):
    """Save DataFrame to CSV."""
    df.to_csv(path, index=False)
