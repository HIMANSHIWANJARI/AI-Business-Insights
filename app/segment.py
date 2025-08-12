import pandas as pd
from sklearn.cluster import KMeans

def create_rfm(df):
    """Create Recency, Frequency, Monetary table"""
    # Ensure 'Date' column is datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Drop rows where 'Date' could not be converted
    df = df.dropna(subset=['Date'])
    
    # Define latest date for recency calculation as next day after max purchase date
    latest_date = df['Date'].max() + pd.Timedelta(days=1)
    
    # Group by CustomerID to calculate RFM metrics
    rfm = df.groupby('CustomerID').agg({
        'Date': lambda x: (latest_date - x.max()).days,  # Recency: days since last purchase
        'CustomerID': 'count',                            # Frequency: number of purchases
        'TotalAmount': 'sum'                              # Monetary: total spending
    }).rename(columns={
        'Date': 'Recency',
        'CustomerID': 'Frequency',
        'TotalAmount': 'Monetary'
    }).reset_index()
    
    return rfm

def assign_segments(rfm_df, n_clusters=3):
    """Apply KMeans clustering on RFM features"""
    features = ['Recency', 'Frequency', 'Monetary']
    
    # Check if all required features exist
    missing_cols = [col for col in features if col not in rfm_df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in RFM dataframe: {missing_cols}")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    rfm_df['Segment'] = kmeans.fit_predict(rfm_df[features])
    
    return rfm_df
