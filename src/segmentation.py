import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import datetime as dt

def calculate_rfm(df):
    """Calculates RFM metrics for each customer."""
    today_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
    
    rfm = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (today_date - x.max()).days,
        'Invoice': lambda x: x.nunique(),
        'TotalPrice': lambda x: x.sum()
    })
    
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    return rfm

def segment_customers(rfm, n_clusters=4):
    """Segments customers using K-Means clustering."""
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    return rfm, kmeans

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.join(os.getcwd(), 'src'))
    from data_loader import load_data
    from preprocessing import preprocess_data
    
    path = "data/online_retail_ii/online_retail_II.xlsx"
    df = load_data(path)
    df_clean = preprocess_data(df)
    rfm = calculate_rfm(df_clean)
    rfm_segmented, model = segment_customers(rfm)
    print("Customer Segmentation Sample:")
    print(rfm_segmented.head())
