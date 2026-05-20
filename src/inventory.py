import pandas as pd
import numpy as np

def abc_analysis(df):
    """Classifies products into A, B, and C categories based on revenue."""
    product_revenue = df.groupby('StockCode')['TotalPrice'].sum().sort_values(ascending=False).reset_index()
    product_revenue['CumulativeRevenue'] = product_revenue['TotalPrice'].cumsum()
    total_revenue = product_revenue['TotalPrice'].sum()
    product_revenue['RevenuePercentage'] = (product_revenue['CumulativeRevenue'] / total_revenue) * 100
    
    def classify(pct):
        if pct <= 80: return 'A'
        elif pct <= 95: return 'B'
        else: return 'C'
        
    product_revenue['ABC_Category'] = product_revenue['RevenuePercentage'].apply(classify)
    return product_revenue

def calculate_eoq(demand, ordering_cost=50, holding_cost=2):
    """Calculates Economic Order Quantity."""
    # EOQ = sqrt((2 * D * S) / H)
    if holding_cost == 0: return 0
    eoq = np.sqrt((2 * demand * ordering_cost) / holding_cost)
    return eoq

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.join(os.getcwd(), 'src'))
    from data_loader import load_data
    from preprocessing import preprocess_data
    
    path = "data/online_retail_ii/online_retail_II.xlsx"
    df = load_data(path)
    df_clean = preprocess_data(df)
    abc = abc_analysis(df_clean)
    print("ABC Analysis Sample:")
    print(abc.head())
