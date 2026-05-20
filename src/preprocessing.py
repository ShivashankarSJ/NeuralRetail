import pandas as pd

def preprocess_data(df):
    """Cleans and prepares the retail data."""
    # Drop rows without Customer ID
    df = df.dropna(subset=['Customer ID'])
    
    # Convert Customer ID to int
    df['Customer ID'] = df['Customer ID'].astype(int)
    
    # Remove cancellations (Invoice starts with 'C')
    df = df[~df['Invoice'].astype(str).str.startswith('C')]
    
    # Remove negative or zero quantity/price
    df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
    
    # Calculate TotalPrice
    df['TotalPrice'] = df['Quantity'] * df['Price']
    
    # Convert InvoiceDate to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    return df

if __name__ == "__main__":
    from data_loader import load_data
    path = "data/online_retail_ii/online_retail_II.xlsx"
    df = load_data(path)
    df_clean = preprocess_data(df)
    print(f"Cleaned data has {df_clean.shape[0]} rows.")
    print(df_clean.head())
