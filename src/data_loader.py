import pandas as pd
import os

def load_data(file_path):
    """Loads the Excel data and returns a DataFrame."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Online Retail II has two sheets: 'Year 2009-2010' and 'Year 2010-2011'
    xl = pd.ExcelFile(file_path)
    df1 = xl.parse('Year 2009-2010')
    df2 = xl.parse('Year 2010-2011')
    
    df = pd.concat([df1, df2], ignore_index=True)
    return df

if __name__ == "__main__":
    path = "data/online_retail_ii/online_retail_II.xlsx"
    df = load_data(path)
    print(f"Loaded data with {df.shape[0]} rows and {df.shape[1]} columns.")
    print(df.head())
