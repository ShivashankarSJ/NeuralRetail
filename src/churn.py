import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

def prepare_churn_data(df, churn_threshold_days=90):
    """Prepares features and labels for churn prediction."""
    # We define churn based on the last transaction in the dataset
    max_date = df['InvoiceDate'].max()
    
    # Calculate RFM as features
    rfm = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (max_date - x.max()).days,
        'Invoice': lambda x: x.nunique(),
        'TotalPrice': lambda x: x.sum()
    })
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    
    # Define churn: 1 if Recency > threshold, else 0
    # Note: This is a simplified proxy for demonstration
    rfm['Churn'] = (rfm['Recency'] > churn_threshold_days).astype(int)
    
    # Use Frequency and Monetary as features to predict Churn
    X = rfm[['Frequency', 'Monetary']]
    y = rfm['Churn']
    
    return X, y

def train_churn_model(X, y):
    """Trains an XGBoost model for churn prediction."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("Churn Model Evaluation:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    
    return model

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.join(os.getcwd(), 'src'))
    from data_loader import load_data
    from preprocessing import preprocess_data
    
    path = "data/online_retail_ii/online_retail_II.xlsx"
    df = load_data(path)
    df_clean = preprocess_data(df)
    X, y = prepare_churn_data(df_clean)
    model = train_churn_model(X, y)
