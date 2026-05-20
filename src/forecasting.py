import pandas as pd
from prophet import Prophet

def prepare_time_series(df):
    """Prepares the data for Prophet (columns 'ds' and 'y')."""
    df_ts = df.groupby('InvoiceDate')['TotalPrice'].sum().reset_index()
    df_ts.columns = ['ds', 'y']
    return df_ts

def train_prophet(df_ts):
    """Trains a Prophet model."""
    model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=True)
    model.fit(df_ts)
    return model

def make_forecast(model, periods=30):
    """Makes a forecast for the specified number of periods (days)."""
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.join(os.getcwd(), 'src'))
    from data_loader import load_data
    from preprocessing import preprocess_data
    
    path = "data/online_retail_ii/online_retail_II.xlsx"
    df = load_data(path)
    df_clean = preprocess_data(df)
    df_ts = prepare_time_series(df_clean)
    model = train_prophet(df_ts)
    forecast = make_forecast(model)
    print("Forecast Sample:")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
