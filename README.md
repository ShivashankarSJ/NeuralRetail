# NeuralRetail: AI-Powered Sales Intelligence Platform

## Project Overview
NeuralRetail is an end-to-end AI-powered sales intelligence platform designed for enterprise clients in retail, FMCG, and e-commerce. The platform provides accurate demand forecasts, customer intelligence, churn predictions, and inventory optimization recommendations.

**Project Code:** AMX-DS-2026-04
**Domain:** Data Science & Analytics
**Prepared for:** Amdox – Engineering Division

## Key Features
- **Executive Overview:** High-level KPIs and revenue trends.
- **Sales Analytics:** In-depth analysis of sales performance and top products.
- **Customer Hub:** RFM-based customer segmentation and behavioral clustering.
- **Demand Explorer:** Time-series forecasting using Prophet for future demand prediction.
- **Churn Risk Assessment:** XGBoost-powered churn prediction to identify at-risk customers.
- **Inventory Health:** ABC analysis and Economic Order Quantity (EOQ) optimization.

## Tech Stack
- **Language:** Python 3.12
- **Data Processing:** Pandas, NumPy, Polars
- **Machine Learning:** Scikit-learn, XGBoost, LightGBM
- **Forecasting:** Prophet
- **Dashboard:** Streamlit, Plotly
- **Containerization:** Docker

## Project Structure
```
NeuralRetail/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container configuration
├── src/                # Source code modules
│   ├── data_loader.py   # Data ingestion
│   ├── preprocessing.py # Data cleaning
│   ├── segmentation.py  # RFM & Clustering
│   ├── forecasting.py   # Prophet forecasting
│   ├── churn.py         # XGBoost churn model
│   └── inventory.py     # ABC & EOQ logic
├── notebooks/          # Jupyter notebooks for EDA and testing
├── docs/               # Original architecture and reference docs
└── data/               # Project datasets (unzipped)
```

## Setup Instructions

### Prerequisites
- Python 3.12
- Docker (optional, for containerized run)

### Local Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

### Docker Deployment
1. Build the Docker image:
   ```bash
   docker build -t neuralretail .
   ```
2. Run the container:
   ```bash
   docker run -p 8501:8501 neuralretail
   ```

## Evaluation Metrics
- **Demand MAPE:** ≤ 10%
- **Churn AUC-ROC:** ≥ 0.90
- **Stockout Reduction:** 30–50% Target

---
*Crafted with precision and modern data science principles for Amdox Technologies.*
