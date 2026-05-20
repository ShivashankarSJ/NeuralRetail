# NeuralRetail Model Cards

## 1. Demand Forecasting Model
- **Algorithm:** Facebook Prophet
- **Task:** Daily Sales (Revenue) Forecasting
- **Features:** Historical daily revenue, yearly/weekly seasonality.
- **Goal:** MAPE ≤ 10%
- **Intended Use:** Supply chain planning and stock level optimization.
- **Limitations:** Does not account for sudden external shocks (e.g., pandemics) unless explicitly added as regressors.

## 2. Customer Segmentation Model
- **Algorithm:** K-Means Clustering + RFM Analysis
- **Task:** Unsupervised Behavioral Grouping
- **Metrics:** Recency, Frequency, Monetary (RFM)
- **Clusters:** 4 (Optimized for business personas)
- **Intended Use:** Personalized marketing and loyalty program design.
- **Limitations:** Cluster definitions may drift as customer behavior changes over time.

## 3. Churn Prediction Model
- **Algorithm:** XGBoost Classifier
- **Task:** Binary Classification (Churn vs. No-Churn)
- **Features:** Frequency, Monetary (RFM derived)
- **Target:** Inactivity > 90 days (Proxy for churn)
- **Threshold:** 0.5 (Adjustable based on cost of false positives/negatives)
- **Intended Use:** Proactive customer retention interventions.
- **Limitations:** Based on historical behavioral patterns; does not capture qualitative reasons for churn.

## 4. Inventory Optimization
- **Methods:** ABC Analysis (Pareto Principle) & EOQ (Economic Order Quantity)
- **Task:** Revenue-based prioritization and optimal reorder calculation.
- **Goal:** 30–50% Stockout Reduction.
- **Intended Use:** Warehouse management and procurement automation.
- **Limitations:** EOQ assumes constant demand and lead times.
