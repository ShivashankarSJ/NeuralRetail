import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from data_loader import load_data
from preprocessing import preprocess_data
from segmentation import calculate_rfm, segment_customers
from forecasting import prepare_time_series, train_prophet, make_forecast
from churn import prepare_churn_data, train_churn_model
from inventory import abc_analysis, calculate_eoq

# PAGE CONFIGURATION
st.set_page_config(page_title="NeuralRetail - Amdox AI Sales Intelligence", layout="wide")

# Custom CSS for Amdox Branding

st.markdown("""
    <style>
    .main {
        background-color: #ffffff;
    }
    .stButton>button {
        background-color: #E84E1B;
        color: white;
    }
    .css-10trblm {
        color: #E84E1B;
    }
    h1, h2, h3 {
        color: #E84E1B;
    }
    </style>
    """, unsafe_allow_html=True)


# DATA LOADING (CACHED)
@st.cache_data
def get_data():
    path = "data/online_retail_ii/online_retail_II.xlsx"
    df = load_data(path)
    df_clean = preprocess_data(df)
    return df_clean


# APP LAYOUT
st.sidebar.image("https://media.licdn.com/dms/image/v2/D560BAQHkxNcin7pyug/company-logo_200_200/B56ZYEaCS_HEAM-/0/1743830657663/amdox_tech_logo?e=1784160000&v=beta&t=XWs3TlYo408hlbg6KFp4WZkyfvHX_BTE70iRe-YWzrk", width=200) # Placeholder for Amdox logo
st.sidebar.title("NeuralRetail")
st.sidebar.markdown("*AI-Powered Sales Intelligence*")

page = st.sidebar.radio(
    "Navigation",
    ["Executive Overview", "Sales Analytics", "Customer Hub", "Demand Explorer", "Churn Risk", "Inventory Health"]
)

try:
    df = get_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# EXECUTIVE OVERVIEW
if page == "Executive Overview":
    st.title("🚀 Executive Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_rev = df['TotalPrice'].sum()
    total_orders = df['Invoice'].nunique()
    total_cust = df['Customer ID'].nunique()
    avg_order = total_rev / total_orders
    
    col1.metric("Total Revenue", f"${total_rev/1e6:.2f}M")
    col2.metric("Total Orders", f"{total_orders:,}")
    col3.metric("Total Customers", f"{total_cust:,}")
    col4.metric("Avg Order Value", f"${avg_order:.2f}")
    
    st.subheader("Revenue by Country")
    country_rev = df.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False).head(10)
    fig = px.bar(country_rev, x=country_rev.index, y='TotalPrice', color_discrete_sequence=['#E84E1B'])
    st.plotly_chart(fig, use_container_width=True)

# SALES ANALYTICS
elif page == "Sales Analytics":
    st.title("📊 Sales Analytics")
    
    df['MonthYear'] = df['InvoiceDate'].dt.to_period('M').astype(str)
    monthly_sales = df.groupby('MonthYear')['TotalPrice'].sum().reset_index()
    
    st.subheader("Monthly Revenue Trend")
    fig = px.line(monthly_sales, x='MonthYear', y='TotalPrice', color_discrete_sequence=['#F7941D'])
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Top Selling Products")
    top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
    fig = px.bar(top_products, x='Quantity', y=top_products.index, orientation='h', color_discrete_sequence=['#FBBA13'])
    st.plotly_chart(fig, use_container_width=True)

# CUSTOMER HUB
elif page == "Customer Hub":
    st.title("👥 Customer Hub")
    
    rfm = calculate_rfm(df)
    rfm_segmented, kmeans = segment_customers(rfm)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Segments")
        fig = px.scatter(rfm_segmented, x='Recency', y='Frequency', color='Cluster', size='Monetary', 
                         hover_data=['Monetary'], color_continuous_scale='Oranges')
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Segment Distribution")
        seg_dist = rfm_segmented['Cluster'].value_counts()
        fig = px.pie(values=seg_dist.values, names=seg_dist.index, color_discrete_sequence=px.colors.sequential.Oranges)
        st.plotly_chart(fig, use_container_width=True)

# DEMAND EXPLORER
elif page == "Demand Explorer":
    st.title("📈 Demand Explorer")
    
    df_ts = prepare_time_series(df)
    
    with st.spinner("Training forecasting model..."):
        model = train_prophet(df_ts)
        forecast = make_forecast(model, periods=90)
    
    st.subheader("Sales Forecast (Next 90 Days)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_ts['ds'], y=df_ts['y'], name='Actual', line=dict(color='#E84E1B')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast', line=dict(color='#F7941D', dash='dash')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', line_color='rgba(232, 78, 27, 0.2)', showlegend=False))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(232, 78, 27, 0.2)', name='Confidence Interval'))
    
    st.plotly_chart(fig, use_container_width=True)

# CHURN RISK
elif page == "Churn Risk":
    st.title("⚠️ Churn Risk Assessment")
    
    X, y = prepare_churn_data(df)
    
    with st.spinner("Analyzing churn patterns..."):
        # Training every time for demo, in production load a pre-trained model
        model = train_churn_model(X, y)
    
    rfm = calculate_rfm(df)
    probs = model.predict_proba(rfm[['Frequency', 'Monetary']])[:, 1]
    rfm['ChurnProbability'] = probs
    
    st.subheader("High Risk Customers")
    high_risk = rfm.sort_values('ChurnProbability', ascending=False).head(10)
    st.dataframe(high_risk[['Recency', 'Frequency', 'Monetary', 'ChurnProbability']])
    
    fig = px.histogram(rfm, x='ChurnProbability', nbins=20, color_discrete_sequence=['#E84E1B'])
    st.subheader("Churn Probability Distribution")
    st.plotly_chart(fig, use_container_width=True)


# INVENTORY HEALTH
elif page == "Inventory Health":
    st.title("📦 Inventory Health & Optimization")
    
    abc = abc_analysis(df)
    
    st.subheader("ABC Analysis Summary")
    abc_summary = abc['ABC_Category'].value_counts()
    fig = px.pie(values=abc_summary.values, names=abc_summary.index, color_discrete_sequence=['#E84E1B', '#F7941D', '#FBBA13'])
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Recommended Reorder Quantities (EOQ)")
    # Calculate EOQ for top A items
    top_a = abc[abc['ABC_Category'] == 'A'].head(10)
    top_a['EOQ'] = top_a['TotalPrice'].apply(lambda x: calculate_eoq(x/12)) # Simplified demand as monthly
    
    st.table(top_a[['StockCode', 'TotalPrice', 'ABC_Category', 'EOQ']])

st.sidebar.markdown("---")
st.sidebar.info("AMX-DS-2026-04 | NeuralRetail v1.0")
