import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests

st.set_page_config(page_title="ML Research Dashboard", layout="wide")

st.title("Task-Oriented Benchmarking: Traditional ML in Stock Markets")
st.markdown("""
This dashboard evaluates traditional ML models based on their **Financial Performance** rather than just statistical accuracy.
""")

# Sidebar
st.sidebar.header("Configuration")
ticker = st.sidebar.selectbox("Select Ticker", ["AAPL", "GOOGL", "MSFT", "TSLA"])
benchmarking_file = f"data/processed/{ticker}_benchmarking_results.csv"

# Load Benchmarking Results
if os.path.exists(benchmarking_file):
    results_df = pd.read_csv(benchmarking_file, index_col=0)
    
    st.header(f"Model Benchmarking Results: {ticker}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Statistical vs. Financial Metrics")
        st.dataframe(results_df.style.highlight_max(axis=0, subset=['Accuracy', 'Sharpe Ratio', 'Total Return']))
        
    with col2:
        st.subheader("Total Return Comparison")
        fig, ax = plt.subplots()
        sns.barplot(x=results_df.index, y=results_df['Total Return'], ax=ax, palette="viridis")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
    st.divider()
    
    # Financial Metrics Comparison
    st.subheader("Metric Distribution")
    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(x=results_df.index, y=results_df['Sharpe Ratio'], ax=ax2[0], palette="magma")
    ax2[0].set_title("Sharpe Ratio")
    sns.barplot(x=results_df.index, y=results_df['Max Drawdown'], ax=ax2[1], palette="rocket")
    ax2[1].set_title("Max Drawdown")
    st.pyplot(fig2)

else:
    st.warning(f"Benchmarking results for {ticker} not found. Please run the training pipeline first.")
    st.info("Run: `python src/train_benchmark.py` after ensuring raw data exists.")

st.divider()

# Live Prediction Simulation (via API)
st.header("Live Analysis Core (FastAPI Integration)")
st.markdown("This section simulates a call to the FastAPI inference engine using the most recent market data.")

if st.button("Run Inference Analysis"):
    # Mock data ingestion for simulation
    raw_path = f"data/raw/{ticker}_2020-01-01_2026-05-11.parquet" # Adjust based on actual file
    if os.path.exists(raw_path):
        df = pd.read_parquet(raw_path).tail(100) # Send recent 100 days
        ohlcv_list = df.reset_index().to_dict(orient='records')
        # Convert timestamp to string
        for item in ohlcv_list:
            if 'Date' in item:
                item['Date'] = str(item['Date'])
        
        try:
            # Note: API must be running for this to work
            # In a real setup, we'd use a real URL
            # For simulation in this demo, we handle the error gracefully
            response = requests.post("http://localhost:8000/analyze", 
                                     json={"ticker": ticker, "ohlcv_data": ohlcv_list},
                                     timeout=2)
            if response.status_code == 200:
                analysis = response.json()
                st.success(f"Market Regime Detected: **{analysis['regime']}**")
                
                # Display individual model signals
                st.write("Model Individual Signals (1: Buy, -1: Sell, 0: Neutral):")
                st.json(analysis['predictions'])
            else:
                st.error("API Error: Ensure the FastAPI backend is running.")
        except requests.exceptions.ConnectionError:
            st.error("Connection Error: FastAPI backend (port 8000) is not reachable.")
    else:
        st.error("Raw data file not found to simulate API request.")

st.divider()
st.subheader("About the Models")
st.markdown("""
- **ARIMA**: High-accuracy short-term forecasting.
- **SVM**: Maximizes margins for high precision classification.
- **Gradient Boosting (XGBoost/LightGBM)**: Corrects residuals sequentially, robust for complex regimes.
- **Random Forest**: Ensemble of trees for stability and accuracy.
""")
