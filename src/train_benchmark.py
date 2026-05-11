import pandas as pd
import numpy as np
import os
import joblib
from src.features.technical_indicators import add_technical_indicators
from src.features.preprocessing import triple_barrier_labeling, prepare_features_and_labels, walk_forward_split
from src.models.model_wrappers import ModelWrapper, calculate_financial_metrics, run_arima_benchmark
from sklearn.metrics import accuracy_score, classification_report

def run_benchmarking(ticker, start_date, end_date):
    # 1. Load Data
    raw_path = f"data/raw/{ticker}_{start_date}_{end_date}.parquet"
    if not os.path.exists(raw_path):
        print(f"Data not found at {raw_path}. Please run data ingestion first.")
        return
    
    df = pd.read_parquet(raw_path)
    
    # 2. Feature Engineering
    df = add_technical_indicators(df)
    
    # 3. Labeling
    df['Label'] = triple_barrier_labeling(df)
    
    # 4. Prepare Features & Labels
    X, y, scaler = prepare_features_and_labels(df)
    
    # 5. Walk-Forward Split
    X_train, X_test, y_train, y_test = walk_forward_split(X, y)
    
    # 6. Benchmarking
    results = {}
    models = ['SVM', 'RandomForest', 'XGBoost', 'LightGBM']
    
    test_prices = df.loc[X_test.index, 'Close']
    
    for model_name in models:
        print(f"Training {model_name}...")
        wrapper = ModelWrapper(model_name)
        wrapper.train(X_train, y_train)
        preds = wrapper.predict(X_test)
        
        # Statistical Metrics
        acc = accuracy_score(y_test, preds)
        
        # Financial Metrics
        fin_metrics = calculate_financial_metrics(y_test, preds, test_prices)
        
        results[model_name] = {
            "Accuracy": acc,
            **fin_metrics
        }
        
        # Save model
        os.makedirs("models", exist_ok=True)
        joblib.dump(wrapper.model, f"models/{model_name}_{ticker}.joblib")
    
    # ARIMA Special Handling
    print("Running ARIMA Benchmark (This may take a while)...")
    # For ARIMA, we use log returns of the close price
    log_returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    train_size = int(len(log_returns) * 0.8)
    arima_preds_raw = run_arima_benchmark(log_returns.values, train_size)
    
    # Convert ARIMA forecast to signals (1 if pos return, -1 if neg)
    arima_signals = np.where(arima_preds_raw > 0, 1, -1)
    # Match labels for accuracy calculation (simplified)
    # y_test_returns = np.where(log_returns.values[train_size:] > 0, 1, -1)
    # arima_acc = accuracy_score(y_test_returns, arima_signals)
    
    # For comparison, we'll just calculate financial metrics for ARIMA
    arima_fin = calculate_financial_metrics(None, arima_signals, df['Close'].iloc[train_size+1:])
    results["ARIMA"] = {
        "Accuracy": "N/A (Forecasting)",
        **arima_fin
    }
    
    # 7. Summary
    results_df = pd.DataFrame(results).T
    print("\nBenchmarking Results:")
    print(results_df)
    
    results_df.to_csv(f"data/processed/{ticker}_benchmarking_results.csv")
    print(f"\nResults saved to data/processed/{ticker}_benchmarking_results.csv")
    
    # Save scaler for API usage
    joblib.dump(scaler, f"models/scaler_{ticker}.joblib")
    
    return results_df

if __name__ == "__main__":
    ticker = "AAPL"
    start = "2020-01-01"
    end = "2026-05-11" # Current session date
    run_benchmarking(ticker, start, end)
