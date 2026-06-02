import os
import pandas as pd
import numpy as np
from src.features.technical_indicators import add_technical_indicators
from src.features.preprocessing import triple_barrier_labeling, prepare_features_and_labels
from src.data_ingestion import fetch_stock_data
from src.train_benchmark import run_benchmarking

def test_pipeline_smoke():
    """
    Smoke test to verify the end-to-end flow and all new trading modules.
    """
    ticker = "AAPL"
    start = "2023-01-01"
    end = "2023-12-31" # Use a fixed period for testing
    
    print("--- Starting Verification Smoke Test ---")
    
    # 1. Test Ingestion
    print("\n1. Testing Data Ingestion...")
    raw_file = fetch_stock_data(ticker, start, end, allow_synthetic_fallback=True)
    if raw_file and os.path.exists(raw_file):
        print("PASS: Data Ingestion")
    else:
        print("FAIL: Data Ingestion")
        return

    # 2. Test Feature Engineering
    print("\n2. Testing Feature Engineering...")
    df = pd.read_parquet(raw_file)
    df_feat = add_technical_indicators(df)
    required_cols = ['RSI', 'MACD', 'BB_Upper', 'Volatility', 'SMA_50']
    if all(col in df_feat.columns for col in required_cols):
        print("PASS: Feature Engineering")
    else:
        print("FAIL: Feature Engineering")
        return

    # 3. Test Market Regime Detector
    print("\n3. Testing Market Regime Detector...")
    from src.models.market_regime import MarketRegimeDetector
    regime_feat = df_feat[['Log_Returns', 'Volatility']].dropna()
    regime_det = MarketRegimeDetector(n_regimes=3)
    regime_det.fit(regime_feat)
    regimes = regime_det.predict(regime_feat)
    regime_names = regime_det.predict_regime_name(regime_feat)
    if len(regimes) == len(regime_feat) and len(regime_names) == len(regime_feat):
        print("PASS: Market Regime Detector")
    else:
        print("FAIL: Market Regime Detector")
        return

    # 4. Test Position Sizing
    print("\n4. Testing Position Sizing...")
    from src.models.position_sizing import calculate_position_sizes
    signals = np.array([1, 0, -1, 1])
    probs = np.array([[0.1, 0.1, 0.8], [0.33, 0.33, 0.33], [0.7, 0.2, 0.1], [0.2, 0.1, 0.7]])
    sizes = calculate_position_sizes(signals, probabilities=probs, method="kelly")
    if len(sizes) == 4 and sizes[1] == 0.0 and sizes[0] > 0.0 and sizes[2] < 0.0:
        print("PASS: Position Sizing")
    else:
        print("FAIL: Position Sizing")
        return

    # 5. Test Portfolio Sizing
    print("\n5. Testing Portfolio Sizing...")
    from src.models.portfolio_sizing import calculate_equal_weights, calculate_risk_parity_weights, calculate_mvo_weights
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    ret = np.array([0.15, 0.20])
    w_ew = calculate_equal_weights(2)
    w_rp = calculate_risk_parity_weights(cov)
    w_mvo = calculate_mvo_weights(ret, cov)
    if len(w_ew) == 2 and len(w_rp) == 2 and len(w_mvo) == 2 and np.isclose(np.sum(w_mvo), 1.0):
        print("PASS: Portfolio Sizing")
    else:
        print("FAIL: Portfolio Sizing")
        return

    # 6. Test Advanced Backtesting
    print("\n6. Testing Advanced Backtesting Simulator...")
    from src.models.backtester import run_advanced_backtest
    sim_res = run_advanced_backtest(
        prices=df_feat['Close'],
        signals=np.ones(len(df_feat)),
        volatilities=df_feat['Volatility'].values,
        sizing_method="volatility"
    )
    if "Total Return" in sim_res and "equity_curve" in sim_res:
        print("PASS: Advanced Backtesting Simulator")
    else:
        print("FAIL: Advanced Backtesting Simulator")
        return

    # 7. Test Benchmarking Pipeline
    print("\n7. Testing Benchmarking Pipeline...")
    results = run_benchmarking(ticker, start, end)
    if results is not None and not results.empty:
        print("PASS: Benchmarking Pipeline")
    else:
        print("FAIL: Benchmarking Pipeline")
        return

    print("\n--- Smoke Test Completed Successfully ---")

if __name__ == "__main__":
    test_pipeline_smoke()

