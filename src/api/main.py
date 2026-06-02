from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
from src.features.technical_indicators import add_technical_indicators
from src.models.model_wrappers import ModelWrapper
from src.models.position_sizing import calculate_position_sizes

app = FastAPI(title="ML Research: Stock Market Benchmark API")

# Global variables for models, scalers, and regime detectors
models = {}
scalers = {}
regime_detectors = {}

def load_resources(ticker="AAPL"):
    """Loads trained models, scalers, and regime detectors for a given ticker."""
    model_names = ['SVM', 'RandomForest', 'XGBoost', 'LightGBM']
    for name in model_names:
        model_path = f"models/{name}_{ticker}.joblib"
        if os.path.exists(model_path):
            raw_model = joblib.load(model_path)
            # Reconstruct ModelWrapper around raw model
            wrapper = ModelWrapper(name)
            wrapper.model = raw_model
            if name == "XGBoost":
                wrapper._class_to_label = {0: -1, 1: 0, 2: 1}
                wrapper._label_to_class = {-1: 0, 0: 1, 1: 2}
            models[name] = wrapper
            
    scaler_path = f"models/scaler_{ticker}.joblib"
    if os.path.exists(scaler_path):
        scalers[ticker] = joblib.load(scaler_path)
        
    regime_path = f"models/regime_detector_{ticker}.joblib"
    if os.path.exists(regime_path):
        regime_detectors[ticker] = joblib.load(regime_path)

@app.on_event("startup")
def startup_event():
    load_resources()

class PredictionRequest(BaseModel):
    ticker: str
    ohlcv_data: list # List of dicts with OHLCV for recent days

class PredictionResponse(BaseModel):
    ticker: str
    predictions: dict
    regime: str
    regime_gmm: str
    position_sizes: dict

@app.get("/")
def read_root():
    return {"message": "Welcome to the Stock Market Benchmark API"}

@app.post("/analyze", response_model=PredictionResponse)
def analyze(request: PredictionRequest):
    ticker = request.ticker
    if ticker not in scalers:
        load_resources(ticker)
        if ticker not in scalers:
            raise HTTPException(status_code=404, detail=f"Models for ticker {ticker} not found.")

    # Convert incoming data to DataFrame
    df = pd.DataFrame(request.ohlcv_data)
    
    # Calculate features (Technical Indicators)
    try:
        df_with_features = add_technical_indicators(df)
        
        # Features are everything except OHLCV
        ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        X = df_with_features.drop(columns=[col for col in ohlcv_cols if col in df_with_features.columns])
        
        # Scale features
        X_scaled = scalers[ticker].transform(X)
        
        # Run inference and position sizing across models
        preds = {}
        position_sizes = {}
        for name, wrapper in models.items():
            # Use the most recent data point for current prediction
            p = wrapper.predict(X_scaled[-1:])
            preds[name] = int(p[0])
            
            # Probability and Position Sizing (Kelly method)
            prob = wrapper.predict_proba(X_scaled[-1:])
            size = calculate_position_sizes(
                signals=[preds[name]], 
                probabilities=prob, 
                method="kelly"
            )[0]
            position_sizes[name] = float(size)
            
        # Determine Market Regime (Simplified consensus)
        consensus = sum(preds.values())
        if consensus > 0:
            regime = "Bullish"
        elif consensus < 0:
            regime = "Bearish"
        else:
            regime = "Neutral"
            
        # Determine GMM-based Market Regime
        regime_gmm = "Unknown"
        if ticker in regime_detectors:
            # Predict regime using last data point of Log_Returns and Volatility
            last_feat = df_with_features[['Log_Returns', 'Volatility']].iloc[-1:]
            regime_gmm = regime_detectors[ticker].predict_regime_name(last_feat)[0]
            
        return PredictionResponse(
            ticker=ticker, 
            predictions=preds, 
            regime=regime, 
            regime_gmm=regime_gmm,
            position_sizes=position_sizes
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

