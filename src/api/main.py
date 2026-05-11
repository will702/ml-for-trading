from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
from src.features.technical_indicators import add_technical_indicators

app = FastAPI(title="ML Research: Stock Market Benchmark API")

# Global variables for models and scalers
models = {}
scalers = {}

def load_resources(ticker="AAPL"):
    """Loads trained models and scalers for a given ticker."""
    model_names = ['SVM', 'RandomForest', 'XGBoost', 'LightGBM']
    for name in model_names:
        model_path = f"models/{name}_{ticker}.joblib"
        if os.path.exists(model_path):
            models[name] = joblib.load(model_path)
            
    scaler_path = f"models/scaler_{ticker}.joblib"
    if os.path.exists(scaler_path):
        scalers[ticker] = joblib.load(scaler_path)

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
        
        # Run inference across models
        preds = {}
        for name, model in models.items():
            # Use the most recent data point for current prediction
            p = model.predict(X_scaled[-1:])
            preds[name] = int(p[0])
            
        # Determine Market Regime (Simplified consensus)
        consensus = sum(preds.values())
        if consensus > 0:
            regime = "Bullish"
        elif consensus < 0:
            regime = "Bearish"
        else:
            regime = "Neutral"
            
        return PredictionResponse(ticker=ticker, predictions=preds, regime=regime)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
