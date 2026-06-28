import pandas as pd
import numpy as np

def calculate_rsi(data, window=14):
    """Calculates Relative Strength Index (RSI)."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, slow=26, fast=12, signal=9):
    """Calculates MACD, Signal line, and MACD Histogram."""
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(data, window=20, num_std=2):
    """Calculates Bollinger Bands (Upper, Middle, Lower)."""
    middle_band = data.rolling(window=window).mean()
    std_dev = data.rolling(window=window).std()
    upper_band = middle_band + (std_dev * num_std)
    lower_band = middle_band - (std_dev * num_std)
    return upper_band, middle_band, lower_band

def add_technical_indicators(df):
    """Adds a set of technical indicators to the OHLCV DataFrame."""
    # Ensure df is a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Handle MultiIndex if data came from yfinance with multiple tickers or newer version
    if isinstance(df.columns, pd.MultiIndex):
        standard_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        new_cols = []
        for col in df.columns:
            if col[0] in standard_cols:
                new_cols.append(col[0])
            elif len(col) > 1 and col[1] in standard_cols:
                new_cols.append(col[1])
            else:
                new_cols.append(col[0])
        df.columns = new_cols

    close = df['Close']
    
    # Simple Moving Averages
    df['SMA_20'] = close.rolling(window=20).mean()
    df['SMA_50'] = close.rolling(window=50).mean()
    
    # RSI
    df['RSI'] = calculate_rsi(close)
    
    # MACD
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(close)
    
    # Bollinger Bands
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(close)
    
    # Momentum
    df['Momentum'] = close.diff(10)
    
    # Volatility (Log Returns)
    df['Log_Returns'] = np.log(close / close.shift(1))
    df['Volatility'] = df['Log_Returns'].rolling(window=20).std()
    
    # Drop rows with NaN values created by rolling windows
    return df.dropna()
