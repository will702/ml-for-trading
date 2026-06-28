import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

def triple_barrier_labeling(df, vertical_barrier=5, profit_taking=0.02, stop_loss=0.01):
    """
    Simplified Triple Barrier Labeling.
    1: Profit Taking hit
    -1: Stop Loss hit
    0: Vertical Barrier (time) hit
    """
    labels = []
    close = df['Close'].values
    
    for i in range(len(close) - vertical_barrier):
        future_prices = close[i+1 : i+1+vertical_barrier]
        initial_price = close[i]
        
        label = 0
        for price in future_prices:
            return_pct = (price - initial_price) / initial_price
            if return_pct >= profit_taking:
                label = 1
                break
            elif return_pct <= -stop_loss:
                label = -1
                break
        labels.append(label)
    
    # Pad with NaN for the last few entries
    labels.extend([np.nan] * vertical_barrier)
    return pd.Series(labels, index=df.index)

def prepare_features_and_labels(df, target_col='Label'):
    """
    Scales features and handles the final feature/label split.
    """
    # Drop rows where Label is NaN
    df = df.dropna(subset=[target_col])
    
    # Features are everything except OHLCV and Label
    ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
    drop_cols = [col for col in ohlcv_cols if col in df.columns] + [target_col]
    
    X = df.drop(columns=drop_cols)
    y = df[target_col]
    
    # Scaling
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    
    return X_scaled, y, scaler

def walk_forward_split(X, y, test_size=0.2):
    """
    Time-series friendly split (no shuffling).
    """
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    return X_train, X_test, y_train, y_test
