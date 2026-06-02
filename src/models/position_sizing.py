import numpy as np
import pandas as pd

def calculate_position_sizes(signals, probabilities=None, volatilities=None, method="constant", target_vol=0.015, win_loss_ratio=2.0):
    """
    Calculate dynamic position sizes based on various risk management methods.
    
    Parameters:
    -----------
    signals : array-like
        Trading signals, where 1 is Long, -1 is Short, and 0 is Neutral.
    probabilities : array-like, shape (n_samples, 3) or (n_samples, n_classes), optional
        Prediction probabilities for classes corresponding to [-1, 0, 1].
    volatilities : array-like, optional
        Rolling asset volatilities (e.g. standard deviation of returns).
    method : str
        The position sizing method: "constant", "volatility", or "kelly".
    target_vol : float
        Target volatility for volatility-adjusted sizing (default is 1.5%).
    win_loss_ratio : float
        Risk-reward ratio (Profit Taking / Stop Loss) for Kelly sizing (default is 2.0).
        
    Returns:
    --------
    sizes : np.ndarray
        Array of calculated position sizes (ranges between -1.0 and 1.0, or 0.0 if neutral).
    """
    signals = np.asarray(signals)
    n_samples = len(signals)
    
    if method == "constant":
        return signals.astype(float)
        
    elif method == "volatility":
        if volatilities is None:
            # Fallback to constant if volatilities not provided
            return signals.astype(float)
        volatilities = np.asarray(volatilities, dtype=float)
        
        # Avoid division by zero
        volatilities = np.where(volatilities == 0, 1e-6, volatilities)
        
        # Scale size: size = target_vol / volatility
        scale = target_vol / volatilities
        
        # Clip scaling factor to prevent extreme leverage (e.g., between 0.1 and 1.5)
        scale = np.clip(scale, 0.1, 1.5)
        
        return signals * scale
        
    elif method == "kelly":
        if probabilities is None:
            # Fallback to constant if probabilities not provided
            return signals.astype(float)
            
        probabilities = np.asarray(probabilities)
        
        # Assuming probabilities has shape (n_samples, 3) corresponding to class indices:
        # Index 0: -1 (Bearish/Short)
        # Index 1: 0 (Neutral/Hold)
        # Index 2: 1 (Bullish/Long)
        if probabilities.shape[1] < 3:
            # If binary class or invalid, fallback
            return signals.astype(float)
            
        sizes = np.zeros(n_samples)
        for i in range(n_samples):
            sig = signals[i]
            if sig == 1:
                # Long: P is probability of Bullish (index 2)
                p = probabilities[i, 2]
                # Kelly formula: f* = p - (1-p)/R
                f_star = p - (1.0 - p) / win_loss_ratio
                sizes[i] = max(0.0, min(1.0, f_star))
            elif sig == -1:
                # Short: P is probability of Bearish (index 0)
                p = probabilities[i, 0]
                # Kelly formula: f* = p - (1-p)/R
                f_star = p - (1.0 - p) / win_loss_ratio
                sizes[i] = -max(0.0, min(1.0, f_star)) # Negative for short
            else:
                sizes[i] = 0.0
                
        return sizes
        
    else:
        raise ValueError(f"Unknown position sizing method: {method}")
