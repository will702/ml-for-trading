import numpy as np
import pandas as pd
from scipy.optimize import minimize

def calculate_equal_weights(n_assets):
    """
    Returns equal weights for n assets.
    """
    if n_assets == 0:
        return np.array([])
    return np.ones(n_assets) / n_assets

def calculate_risk_parity_weights(cov_matrix):
    """
    Calculates weights such that each asset contributes equally to the portfolio volatility.
    Under a simplified assumption, weights are proportional to 1 / asset_volatility.
    """
    cov_matrix = np.asarray(cov_matrix)
    n_assets = cov_matrix.shape[0]
    if n_assets == 0:
        return np.array([])
        
    vols = np.sqrt(np.diag(cov_matrix))
    # Handle zero volatility case
    vols = np.where(vols == 0, 1e-6, vols)
    
    inv_vols = 1.0 / vols
    weights = inv_vols / np.sum(inv_vols)
    return weights

def calculate_mvo_weights(expected_returns, cov_matrix, risk_free_rate=0.0):
    """
    Performs Mean-Variance Optimization to maximize the Sharpe Ratio.
    Constraints: weights sum to 1, no short selling (0 <= w_i <= 1).
    """
    expected_returns = np.asarray(expected_returns)
    cov_matrix = np.asarray(cov_matrix)
    n_assets = len(expected_returns)
    
    if n_assets == 0:
        return np.array([])
    if n_assets == 1:
        return np.array([1.0])
        
    # Objective function: negative Sharpe Ratio
    def neg_sharpe(weights):
        port_return = np.dot(weights, expected_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        if port_vol == 0:
            return 0.0
        return - (port_return - risk_free_rate) / port_vol

    # Initial guess
    init_weights = np.ones(n_assets) / n_assets
    
    # Constraints & Bounds
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    
    # Run optimizer
    result = minimize(
        neg_sharpe, 
        init_weights, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints
    )
    
    if result.success:
        return result.x
    else:
        # Fallback to Equal Weight if optimization fails
        return init_weights
