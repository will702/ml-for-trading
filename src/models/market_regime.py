import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

class MarketRegimeDetector:
    """
    Gaussian Mixture Model (GMM) based market regime detector.
    Clusters market conditions into distinct regimes based on returns and volatility.
    """
    def __init__(self, n_regimes=3, random_state=42):
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.gmm = GaussianMixture(n_components=n_regimes, random_state=random_state, n_init=10)
        self.regime_order = None  # maps GMM component ID -> sorted regime index (0=Low, 1=Medium, 2=High Vol)

    def fit(self, X):
        """
        Fit the GMM model.
        X should be a 2D array-like containing [Log_Returns, Volatility]
        """
        X_arr = np.asarray(X)
        # Drop rows with NaN or Inf
        X_arr = X_arr[np.isfinite(X_arr).all(axis=1)]
        
        self.gmm.fit(X_arr)
        
        # Map clusters consistently based on average volatility (feature index 1)
        means = self.gmm.means_
        vol_means = means[:, 1]
        
        # regime_order[0] will be raw label of cluster with lowest vol
        # regime_order[2] will be raw label of cluster with highest vol
        self.regime_order = np.argsort(vol_means)
        return self

    def predict(self, X):
        """
        Predict the ordered regime (0=Low Volatility, 1=Medium Volatility, 2=High Volatility).
        """
        X_arr = np.asarray(X)
        preds = self.gmm.predict(X_arr)
        
        # Map raw GMM prediction to ordered regime
        lookup = {raw_label: rank for rank, raw_label in enumerate(self.regime_order)}
        ordered_preds = np.array([lookup[p] for p in preds])
        return ordered_preds

    def predict_regime_name(self, X):
        """
        Predict human-readable regime name.
        """
        ordered_preds = self.predict(X)
        names = {
            0: "Low Volatility (Steady/Trend)",
            1: "Moderate Volatility (Sideways/Range)",
            2: "High Volatility (Turbulent/Stress)"
        }
        return [names[p] for p in ordered_preds]
