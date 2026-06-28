import numpy as np
import pandas as pd
from src.models.position_sizing import calculate_position_sizes

def run_advanced_backtest(
    prices,
    signals,
    probabilities=None,
    volatilities=None,
    trend_filter=None,
    sizing_method="constant",
    target_vol=0.015,
    stop_loss=0.0,
    profit_taking=0.0,
    trailing_stop=0.0,
    time_barrier=0,
    win_loss_ratio=2.0
):
    """
    Simulates trading with realistic entry/exit rules and position sizing.
    
    Parameters:
    -----------
    prices : pd.Series
        Historical prices (Close).
    signals : np.ndarray or pd.Series
        Model signals (-1 for Short, 1 for Long, 0 for Neutral).
    probabilities : np.ndarray, optional
        Confidence probabilities for Kelly sizing.
    volatilities : np.ndarray, optional
        Historical volatility for volatility sizing.
    trend_filter : pd.Series or np.ndarray, optional
        Boolean series indicating if trend is bullish (True) or bearish/neutral (False).
        If provided, Long entries are only allowed when trend_filter is True, 
        and Short entries when trend_filter is False.
    sizing_method : str
        Position sizing: "constant", "volatility", or "kelly".
    target_vol : float
        Volatility target.
    stop_loss : float
        Stop loss percentage (e.g. 0.02 for 2%). 0 to disable.
    profit_taking : float
        Profit taking percentage (e.g. 0.04 for 4%). 0 to disable.
    trailing_stop : float
        Trailing stop percentage (e.g. 0.02 for 2% from peak). 0 to disable.
    time_barrier : int
        Max holding period in days. 0 to disable.
    win_loss_ratio : float
        Risk-reward ratio for Kelly sizing.
        
    Returns:
    --------
    dict containing:
        - "Total Return": float
        - "Sharpe Ratio": float
        - "Max Drawdown": float
        - "strategy_returns": pd.Series
        - "equity_curve": pd.Series
        - "trades": list of dicts (logs of completed trades)
    """
    prices = pd.Series(prices).astype(float)
    signals = np.asarray(signals)
    n_samples = len(prices)
    
    # Calculate all position sizes upfront
    raw_sizes = calculate_position_sizes(
        signals=signals,
        probabilities=probabilities,
        volatilities=volatilities,
        method=sizing_method,
        target_vol=target_vol,
        win_loss_ratio=win_loss_ratio
    )
    
    # Align data
    strategy_returns = pd.Series(0.0, index=prices.index)
    trades = []
    
    # State variables
    position = 0          # Current position: 1=Long, -1=Short, 0=None
    entry_price = 0.0
    entry_idx = -1
    highest_price = 0.0
    lowest_price = 0.0
    holding_days = 0
    trade_size = 0.0
    
    price_values = prices.values
    price_index = prices.index
    
    for t in range(1, n_samples):
        # 1. If we hold a position, calculate return from t-1 to t
        if position != 0:
            holding_days += 1
            asset_return = (price_values[t] - price_values[t-1]) / price_values[t-1]
            strategy_returns.iloc[t] = position * trade_size * asset_return
            
            # Update peak/trough prices for trailing stop
            if position == 1:
                highest_price = max(highest_price, price_values[t])
            else:
                lowest_price = min(lowest_price, price_values[t])
                
            # 2. Check Exit Conditions on day t
            exit_triggered = False
            exit_reason = ""
            
            # Stop Loss (SL)
            if stop_loss > 0:
                if position == 1 and price_values[t] <= entry_price * (1 - stop_loss):
                    exit_triggered = True
                    exit_reason = "Stop Loss"
                elif position == -1 and price_values[t] >= entry_price * (1 + stop_loss):
                    exit_triggered = True
                    exit_reason = "Stop Loss"
                    
            # Profit Taking (PT)
            if profit_taking > 0 and not exit_triggered:
                if position == 1 and price_values[t] >= entry_price * (1 + profit_taking):
                    exit_triggered = True
                    exit_reason = "Profit Target"
                elif position == -1 and price_values[t] <= entry_price * (1 - profit_taking):
                    exit_triggered = True
                    exit_reason = "Profit Target"
                    
            # Trailing Stop
            if trailing_stop > 0 and not exit_triggered:
                if position == 1 and price_values[t] <= highest_price * (1 - trailing_stop):
                    exit_triggered = True
                    exit_reason = "Trailing Stop"
                elif position == -1 and price_values[t] >= lowest_price * (1 + trailing_stop):
                    exit_triggered = True
                    exit_reason = "Trailing Stop"
                    
            # Time Barrier
            if time_barrier > 0 and holding_days >= time_barrier and not exit_triggered:
                exit_triggered = True
                exit_reason = "Time Barrier"
                
            # Signal Reversal
            if not exit_triggered:
                current_sig = signals[t]
                if current_sig != 0 and current_sig != position:
                    exit_triggered = True
                    exit_reason = "Signal Reversal"
                    
            # Handle Exit Execution
            if exit_triggered:
                pnl = (price_values[t] - entry_price) / entry_price if position == 1 else (entry_price - price_values[t]) / entry_price
                realized_pnl = pnl * trade_size
                trades.append({
                    "entry_date": price_index[entry_idx],
                    "exit_date": price_index[t],
                    "position": "Long" if position == 1 else "Short",
                    "entry_price": entry_price,
                    "exit_price": price_values[t],
                    "size": trade_size,
                    "reason": exit_reason,
                    "pnl": realized_pnl
                })
                position = 0
                trade_size = 0.0
                
        # 3. If we are neutral, check for Entry
        if position == 0:
            current_sig = signals[t]
            if current_sig != 0:
                # Apply Trend Filter if provided
                allowed = True
                if trend_filter is not None:
                    is_bullish = bool(trend_filter[t])
                    if current_sig == 1 and not is_bullish:
                        allowed = False
                    elif current_sig == -1 and is_bullish:
                        allowed = False
                        
                if allowed:
                    position = int(current_sig)
                    entry_price = price_values[t]
                    entry_idx = t
                    highest_price = entry_price
                    lowest_price = entry_price
                    holding_days = 0
                    trade_size = abs(raw_sizes[t])
                    
    # Calculate performance metrics
    equity_curve = (1.0 + strategy_returns).cumprod()
    total_return = equity_curve.iloc[-1] - 1.0 if len(equity_curve) > 0 else 0.0
    
    vol = strategy_returns.std()
    sharpe_ratio = 0.0
    if vol > 0:
        sharpe_ratio = (strategy_returns.mean() / vol) * np.sqrt(252)
        
    drawdown = equity_curve / equity_curve.cummax() - 1.0
    max_drawdown = drawdown.min()
    
    return {
        "Total Return": float(total_return),
        "Sharpe Ratio": float(sharpe_ratio),
        "Max Drawdown": float(max_drawdown),
        "strategy_returns": strategy_returns,
        "equity_curve": equity_curve,
        "trades": trades
    }
