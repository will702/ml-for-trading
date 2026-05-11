import yfinance as yf
import pandas as pd
import os
from datetime import datetime

def fetch_stock_data(ticker, start_date, end_date, output_dir="data/raw"):
    """
    Fetches historical OHLCV data for a given ticker from Yahoo Finance.
    """
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            print(f"No data found for {ticker}.")
            return None
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to Parquet for better performance/storage
        filename = f"{ticker}_{start_date}_{end_date}.parquet"
        filepath = os.path.join(output_dir, filename)
        data.to_parquet(filepath)
        
        print(f"Successfully saved data to {filepath}")
        return filepath
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    ticker_symbol = "AAPL"
    start = "2020-01-01"
    end = datetime.now().strftime("%Y-%m-%d")
    fetch_stock_data(ticker_symbol, start, end)
