# Implementation Plan: Task-Oriented Benchmarking in Stock Market Applications

## Background & Motivation
The project aims to benchmark traditional ML models (ARIMA, SVM, Gradient Boosting, Random Forest) in stock market domains to bridge the gap between statistical accuracy and practical financial performance. The goal is to provide empirical evidence for model selection across different trading tasks, avoiding the "high accuracy, low alpha" trap.

## Scope & Impact
- **Data Ingestion & Preprocessing**: Automated pipelines to download historical stock data (`yfinance`) and compute features.
- **Model Training**: A robust benchmarking pipeline using time-series walk-forward validation.
- **Backend API**: A FastAPI application to serve the models for live inference.
- **Frontend Dashboard**: A Streamlit dashboard to visualize predictions, backtesting results, and performance metrics.

## Proposed Solution
- **Architecture**: A **Modular Pipeline** separating concerns (`data/`, `src/features/`, `src/models/`, `src/api/`, `src/ui/`).
- **Data Storage**: Using **Local CSV/Parquet** files for simplicity, fast I/O, and zero external dependencies.
- **Training Strategy**: A **Unified Training Script** that sequentially trains, evaluates, and benchmarks all models, outputting a consolidated performance matrix.

## Alternatives Considered
- *Monolithic scripts*: Rejected in favor of a modular approach to improve scalability and testing.
- *SQLite Database*: Rejected in favor of CSV/Parquet to reduce setup complexity.
- *Separate model scripts*: Rejected to streamline the benchmarking process and ensure consistent evaluation across all models in one go.

## Phased Implementation Plan

- **Phase 1: Project Setup & Data Ingestion**
  - Scaffold the modular directory structure.
  - Implement data ingestion scripts fetching OHLCV data from `yfinance`.
  - Save raw data to local Parquet/CSV storage.

- **Phase 2: Feature Engineering & Preprocessing**
  - Compute technical indicators (RSI, MACD, Bollinger Bands, Moving Averages).
  - Implement labeling strategies (Triple Barrier or Fixed-Time Horizon).
  - Apply data scaling (`RobustScaler` / `MinMaxScaler`) and prepare Time-Series Walk-Forward splits.

- **Phase 3: Model Training & Benchmarking**
  - Implement wrapper classes for ARIMA, SVM, LightGBM/XGBoost, and Random Forest.
  - Develop the unified training script to run walk-forward validation across all models.
  - Calculate Statistical Metrics (MAE, RMSE, Hit Rate) and Financial Metrics (Sharpe Ratio, Max Drawdown, Cumulative Returns).
  - Generate a combined performance matrix report.

- **Phase 4: Backend API (FastAPI)**
  - Develop FastAPI endpoints to run inference using the trained and serialized models.
  - Define JSON response schemas for trading signals (e.g., Regime, Action).

- **Phase 5: Frontend Dashboard (Streamlit)**
  - Build the UI for backtesting visualization (Cumulative Returns & Drawdowns).
  - Integrate live prediction tracking communicating with the FastAPI backend.
  - Add interactive parameter tuning controls.

## Verification
- Unit tests for feature calculation and labeling logic.
- Strict validation of the walk-forward split to ensure zero data leakage.
- End-to-end integration test validating the flow from API inference to Streamlit visualization.

## Migration & Rollback
- As a greenfield implementation, no data migration is required.
- Git version control will be utilized with feature branches corresponding to each implementation phase to ensure safe rollbacks.