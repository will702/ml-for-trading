# ML Research: Task-Oriented Benchmarking in Stock Market Applications

## Project Overview
This project benchmarks traditional Machine Learning models (ARIMA, SVM, Gradient Boosting, Random Forest) within the stock market domain. The primary goal is to bridge the gap between statistical precision (MAE, RMSE) and practical financial performance (Sharpe Ratio, Max Drawdown). It seeks to identify "best-fit" models for specific trading tasks such as portfolio sizing, entry/exit signals, and market regime detection.

### Core Objectives
- **Metric Correlation:** Investigate the relationship between statistical accuracy (F1, ROC-AUC) and financial metrics.
- **Model Comparison:** Systematically benchmark models like ARIMA (short-term), SVM (accuracy), and Ensembles (regime detection).
- **Demystification:** Uncover the logic behind quant systems and traditional ML algorithms in finance.

## Tech Stack
- **Languages:** Python
- **Backend API:** FastAPI (Analysis Core)
- **Frontend Dashboard:** Streamlit (Visualization & Backtesting)
- **Data Sourcing:** `yfinance` (Primary), `StockPro`, `idx.co.id`
- **Machine Learning:** `scikit-learn`, `XGBoost`, `LightGBM`, `ARIMA`
- **Methodology Tools:** ADF Test (Stationarity), Triple Barrier Labeling, Time-Series Walk-Forward Validation.

## Development Conventions

### Data Preprocessing
- **Feature Engineering:** Technical indicators (RSI, MACD, Bollinger Bands, Moving Averages).
- **Scaling:** `RobustScaler` or `MinMaxScaler` to handle financial outliers.
- **Validation:** Always use **Time-Series Walk-Forward Validation** to prevent data leakage.

### Implementation Architecture
- **Inference:** Models should be served via FastAPI returning JSON analysis (e.g., `{"Regime": "Bullish", "Action": "Buy"}`).
- **UI:** Streamlit dashboard for visualizing predictions vs. actual price action and backtesting performance.

## Key Files
- `props_research.md`: Original research proposal and problem statement.
- `GEMINI.md`: This instructional context file.
- `TODO`: Implementation of FastAPI backend and Streamlit dashboard.

## Usage & Workflows
- **Research Phase:** Exploration of datasets and initial feature engineering.
- **Benchmarking Phase:** Training and evaluating models against both statistical and financial metrics.
- **Deployment Phase:** Integrating models into the FastAPI core and Streamlit UI.

---
*Note: This project is authored by Group 9 (Gregorius Willson, Marco Oden Leo, Yoel Nathanael).*
