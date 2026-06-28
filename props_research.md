## **Task-Oriented Benchmarking of Traditional ML Models in Stock Market Applications**

**Group 9**  
Members : 

* Gregorius Willson—2802449846  
* Marco Oden Leo—2802429453  
* Yoel Nathanael—2802445766

### **2\. Problem Statement:** 

	Right now, in this modern developed world, data from, like, the stock market flow within milliseconds right into everyone's hands; therefore, we should also be able to analyze the market within proper algorithms and pipelines in order to achieve superior results. However, there are several critical gaps in implementing ML for stock market domains. 

1. Disconnected Metrics**:** Most of the ML models that we learn only can be optimized for metrics like MAE and RMSE, which are statistical; however, in the stock market, statistical error doesn't really correlate positively with financial metrics like Sharpe ratio and maximum drawdown. Without proper usage of the model for specific tasks, the model could achieve high accuracy but very low alpha.  
2. Wrong usage of certain models**:** in the stock market, there is rarely a single model to beat every model possible. Like, for example, ARIMA, which dominates short-term predictions, but failed badly in predicting *market regimes* compared to an *ensemble.* (XGBoost/Random Forest). The lack of systematic benchmark also might make us use wrong model in certain tasks.   
3. Black box understanding: Most students who study ML or retail traders who use certain AI tools for prediction don't really understand the logic and concept behind the algorithms. Through this research, we are going to try to uncovered the magic behind those numbers

### **3\. Project Purpose:** 

This project is meant to demystify relations between trading and quant optimizations since many out there think quant systems are like black magic that has never been seen before; therefore, by this project existing, we mean to apply our skills and knowledge in ML, specifically traditional ML models, to help evaluate and rank them based on their usage so that during real-world applications like stock trading, it gives the highest alpha (advantages) for certain tasks (portfolio sizing, entry/exit, and regime detection). And also we wanted to seek out some relevancy between statistical metrics like F1, ROC-AUC, etc. and financial metrics (Sharpe Ratio, Max Drawdowns , and so on) 

Furthermore, this project seeks to bridge the critical gap between theoretical statistical precision and practical financial performance. Traditional education often emphasizes the minimization of metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE); however, these do not always correlate with a healthy **Sharpe Ratio** or a manageable **Maximum Drawdown** in a live trading environment. This research will systematically benchmark these models to ensure they balance mathematical accuracy with robust economic value, thereby preventing the "high accuracy, low alpha" trap often found in disconnected models

### **4\. Dataset Source & Description**

* Primary Source: [yfinance](https://pypi.org/project/yfinance/) (Yahoo Finance API) for historical OHLCV (Open, High, Low, Close, Volume) data.  
* Supplementary Source: [StockPro](https://ipo-tracker.site/)  for fundamental, price ticker (OHLCV), technical analysis.  
* Supplementary Source: [idx.co.id](http://idx.co.id) ( Indonesian Stock Market Data Provider / BEI ) 

### **5\. Initial Plan for Model / Methodology and Baseline**

A. Data Preprocessing:

To transform raw OHLCV data into machine-learnable features, we will implement the following:

* **Feature Engineering:** Calculation of technical indicators (RSI, MACD, Bollinger Bands, and MA).  
* **Stationarity Checks:** Applying the Augmented Dickey-Fuller (ADF) test, especially for ARIMA.  
* **Scaling & Normalization:** Using RobustScaler or MinMaxScaler to handle financial outliers.  
* **Labeling:** Implementing "Triple Barrier Labeling" or "Fixed-Time Horizon" labeling for entry/exit signals.  
* **Data Splitting:** Utilizing **Time-Series Walk-Forward Validation** to prevent data leakage.

B. Benchmarking Models:  
We will evaluate four distinct mathematical approaches to determine their "Best-Fit" task:

* **ARIMA:** Applicable model for predicting stock-market, this is due to its high-accuracy short-term forecast capability, it consists of **AR** (autoregression), **I** (integration), and **MA** (moving average).  
* **SVM (Support Vector Machine):** Maximizes the margin between the support vectors or nearest data points of different classes, to ensure high accuracy.  
* **Gradient Boosting (LightGBM / XGBoost):** **LightGBM** (Light Gradient Boosting Machine) is extremely fast and uses low memory, meanwhile **XGBoost** (Xtreme Gradient Boosting) is more robust and precisely accurate. **LightGBM** is good for large-scale datasets with millions of rows or thousands of features, while **XGBoost** best case is a moderate-sized datasets with stable performance and tuning simplicity as priorities. Both use an ensemble of [decision trees](https://machinelearningmastery.com/light-gradient-boosted-machine-lightgbm-ensemble/) to do predictions by sequential training, each new tree eventually correcting the errors or residuals of the previous trees.  
* **Random Forest:** Combining multiple decision trees to get more stable and accurate result/prediction. It also has two best use cases, for classification every tree in the forest votes for a class, and the class with the most votes becomes the final result/prediction, for regression the forest calculates average of every tree outputs, determining the final prediction or numeric value.

C. Deployment & Integration

The final phase involves transitioning the best-performing models into a production-ready environment:

* **Analysis Core (FastAPI):** We will build a high-performance backend using **FastAPI**. This API will ingest real-time data, run the inference through the saved model weights, and return JSON-formatted analysis (e.g., *Regime: Bullish, Action: Buy*).  
* **User Interface (Streamlit):** A web-based dashboard built with **Streamlit** to visualize:  
  * Live model predictions vs. actual price action.  
  * Backtesting performance charts (Cumulative Returns & Drawdowns).  
  * Interactive parameter tuning for "Portfolio Sizing."

### **6\. Planned Evaluation metrics**

Statistical Metrics (Model Accuracy) : 

* **MAE (Mean Absolute Error):** To measure average prediction deviation.  
* **RMSE (Root Mean Squared Error):** To penalize large outliers in price prediction.  
* **Directional Accuracy (Hit Rate):** Percentage of correctly predicted price movements (Up/Down).


Financial Metrics (Economic Value): 

* **Sharpe Ratio:** To measure risk-adjusted returns.  
* **MDD (Maximum Drawdowns):** To assess the worst-case peak-to-trough decline.  
* **Cumulative Returns:** The total "Alpha" generated by the model over the testing period.

### **7\. Success Criteria**

The project is considered successful if we can produce a task-performance matrix that identifies which model provides the highest risk-adjusted return for each specific operation, rather than just identifying the model with the lowest statistical error.

### **References**

* De Prado, M. L. (2018). *Advances in Financial Machine Learning*. Wiley. (Reference for **Triple Barrier Labeling** and **Walk-Forward Validation**) .  
* Aronson, D. R. (2006). *Evidence-Based Technical Analysis*. Wiley. (Basis for benchmarking statistical vs. financial metrics) .  
* Hyndman, R. J., & Athanasopoulos, G. (2018). *Forecasting: Principles and Practice*. OTexts. (Technical foundation for **ARIMA** models) .  
* Murphy, J. J. (1999). *Technical Analysis of the Financial Markets*. New York Institute of Finance. (Reference for **RSI, MACD, and Bollinger Bands**) .  
* Yahoo Finance. (n.d.). *yfinance Python Library Documentation*. Retrieved from [https://pypi.org/project/yfinance/](https://pypi.org/project/yfinance/).  
* Stockpro. (n.d.). *IPO-Tracker and Market Analysis Data*. Retrieved from [https://ipo-tracker.site/](https://ipo-tracker.site/).  
* Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. Proceedings of the 22nd ACM SIGKDD. (Documentation for **Gradient Boosting** implementation) .  
* Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research. (Reference for **SVM, Random Forest, and Scaling** methods) .  
* Tiangolo, S. (n.d.). *FastAPI Framework Documentation*. Retrieved from [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/).  
* Streamlit Inc. (n.d.). *Streamlit: The fastest way to build and share data apps*. Retrieved from [https://streamlit.io/](https://streamlit.io/).

  