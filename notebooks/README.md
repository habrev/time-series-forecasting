## Task 1: Preprocess and Explore the Data

### Objective
This task involves loading, cleaning, and exploring historical financial data for **Tesla (TSLA), Vanguard Total Bond Market ETF (BND), and SPDR S&P 500 ETF Trust (SPY)** using `yfinance`. These assets represent different risk-return profiles:
- **TSLA**: High returns with high volatility.
- **BND**: Stability with low risk.
- **SPY**: Diversified, moderate-risk market exposure.

The goal is to prepare the data for modeling by performing data cleaning, handling missing values, normalizing data if needed, and conducting exploratory data analysis (EDA).

---

## Data Collection
We extract historical stock data using the **Yahoo Finance API** (`yfinance`).


# Task-2:Time Series Forecasting for Tesla Stock Prices

This project aims to develop a time series forecasting model to predict Tesla's future stock prices using different forecasting techniques such as ARIMA, SARIMA, and LSTM (Long Short-Term Memory). We will evaluate the model's performance based on various metrics and refine it by optimizing model parameters.

## Steps to Develop the Forecasting Model

### 1. Data Collection
- Collect Tesla's historical stock price data. You can use libraries like `yfinance` or any other API to download stock price data.
- Ensure that the dataset includes the stock prices at regular time intervals (e.g., daily, weekly).

### 2. Preprocessing
- Clean the data, handle any missing values, and ensure the dataset is in the correct format for time series analysis.
- Visualize the data to understand any trends or seasonality present in the time series.

### 3. Split Data into Training and Testing Sets
- Split the dataset into training and testing sets. Typically, use the first 80-90% of the data for training and the remaining 10-20% for testing.

### 4. Model Selection
You can choose one of the following models:

#### ARIMA (AutoRegressive Integrated Moving Average)
- ARIMA is suitable for univariate time series data without seasonality.
- It requires three parameters: `p`, `d`, and `q` (AutoRegressive order, Integrated order, and Moving Average order).
- Model the data using ARIMA after making it stationary.

#### SARIMA (Seasonal ARIMA)
- SARIMA extends ARIMA by including seasonal components.
- It requires additional parameters: `P`, `D`, `Q`, and `s` (Seasonal AutoRegressive order, Seasonal Integrated order, Seasonal Moving Average order, and the length of the seasonality).
- Use SARIMA when your data exhibits seasonality.

#### LSTM (Long Short-Term Memory)
- LSTM is a deep learning model that is effective for capturing long-term dependencies in time series data.
- It requires reshaping the dataset into a format that can be used by the neural network.

### 5. Model Training
Train the selected model using the training dataset. Depending on the model you choose:

- **ARIMA/SARIMA**: Fit the model to the data and perform parameter tuning.
- **LSTM**: Train the neural network with the prepared data, optimizing the architecture and training parameters.

### 6. Model Evaluation
Evaluate the model using the following metrics:
- **Mean Absolute Error (MAE)**: Measures the average magnitude of the errors in a set of predictions, without considering their direction.
- **Root Mean Squared Error (RMSE)**: Measures the average magnitude of the errors, giving higher weight to larger errors.
- **Mean Absolute Percentage Error (MAPE)**: Measures the accuracy of the predictions as a percentage of the actual values.

### 7. Hyperparameter Optimization
- **ARIMA/SARIMA**: Use `auto_arima` from the `pmdarima` library or grid search techniques to find the optimal `(p, d, q)` or `(P, D, Q, s)` parameters.
- **LSTM**: Tune hyperparameters such as the number of layers, neurons per layer, and learning rate.

### 8. Forecasting
- Use the trained model to forecast future Tesla stock prices.
- Compare the forecasted values with the test dataset to assess the model's accuracy.

### 9. Model Refinement
Refine the model by:
- Adjusting parameters based on evaluation results.
- Trying different models and comparing their performance.
