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
