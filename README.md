

# StockAIForecast

A Streamlit web app for stock price prediction, portfolio optimisation, and strategy backtesting using machine learning.

## Features

**Single Stock Analysis**
- Fetches live OHLCV data from Yahoo Finance (cached for 1 hour)
- Technical indicators: RSI, MACD, Bollinger Bands, Stochastic, ATR, Williams %R, CCI
- ML models: Linear Regression, Random Forest, LSTM, and an optimised Ensemble
- Walk-forward cross-validation realistic out-of-sample performance per fold
- Downloadable predictions, metrics, and summary reports

**Portfolio Optimisation**
- Multi-stock analysis with Modern Portfolio Theory
- Three optimisation objectives: Max Sharpe Ratio, Min Variance, Max Return
- Efficient frontier visualisation
- Returns-based correlation matrix (not price-based)

**Strategy Backtesting**
- Four strategies: SMA Crossover, RSI, MACD, Bollinger Bands
- Configurable parameters and initial capital
- Metrics: total return, Sharpe ratio, max drawdown, win rate, trade count
- Compared against a buy-and-hold benchmark

## Project Structure

app.py                  # Entry point — page config and tab routing
ui_single_stock.py      # Single stock analysis UI
ui_portfolio.py         # Portfolio optimisation UI
ui_backtesting.py       # Backtesting UI
data_fetcher.py         # Yahoo Finance data fetching (cached)
data_processor.py       # Data cleaning and feature engineering
technical_indicators.py # RSI, MACD, Bollinger Bands, etc.
models.py               # ML model training, prediction, walk-forward validation
portfolio.py            # Portfolio optimisation and visualisation
backtesting.py          # Signal generation and backtest engine
visualizations.py       # All Plotly charts
report_generator.py     # CSV and text report generation
