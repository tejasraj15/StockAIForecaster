import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime

class StockDataFetcher:
    """Class to handle stock data fetching from Yahoo Finance"""
    
    def __init__(self):
        pass
    
    def fetch_data(self, ticker, start_date, end_date):
        """
        Fetch stock data for given ticker and date range
        
        Args:
            ticker (str): Stock ticker symbol
            start_date (datetime): Start date for data
            end_date (datetime): End date for data
            
        Returns:
            pd.DataFrame: Stock data with OHLCV columns
        """
        try:
            # Convert dates to string format for yfinance
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Create ticker object
            stock = yf.Ticker(ticker)
            
            # Fetch historical data
            data = stock.history(start=start_str, end=end_str)
            
            if data.empty:
                st.error(f"No data found for ticker {ticker} in the specified date range.")
                return None
            
            # Reset index to make Date a column
            data.reset_index(inplace=True)
            
            # Ensure we have the required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_columns):
                st.error("Retrieved data is missing required columns.")
                return None
            
            return data
            
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {str(e)}")
            return None
    
    def get_stock_info(self, ticker):
        """
        Get basic stock information
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            dict: Stock information
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return {
                'name': info.get('longName', ticker),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0)
            }
        except Exception as e:
            return {
                'name': ticker,
                'sector': 'N/A',
                'industry': 'N/A',
                'market_cap': 0,
                'pe_ratio': 0
            }
    
    def validate_ticker(self, ticker):
        """
        Validate if ticker exists
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            bool: True if ticker is valid, False otherwise
        """
        try:
            stock = yf.Ticker(ticker)
            # Try to get recent data (last 5 days)
            data = stock.history(period="5d")
            return not data.empty
        except:
            return False
