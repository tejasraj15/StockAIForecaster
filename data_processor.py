import pandas as pd
import numpy as np
from scipy import stats
import streamlit as st

class DataProcessor:
    """Class to handle data cleaning and preprocessing"""
    
    def __init__(self):
        pass
    
    def clean_data(self, data):
        """
        Clean and preprocess stock data
        
        Args:
            data (pd.DataFrame): Raw stock data
            
        Returns:
            pd.DataFrame: Cleaned data
        """
        try:
            # Make a copy to avoid modifying original data
            cleaned_data = data.copy()
            
            # Convert Date column to datetime and set as index
            if 'Date' in cleaned_data.columns:
                cleaned_data['Date'] = pd.to_datetime(cleaned_data['Date'])
                cleaned_data.set_index('Date', inplace=True)
            
            # Sort by date
            cleaned_data.sort_index(inplace=True)
            
            # Remove any duplicate dates
            cleaned_data = cleaned_data[~cleaned_data.index.duplicated(keep='first')]
            
            # Handle missing values
            cleaned_data = self.handle_missing_values(cleaned_data)
            
            # Remove outliers
            cleaned_data = self.remove_outliers(cleaned_data)
            
            # Add basic derived features
            cleaned_data = self.add_basic_features(cleaned_data)
            
            return cleaned_data
            
        except Exception as e:
            st.error(f"Error cleaning data: {str(e)}")
            return data
    
    def handle_missing_values(self, data):
        """
        Handle missing values in the dataset
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with missing values handled
        """
        # Forward fill missing values (use previous day's value)
        data = data.ffill()
        
        # If there are still missing values at the beginning, backward fill
        data = data.bfill()
        
        # Remove any rows that still have missing values
        data = data.dropna()
        
        return data
    
    def remove_outliers(self, data, threshold=3):
        """
        Remove outliers using Z-score method
        
        Args:
            data (pd.DataFrame): Input data
            threshold (float): Z-score threshold for outlier detection
            
        Returns:
            pd.DataFrame: Data with outliers removed
        """
        try:
            # Calculate Z-scores for price columns
            price_columns = ['Open', 'High', 'Low', 'Close']
            
            for col in price_columns:
                if col in data.columns:
                    z_scores = np.abs(stats.zscore(data[col]))
                    # Keep rows where Z-score is below threshold
                    data = data[z_scores < threshold]
            
            return data
            
        except Exception as e:
            st.warning(f"Error removing outliers: {str(e)}")
            return data
    
    def add_basic_features(self, data):
        """
        Add basic derived features
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with additional features
        """
        try:
            # Price changes
            data['Price_Change'] = data['Close'].diff()
            data['Price_Change_Pct'] = data['Close'].pct_change()
            
            # High-Low spread
            data['HL_Spread'] = data['High'] - data['Low']
            data['HL_Spread_Pct'] = (data['High'] - data['Low']) / data['Close']
            
            # Open-Close spread
            data['OC_Spread'] = data['Close'] - data['Open']
            data['OC_Spread_Pct'] = (data['Close'] - data['Open']) / data['Open']
            
            # Intraday return
            data['Intraday_Return'] = (data['Close'] - data['Open']) / data['Open']
            
            # Volume-Price trend
            data['Volume_Price_Trend'] = data['Volume'] * data['Price_Change_Pct']
            
            # Price range as percentage of close
            data['Range_Pct'] = (data['High'] - data['Low']) / data['Close']
            
            return data
            
        except Exception as e:
            st.warning(f"Error adding basic features: {str(e)}")
            return data
    
    def create_lagged_features(self, data, columns, lags):
        """
        Create lagged features for time series analysis
        
        Args:
            data (pd.DataFrame): Input data
            columns (list): Columns to create lags for
            lags (list): List of lag periods
            
        Returns:
            pd.DataFrame: Data with lagged features
        """
        try:
            for col in columns:
                if col in data.columns:
                    for lag in lags:
                        data[f'{col}_lag_{lag}'] = data[col].shift(lag)
            
            # Remove rows with NaN values created by lagging
            data = data.dropna()
            
            return data
            
        except Exception as e:
            st.warning(f"Error creating lagged features: {str(e)}")
            return data
    
    def create_rolling_features(self, data, columns, windows):
        """
        Create rolling window features
        
        Args:
            data (pd.DataFrame): Input data
            columns (list): Columns to create rolling features for
            windows (list): List of window sizes
            
        Returns:
            pd.DataFrame: Data with rolling features
        """
        try:
            for col in columns:
                if col in data.columns:
                    for window in windows:
                        data[f'{col}_roll_mean_{window}'] = data[col].rolling(window).mean()
                        data[f'{col}_roll_std_{window}'] = data[col].rolling(window).std()
                        data[f'{col}_roll_min_{window}'] = data[col].rolling(window).min()
                        data[f'{col}_roll_max_{window}'] = data[col].rolling(window).max()
            
            # Remove rows with NaN values created by rolling windows
            data = data.dropna()
            
            return data
            
        except Exception as e:
            st.warning(f"Error creating rolling features: {str(e)}")
            return data
    
    def normalize_features(self, data, columns, method='minmax'):
        """
        Normalize specified columns
        
        Args:
            data (pd.DataFrame): Input data
            columns (list): Columns to normalize
            method (str): Normalization method ('minmax' or 'zscore')
            
        Returns:
            pd.DataFrame: Data with normalized features
        """
        try:
            from sklearn.preprocessing import MinMaxScaler, StandardScaler
            
            if method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'zscore':
                scaler = StandardScaler()
            else:
                st.warning(f"Unknown normalization method: {method}")
                return data
            
            for col in columns:
                if col in data.columns:
                    data[f'{col}_normalized'] = scaler.fit_transform(data[[col]])
            
            return data
            
        except Exception as e:
            st.warning(f"Error normalizing features: {str(e)}")
            return data
