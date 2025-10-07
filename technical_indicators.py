import pandas as pd
import numpy as np
import streamlit as st

class TechnicalIndicators:
    """Class to calculate various technical indicators"""
    
    def __init__(self):
        pass
    
    def add_all_indicators(self, data):
        """
        Add all technical indicators to the dataset
        
        Args:
            data (pd.DataFrame): Stock data
            
        Returns:
            pd.DataFrame: Data with technical indicators
        """
        try:
            # Moving averages
            data = self.add_moving_averages(data)
            
            # RSI
            data = self.add_rsi(data)
            
            # MACD
            data = self.add_macd(data)
            
            # Bollinger Bands
            data = self.add_bollinger_bands(data)
            
            # Stochastic Oscillator
            data = self.add_stochastic(data)
            
            # Average True Range (ATR)
            data = self.add_atr(data)
            
            # Williams %R
            data = self.add_williams_r(data)
            
            # Commodity Channel Index (CCI)
            data = self.add_cci(data)
            
            return data
            
        except Exception as e:
            st.warning(f"Error adding technical indicators: {str(e)}")
            return data
    
    def add_moving_averages(self, data, periods=[5, 10, 20, 50, 200]):
        """
        Add Simple Moving Averages (SMA) and Exponential Moving Averages (EMA)
        
        Args:
            data (pd.DataFrame): Stock data
            periods (list): Periods for moving averages
            
        Returns:
            pd.DataFrame: Data with moving averages
        """
        for period in periods:
            if len(data) >= period:
                data[f'SMA_{period}'] = data['Close'].rolling(window=period).mean()
                data[f'EMA_{period}'] = data['Close'].ewm(span=period).mean()
        
        return data
    
    def add_rsi(self, data, period=14):
        """
        Add Relative Strength Index (RSI)
        
        Args:
            data (pd.DataFrame): Stock data
            period (int): RSI period
            
        Returns:
            pd.DataFrame: Data with RSI
        """
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        return data
    
    def add_macd(self, data, fast=12, slow=26, signal=9):
        """
        Add MACD (Moving Average Convergence Divergence)
        
        Args:
            data (pd.DataFrame): Stock data
            fast (int): Fast EMA period
            slow (int): Slow EMA period
            signal (int): Signal line EMA period
            
        Returns:
            pd.DataFrame: Data with MACD
        """
        exp1 = data['Close'].ewm(span=fast).mean()
        exp2 = data['Close'].ewm(span=slow).mean()
        
        data['MACD'] = exp1 - exp2
        data['MACD_Signal'] = data['MACD'].ewm(span=signal).mean()
        data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
        
        return data
    
    def add_bollinger_bands(self, data, period=20, std_dev=2):
        """
        Add Bollinger Bands
        
        Args:
            data (pd.DataFrame): Stock data
            period (int): Moving average period
            std_dev (float): Standard deviation multiplier
            
        Returns:
            pd.DataFrame: Data with Bollinger Bands
        """
        sma = data['Close'].rolling(window=period).mean()
        std = data['Close'].rolling(window=period).std()
        
        data['BB_Upper'] = sma + (std * std_dev)
        data['BB_Lower'] = sma - (std * std_dev)
        data['BB_Middle'] = sma
        data['BB_Width'] = data['BB_Upper'] - data['BB_Lower']
        data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
        
        return data
    
    def add_stochastic(self, data, k_period=14, d_period=3):
        """
        Add Stochastic Oscillator
        
        Args:
            data (pd.DataFrame): Stock data
            k_period (int): %K period
            d_period (int): %D period
            
        Returns:
            pd.DataFrame: Data with Stochastic Oscillator
        """
        lowest_low = data['Low'].rolling(window=k_period).min()
        highest_high = data['High'].rolling(window=k_period).max()
        
        data['Stoch_K'] = 100 * ((data['Close'] - lowest_low) / (highest_high - lowest_low))
        data['Stoch_D'] = data['Stoch_K'].rolling(window=d_period).mean()
        
        return data
    
    def add_atr(self, data, period=14):
        """
        Add Average True Range (ATR)
        
        Args:
            data (pd.DataFrame): Stock data
            period (int): ATR period
            
        Returns:
            pd.DataFrame: Data with ATR
        """
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        data['ATR'] = true_range.rolling(window=period).mean()
        
        return data
    
    def add_williams_r(self, data, period=14):
        """
        Add Williams %R
        
        Args:
            data (pd.DataFrame): Stock data
            period (int): Williams %R period
            
        Returns:
            pd.DataFrame: Data with Williams %R
        """
        highest_high = data['High'].rolling(window=period).max()
        lowest_low = data['Low'].rolling(window=period).min()
        
        data['Williams_R'] = -100 * ((highest_high - data['Close']) / (highest_high - lowest_low))
        
        return data
    
    def add_cci(self, data, period=20):
        """
        Add Commodity Channel Index (CCI)
        
        Args:
            data (pd.DataFrame): Stock data
            period (int): CCI period
            
        Returns:
            pd.DataFrame: Data with CCI
        """
        tp = (data['High'] + data['Low'] + data['Close']) / 3
        sma_tp = tp.rolling(window=period).mean()
        mean_deviation = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        
        data['CCI'] = (tp - sma_tp) / (0.015 * mean_deviation)
        
        return data
    
    def add_volume_indicators(self, data):
        """
        Add volume-based indicators
        
        Args:
            data (pd.DataFrame): Stock data
            
        Returns:
            pd.DataFrame: Data with volume indicators
        """
        # Volume Moving Average
        data['Volume_SMA_10'] = data['Volume'].rolling(window=10).mean()
        data['Volume_SMA_20'] = data['Volume'].rolling(window=20).mean()
        
        # Volume Rate of Change
        data['Volume_ROC'] = data['Volume'].pct_change(periods=10) * 100
        
        # On Balance Volume (OBV)
        data['OBV'] = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
        
        # Volume Price Trend (VPT)
        data['VPT'] = (data['Volume'] * data['Close'].pct_change()).fillna(0).cumsum()
        
        return data
