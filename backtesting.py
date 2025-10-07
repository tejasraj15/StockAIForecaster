import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class BacktestingFramework:
    """Class to handle backtesting of trading strategies"""
    
    def __init__(self):
        self.initial_capital = 10000
    
    def simple_moving_average_strategy(self, data, short_window=20, long_window=50):
        """
        Simple Moving Average Crossover Strategy
        
        Args:
            data (pd.DataFrame): Stock data with 'Close' prices
            short_window (int): Short MA window
            long_window (int): Long MA window
            
        Returns:
            pd.DataFrame: Data with signals
        """
        signals = data.copy()
        
        # Calculate moving averages
        signals['short_ma'] = data['Close'].rolling(window=short_window).mean()
        signals['long_ma'] = data['Close'].rolling(window=long_window).mean()
        
        # Generate signals
        signals['signal'] = 0
        signals.loc[signals['short_ma'] > signals['long_ma'], 'signal'] = 1  # Buy
        signals.loc[signals['short_ma'] <= signals['long_ma'], 'signal'] = -1  # Sell
        
        # Position changes
        signals['position'] = signals['signal'].diff()
        
        return signals
    
    def rsi_strategy(self, data, rsi_column='RSI', oversold=30, overbought=70):
        """
        RSI-based Strategy
        
        Args:
            data (pd.DataFrame): Stock data with RSI column
            rsi_column (str): Name of RSI column
            oversold (int): Oversold threshold
            overbought (int): Overbought threshold
            
        Returns:
            pd.DataFrame: Data with signals
        """
        signals = data.copy()
        
        if rsi_column not in signals.columns:
            st.error(f"RSI column '{rsi_column}' not found in data")
            return signals
        
        # Generate signals
        signals['signal'] = 0
        signals.loc[signals[rsi_column] < oversold, 'signal'] = 1  # Buy (oversold)
        signals.loc[signals[rsi_column] > overbought, 'signal'] = -1  # Sell (overbought)
        
        # Position changes
        signals['position'] = signals['signal'].diff()
        
        return signals
    
    def macd_strategy(self, data):
        """
        MACD Strategy
        
        Args:
            data (pd.DataFrame): Stock data with MACD and Signal columns
            
        Returns:
            pd.DataFrame: Data with signals
        """
        signals = data.copy()
        
        if 'MACD' not in signals.columns or 'MACD_Signal' not in signals.columns:
            st.error("MACD columns not found in data")
            return signals
        
        # Generate signals
        signals['signal'] = 0
        signals.loc[signals['MACD'] > signals['MACD_Signal'], 'signal'] = 1  # Buy
        signals.loc[signals['MACD'] <= signals['MACD_Signal'], 'signal'] = -1  # Sell
        
        # Position changes
        signals['position'] = signals['signal'].diff()
        
        return signals
    
    def bollinger_bands_strategy(self, data):
        """
        Bollinger Bands Strategy
        
        Args:
            data (pd.DataFrame): Stock data with Bollinger Bands
            
        Returns:
            pd.DataFrame: Data with signals
        """
        signals = data.copy()
        
        if not all(col in signals.columns for col in ['BB_Upper', 'BB_Lower']):
            st.error("Bollinger Bands columns not found in data")
            return signals
        
        # Generate signals
        signals['signal'] = 0
        signals.loc[signals['Close'] < signals['BB_Lower'], 'signal'] = 1  # Buy (below lower band)
        signals.loc[signals['Close'] > signals['BB_Upper'], 'signal'] = -1  # Sell (above upper band)
        
        # Position changes
        signals['position'] = signals['signal'].diff()
        
        return signals
    
    def backtest_strategy(self, signals, initial_capital=None):
        """
        Backtest a trading strategy
        
        Args:
            signals (pd.DataFrame): DataFrame with trading signals
            initial_capital (float): Initial capital for backtesting
            
        Returns:
            dict: Backtesting results
        """
        if initial_capital is None:
            initial_capital = self.initial_capital
        else:
            self.initial_capital = initial_capital  # Update instance variable
        
        # Initialize portfolio
        portfolio = signals.copy()
        portfolio['holdings'] = 0.0
        portfolio['cash'] = initial_capital
        portfolio['total'] = initial_capital
        portfolio['returns'] = 0.0
        
        # Track position
        position = 0
        shares = 0
        cash = initial_capital
        
        for i in range(len(portfolio)):
            if pd.isna(portfolio['position'].iloc[i]):
                portfolio.loc[portfolio.index[i], 'holdings'] = shares * portfolio['Close'].iloc[i]
                portfolio.loc[portfolio.index[i], 'cash'] = cash
                portfolio.loc[portfolio.index[i], 'total'] = cash + (shares * portfolio['Close'].iloc[i])
                continue
            
            # Buy signal
            if portfolio['position'].iloc[i] > 0 and cash > 0:
                shares_to_buy = int(cash / portfolio['Close'].iloc[i])
                if shares_to_buy > 0:
                    shares += shares_to_buy
                    cash -= shares_to_buy * portfolio['Close'].iloc[i]
                    position = 1
            
            # Sell signal
            elif portfolio['position'].iloc[i] < 0 and shares > 0:
                cash += shares * portfolio['Close'].iloc[i]
                shares = 0
                position = 0
            
            # Update portfolio values
            portfolio.loc[portfolio.index[i], 'holdings'] = shares * portfolio['Close'].iloc[i]
            portfolio.loc[portfolio.index[i], 'cash'] = cash
            portfolio.loc[portfolio.index[i], 'total'] = cash + (shares * portfolio['Close'].iloc[i])
        
        # Calculate returns
        portfolio['returns'] = portfolio['total'].pct_change()
        
        # Calculate metrics
        total_return = (portfolio['total'].iloc[-1] - initial_capital) / initial_capital
        
        # Buy and hold return
        buy_hold_return = (portfolio['Close'].iloc[-1] - portfolio['Close'].iloc[0]) / portfolio['Close'].iloc[0]
        
        # Calculate Sharpe ratio
        sharpe_ratio = (portfolio['returns'].mean() / portfolio['returns'].std()) * np.sqrt(252) if portfolio['returns'].std() > 0 else 0
        
        # Maximum drawdown
        rolling_max = portfolio['total'].expanding().max()
        drawdown = (portfolio['total'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Number of trades (exclude NaN values)
        num_trades = (portfolio['position'].dropna() != 0).sum()
        
        # Win rate
        trade_returns = []
        entry_price = None
        
        for i in range(len(portfolio)):
            if portfolio['position'].iloc[i] > 0:  # Buy
                entry_price = portfolio['Close'].iloc[i]
            elif portfolio['position'].iloc[i] < 0 and entry_price is not None:  # Sell
                trade_return = (portfolio['Close'].iloc[i] - entry_price) / entry_price
                trade_returns.append(trade_return)
                entry_price = None
        
        win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns) if trade_returns else 0
        
        return {
            'portfolio': portfolio,
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'final_value': portfolio['total'].iloc[-1],
            'initial_capital': initial_capital
        }
    
    def plot_backtest_results(self, backtest_results, strategy_name='Strategy'):
        """
        Plot backtesting results
        
        Args:
            backtest_results (dict): Backtesting results
            strategy_name (str): Name of the strategy
            
        Returns:
            plotly.graph_objects.Figure: Backtesting plot
        """
        portfolio = backtest_results['portfolio']
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=['Portfolio Value vs Buy & Hold', 'Price & Signals', 'Returns Distribution']
        )
        
        # Portfolio value
        fig.add_trace(
            go.Scatter(
                x=portfolio.index,
                y=portfolio['total'],
                mode='lines',
                name=f'{strategy_name} Portfolio',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Buy and hold (use actual initial capital from backtest)
        initial_capital = backtest_results.get('initial_capital', self.initial_capital)
        buy_hold_value = initial_capital * (portfolio['Close'] / portfolio['Close'].iloc[0])
        fig.add_trace(
            go.Scatter(
                x=portfolio.index,
                y=buy_hold_value,
                mode='lines',
                name='Buy & Hold',
                line=dict(color='gray', dash='dash')
            ),
            row=1, col=1
        )
        
        # Price with signals
        fig.add_trace(
            go.Scatter(
                x=portfolio.index,
                y=portfolio['Close'],
                mode='lines',
                name='Price',
                line=dict(color='black')
            ),
            row=2, col=1
        )
        
        # Buy signals
        buy_signals = portfolio[portfolio['position'] > 0]
        fig.add_trace(
            go.Scatter(
                x=buy_signals.index,
                y=buy_signals['Close'],
                mode='markers',
                name='Buy Signal',
                marker=dict(symbol='triangle-up', size=10, color='green')
            ),
            row=2, col=1
        )
        
        # Sell signals
        sell_signals = portfolio[portfolio['position'] < 0]
        fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=sell_signals['Close'],
                mode='markers',
                name='Sell Signal',
                marker=dict(symbol='triangle-down', size=10, color='red')
            ),
            row=2, col=1
        )
        
        # Returns histogram
        fig.add_trace(
            go.Histogram(
                x=portfolio['returns'].dropna() * 100,
                name='Returns',
                nbinsx=50,
                marker_color='steelblue'
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            title=f'{strategy_name} Backtesting Results',
            height=900,
            showlegend=True
        )
        
        fig.update_yaxes(title_text='Value ($)', row=1, col=1)
        fig.update_yaxes(title_text='Price ($)', row=2, col=1)
        fig.update_xaxes(title_text='Date', row=3, col=1)
        fig.update_yaxes(title_text='Frequency', row=3, col=1)
        
        return fig
