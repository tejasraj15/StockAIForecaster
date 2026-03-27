import pandas as pd
import numpy as np
from scipy.optimize import minimize
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from data_fetcher import StockDataFetcher
from data_processor import DataProcessor

class PortfolioOptimizer:
    """Class to handle portfolio optimization and multi-stock analysis"""
    
    def __init__(self):
        self.fetcher = StockDataFetcher()
        self.processor = DataProcessor()
    
    def fetch_multiple_stocks(self, tickers, start_date, end_date):
        """
        Fetch data for multiple stocks
        
        Args:
            tickers (list): List of stock ticker symbols
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            dict: Dictionary of stock data
        """
        stock_data = {}
        
        for ticker in tickers:
            data = self.fetcher.fetch_data(ticker, start_date, end_date)
            if data is not None and not data.empty:
                processed = self.processor.clean_data(data)
                stock_data[ticker] = processed
        
        return stock_data
    
    def calculate_returns(self, stock_data):
        """
        Calculate daily returns for all stocks
        
        Args:
            stock_data (dict): Dictionary of stock dataframes
            
        Returns:
            pd.DataFrame: DataFrame of daily returns
        """
        returns = {}
        
        for ticker, data in stock_data.items():
            if 'Close' in data.columns:
                returns[ticker] = data['Close'].pct_change()
        
        returns_df = pd.DataFrame(returns)
        returns_df = returns_df.dropna()
        
        return returns_df
    
    def calculate_portfolio_metrics(self, returns_df, weights):
        """
        Calculate portfolio performance metrics
        
        Args:
            returns_df (pd.DataFrame): DataFrame of stock returns
            weights (np.array or dict): Portfolio weights
            
        Returns:
            dict: Portfolio metrics
        """
        if isinstance(weights, dict):
            weights = np.array([weights[ticker] for ticker in returns_df.columns])
        
        # Portfolio returns
        portfolio_return = np.sum(returns_df.mean() * weights) * 252  # Annualized
        
        # Portfolio volatility
        cov_matrix = returns_df.cov() * 252  # Annualized
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # Sharpe ratio (assuming risk-free rate of 0.02)
        risk_free_rate = 0.02
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        return {
            'return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio
        }
    
    def optimize_portfolio(self, returns_df, method='sharpe'):
        """
        Optimize portfolio weights
        
        Args:
            returns_df (pd.DataFrame): DataFrame of stock returns
            method (str): Optimization method ('sharpe', 'min_variance', 'max_return')
            
        Returns:
            dict: Optimized weights
        """
        n_assets = len(returns_df.columns)
        
        # Guard against insufficient data
        if len(returns_df) < 30 or returns_df.isnull().all().any():
            st.warning("Insufficient data for optimization. Using equal weights.")
            return {ticker: 1.0 / n_assets for ticker in returns_df.columns}
        
        # Objective functions
        def neg_sharpe(weights):
            metrics = self.calculate_portfolio_metrics(returns_df, weights)
            return -metrics['sharpe_ratio']
        
        def portfolio_variance(weights):
            cov_matrix = returns_df.cov() * 252
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        def neg_return(weights):
            return -np.sum(returns_df.mean() * weights) * 252
        
        # Constraints: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        # Bounds: weights between 0 and 1 (no short selling)
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess: equal weights
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        # Select objective function
        if method == 'sharpe':
            objective = neg_sharpe
        elif method == 'min_variance':
            objective = portfolio_variance
        elif method == 'max_return':
            objective = neg_return
        else:
            objective = neg_sharpe
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            optimized_weights = {ticker: float(weight) 
                               for ticker, weight in zip(returns_df.columns, result.x)}
            return optimized_weights
        else:
            # Return equal weights if optimization fails
            return {ticker: 1.0 / n_assets for ticker in returns_df.columns}
    
    def generate_efficient_frontier(self, returns_df, n_portfolios=100):
        """
        Generate efficient frontier data
        
        Args:
            returns_df (pd.DataFrame): DataFrame of stock returns
            n_portfolios (int): Number of random portfolios to generate
            
        Returns:
            pd.DataFrame: Efficient frontier data
        """
        n_assets = len(returns_df.columns)
        results = []
        
        for _ in range(n_portfolios):
            # Random weights
            weights = np.random.random(n_assets)
            weights /= np.sum(weights)
            
            # Calculate metrics
            metrics = self.calculate_portfolio_metrics(returns_df, weights)
            
            results.append({
                'return': metrics['return'],
                'volatility': metrics['volatility'],
                'sharpe_ratio': metrics['sharpe_ratio']
            })
        
        return pd.DataFrame(results)
    
    def calculate_correlation_matrix(self, stock_data):
        """
        Calculate correlation matrix for stock daily returns.

        Returns are used instead of prices because price-level correlation
        is spurious for non-stationary series — two unrelated trending stocks
        will appear highly correlated simply because both prices go up over time.

        Args:
            stock_data (dict): Dictionary of stock dataframes

        Returns:
            pd.DataFrame: Correlation matrix of daily returns
        """
        returns = {}

        for ticker, data in stock_data.items():
            if 'Close' in data.columns:
                returns[ticker] = data['Close'].pct_change()

        returns_df = pd.DataFrame(returns).dropna()
        correlation_matrix = returns_df.corr()

        return correlation_matrix
    
    def plot_efficient_frontier(self, frontier_data, optimal_portfolio_metrics=None):
        """
        Plot efficient frontier
        
        Args:
            frontier_data (pd.DataFrame): Efficient frontier data
            optimal_portfolio_metrics (dict): Optimal portfolio metrics to highlight
            
        Returns:
            plotly.graph_objects.Figure: Efficient frontier plot
        """
        fig = go.Figure()
        
        # Plot random portfolios
        fig.add_trace(go.Scatter(
            x=frontier_data['volatility'],
            y=frontier_data['return'],
            mode='markers',
            marker=dict(
                size=5,
                color=frontier_data['sharpe_ratio'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Sharpe Ratio')
            ),
            text=[f"Sharpe: {s:.2f}" for s in frontier_data['sharpe_ratio']],
            hovertemplate='Volatility: %{x:.2%}<br>Return: %{y:.2%}<br>%{text}<extra></extra>',
            name='Random Portfolios'
        ))
        
        # Highlight optimal portfolio if provided
        if optimal_portfolio_metrics:
            fig.add_trace(go.Scatter(
                x=[optimal_portfolio_metrics['volatility']],
                y=[optimal_portfolio_metrics['return']],
                mode='markers',
                marker=dict(size=15, color='red', symbol='star'),
                name='Optimal Portfolio',
                hovertemplate='Volatility: %{x:.2%}<br>Return: %{y:.2%}<br>Sharpe: ' + 
                             f"{optimal_portfolio_metrics['sharpe_ratio']:.2f}<extra></extra>"
            ))
        
        fig.update_layout(
            title='Efficient Frontier',
            xaxis_title='Volatility (Annual)',
            yaxis_title='Expected Return (Annual)',
            height=500,
            xaxis=dict(tickformat='.0%'),
            yaxis=dict(tickformat='.0%')
        )
        
        return fig
    
    def plot_correlation_heatmap(self, correlation_matrix):
        """
        Plot correlation heatmap
        
        Args:
            correlation_matrix (pd.DataFrame): Correlation matrix
            
        Returns:
            plotly.graph_objects.Figure: Correlation heatmap
        """
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title='Stock Returns Correlation Matrix',
            height=500
        )
        
        return fig
    
    def plot_cumulative_returns(self, stock_data):
        """
        Plot cumulative returns for all stocks
        
        Args:
            stock_data (dict): Dictionary of stock dataframes
            
        Returns:
            plotly.graph_objects.Figure: Cumulative returns plot
        """
        fig = go.Figure()
        
        for ticker, data in stock_data.items():
            if 'Close' in data.columns:
                cumulative_return = (1 + data['Close'].pct_change()).cumprod() - 1
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=cumulative_return * 100,
                    mode='lines',
                    name=ticker
                ))
        
        fig.update_layout(
            title='Cumulative Returns Comparison',
            xaxis_title='Date',
            yaxis_title='Cumulative Return (%)',
            height=500,
            hovermode='x unified'
        )
        
        return fig
