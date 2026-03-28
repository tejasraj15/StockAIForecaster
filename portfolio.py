import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go

from data_fetcher import StockDataFetcher
from data_processor import DataProcessor


class PortfolioOptimizer:
    """Fetches multi-stock data and optimises portfolio weights."""

    def __init__(self):
        self._fetcher = StockDataFetcher()
        self._processor = DataProcessor()

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def fetch_multiple_stocks(self, tickers: list, start_date, end_date) -> dict:
        """Return a dict of {ticker: cleaned_DataFrame} for each ticker that loads successfully."""
        stock_data = {}
        for ticker in tickers:
            data = self._fetcher.fetch_data(ticker, start_date, end_date)
            if data is not None and not data.empty:
                stock_data[ticker] = self._processor.clean_data(data)
        return stock_data

    def calculate_returns(self, stock_data: dict) -> pd.DataFrame:
        """Return a DataFrame of daily percentage returns, NaN rows dropped."""
        returns = {ticker: data["Close"].pct_change()
                   for ticker, data in stock_data.items() if "Close" in data.columns}
        return pd.DataFrame(returns).dropna()

    def calculate_correlation_matrix(self, stock_data: dict) -> pd.DataFrame:
        """
        Return the correlation matrix of daily *returns* (not prices).

        Price-level correlation is spurious for non-stationary series — two
        unrelated trending stocks appear highly correlated simply because both
        prices go up over time. Return-based correlation reflects true co-movement.
        """
        returns = {ticker: data["Close"].pct_change()
                   for ticker, data in stock_data.items() if "Close" in data.columns}
        return pd.DataFrame(returns).dropna().corr()

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------

    def calculate_portfolio_metrics(self, returns_df: pd.DataFrame, weights) -> dict:
        """
        Return annualised return, volatility, and Sharpe ratio for the given weights.
        *weights* may be a dict or a numpy array ordered like ``returns_df.columns``.
        """
        if isinstance(weights, dict):
            weights = np.array([weights[t] for t in returns_df.columns])

        ann_return = float(np.sum(returns_df.mean() * weights) * 252)
        cov = returns_df.cov() * 252
        ann_vol = float(np.sqrt(weights @ cov.values @ weights))
        sharpe = (ann_return - 0.02) / ann_vol if ann_vol > 0 else 0.0

        return {"return": ann_return, "volatility": ann_vol, "sharpe_ratio": sharpe}

    def optimize_portfolio(self, returns_df: pd.DataFrame, method: str = "sharpe") -> dict:
        """
        Optimise portfolio weights for *method* (``'sharpe'``, ``'min_variance'``, or
        ``'max_return'``).  Falls back to equal weights if data is insufficient or
        the solver fails.
        """
        n = len(returns_df.columns)
        equal_weights = {t: 1.0 / n for t in returns_df.columns}

        if len(returns_df) < 30 or returns_df.isnull().all().any():
            return equal_weights

        objectives = {
            "sharpe": lambda w: -self.calculate_portfolio_metrics(returns_df, w)["sharpe_ratio"],
            "min_variance": lambda w: float(w @ (returns_df.cov() * 252).values @ w),
            "max_return": lambda w: -float(np.sum(returns_df.mean() * w) * 252),
        }
        objective = objectives.get(method, objectives["sharpe"])

        result = minimize(
            objective,
            np.ones(n) / n,
            method="SLSQP",
            bounds=[(0, 1)] * n,
            constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},
        )
        if result.success:
            return {t: float(w) for t, w in zip(returns_df.columns, result.x)}
        return equal_weights

    def generate_efficient_frontier(self, returns_df: pd.DataFrame, n_portfolios: int = 100) -> pd.DataFrame:
        """Sample *n_portfolios* random weight vectors and return their risk/return metrics."""
        n = len(returns_df.columns)
        records = []
        for _ in range(n_portfolios):
            w = np.random.dirichlet(np.ones(n))
            records.append(self.calculate_portfolio_metrics(returns_df, w))
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_efficient_frontier(self, frontier_data: pd.DataFrame,
                                 optimal_portfolio_metrics: dict = None) -> go.Figure:
        """Scatter plot of sampled portfolios coloured by Sharpe ratio, with the optimal marked."""
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=frontier_data["volatility"],
            y=frontier_data["return"],
            mode="markers",
            marker=dict(size=5, color=frontier_data["sharpe_ratio"],
                        colorscale="Viridis", showscale=True,
                        colorbar=dict(title="Sharpe Ratio")),
            text=[f"Sharpe: {s:.2f}" for s in frontier_data["sharpe_ratio"]],
            hovertemplate="Volatility: %{x:.2%}<br>Return: %{y:.2%}<br>%{text}<extra></extra>",
            name="Random Portfolios",
        ))

        if optimal_portfolio_metrics:
            fig.add_trace(go.Scatter(
                x=[optimal_portfolio_metrics["volatility"]],
                y=[optimal_portfolio_metrics["return"]],
                mode="markers",
                marker=dict(size=15, color="red", symbol="star"),
                name="Optimal Portfolio",
                hovertemplate=(
                    "Volatility: %{x:.2%}<br>Return: %{y:.2%}<br>"
                    f"Sharpe: {optimal_portfolio_metrics['sharpe_ratio']:.2f}<extra></extra>"
                ),
            ))

        fig.update_layout(
            title="Efficient Frontier",
            xaxis_title="Volatility (Annual)",
            yaxis_title="Expected Return (Annual)",
            height=500,
            xaxis=dict(tickformat=".0%"),
            yaxis=dict(tickformat=".0%"),
        )
        return fig

    def plot_correlation_heatmap(self, correlation_matrix: pd.DataFrame) -> go.Figure:
        """Heatmap of the returns correlation matrix."""
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale="RdBu",
            zmid=0,
            text=correlation_matrix.values.round(2),
            texttemplate="%{text}",
            textfont={"size": 10},
        ))
        fig.update_layout(title="Stock Returns Correlation Matrix", height=500)
        return fig

    def plot_cumulative_returns(self, stock_data: dict) -> go.Figure:
        """Line chart of cumulative return for each stock."""
        fig = go.Figure()
        for ticker, data in stock_data.items():
            if "Close" in data.columns:
                cum_ret = (1 + data["Close"].pct_change()).cumprod() - 1
                fig.add_trace(go.Scatter(
                    x=data.index, y=cum_ret * 100,
                    mode="lines", name=ticker,
                ))
        fig.update_layout(
            title="Cumulative Returns Comparison",
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
            height=500,
            hovermode="x unified",
        )
        return fig
