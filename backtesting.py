import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class BacktestingFramework:
    """Generates trading signals and backtests them against historical price data."""

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def simple_moving_average_strategy(self, data: pd.DataFrame,
                                        short_window: int = 20,
                                        long_window: int = 50) -> pd.DataFrame:
        """Buy when short MA crosses above long MA; sell when it crosses below."""
        signals = data.copy()
        signals["short_ma"] = data["Close"].rolling(short_window).mean()
        signals["long_ma"] = data["Close"].rolling(long_window).mean()
        signals["signal"] = np.where(signals["short_ma"] > signals["long_ma"], 1, -1)
        signals["position"] = signals["signal"].diff()
        return signals

    def rsi_strategy(self, data: pd.DataFrame,
                     rsi_column: str = "RSI",
                     oversold: int = 30,
                     overbought: int = 70) -> pd.DataFrame:
        """Buy when RSI dips below *oversold*; sell when it rises above *overbought*."""
        if rsi_column not in data.columns:
            raise ValueError(f"Column '{rsi_column}' not found. Run add_rsi() first.")
        signals = data.copy()
        signals["signal"] = 0
        signals.loc[signals[rsi_column] < oversold, "signal"] = 1
        signals.loc[signals[rsi_column] > overbought, "signal"] = -1
        signals["position"] = signals["signal"].diff()
        return signals

    def macd_strategy(self, data: pd.DataFrame) -> pd.DataFrame:
        """Buy when MACD crosses above its signal line; sell when it crosses below."""
        for col in ("MACD", "MACD_Signal"):
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found. Run add_macd() first.")
        signals = data.copy()
        signals["signal"] = np.where(signals["MACD"] > signals["MACD_Signal"], 1, -1)
        signals["position"] = signals["signal"].diff()
        return signals

    def bollinger_bands_strategy(self, data: pd.DataFrame) -> pd.DataFrame:
        """Buy when price falls below the lower band; sell when it rises above the upper band."""
        for col in ("BB_Upper", "BB_Lower"):
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found. Run add_bollinger_bands() first.")
        signals = data.copy()
        signals["signal"] = 0
        signals.loc[signals["Close"] < signals["BB_Lower"], "signal"] = 1
        signals.loc[signals["Close"] > signals["BB_Upper"], "signal"] = -1
        signals["position"] = signals["signal"].diff()
        return signals

    # ------------------------------------------------------------------
    # Backtest engine
    # ------------------------------------------------------------------

    def backtest_strategy(self, signals: pd.DataFrame, initial_capital: float = 10_000) -> dict:
        """
        Simulate a long-only strategy on *signals* starting with *initial_capital*.

        Returns a dict with keys: portfolio, total_return, buy_hold_return,
        sharpe_ratio, max_drawdown, num_trades, win_rate, final_value, initial_capital.
        """
        portfolio = signals.copy()
        portfolio[["holdings", "cash", "total", "returns"]] = 0.0

        position, shares, cash = 0, 0, initial_capital

        for i in range(len(portfolio)):
            pos_change = portfolio["position"].iloc[i]

            if not pd.isna(pos_change):
                if pos_change > 0 and cash > 0:          # buy signal
                    shares_to_buy = int(cash / portfolio["Close"].iloc[i])
                    if shares_to_buy > 0:
                        shares += shares_to_buy
                        cash -= shares_to_buy * portfolio["Close"].iloc[i]
                        position = 1
                elif pos_change < 0 and shares > 0:      # sell signal
                    cash += shares * portfolio["Close"].iloc[i]
                    shares = 0
                    position = 0

            portfolio.loc[portfolio.index[i], "holdings"] = shares * portfolio["Close"].iloc[i]
            portfolio.loc[portfolio.index[i], "cash"] = cash
            portfolio.loc[portfolio.index[i], "total"] = cash + shares * portfolio["Close"].iloc[i]

        portfolio["returns"] = portfolio["total"].pct_change()

        total_return = (portfolio["total"].iloc[-1] - initial_capital) / initial_capital
        buy_hold_return = (portfolio["Close"].iloc[-1] - portfolio["Close"].iloc[0]) / portfolio["Close"].iloc[0]

        sharpe = (portfolio["returns"].mean() / portfolio["returns"].std() * np.sqrt(252)
                  if portfolio["returns"].std() > 0 else 0.0)

        rolling_max = portfolio["total"].expanding().max()
        max_drawdown = ((portfolio["total"] - rolling_max) / rolling_max).min()

        num_trades = int((portfolio["position"].dropna() != 0).sum())

        trade_returns, entry_price = [], None
        for i in range(len(portfolio)):
            p = portfolio["position"].iloc[i]
            if p > 0:
                entry_price = portfolio["Close"].iloc[i]
            elif p < 0 and entry_price is not None:
                trade_returns.append((portfolio["Close"].iloc[i] - entry_price) / entry_price)
                entry_price = None

        win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns) if trade_returns else 0.0

        return {
            "portfolio": portfolio,
            "total_return": total_return,
            "buy_hold_return": buy_hold_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "num_trades": num_trades,
            "win_rate": win_rate,
            "final_value": portfolio["total"].iloc[-1],
            "initial_capital": initial_capital,
        }

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_backtest_results(self, backtest_results: dict, strategy_name: str = "Strategy") -> go.Figure:
        """Return a three-panel Plotly figure: portfolio value, price/signals, returns histogram."""
        portfolio = backtest_results["portfolio"]
        initial_capital = backtest_results.get("initial_capital", 10_000)

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=["Portfolio Value vs Buy & Hold", "Price & Signals", "Returns Distribution"],
        )

        # Portfolio value
        fig.add_trace(go.Scatter(
            x=portfolio.index, y=portfolio["total"],
            mode="lines", name=f"{strategy_name} Portfolio",
            line=dict(color="blue"),
        ), row=1, col=1)

        # Buy-and-hold benchmark
        buy_hold = initial_capital * (portfolio["Close"] / portfolio["Close"].iloc[0])
        fig.add_trace(go.Scatter(
            x=portfolio.index, y=buy_hold,
            mode="lines", name="Buy & Hold",
            line=dict(color="gray", dash="dash"),
        ), row=1, col=1)

        # Price
        fig.add_trace(go.Scatter(
            x=portfolio.index, y=portfolio["Close"],
            mode="lines", name="Price", line=dict(color="black"),
        ), row=2, col=1)

        # Buy/sell markers
        buy_pts = portfolio[portfolio["position"] > 0]
        fig.add_trace(go.Scatter(
            x=buy_pts.index, y=buy_pts["Close"], mode="markers",
            name="Buy", marker=dict(symbol="triangle-up", size=10, color="green"),
        ), row=2, col=1)

        sell_pts = portfolio[portfolio["position"] < 0]
        fig.add_trace(go.Scatter(
            x=sell_pts.index, y=sell_pts["Close"], mode="markers",
            name="Sell", marker=dict(symbol="triangle-down", size=10, color="red"),
        ), row=2, col=1)

        # Returns histogram
        fig.add_trace(go.Histogram(
            x=portfolio["returns"].dropna() * 100,
            name="Daily Returns", nbinsx=50, marker_color="steelblue",
        ), row=3, col=1)

        fig.update_layout(
            title=f"{strategy_name} Backtesting Results",
            height=900, showlegend=True,
        )
        fig.update_yaxes(title_text="Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Price ($)", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Frequency", row=3, col=1)

        return fig
