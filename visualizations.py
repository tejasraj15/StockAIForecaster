import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


class StockVisualizer:
    """Builds Plotly figures for all stock analysis views."""

    _MA_COLORS = ["orange", "red", "green", "purple", "brown"]
    _PRED_COLORS = ["red", "blue", "green", "orange", "purple", "brown"]

    # ------------------------------------------------------------------
    # Single-stock charts
    # ------------------------------------------------------------------

    def plot_price_with_indicators(self, data: pd.DataFrame, ticker: str) -> go.Figure:
        """Four-panel chart: candlesticks + MAs, volume, RSI, MACD."""
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.15, 0.2, 0.15],
            subplot_titles=["Price & Moving Averages", "Volume", "RSI", "MACD"],
        )

        # Candlesticks
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data["Open"], high=data["High"],
            low=data["Low"], close=data["Close"],
            name="Price",
        ), row=1, col=1)

        # Moving averages
        ma_cols = [(c, c.replace("_", " ")) for c in data.columns if "SMA_" in c or "EMA_" in c]
        for i, (col, label) in enumerate(ma_cols[:5]):
            fig.add_trace(go.Scatter(
                x=data.index, y=data[col], mode="lines",
                name=label, line=dict(color=self._MA_COLORS[i % len(self._MA_COLORS)], width=1),
            ), row=1, col=1)

        # Bollinger Bands
        if {"BB_Upper", "BB_Lower"}.issubset(data.columns):
            fig.add_trace(go.Scatter(
                x=data.index, y=data["BB_Upper"], mode="lines",
                name="BB Upper", line=dict(color="gray", width=1, dash="dash"), showlegend=False,
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=data.index, y=data["BB_Lower"], mode="lines",
                name="BB Lower", line=dict(color="gray", width=1, dash="dash"),
                fill="tonexty", fillcolor="rgba(128,128,128,0.1)",
            ), row=1, col=1)

        # Volume
        vol_colors = ["green" if row["Close"] >= row["Open"] else "red"
                      for _, row in data.iterrows()]
        fig.add_trace(go.Bar(
            x=data.index, y=data["Volume"], name="Volume", marker_color=vol_colors,
        ), row=2, col=1)

        # RSI
        if "RSI" in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data["RSI"], mode="lines",
                name="RSI", line=dict(color="purple"),
            ), row=3, col=1)
            for level, color in [(70, "red"), (30, "green"), (50, "gray")]:
                fig.add_hline(y=level, line_dash="dash", line_color=color, row=3, col=1)

        # MACD
        if {"MACD", "MACD_Signal"}.issubset(data.columns):
            fig.add_trace(go.Scatter(
                x=data.index, y=data["MACD"], mode="lines",
                name="MACD", line=dict(color="blue"),
            ), row=4, col=1)
            fig.add_trace(go.Scatter(
                x=data.index, y=data["MACD_Signal"], mode="lines",
                name="Signal", line=dict(color="red"),
            ), row=4, col=1)
            if "MACD_Histogram" in data.columns:
                hist_colors = ["green" if x >= 0 else "red" for x in data["MACD_Histogram"]]
                fig.add_trace(go.Bar(
                    x=data.index, y=data["MACD_Histogram"],
                    name="Histogram", marker_color=hist_colors,
                ), row=4, col=1)

        fig.update_layout(
            title=f"{ticker} — Comprehensive Technical Analysis",
            xaxis_rangeslider_visible=False,
            height=800, showlegend=True,
        )
        return fig

    def plot_volume_analysis(self, data: pd.DataFrame) -> go.Figure:
        """Two-panel chart: coloured volume bars and volume moving averages."""
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
            subplot_titles=["Volume vs Price", "Volume Moving Averages"],
        )

        vol_colors = ["green" if row["Close"] >= row["Open"] else "red"
                      for _, row in data.iterrows()]
        fig.add_trace(go.Bar(
            x=data.index, y=data["Volume"], name="Volume", marker_color=vol_colors,
        ), row=1, col=1)

        for col, color, label in [("Volume_SMA_10", "orange", "Volume SMA 10"),
                                   ("Volume_SMA_20", "red", "Volume SMA 20")]:
            if col in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index, y=data[col], mode="lines",
                    name=label, line=dict(color=color),
                ), row=2, col=1)

        fig.update_layout(title="Volume Analysis", height=400, showlegend=True)
        return fig

    def plot_volatility_analysis(self, data: pd.DataFrame) -> go.Figure:
        """Rolling 10-day and 30-day annualised volatility, plus ATR (%)."""
        returns = data["Close"].pct_change()
        fig = go.Figure()

        for window, color, label in [(10, "red", "10-day"), (30, "blue", "30-day")]:
            vol = returns.rolling(window).std() * np.sqrt(252) * 100
            fig.add_trace(go.Scatter(
                x=data.index, y=vol, mode="lines",
                name=f"{label} Volatility", line=dict(color=color),
            ))

        if "ATR" in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data["ATR"] / data["Close"] * 100,
                mode="lines", name="ATR (%)", line=dict(color="green"),
            ))

        fig.update_layout(
            title="Volatility Analysis",
            yaxis_title="Volatility (%)", height=400,
        )
        return fig

    def plot_correlation_heatmap(self, data: pd.DataFrame) -> go.Figure:
        """Heatmap of feature correlations for a single stock's feature DataFrame."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        base = ["Open", "High", "Low", "Close", "Volume"]
        tech = [c for c in numeric_cols
                if any(ind in c for ind in ["SMA", "EMA", "RSI", "MACD", "BB", "Stoch", "ATR", "Williams", "CCI"])]
        cols = [c for c in base + tech[:10] if c in data.columns]

        if len(cols) < 2:
            return go.Figure()

        corr = data[cols].corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale="RdBu", zmid=0,
            text=corr.values.round(2),
            texttemplate="%{text}", textfont={"size": 8},
        ))
        fig.update_layout(title="Feature Correlation Matrix", height=500)
        return fig

    # ------------------------------------------------------------------
    # Prediction charts
    # ------------------------------------------------------------------

    def plot_predictions_comparison(self, results: dict, test_dates: pd.Index) -> go.Figure:
        """Overlay actual prices and all model predictions on a single chart."""
        fig = go.Figure()

        actual = next((r["actual"] for r in results.values() if len(r["actual"]) > 0), None)
        if actual is None:
            return fig

        n = min(len(actual), len(test_dates))
        actual = actual[:n]
        dates = test_dates[:n]

        fig.add_trace(go.Scatter(
            x=dates, y=actual, mode="lines",
            name="Actual", line=dict(color="black", width=2),
        ))

        for i, (name, result) in enumerate(results.items()):
            if len(result["predictions"]) > 0:
                preds = result["predictions"][:n]
                fig.add_trace(go.Scatter(
                    x=dates, y=preds, mode="lines",
                    name=f"{name} Prediction",
                    line=dict(color=self._PRED_COLORS[i % len(self._PRED_COLORS)], width=1, dash="dash"),
                ))

        fig.update_layout(
            title="Actual vs Predicted Stock Prices",
            xaxis_title="Date", yaxis_title="Price ($)",
            height=500, hovermode="x unified",
        )
        return fig

    def plot_metrics_comparison(self, metrics: dict, metric_name: str) -> go.Figure:
        """Bar chart comparing *metric_name* across all models. Best bar is green."""
        models = list(metrics.keys())
        values = [metrics[m].get(metric_name, 0) for m in models]

        best = min(values) if metric_name in ("RMSE", "MAE") else max(values)
        colors = ["green" if v == best else "red" for v in values]

        fig = go.Figure(data=[go.Bar(
            x=models, y=values,
            marker_color=colors,
            text=[f"{v:.4f}" for v in values],
            textposition="auto",
        )])
        fig.update_layout(
            title=f"{metric_name} Comparison Across Models",
            yaxis_title=metric_name, height=400,
        )
        return fig

    def plot_feature_importance(self, model, feature_names: list = None) -> go.Figure:
        """
        Horizontal bar chart of Random Forest feature importances.

        Args:
            model: A trained model with a ``feature_importances_`` attribute.
            feature_names: Column names matching the training features.
                           Falls back to generic labels if not provided.
        """
        if not hasattr(model, "feature_importances_"):
            return go.Figure()

        importances = model.feature_importances_
        if feature_names is None or len(feature_names) != len(importances):
            feature_names = [f"Feature_{i}" for i in range(len(importances))]

        indices = np.argsort(importances)[::-1][:20]  # top 20

        fig = go.Figure(data=[go.Bar(
            x=[feature_names[i] for i in indices],
            y=[importances[i] for i in indices],
            marker_color="steelblue",
        )])
        fig.update_layout(
            title="Feature Importance (Random Forest)",
            xaxis_title="Feature", yaxis_title="Importance",
            height=400, xaxis_tickangle=-45,
        )
        return fig
