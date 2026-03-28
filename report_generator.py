import pandas as pd
from datetime import datetime


class ReportGenerator:
    """Generates downloadable CSV and text reports for analysis results."""

    def generate_prediction_csv(self, results: dict, metrics: dict,
                                 test_dates: pd.DatetimeIndex, ticker: str) -> str:
        """CSV with one row per (model, date) containing actual price, prediction, and errors."""
        rows = []
        for model_name, result in results.items():
            preds = result["predictions"]
            actual = result["actual"]
            n = min(len(preds), len(actual), len(test_dates))
            for i in range(n):
                err = actual[i] - preds[i]
                rows.append({
                    "Date": test_dates[i].strftime("%Y-%m-%d"),
                    "Ticker": ticker,
                    "Model": model_name,
                    "Actual_Price": actual[i],
                    "Predicted_Price": preds[i],
                    "Error": err,
                    "Absolute_Error": abs(err),
                    "Percentage_Error": err / actual[i] * 100 if actual[i] != 0 else 0,
                })
        return pd.DataFrame(rows).to_csv(index=False)

    def generate_metrics_csv(self, metrics: dict, ticker: str) -> str:
        """CSV with one row per model and its performance metrics."""
        df = pd.DataFrame(metrics).T
        df.insert(0, "Ticker", ticker)
        df.insert(1, "Model", df.index)
        return df.reset_index(drop=True).to_csv(index=False)

    def generate_portfolio_csv(self, optimal_weights: dict,
                                performance_metrics: dict, tickers: list) -> str:
        """CSV with portfolio weights followed by summary metrics."""
        weight_rows = [{"Ticker": t, "Weight": w, "Weight_Percentage": w * 100}
                       for t, w in optimal_weights.items()]
        df = pd.DataFrame(weight_rows)

        metric_rows = [{"Ticker": k, "Weight": v, "Weight_Percentage": ""}
                       for k, v in performance_metrics.items()]
        spacer = pd.DataFrame([{"Ticker": "", "Weight": "", "Weight_Percentage": ""}])
        df = pd.concat([df, spacer, pd.DataFrame(metric_rows)], ignore_index=True)

        return df.to_csv(index=False)

    def generate_backtesting_csv(self, backtest_results: dict,
                                  strategy_name: str, ticker: str) -> str:
        """CSV with the full portfolio time series followed by summary metrics."""
        portfolio = backtest_results["portfolio"]
        df = portfolio[["Close", "signal", "position", "holdings", "cash", "total"]].copy()
        df.insert(0, "Date", df.index)
        df.insert(1, "Ticker", ticker)
        df.insert(2, "Strategy", strategy_name)
        df = df.reset_index(drop=True)

        metric_keys = ["total_return", "buy_hold_return", "sharpe_ratio",
                       "max_drawdown", "num_trades", "win_rate", "final_value"]
        metric_rows = [
            {c: ("METRICS" if c == "Date" else k if c == "Ticker" else backtest_results[k]
                 if c == "Strategy" else "")
             for c in df.columns}
            for k in metric_keys
        ]
        spacer = pd.DataFrame([{c: "" for c in df.columns}])
        df = pd.concat([df, spacer, pd.DataFrame(metric_rows)], ignore_index=True)

        return df.to_csv(index=False)

    def generate_summary_report(self, ticker: str, data_summary: dict,
                                 metrics: dict = None,
                                 portfolio_weights: dict = None,
                                 backtest_metrics: dict = None) -> str:
        """Plain-text summary report suitable for download."""

        def fmt(value, fmt_str):
            if value in ("N/A", None):
                return "N/A"
            try:
                return fmt_str.format(value)
            except (ValueError, TypeError):
                return str(value)

        lines = [
            "STOCK ANALYSIS REPORT",
            "=" * 60,
            "",
            f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Ticker    : {ticker}",
            "",
            "DATA SUMMARY",
            "-" * 60,
            f"Date Range : {data_summary.get('start_date', 'N/A')} to {data_summary.get('end_date', 'N/A')}",
            f"Total Days : {data_summary.get('total_days', 'N/A')}",
            f"Latest Price      : {fmt(data_summary.get('latest_price'), '${:.2f}')}",
            f"Price Change      : {fmt(data_summary.get('price_change'), '{:.2f}%')}",
            f"Volatility (Ann.) : {fmt(data_summary.get('volatility'), '{:.2f}%')}",
            "",
        ]

        if metrics:
            lines += ["MODEL PERFORMANCE METRICS", "-" * 60]
            for model_name, m in metrics.items():
                lines.append(f"\n{model_name}:")
                for k, v in m.items():
                    lines.append(f"  {k}: {v:.4f}")
            best = min(metrics.items(), key=lambda x: x[1].get("RMSE", float("inf")))
            lines.append(f"\nBest Model: {best[0]} (RMSE: {best[1].get('RMSE', 'N/A'):.4f})")
            lines.append("")

        if portfolio_weights:
            lines += ["PORTFOLIO OPTIMIZATION", "-" * 60, "Optimal Allocation:"]
            for sym, w in portfolio_weights.items():
                lines.append(f"  {sym}: {w * 100:.2f}%")
            lines.append("")

        if backtest_metrics:
            lines += [
                "BACKTESTING RESULTS",
                "-" * 60,
                f"Total Return    : {fmt(backtest_metrics.get('total_return'), '{:.2%}')}",
                f"Buy&Hold Return : {fmt(backtest_metrics.get('buy_hold_return'), '{:.2%}')}",
                f"Sharpe Ratio    : {fmt(backtest_metrics.get('sharpe_ratio'), '{:.2f}')}",
                f"Max Drawdown    : {fmt(backtest_metrics.get('max_drawdown'), '{:.2%}')}",
                f"Num Trades      : {backtest_metrics.get('num_trades', 'N/A')}",
                f"Win Rate        : {fmt(backtest_metrics.get('win_rate'), '{:.2%}')}",
                f"Final Value     : {fmt(backtest_metrics.get('final_value'), '${:,.2f}')}",
                "",
            ]

        lines += ["=" * 60, "End of Report"]
        return "\n".join(lines)
