import pandas as pd
import numpy as np
from datetime import datetime
import io
import base64

class ReportGenerator:
    """Generate downloadable reports for stock analysis and predictions"""
    
    def __init__(self):
        pass
    
    def generate_prediction_csv(self, results, metrics, test_dates, ticker):
        """
        Generate CSV report for prediction results
        
        Args:
            results (dict): Prediction results from models
            metrics (dict): Performance metrics
            test_dates (pd.DatetimeIndex): Test set dates
            ticker (str): Stock ticker symbol
            
        Returns:
            str: CSV data as string
        """
        csv_data = []
        
        for model_name, result in results.items():
            predictions = result['predictions']
            actual = result['actual']
            
            min_len = min(len(predictions), len(actual), len(test_dates))
            
            for i in range(min_len):
                csv_data.append({
                    'Date': test_dates[i].strftime('%Y-%m-%d'),
                    'Ticker': ticker,
                    'Model': model_name,
                    'Actual_Price': actual[i],
                    'Predicted_Price': predictions[i],
                    'Error': actual[i] - predictions[i],
                    'Absolute_Error': abs(actual[i] - predictions[i]),
                    'Percentage_Error': ((actual[i] - predictions[i]) / actual[i] * 100) if actual[i] != 0 else 0
                })
        
        df = pd.DataFrame(csv_data)
        return df.to_csv(index=False)
    
    def generate_metrics_csv(self, metrics, ticker):
        """
        Generate CSV report for model performance metrics
        
        Args:
            metrics (dict): Performance metrics for models
            ticker (str): Stock ticker symbol
            
        Returns:
            str: CSV data as string
        """
        df = pd.DataFrame(metrics).T
        df['Ticker'] = ticker
        df['Model'] = df.index
        df = df.reset_index(drop=True)
        
        cols = ['Ticker', 'Model'] + [col for col in df.columns if col not in ['Ticker', 'Model']]
        df = df[cols]
        
        return df.to_csv(index=False)
    
    def generate_portfolio_csv(self, optimal_weights, performance_metrics, tickers):
        """
        Generate CSV report for portfolio optimization results
        
        Args:
            optimal_weights (dict): Optimal portfolio weights
            performance_metrics (dict): Portfolio performance metrics
            tickers (list): List of ticker symbols
            
        Returns:
            str: CSV data as string
        """
        data = []
        
        for ticker, weight in optimal_weights.items():
            data.append({
                'Ticker': ticker,
                'Weight': weight,
                'Weight_Percentage': weight * 100
            })
        
        df = pd.DataFrame(data)
        
        # Add metrics as additional rows in proper CSV format
        metrics_rows = []
        for key, value in performance_metrics.items():
            metrics_rows.append({
                'Ticker': key,
                'Weight': value,
                'Weight_Percentage': ''
            })
        
        if metrics_rows:
            metrics_df = pd.DataFrame(metrics_rows)
            df = pd.concat([df, pd.DataFrame([{'Ticker': '', 'Weight': '', 'Weight_Percentage': ''}]), metrics_df], ignore_index=True)
        
        return df.to_csv(index=False)
    
    def generate_backtesting_csv(self, backtest_results, strategy_name, ticker):
        """
        Generate CSV report for backtesting results
        
        Args:
            backtest_results (dict): Backtesting results
            strategy_name (str): Name of trading strategy
            ticker (str): Stock ticker symbol
            
        Returns:
            str: CSV data as string
        """
        portfolio = backtest_results['portfolio']
        
        df = portfolio[['Close', 'signal', 'position', 'holdings', 'cash', 'total']].copy()
        df['Ticker'] = ticker
        df['Strategy'] = strategy_name
        df['Date'] = df.index
        df = df.reset_index(drop=True)
        
        cols = ['Date', 'Ticker', 'Strategy'] + [col for col in df.columns if col not in ['Date', 'Ticker', 'Strategy']]
        df = df[cols]
        
        # Add metrics as additional rows in proper CSV format
        metrics_data = [
            {'Date': 'METRICS', 'Ticker': 'Total Return', 'Strategy': backtest_results['total_return'], 'Close': '', 'signal': '', 'position': '', 'holdings': '', 'cash': '', 'total': ''},
            {'Date': 'METRICS', 'Ticker': 'Buy & Hold Return', 'Strategy': backtest_results['buy_hold_return'], 'Close': '', 'signal': '', 'position': '', 'holdings': '', 'cash': '', 'total': ''},
            {'Date': 'METRICS', 'Ticker': 'Sharpe Ratio', 'Strategy': backtest_results['sharpe_ratio'], 'Close': '', 'signal': '', 'position': '', 'holdings': '', 'cash': '', 'total': ''},
            {'Date': 'METRICS', 'Ticker': 'Max Drawdown', 'Strategy': backtest_results['max_drawdown'], 'Close': '', 'signal': '', 'position': '', 'holdings': '', 'cash': '', 'total': ''},
            {'Date': 'METRICS', 'Ticker': 'Number of Trades', 'Strategy': backtest_results['num_trades'], 'Close': '', 'signal': '', 'position': '', 'holdings': '', 'cash': '', 'total': ''},
            {'Date': 'METRICS', 'Ticker': 'Win Rate', 'Strategy': backtest_results['win_rate'], 'Close': '', 'signal': '', 'position': '', 'holdings': '', 'cash': '', 'total': ''},
            {'Date': 'METRICS', 'Ticker': 'Final Value', 'Strategy': backtest_results['final_value'], 'Close': '', 'signal': '', 'position': '', 'holdings': '', 'cash': '', 'total': ''}
        ]
        
        metrics_df = pd.DataFrame(metrics_data)
        df = pd.concat([df, pd.DataFrame([{col: '' for col in df.columns}]), metrics_df], ignore_index=True)
        
        return df.to_csv(index=False)
    
    def generate_summary_report(self, ticker, data_summary, metrics=None, portfolio_weights=None, backtest_metrics=None):
        """
        Generate a text summary report
        
        Args:
            ticker (str): Stock ticker symbol
            data_summary (dict): Summary of stock data
            metrics (dict, optional): Model performance metrics
            portfolio_weights (dict, optional): Portfolio allocation weights
            backtest_metrics (dict, optional): Backtesting performance metrics
            
        Returns:
            str: Summary report as text
        """
        # Format numeric fields safely
        def safe_format(value, format_str):
            if value == 'N/A' or value is None:
                return 'N/A'
            try:
                return format_str.format(value)
            except (ValueError, TypeError):
                return str(value)
        
        report = f"""
STOCK ANALYSIS REPORT
{'=' * 60}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Ticker: {ticker}

DATA SUMMARY
{'-' * 60}
Date Range: {data_summary.get('start_date', 'N/A')} to {data_summary.get('end_date', 'N/A')}
Total Days: {data_summary.get('total_days', 'N/A')}
Latest Price: {safe_format(data_summary.get('latest_price', 'N/A'), '${:.2f}')}
Price Change: {safe_format(data_summary.get('price_change', 'N/A'), '{:.2f}%')}
Volatility (Annual): {safe_format(data_summary.get('volatility', 'N/A'), '{:.2f}%')}

"""
        
        if metrics:
            report += f"""
MODEL PERFORMANCE METRICS
{'-' * 60}
"""
            for model_name, model_metrics in metrics.items():
                report += f"\n{model_name}:\n"
                for metric_name, value in model_metrics.items():
                    report += f"  {metric_name}: {value:.4f}\n"
            
            best_model = min(metrics.items(), key=lambda x: x[1].get('RMSE', float('inf')))
            report += f"\nBest Performing Model: {best_model[0]} (RMSE: {best_model[1].get('RMSE', 'N/A'):.4f})\n"
        
        if portfolio_weights:
            report += f"""
PORTFOLIO OPTIMIZATION
{'-' * 60}
Optimal Allocation:
"""
            for ticker_sym, weight in portfolio_weights.items():
                report += f"  {ticker_sym}: {weight*100:.2f}%\n"
        
        if backtest_metrics:
            report += f"""
BACKTESTING RESULTS
{'-' * 60}
Total Return: {safe_format(backtest_metrics.get('total_return', 'N/A'), '{:.2%}')}
Buy & Hold Return: {safe_format(backtest_metrics.get('buy_hold_return', 'N/A'), '{:.2%}')}
Sharpe Ratio: {safe_format(backtest_metrics.get('sharpe_ratio', 'N/A'), '{:.2f}')}
Max Drawdown: {safe_format(backtest_metrics.get('max_drawdown', 'N/A'), '{:.2%}')}
Number of Trades: {backtest_metrics.get('num_trades', 'N/A')}
Win Rate: {safe_format(backtest_metrics.get('win_rate', 'N/A'), '{:.2%}')}
Final Value: {safe_format(backtest_metrics.get('final_value', 'N/A'), '${:,.2f}')}
"""
        
        report += f"""
{'=' * 60}
End of Report
"""
        return report
    
    def create_download_link(self, data, filename, file_label):
        """
        Create a download link for data
        
        Args:
            data (str): Data to download
            filename (str): Name of the file
            file_label (str): Label for the download button
            
        Returns:
            str: HTML download link
        """
        b64 = base64.b64encode(data.encode()).decode()
        href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{file_label}</a>'
        return href
