import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from data_fetcher import StockDataFetcher
from data_processor import DataProcessor
from models import StockPredictor
from visualizations import StockVisualizer
from technical_indicators import TechnicalIndicators
from portfolio import PortfolioOptimizer
from backtesting import BacktestingFramework
from report_generator import ReportGenerator

# Page configuration
st.set_page_config(
    page_title="Stock Analysis & Prediction ML App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'predictions_made' not in st.session_state:
    st.session_state.predictions_made = False

def main():
    st.title("📈 Stock Price Analysis & Prediction")
    st.markdown("### Machine Learning Application for Stock Market Analysis")
    
    # Tabs for different features
    tab1, tab2, tab3 = st.tabs(["📊 Single Stock Analysis", "💼 Portfolio Optimization", "🔄 Backtesting"])
    
    with tab1:
        single_stock_analysis()
    
    with tab2:
        portfolio_analysis()
    
    with tab3:
        backtesting_analysis()

def single_stock_analysis():
    """Single stock analysis and prediction"""
    # Sidebar for parameters
    st.sidebar.header("Configuration")
    
    # Stock symbol selection
    ticker = st.sidebar.text_input(
        "Stock Symbol (e.g., AAPL, GOOGL, MSFT)",
        value="AAPL",
        help="Enter a valid stock ticker symbol"
    ).upper()
    
    # Date range selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365*2),
            max_value=datetime.now()
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            max_value=datetime.now()
        )
    
    # Prediction parameters
    st.sidebar.subheader("Prediction Settings")
    prediction_days = st.sidebar.slider("Days to predict ahead", 1, 30, 5)
    test_size = st.sidebar.slider("Test set size (%)", 10, 40, 20) / 100
    
    # Model selection
    models_to_use = st.sidebar.multiselect(
        "Select Models",
        ["Linear Regression", "Random Forest", "LSTM", "Ensemble (All Models)"],
        default=["Linear Regression", "Random Forest"]
    )
    
    # Ensemble optimization
    use_optimized_weights = st.sidebar.checkbox(
        "Use Optimized Ensemble Weights",
        value=True,
        help="Optimize ensemble weights using validation set"
    )
    
    # Baseline models
    baseline_models = st.sidebar.multiselect(
        "Select Baseline Models",
        ["Random Walk", "Simple Moving Average"],
        default=["Random Walk", "Simple Moving Average"]
    )
    
    # Auto-refresh settings
    st.sidebar.subheader("Auto-Refresh Settings")
    enable_auto_refresh = st.sidebar.checkbox(
        "Enable Auto-Refresh",
        value=False,
        help="Automatically fetch latest data and update predictions"
    )
    
    if enable_auto_refresh:
        refresh_interval = st.sidebar.selectbox(
            "Refresh Interval",
            [60, 300, 900, 1800, 3600],
            format_func=lambda x: f"{x//60} minute{'s' if x//60 != 1 else ''}" if x < 3600 else "1 hour",
            index=1,
            help="How often to refresh data and predictions"
        )
        st.session_state.auto_refresh_enabled = True
        st.session_state.refresh_interval = refresh_interval
    else:
        st.session_state.auto_refresh_enabled = False
    
    # Load data button
    if st.sidebar.button("Load & Analyze Data", type="primary"):
        with st.spinner("Fetching stock data..."):
            try:
                # Initialize components
                fetcher = StockDataFetcher()
                processor = DataProcessor()
                predictor = StockPredictor()
                visualizer = StockVisualizer()
                tech_indicators = TechnicalIndicators()
                
                # Fetch data
                raw_data = fetcher.fetch_data(ticker, start_date, end_date)
                
                if raw_data is None or raw_data.empty:
                    st.error("Failed to fetch data. Please check the ticker symbol and date range.")
                    return
                
                # Process data
                processed_data = processor.clean_data(raw_data)
                
                # Add technical indicators
                processed_data = tech_indicators.add_all_indicators(processed_data)
                
                # Store in session state
                st.session_state.raw_data = raw_data
                st.session_state.processed_data = processed_data
                st.session_state.ticker = ticker
                st.session_state.data_loaded = True
                st.session_state.predictions_made = False
                st.session_state.last_refresh_time = datetime.now()
                
                st.success(f"Successfully loaded data for {ticker}")
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                return
    
    # Display analysis if data is loaded
    if st.session_state.data_loaded:
        display_analysis()
        
        # Model training and prediction
        if st.button("Train Models & Make Predictions", type="primary"):
            with st.spinner("Training models and making predictions..."):
                try:
                    make_predictions(
                        models_to_use, 
                        baseline_models, 
                        prediction_days, 
                        test_size,
                        use_optimized_weights
                    )
                    st.session_state.predictions_made = True
                    st.success("Models trained and predictions completed!")
                except Exception as e:
                    st.error(f"Error in model training: {str(e)}")
        
        # Display predictions if available
        if st.session_state.predictions_made:
            display_predictions()
    
    # Auto-refresh logic
    if st.session_state.get('auto_refresh_enabled', False) and st.session_state.data_loaded:
        import time
        
        # Display auto-refresh status
        last_update = st.session_state.get('last_refresh_time', datetime.now())
        time_since_update = (datetime.now() - last_update).total_seconds()
        time_until_refresh = max(0, int(st.session_state.refresh_interval - time_since_update))
        
        st.sidebar.divider()
        st.sidebar.subheader("🔄 Auto-Refresh Status")
        st.sidebar.info(f"Last updated: {last_update.strftime('%H:%M:%S')}")
        if time_until_refresh > 0:
            st.sidebar.info(f"Next update in: {time_until_refresh} seconds")
        else:
            st.sidebar.info("Refreshing now...")
        
        # Auto-refresh if interval has passed
        if time_since_update >= st.session_state.refresh_interval:
            with st.spinner("Auto-refreshing data..."):
                try:
                    # Fetch latest data
                    fetcher = StockDataFetcher()
                    processor = DataProcessor()
                    tech_indicators = TechnicalIndicators()
                    
                    # Update end date to now
                    raw_data = fetcher.fetch_data(ticker, start_date, datetime.now())
                    
                    if raw_data is not None and not raw_data.empty:
                        # Process data
                        processed_data = processor.clean_data(raw_data)
                        processed_data = tech_indicators.add_all_indicators(processed_data)
                        
                        # Update session state
                        st.session_state.raw_data = raw_data
                        st.session_state.processed_data = processed_data
                        st.session_state.last_refresh_time = datetime.now()
                        
                        # Auto-retrain if predictions were made
                        if st.session_state.predictions_made:
                            make_predictions(
                                models_to_use, 
                                baseline_models, 
                                prediction_days, 
                                test_size,
                                use_optimized_weights
                            )
                            st.success("Data and predictions auto-updated!")
                        else:
                            st.success("Data auto-updated!")
                        
                        st.rerun()
                except Exception as e:
                    st.error(f"Auto-refresh error: {str(e)}")
        else:
            # Trigger rerun to update countdown
            time.sleep(1)
            st.rerun()

def display_analysis():
    """Display data analysis and visualizations"""
    data = st.session_state.processed_data
    ticker = st.session_state.ticker
    visualizer = StockVisualizer()
    
    st.header(f"📊 Data Analysis for {ticker}")
    
    # Data overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(data))
    with col2:
        st.metric("Date Range", f"{len(data)} days")
    with col3:
        latest_price = data['Close'].iloc[-1]
        price_change = ((data['Close'].iloc[-1] / data['Close'].iloc[-2]) - 1) * 100
        st.metric("Latest Price", f"${latest_price:.2f}", f"{price_change:+.2f}%")
    with col4:
        volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100
        st.metric("Volatility (Annual)", f"{volatility:.1f}%")
    
    # Price chart with technical indicators
    st.subheader("📈 Price Chart with Technical Indicators")
    price_fig = visualizer.plot_price_with_indicators(data, ticker)
    st.plotly_chart(price_fig, use_container_width=True)
    
    # Technical indicators
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Volume Analysis")
        volume_fig = visualizer.plot_volume_analysis(data)
        st.plotly_chart(volume_fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Volatility Analysis")
        volatility_fig = visualizer.plot_volatility_analysis(data)
        st.plotly_chart(volatility_fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("🔗 Feature Correlation Matrix")
    correlation_fig = visualizer.plot_correlation_heatmap(data)
    st.plotly_chart(correlation_fig, use_container_width=True)
    
    # Data table
    st.subheader("📋 Recent Data")
    st.dataframe(data.tail(10), use_container_width=True)

def make_predictions(models_to_use, baseline_models, prediction_days, test_size, use_optimized_weights=True):
    """Train models and make predictions"""
    data = st.session_state.processed_data
    predictor = StockPredictor()
    
    # Prepare features and targets
    features = predictor.prepare_features(data)
    X, y = predictor.create_features_and_targets(features, prediction_days)
    
    # Train-test-validation split
    split_idx = int(len(X) * (1 - test_size))
    val_size = int(split_idx * 0.15)  # 15% of training data for validation
    
    X_train, X_val, X_test = X[:split_idx-val_size], X[split_idx-val_size:split_idx], X[split_idx:]
    y_train, y_val, y_test = y[:split_idx-val_size], y[split_idx-val_size:split_idx], y[split_idx:]
    
    results = {}
    ensemble_models = {}
    
    # Check if Ensemble is requested
    use_ensemble = "Ensemble (All Models)" in models_to_use
    
    # Train ML models
    for model_name in models_to_use:
        if model_name == "Ensemble (All Models)":
            continue  # Handle ensemble separately
            
        if model_name == "Linear Regression":
            model = predictor.train_linear_regression(X_train, y_train)
            predictions = predictor.predict(model, X_test)
            results[model_name] = {
                'predictions': predictions,
                'actual': y_test,
                'model': model
            }
            if use_ensemble:
                ensemble_models['LinearRegression'] = model
        
        elif model_name == "Random Forest":
            model = predictor.train_random_forest(X_train, y_train)
            predictions = predictor.predict(model, X_test)
            results[model_name] = {
                'predictions': predictions,
                'actual': y_test,
                'model': model
            }
            if use_ensemble:
                ensemble_models['RandomForest'] = model
        
        elif model_name == "LSTM":
            model = predictor.train_lstm(X_train, y_train, X_val, y_val)
            predictions = predictor.predict_lstm(model, X_test)
            results[model_name] = {
                'predictions': predictions,
                'actual': y_test,
                'model': model
            }
            if use_ensemble:
                ensemble_models['LSTM'] = model
    
    # Train ensemble if requested
    if use_ensemble:
        # If no individual models selected, train all for ensemble
        if not ensemble_models:
            ensemble_models = predictor.train_ensemble(X_train, y_train, X_val, y_val)
        
        # Optimize weights if requested
        if use_optimized_weights and len(ensemble_models) > 1 and len(X_val) > 0:
            weights = predictor.optimize_ensemble_weights(ensemble_models, X_val, y_val)
            st.session_state.ensemble_weights = weights
        else:
            weights = None
            st.session_state.ensemble_weights = {name: 1.0/len(ensemble_models) for name in ensemble_models}
        
        # Make ensemble predictions
        ensemble_predictions = predictor.predict_ensemble(ensemble_models, X_test, weights)
        # Align actual values to match ensemble predictions length
        aligned_y_test = y_test[-len(ensemble_predictions):] if len(y_test) > len(ensemble_predictions) else y_test
        results["Ensemble (All Models)"] = {
            'predictions': ensemble_predictions,
            'actual': aligned_y_test,
            'model': ensemble_models
        }
    
    # Train baseline models
    for baseline_name in baseline_models:
        if baseline_name == "Random Walk":
            predictions = predictor.random_walk_baseline(y_test)
            results[baseline_name] = {
                'predictions': predictions,
                'actual': y_test,
                'model': None
            }
        
        elif baseline_name == "Simple Moving Average":
            predictions = predictor.moving_average_baseline(
                st.session_state.processed_data['Close'], 
                len(y_test)
            )
            results[baseline_name] = {
                'predictions': predictions,
                'actual': y_test,
                'model': None
            }
    
    # Calculate metrics
    metrics = {}
    for name, result in results.items():
        metrics[name] = predictor.calculate_metrics(
            result['actual'], 
            result['predictions']
        )
    
    # Store results
    st.session_state.prediction_results = results
    st.session_state.prediction_metrics = metrics
    st.session_state.test_dates = data.index[split_idx + prediction_days:]

def display_predictions():
    """Display prediction results and metrics"""
    results = st.session_state.prediction_results
    metrics = st.session_state.prediction_metrics
    test_dates = st.session_state.test_dates
    visualizer = StockVisualizer()
    report_gen = ReportGenerator()
    ticker = st.session_state.get('ticker', 'STOCK')
    
    st.header("🎯 Model Predictions & Performance")
    
    # Download buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        predictions_csv = report_gen.generate_prediction_csv(results, metrics, test_dates, ticker)
        st.download_button(
            label="📥 Download Predictions CSV",
            data=predictions_csv,
            file_name=f"{ticker}_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        metrics_csv = report_gen.generate_metrics_csv(metrics, ticker)
        st.download_button(
            label="📥 Download Metrics CSV",
            data=metrics_csv,
            file_name=f"{ticker}_metrics_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col3:
        data_summary = {
            'start_date': st.session_state.processed_data.index[0].strftime('%Y-%m-%d'),
            'end_date': st.session_state.processed_data.index[-1].strftime('%Y-%m-%d'),
            'total_days': len(st.session_state.processed_data),
            'latest_price': st.session_state.processed_data['Close'].iloc[-1],
            'price_change': ((st.session_state.processed_data['Close'].iloc[-1] / st.session_state.processed_data['Close'].iloc[-2]) - 1) * 100,
            'volatility': st.session_state.processed_data['Close'].pct_change().std() * np.sqrt(252) * 100
        }
        summary_report = report_gen.generate_summary_report(ticker, data_summary, metrics)
        st.download_button(
            label="📥 Download Summary Report",
            data=summary_report,
            file_name=f"{ticker}_summary_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    # Performance metrics table
    st.subheader("📈 Model Performance Metrics")
    metrics_df = pd.DataFrame(metrics).T
    metrics_df = metrics_df.round(4)
    st.dataframe(metrics_df, use_container_width=True)
    
    # Best performing model
    best_model = metrics_df['RMSE'].idxmin()
    st.success(f"🏆 Best performing model: **{best_model}** (Lowest RMSE: {metrics_df.loc[best_model, 'RMSE']:.4f})")
    
    # Display ensemble weights if available
    if 'ensemble_weights' in st.session_state and st.session_state.ensemble_weights:
        st.subheader("⚖️ Ensemble Model Weights")
        weights_df = pd.DataFrame(
            list(st.session_state.ensemble_weights.items()),
            columns=['Model', 'Weight']
        )
        weights_df['Weight (%)'] = (weights_df['Weight'] * 100).round(2)
        st.dataframe(weights_df, use_container_width=True, hide_index=True)
    
    # Predictions vs actual plot
    st.subheader("📊 Actual vs Predicted Prices")
    prediction_fig = visualizer.plot_predictions_comparison(results, test_dates)
    st.plotly_chart(prediction_fig, use_container_width=True)
    
    # Model performance comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 RMSE Comparison")
        rmse_fig = visualizer.plot_metrics_comparison(metrics, 'RMSE')
        st.plotly_chart(rmse_fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Directional Accuracy")
        accuracy_fig = visualizer.plot_metrics_comparison(metrics, 'Directional_Accuracy')
        st.plotly_chart(accuracy_fig, use_container_width=True)
    
    # Feature importance (for tree-based models)
    for model_name, result in results.items():
        if model_name == "Random Forest" and result['model'] is not None:
            st.subheader(f"🎯 Feature Importance - {model_name}")
            importance_fig = visualizer.plot_feature_importance(result['model'])
            st.plotly_chart(importance_fig, use_container_width=True)
            break

def portfolio_analysis():
    """Portfolio optimization and multi-stock analysis"""
    st.header("💼 Portfolio Optimization")
    st.markdown("Analyze multiple stocks and optimize portfolio allocation")
    
    # Portfolio configuration
    col1, col2 = st.columns([2, 1])
    
    with col1:
        tickers_input = st.text_input(
            "Enter stock symbols (comma-separated)",
            value="AAPL,GOOGL,MSFT,AMZN",
            help="Enter 2 or more stock symbols separated by commas"
        )
    
    with col2:
        optimization_method = st.selectbox(
            "Optimization Method",
            ["sharpe", "min_variance", "max_return"],
            format_func=lambda x: {
                "sharpe": "Max Sharpe Ratio",
                "min_variance": "Min Variance",
                "max_return": "Max Return"
            }[x]
        )
    
    # Date range
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365),
            max_value=datetime.now(),
            key="portfolio_start"
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            max_value=datetime.now(),
            key="portfolio_end"
        )
    
    # Analyze portfolio button
    if st.button("Analyze Portfolio", type="primary"):
        tickers = [t.strip().upper() for t in tickers_input.split(',')]
        
        if len(tickers) < 2:
            st.error("Please enter at least 2 stock symbols")
            return
        
        with st.spinner("Fetching stock data and optimizing portfolio..."):
            try:
                # Initialize portfolio optimizer
                portfolio_opt = PortfolioOptimizer()
                
                # Fetch stock data
                stock_data = portfolio_opt.fetch_multiple_stocks(tickers, start_date, end_date)
                
                if len(stock_data) < 2:
                    st.error("Could not fetch data for at least 2 stocks. Please check ticker symbols.")
                    return
                
                # Calculate returns
                returns_df = portfolio_opt.calculate_returns(stock_data)
                
                # Optimize portfolio
                optimal_weights = portfolio_opt.optimize_portfolio(returns_df, method=optimization_method)
                optimal_metrics = portfolio_opt.calculate_portfolio_metrics(returns_df, optimal_weights)
                
                # Generate efficient frontier
                frontier_data = portfolio_opt.generate_efficient_frontier(returns_df, n_portfolios=500)
                
                # Calculate correlation matrix
                correlation_matrix = portfolio_opt.calculate_correlation_matrix(stock_data)
                
                # Store in session state
                st.session_state.portfolio_data = {
                    'stock_data': stock_data,
                    'returns_df': returns_df,
                    'optimal_weights': optimal_weights,
                    'optimal_metrics': optimal_metrics,
                    'frontier_data': frontier_data,
                    'correlation_matrix': correlation_matrix,
                    'tickers': list(stock_data.keys())
                }
                
                st.success("Portfolio analysis completed!")
                
            except Exception as e:
                st.error(f"Error in portfolio analysis: {str(e)}")
                return
    
    # Display results if available
    if 'portfolio_data' in st.session_state:
        data = st.session_state.portfolio_data
        report_gen = ReportGenerator()
        
        # Download buttons
        col1, col2 = st.columns(2)
        
        with col1:
            portfolio_csv = report_gen.generate_portfolio_csv(
                data['optimal_weights'],
                data['optimal_metrics'],
                data['tickers']
            )
            st.download_button(
                label="📥 Download Portfolio Report CSV",
                data=portfolio_csv,
                file_name=f"portfolio_optimization_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            data_summary = {
                'start_date': 'N/A',
                'end_date': 'N/A',
                'total_days': 'N/A',
                'latest_price': 'N/A',
                'price_change': 'N/A',
                'volatility': 'N/A'
            }
            portfolio_summary = report_gen.generate_summary_report(
                'Portfolio',
                data_summary,
                portfolio_weights=data['optimal_weights']
            )
            st.download_button(
                label="📥 Download Portfolio Summary",
                data=portfolio_summary,
                file_name=f"portfolio_summary_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        # Optimal portfolio weights
        st.subheader("🎯 Optimal Portfolio Allocation")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            weights_df = pd.DataFrame(
                list(data['optimal_weights'].items()),
                columns=['Stock', 'Weight']
            )
            weights_df['Weight (%)'] = (weights_df['Weight'] * 100).round(2)
            st.dataframe(weights_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.metric("Expected Return (Annual)", f"{data['optimal_metrics']['return']:.2%}")
            st.metric("Volatility (Annual)", f"{data['optimal_metrics']['volatility']:.2%}")
            st.metric("Sharpe Ratio", f"{data['optimal_metrics']['sharpe_ratio']:.2f}")
        
        # Efficient frontier
        st.subheader("📈 Efficient Frontier")
        portfolio_opt = PortfolioOptimizer()
        frontier_fig = portfolio_opt.plot_efficient_frontier(
            data['frontier_data'],
            data['optimal_metrics']
        )
        st.plotly_chart(frontier_fig, use_container_width=True)
        
        # Correlation matrix
        st.subheader("🔗 Stock Correlation Matrix")
        correlation_fig = portfolio_opt.plot_correlation_heatmap(data['correlation_matrix'])
        st.plotly_chart(correlation_fig, use_container_width=True)
        
        # Cumulative returns comparison
        st.subheader("📊 Cumulative Returns Comparison")
        returns_fig = portfolio_opt.plot_cumulative_returns(data['stock_data'])
        st.plotly_chart(returns_fig, use_container_width=True)
        
        # Individual stock statistics
        st.subheader("📋 Individual Stock Statistics")
        stats_data = []
        for ticker in data['tickers']:
            returns = data['returns_df'][ticker]
            stats_data.append({
                'Stock': ticker,
                'Mean Return (Annual)': f"{returns.mean() * 252:.2%}",
                'Volatility (Annual)': f"{returns.std() * np.sqrt(252):.2%}",
                'Min Return': f"{returns.min():.2%}",
                'Max Return': f"{returns.max():.2%}"
            })
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

def backtesting_analysis():
    """Backtesting trading strategies"""
    st.header("🔄 Strategy Backtesting")
    st.markdown("Test trading strategies on historical data")
    
    # Strategy configuration
    col1, col2 = st.columns([2, 1])
    
    with col1:
        ticker = st.text_input(
            "Stock Symbol",
            value="AAPL",
            key="backtest_ticker"
        ).upper()
    
    with col2:
        strategy = st.selectbox(
            "Trading Strategy",
            ["SMA Crossover", "RSI", "MACD", "Bollinger Bands"]
        )
    
    # Date range
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365*2),
            max_value=datetime.now(),
            key="backtest_start"
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            max_value=datetime.now(),
            key="backtest_end"
        )
    
    # Strategy parameters
    st.subheader("Strategy Parameters")
    
    if strategy == "SMA Crossover":
        col1, col2 = st.columns(2)
        with col1:
            short_window = st.slider("Short MA Window", 5, 50, 20)
        with col2:
            long_window = st.slider("Long MA Window", 20, 200, 50)
    elif strategy == "RSI":
        col1, col2 = st.columns(2)
        with col1:
            oversold = st.slider("Oversold Threshold", 10, 40, 30)
        with col2:
            overbought = st.slider("Overbought Threshold", 60, 90, 70)
    
    # Initial capital
    initial_capital = st.number_input(
        "Initial Capital ($)",
        min_value=1000,
        max_value=1000000,
        value=10000,
        step=1000
    )
    
    # Run backtest button
    if st.button("Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            try:
                # Fetch data
                fetcher = StockDataFetcher()
                processor = DataProcessor()
                tech_indicators = TechnicalIndicators()
                backtester = BacktestingFramework()
                
                raw_data = fetcher.fetch_data(ticker, start_date, end_date)
                
                if raw_data is None or raw_data.empty:
                    st.error("Failed to fetch data. Please check the ticker symbol.")
                    return
                
                # Process data
                processed_data = processor.clean_data(raw_data)
                processed_data = tech_indicators.add_all_indicators(processed_data)
                
                # Generate signals based on strategy
                if strategy == "SMA Crossover":
                    signals = backtester.simple_moving_average_strategy(
                        processed_data, short_window, long_window
                    )
                elif strategy == "RSI":
                    signals = backtester.rsi_strategy(
                        processed_data, oversold=oversold, overbought=overbought
                    )
                elif strategy == "MACD":
                    signals = backtester.macd_strategy(processed_data)
                elif strategy == "Bollinger Bands":
                    signals = backtester.bollinger_bands_strategy(processed_data)
                else:
                    st.error("Unknown strategy")
                    return
                
                # Run backtest
                results = backtester.backtest_strategy(signals, initial_capital)
                
                # Store results
                st.session_state.backtest_results = results
                st.session_state.backtest_strategy = strategy
                st.session_state.backtest_ticker = ticker
                
                st.success("Backtest completed!")
                
            except Exception as e:
                st.error(f"Error in backtesting: {str(e)}")
                return
    
    # Display results if available
    if 'backtest_results' in st.session_state:
        results = st.session_state.backtest_results
        strategy_name = st.session_state.backtest_strategy
        ticker_name = st.session_state.backtest_ticker
        report_gen = ReportGenerator()
        
        # Download buttons
        col1, col2 = st.columns(2)
        
        with col1:
            backtest_csv = report_gen.generate_backtesting_csv(
                results,
                strategy_name,
                ticker_name
            )
            st.download_button(
                label="📥 Download Backtest Results CSV",
                data=backtest_csv,
                file_name=f"{ticker_name}_backtest_{strategy_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            data_summary = {
                'start_date': 'N/A',
                'end_date': 'N/A',
                'total_days': 'N/A',
                'latest_price': 'N/A',
                'price_change': 'N/A',
                'volatility': 'N/A'
            }
            backtest_summary = report_gen.generate_summary_report(
                ticker_name,
                data_summary,
                backtest_metrics=results
            )
            st.download_button(
                label="📥 Download Backtest Summary",
                data=backtest_summary,
                file_name=f"{ticker_name}_backtest_summary_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        # Performance metrics
        st.subheader(f"📊 {strategy_name} Performance - {ticker_name}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Return",
                f"{results['total_return']:.2%}",
                delta=f"{(results['total_return'] - results['buy_hold_return']):.2%} vs Buy & Hold"
            )
        
        with col2:
            st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
        
        with col3:
            st.metric("Max Drawdown", f"{results['max_drawdown']:.2%}")
        
        with col4:
            st.metric("Win Rate", f"{results['win_rate']:.2%}")
        
        # Additional metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Number of Trades", f"{results['num_trades']}")
        
        with col2:
            st.metric("Final Value", f"${results['final_value']:,.2f}")
        
        with col3:
            st.metric("Buy & Hold Return", f"{results['buy_hold_return']:.2%}")
        
        # Backtest visualization
        st.subheader("📈 Backtest Visualization")
        backtester = BacktestingFramework()
        backtest_fig = backtester.plot_backtest_results(results, strategy_name)
        st.plotly_chart(backtest_fig, use_container_width=True)
        
        # Performance comparison
        st.subheader("📊 Strategy vs Buy & Hold")
        
        comparison_data = {
            'Strategy': [strategy_name, 'Buy & Hold'],
            'Total Return': [
                f"{results['total_return']:.2%}",
                f"{results['buy_hold_return']:.2%}"
            ],
            'Final Value': [
                f"${results['final_value']:,.2f}",
                f"${initial_capital * (1 + results['buy_hold_return']):,.2f}"
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
