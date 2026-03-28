import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from data_fetcher import StockDataFetcher
from data_processor import DataProcessor
from models import StockPredictor
from visualizations import StockVisualizer
from technical_indicators import TechnicalIndicators
from report_generator import ReportGenerator


def single_stock_analysis():
    """Single stock analysis and prediction"""
    st.sidebar.header("Configuration")

    ticker = st.sidebar.text_input(
        "Stock Symbol (e.g., AAPL, GOOGL, MSFT)",
        value="AAPL",
        help="Enter a valid stock ticker symbol"
    ).upper()

    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365 * 2),
            max_value=datetime.now()
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            max_value=datetime.now()
        )

    st.sidebar.subheader("Prediction Settings")
    prediction_days = st.sidebar.slider("Days to predict ahead", 1, 30, 5)
    test_size = st.sidebar.slider("Test set size (%)", 10, 40, 20) / 100

    models_to_use = st.sidebar.multiselect(
        "Select Models",
        ["Linear Regression", "Random Forest", "LSTM", "Ensemble (All Models)"],
        default=["Linear Regression", "Random Forest"]
    )

    use_optimized_weights = st.sidebar.checkbox(
        "Use Optimized Ensemble Weights",
        value=True,
        help="Optimize ensemble weights using validation set"
    )

    baseline_models = st.sidebar.multiselect(
        "Select Baseline Models",
        ["Random Walk", "Simple Moving Average"],
        default=["Random Walk", "Simple Moving Average"]
    )

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
            format_func=lambda x: f"{x // 60} minute{'s' if x // 60 != 1 else ''}" if x < 3600 else "1 hour",
            index=1,
        )
        st.session_state.auto_refresh_enabled = True
        st.session_state.refresh_interval = refresh_interval
    else:
        st.session_state.auto_refresh_enabled = False

    if st.sidebar.button("Load & Analyze Data", type="primary"):
        with st.spinner("Fetching stock data..."):
            try:
                fetcher = StockDataFetcher()
                processor = DataProcessor()
                tech_indicators = TechnicalIndicators()

                raw_data = fetcher.fetch_data(ticker, start_date, end_date)

                if raw_data is None or raw_data.empty:
                    st.error("Failed to fetch data. Please check the ticker symbol and date range.")
                    return

                processed_data = processor.clean_data(raw_data)
                processed_data = tech_indicators.add_all_indicators(processed_data)

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

    if st.session_state.data_loaded:
        _display_analysis()

        if st.button("Train Models & Make Predictions", type="primary"):
            with st.spinner("Training models and making predictions..."):
                try:
                    _make_predictions(
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

        if st.session_state.predictions_made:
            _display_predictions()

    # Auto-refresh logic
    if st.session_state.get('auto_refresh_enabled', False) and st.session_state.data_loaded:
        import time

        last_update = st.session_state.get('last_refresh_time', datetime.now())
        time_since_update = (datetime.now() - last_update).total_seconds()
        time_until_refresh = max(0, int(st.session_state.refresh_interval - time_since_update))

        st.sidebar.divider()
        st.sidebar.subheader("Auto-Refresh Status")
        st.sidebar.info(f"Last updated: {last_update.strftime('%H:%M:%S')}")
        if time_until_refresh > 0:
            st.sidebar.info(f"Next update in: {time_until_refresh} seconds")
        else:
            st.sidebar.info("Refreshing now...")

        if time_since_update >= st.session_state.refresh_interval:
            with st.spinner("Auto-refreshing data..."):
                try:
                    fetcher = StockDataFetcher()
                    processor = DataProcessor()
                    tech_indicators = TechnicalIndicators()

                    raw_data = fetcher.fetch_data(ticker, start_date, datetime.now())

                    if raw_data is not None and not raw_data.empty:
                        processed_data = processor.clean_data(raw_data)
                        processed_data = tech_indicators.add_all_indicators(processed_data)

                        st.session_state.raw_data = raw_data
                        st.session_state.processed_data = processed_data
                        st.session_state.last_refresh_time = datetime.now()

                        if st.session_state.predictions_made:
                            _make_predictions(
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
            time.sleep(1)
            st.rerun()


def _display_analysis():
    """Display data analysis and visualizations"""
    data = st.session_state.processed_data
    ticker = st.session_state.ticker
    visualizer = StockVisualizer()

    st.header(f"Data Analysis for {ticker}")

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

    st.subheader("Price Chart with Technical Indicators")
    price_fig = visualizer.plot_price_with_indicators(data, ticker)
    st.plotly_chart(price_fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Volume Analysis")
        volume_fig = visualizer.plot_volume_analysis(data)
        st.plotly_chart(volume_fig, use_container_width=True)
    with col2:
        st.subheader("Volatility Analysis")
        volatility_fig = visualizer.plot_volatility_analysis(data)
        st.plotly_chart(volatility_fig, use_container_width=True)

    st.subheader("Feature Correlation Matrix")
    correlation_fig = visualizer.plot_correlation_heatmap(data)
    st.plotly_chart(correlation_fig, use_container_width=True)

    st.subheader("Recent Data")
    st.dataframe(data.tail(10), use_container_width=True)


def _make_predictions(models_to_use, baseline_models, prediction_days, test_size, use_optimized_weights=True):
    """Train models and make predictions"""
    data = st.session_state.processed_data
    predictor = StockPredictor()

    features = predictor.prepare_features(data)
    X, y = predictor.create_features_and_targets(features, prediction_days)

    split_idx = int(len(X) * (1 - test_size))
    val_size = int(split_idx * 0.15)

    X_train = X[:split_idx - val_size]
    X_val = X[split_idx - val_size:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx - val_size]
    y_val = y[split_idx - val_size:split_idx]
    y_test = y[split_idx:]

    results = {}
    ensemble_models = {}
    use_ensemble = "Ensemble (All Models)" in models_to_use

    for model_name in models_to_use:
        if model_name == "Ensemble (All Models)":
            continue

        if model_name == "Linear Regression":
            model = predictor.train_linear_regression(X_train, y_train)
            predictions = predictor.predict(model, X_test)
            results[model_name] = {'predictions': predictions, 'actual': y_test, 'model': model}
            if use_ensemble:
                ensemble_models['LinearRegression'] = model

        elif model_name == "Random Forest":
            model = predictor.train_random_forest(X_train, y_train)
            predictions = predictor.predict(model, X_test)
            results[model_name] = {'predictions': predictions, 'actual': y_test, 'model': model}
            if use_ensemble:
                ensemble_models['RandomForest'] = model

        elif model_name == "LSTM":
            model = predictor.train_lstm(X_train, y_train, X_val, y_val)
            predictions = predictor.predict_lstm(model, X_test)
            results[model_name] = {'predictions': predictions, 'actual': y_test, 'model': model}
            if use_ensemble:
                ensemble_models['LSTM'] = model

    if use_ensemble:
        if not ensemble_models:
            ensemble_models = predictor.train_ensemble(X_train, y_train, X_val, y_val)

        if use_optimized_weights and len(ensemble_models) > 1 and len(X_val) > 0:
            weights = predictor.optimize_ensemble_weights(ensemble_models, X_val, y_val)
            st.session_state.ensemble_weights = weights
        else:
            weights = None
            st.session_state.ensemble_weights = {name: 1.0 / len(ensemble_models) for name in ensemble_models}

        ensemble_predictions = predictor.predict_ensemble(ensemble_models, X_test, weights)
        aligned_y_test = y_test[-len(ensemble_predictions):] if len(y_test) > len(ensemble_predictions) else y_test
        results["Ensemble (All Models)"] = {
            'predictions': ensemble_predictions,
            'actual': aligned_y_test,
            'model': ensemble_models
        }

    for baseline_name in baseline_models:
        if baseline_name == "Random Walk":
            predictions = predictor.random_walk_baseline(y_test)
            results[baseline_name] = {'predictions': predictions, 'actual': y_test, 'model': None}
        elif baseline_name == "Simple Moving Average":
            predictions = predictor.moving_average_baseline(
                st.session_state.processed_data['Close'],
                len(y_test)
            )
            results[baseline_name] = {'predictions': predictions, 'actual': y_test, 'model': None}

    metrics = {name: predictor.calculate_metrics(r['actual'], r['predictions']) for name, r in results.items()}

    # Walk-forward validation for non-LSTM models
    wf_results = {}
    for model_name in models_to_use:
        if model_name in ("Linear Regression", "Random Forest") and len(X) >= 50:
            model_type = 'linear_regression' if model_name == "Linear Regression" else 'random_forest'
            wf_results[model_name] = predictor.walk_forward_validate(X, y, n_splits=5, model_type=model_type)

    st.session_state.prediction_results = results
    st.session_state.prediction_metrics = metrics
    st.session_state.walk_forward_results = wf_results
    st.session_state.test_dates = data.index[split_idx + prediction_days:]


def _display_predictions():
    """Display prediction results and metrics"""
    results = st.session_state.prediction_results
    metrics = st.session_state.prediction_metrics
    test_dates = st.session_state.test_dates
    wf_results = st.session_state.get('walk_forward_results', {})
    visualizer = StockVisualizer()
    report_gen = ReportGenerator()
    ticker = st.session_state.get('ticker', 'STOCK')

    st.header("Model Predictions & Performance")

    col1, col2, col3 = st.columns(3)
    with col1:
        predictions_csv = report_gen.generate_prediction_csv(results, metrics, test_dates, ticker)
        st.download_button(
            label="Download Predictions CSV",
            data=predictions_csv,
            file_name=f"{ticker}_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col2:
        metrics_csv = report_gen.generate_metrics_csv(metrics, ticker)
        st.download_button(
            label="Download Metrics CSV",
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
            'price_change': ((st.session_state.processed_data['Close'].iloc[-1] /
                              st.session_state.processed_data['Close'].iloc[-2]) - 1) * 100,
            'volatility': st.session_state.processed_data['Close'].pct_change().std() * np.sqrt(252) * 100
        }
        summary_report = report_gen.generate_summary_report(ticker, data_summary, metrics)
        st.download_button(
            label="Download Summary Report",
            data=summary_report,
            file_name=f"{ticker}_summary_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True
        )

    st.subheader("Model Performance Metrics")
    metrics_df = pd.DataFrame(metrics).T.round(4)
    st.dataframe(metrics_df, use_container_width=True)

    best_model = metrics_df['RMSE'].idxmin()
    st.success(f"Best performing model: **{best_model}** (Lowest RMSE: {metrics_df.loc[best_model, 'RMSE']:.4f})")

    if 'ensemble_weights' in st.session_state and st.session_state.ensemble_weights:
        st.subheader("Ensemble Model Weights")
        weights_df = pd.DataFrame(
            list(st.session_state.ensemble_weights.items()),
            columns=['Model', 'Weight']
        )
        weights_df['Weight (%)'] = (weights_df['Weight'] * 100).round(2)
        st.dataframe(weights_df, use_container_width=True, hide_index=True)

    st.subheader("Actual vs Predicted Prices")
    prediction_fig = visualizer.plot_predictions_comparison(results, test_dates)
    st.plotly_chart(prediction_fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("RMSE Comparison")
        rmse_fig = visualizer.plot_metrics_comparison(metrics, 'RMSE')
        st.plotly_chart(rmse_fig, use_container_width=True)
    with col2:
        st.subheader("Directional Accuracy")
        accuracy_fig = visualizer.plot_metrics_comparison(metrics, 'Directional_Accuracy')
        st.plotly_chart(accuracy_fig, use_container_width=True)

    # Walk-forward validation results
    if wf_results:
        st.subheader("Walk-Forward Cross-Validation")
        st.caption(
            "Each fold trains only on past data and tests on the next unseen window — "
            "a realistic simulation of out-of-sample performance."
        )
        for model_name, wf_df in wf_results.items():
            with st.expander(f"{model_name} — walk-forward folds"):
                st.dataframe(wf_df.round(4), use_container_width=True, hide_index=True)
                avg = wf_df[['RMSE', 'MAE', 'R2', 'Directional_Accuracy']].mean().round(4)
                st.write("**Mean across folds:**", avg.to_dict())

    for model_name, result in results.items():
        if model_name == "Random Forest" and result['model'] is not None:
            st.subheader(f"Feature Importance — {model_name}")
            # Derive feature names the same way the predictor does so the chart labels match
            feature_names = list(StockPredictor().prepare_features(st.session_state.processed_data).columns)
            importance_fig = visualizer.plot_feature_importance(result['model'], feature_names)
            st.plotly_chart(importance_fig, use_container_width=True)
            break
