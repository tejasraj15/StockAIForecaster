import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

from data_fetcher import StockDataFetcher
from data_processor import DataProcessor
from technical_indicators import TechnicalIndicators
from backtesting import BacktestingFramework
from report_generator import ReportGenerator


def backtesting_analysis():
    """Backtesting trading strategies"""
    st.header("Strategy Backtesting")
    st.markdown("Test trading strategies on historical data")

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

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365 * 2),
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

    st.subheader("Strategy Parameters")

    short_window = overbought = oversold = None
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

    initial_capital = st.number_input(
        "Initial Capital ($)",
        min_value=1000,
        max_value=1000000,
        value=10000,
        step=1000
    )

    if st.button("Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            try:
                fetcher = StockDataFetcher()
                processor = DataProcessor()
                tech_indicators = TechnicalIndicators()
                backtester = BacktestingFramework()

                raw_data = fetcher.fetch_data(ticker, start_date, end_date)

                if raw_data is None or raw_data.empty:
                    st.error("Failed to fetch data. Please check the ticker symbol.")
                    return

                processed_data = processor.clean_data(raw_data)
                processed_data = tech_indicators.add_all_indicators(processed_data)

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

                results = backtester.backtest_strategy(signals, initial_capital)

                st.session_state.backtest_results = results
                st.session_state.backtest_strategy = strategy
                st.session_state.backtest_ticker = ticker
                st.session_state.backtest_initial_capital = initial_capital

                st.success("Backtest completed!")

            except Exception as e:
                st.error(f"Error in backtesting: {str(e)}")
                return

    if 'backtest_results' not in st.session_state:
        return

    results = st.session_state.backtest_results
    strategy_name = st.session_state.backtest_strategy
    ticker_name = st.session_state.backtest_ticker
    capital = st.session_state.get('backtest_initial_capital', initial_capital)
    report_gen = ReportGenerator()

    col1, col2 = st.columns(2)
    with col1:
        backtest_csv = report_gen.generate_backtesting_csv(results, strategy_name, ticker_name)
        st.download_button(
            label="Download Backtest Results CSV",
            data=backtest_csv,
            file_name=f"{ticker_name}_backtest_{strategy_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col2:
        data_summary = {
            'start_date': 'N/A', 'end_date': 'N/A', 'total_days': 'N/A',
            'latest_price': 'N/A', 'price_change': 'N/A', 'volatility': 'N/A'
        }
        backtest_summary = report_gen.generate_summary_report(
            ticker_name, data_summary, backtest_metrics=results
        )
        st.download_button(
            label="Download Backtest Summary",
            data=backtest_summary,
            file_name=f"{ticker_name}_backtest_summary_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True
        )

    st.subheader(f"{strategy_name} Performance — {ticker_name}")

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

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of Trades", f"{results['num_trades']}")
    with col2:
        st.metric("Final Value", f"${results['final_value']:,.2f}")
    with col3:
        st.metric("Buy & Hold Return", f"{results['buy_hold_return']:.2%}")

    st.subheader("Backtest Visualization")
    backtester = BacktestingFramework()
    backtest_fig = backtester.plot_backtest_results(results, strategy_name)
    st.plotly_chart(backtest_fig, use_container_width=True)

    st.subheader("Strategy vs Buy & Hold")
    comparison_df = pd.DataFrame({
        'Strategy': [strategy_name, 'Buy & Hold'],
        'Total Return': [
            f"{results['total_return']:.2%}",
            f"{results['buy_hold_return']:.2%}"
        ],
        'Final Value': [
            f"${results['final_value']:,.2f}",
            f"${capital * (1 + results['buy_hold_return']):,.2f}"
        ]
    })
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
