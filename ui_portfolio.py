import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from portfolio import PortfolioOptimizer
from report_generator import ReportGenerator


def portfolio_analysis():
    """Portfolio optimization and multi-stock analysis"""
    st.header("Portfolio Optimization")
    st.markdown("Analyze multiple stocks and optimize portfolio allocation")

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

    if st.button("Analyze Portfolio", type="primary"):
        tickers = [t.strip().upper() for t in tickers_input.split(',')]

        if len(tickers) < 2:
            st.error("Please enter at least 2 stock symbols")
            return

        with st.spinner("Fetching stock data and optimizing portfolio..."):
            try:
                portfolio_opt = PortfolioOptimizer()
                stock_data = portfolio_opt.fetch_multiple_stocks(tickers, start_date, end_date)

                if len(stock_data) < 2:
                    st.error("Could not fetch data for at least 2 stocks. Please check ticker symbols.")
                    return

                returns_df = portfolio_opt.calculate_returns(stock_data)
                optimal_weights = portfolio_opt.optimize_portfolio(returns_df, method=optimization_method)
                optimal_metrics = portfolio_opt.calculate_portfolio_metrics(returns_df, optimal_weights)
                frontier_data = portfolio_opt.generate_efficient_frontier(returns_df, n_portfolios=500)
                correlation_matrix = portfolio_opt.calculate_correlation_matrix(stock_data)

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

    if 'portfolio_data' not in st.session_state:
        return

    data = st.session_state.portfolio_data
    report_gen = ReportGenerator()

    col1, col2 = st.columns(2)
    with col1:
        portfolio_csv = report_gen.generate_portfolio_csv(
            data['optimal_weights'], data['optimal_metrics'], data['tickers']
        )
        st.download_button(
            label="Download Portfolio Report CSV",
            data=portfolio_csv,
            file_name=f"portfolio_optimization_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col2:
        data_summary = {
            'start_date': 'N/A', 'end_date': 'N/A', 'total_days': 'N/A',
            'latest_price': 'N/A', 'price_change': 'N/A', 'volatility': 'N/A'
        }
        portfolio_summary = report_gen.generate_summary_report(
            'Portfolio', data_summary, portfolio_weights=data['optimal_weights']
        )
        st.download_button(
            label="Download Portfolio Summary",
            data=portfolio_summary,
            file_name=f"portfolio_summary_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True
        )

    st.subheader("Optimal Portfolio Allocation")
    col1, col2 = st.columns([2, 1])
    with col1:
        weights_df = pd.DataFrame(
            list(data['optimal_weights'].items()), columns=['Stock', 'Weight']
        )
        weights_df['Weight (%)'] = (weights_df['Weight'] * 100).round(2)
        st.dataframe(weights_df, use_container_width=True, hide_index=True)
    with col2:
        st.metric("Expected Return (Annual)", f"{data['optimal_metrics']['return']:.2%}")
        st.metric("Volatility (Annual)", f"{data['optimal_metrics']['volatility']:.2%}")
        st.metric("Sharpe Ratio", f"{data['optimal_metrics']['sharpe_ratio']:.2f}")

    st.subheader("Efficient Frontier")
    portfolio_opt = PortfolioOptimizer()
    frontier_fig = portfolio_opt.plot_efficient_frontier(data['frontier_data'], data['optimal_metrics'])
    st.plotly_chart(frontier_fig, use_container_width=True)

    st.subheader("Stock Returns Correlation Matrix")
    correlation_fig = portfolio_opt.plot_correlation_heatmap(data['correlation_matrix'])
    st.plotly_chart(correlation_fig, use_container_width=True)

    st.subheader("Cumulative Returns Comparison")
    returns_fig = portfolio_opt.plot_cumulative_returns(data['stock_data'])
    st.plotly_chart(returns_fig, use_container_width=True)

    st.subheader("Individual Stock Statistics")
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
    st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
