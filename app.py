import streamlit as st
import warnings
warnings.filterwarnings('ignore')

from ui_single_stock import single_stock_analysis
from ui_portfolio import portfolio_analysis
from ui_backtesting import backtesting_analysis

st.set_page_config(
    page_title="Stock Analysis & Prediction ML App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'predictions_made' not in st.session_state:
    st.session_state.predictions_made = False


def main():
    st.title("Stock Price Analysis & Prediction")
    st.markdown("### Machine Learning Application for Stock Market Analysis")

    tab1, tab2, tab3 = st.tabs(["Single Stock Analysis", "Portfolio Optimization", "Backtesting"])

    with tab1:
        single_stock_analysis()

    with tab2:
        portfolio_analysis()

    with tab3:
        backtesting_analysis()


if __name__ == "__main__":
    main()
