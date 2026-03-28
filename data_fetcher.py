import yfinance as yf
import pandas as pd
import streamlit as st


# Module-level cache so repeated fetches for the same ticker/range are free.
# ttl=3600 means cached data expires after 1 hour.
@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_raw_history(ticker: str, start_str: str, end_str: str) -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    return stock.history(start=start_str, end=end_str)


class StockDataFetcher:
    """Fetches OHLCV stock data from Yahoo Finance."""

    _REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

    def fetch_data(self, ticker: str, start_date, end_date) -> pd.DataFrame | None:
        """
        Return a DataFrame of daily OHLCV data, or None on failure.
        Errors are silent so the UI layer decides how to report them.
        """
        try:
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

            data = _fetch_raw_history(ticker, start_str, end_str)

            if data.empty:
                return None

            data = data.reset_index()

            if not all(col in data.columns for col in self._REQUIRED_COLUMNS):
                return None

            return data

        except Exception:
            return None

    def get_stock_info(self, ticker: str) -> dict:
        """Return basic company metadata. Falls back to safe defaults on error."""
        try:
            info = yf.Ticker(ticker).info
            return {
                "name": info.get("longName", ticker),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("trailingPE", 0),
            }
        except Exception:
            return {"name": ticker, "sector": "N/A", "industry": "N/A",
                    "market_cap": 0, "pe_ratio": 0}

    def validate_ticker(self, ticker: str) -> bool:
        """Return True if *ticker* has recent tradeable data available."""
        try:
            return not yf.Ticker(ticker).history(period="5d").empty
        except Exception:
            return False
