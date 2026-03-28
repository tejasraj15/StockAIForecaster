import pandas as pd
import numpy as np
from scipy import stats


class DataProcessor:
    """Cleans and enriches raw OHLCV stock data."""

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Return a cleaned copy of *data* with a DatetimeIndex, no duplicates,
        missing values filled, outliers removed, and basic derived features added.
        """
        cleaned = data.copy()

        if "Date" in cleaned.columns:
            cleaned["Date"] = pd.to_datetime(cleaned["Date"])
            cleaned = cleaned.set_index("Date")

        cleaned = cleaned.sort_index()
        cleaned = cleaned[~cleaned.index.duplicated(keep="first")]
        cleaned = self._fill_missing(cleaned)
        cleaned = self._remove_outliers(cleaned)
        cleaned = self._add_basic_features(cleaned)
        return cleaned

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fill_missing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill then backward-fill; drop any remaining NaNs."""
        return data.ffill().bfill().dropna()

    def _remove_outliers(self, data: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """Drop rows where any price column has a Z-score above *threshold*."""
        for col in ["Open", "High", "Low", "Close"]:
            if col in data.columns:
                z_scores = np.abs(stats.zscore(data[col]))
                data = data[z_scores < threshold]
        return data

    def _add_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add price-derived columns useful as ML features."""
        data["Price_Change"] = data["Close"].diff()
        data["Price_Change_Pct"] = data["Close"].pct_change()
        data["HL_Spread"] = data["High"] - data["Low"]
        data["HL_Spread_Pct"] = (data["High"] - data["Low"]) / data["Close"]
        data["OC_Spread"] = data["Close"] - data["Open"]
        data["OC_Spread_Pct"] = (data["Close"] - data["Open"]) / data["Open"]
        data["Intraday_Return"] = (data["Close"] - data["Open"]) / data["Open"]
        data["Volume_Price_Trend"] = data["Volume"] * data["Price_Change_Pct"]
        data["Range_Pct"] = (data["High"] - data["Low"]) / data["Close"]
        return data

    # ------------------------------------------------------------------
    # Public utility methods (used selectively by callers)
    # ------------------------------------------------------------------

    def create_lagged_features(self, data: pd.DataFrame, columns: list, lags: list) -> pd.DataFrame:
        """Add lag columns for each (column, lag) pair and drop resulting NaNs."""
        for col in columns:
            if col in data.columns:
                for lag in lags:
                    data[f"{col}_lag_{lag}"] = data[col].shift(lag)
        return data.dropna()

    def create_rolling_features(self, data: pd.DataFrame, columns: list, windows: list) -> pd.DataFrame:
        """Add rolling mean, std, min, max columns and drop resulting NaNs."""
        for col in columns:
            if col in data.columns:
                for window in windows:
                    data[f"{col}_roll_mean_{window}"] = data[col].rolling(window).mean()
                    data[f"{col}_roll_std_{window}"] = data[col].rolling(window).std()
                    data[f"{col}_roll_min_{window}"] = data[col].rolling(window).min()
                    data[f"{col}_roll_max_{window}"] = data[col].rolling(window).max()
        return data.dropna()

    def normalize_features(self, data: pd.DataFrame, columns: list, method: str = "minmax") -> pd.DataFrame:
        """
        Append normalised copies of *columns* (suffixed with ``_normalized``).
        *method* is ``'minmax'`` or ``'zscore'``.
        """
        from sklearn.preprocessing import MinMaxScaler, StandardScaler

        scaler = MinMaxScaler() if method == "minmax" else StandardScaler()
        for col in columns:
            if col in data.columns:
                data[f"{col}_normalized"] = scaler.fit_transform(data[[col]])
        return data
