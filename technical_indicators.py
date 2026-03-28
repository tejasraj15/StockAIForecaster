import pandas as pd
import numpy as np


class TechnicalIndicators:
    """Calculates and appends technical indicators to a stock DataFrame."""

    def add_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add all supported indicators. Silently skips on error."""
        for method in (
            self.add_moving_averages,
            self.add_rsi,
            self.add_macd,
            self.add_bollinger_bands,
            self.add_stochastic,
            self.add_atr,
            self.add_williams_r,
            self.add_cci,
        ):
            try:
                data = method(data)
            except Exception:
                pass
        return data

    def add_moving_averages(self, data: pd.DataFrame, periods: list = [5, 10, 20, 50, 200]) -> pd.DataFrame:
        """Add SMA and EMA columns for each period in *periods*."""
        for p in periods:
            if len(data) >= p:
                data[f"SMA_{p}"] = data["Close"].rolling(p).mean()
                data[f"EMA_{p}"] = data["Close"].ewm(span=p).mean()
        return data

    def add_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Relative Strength Index (RSI)."""
        delta = data["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        data["RSI"] = 100 - (100 / (1 + gain / loss))
        return data

    def add_macd(self, data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Add MACD line, signal line, and histogram."""
        exp_fast = data["Close"].ewm(span=fast).mean()
        exp_slow = data["Close"].ewm(span=slow).mean()
        data["MACD"] = exp_fast - exp_slow
        data["MACD_Signal"] = data["MACD"].ewm(span=signal).mean()
        data["MACD_Histogram"] = data["MACD"] - data["MACD_Signal"]
        return data

    def add_bollinger_bands(self, data: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """Add upper/lower Bollinger Bands, mid-line, width, and %B position."""
        sma = data["Close"].rolling(period).mean()
        std = data["Close"].rolling(period).std()
        data["BB_Upper"] = sma + std * std_dev
        data["BB_Lower"] = sma - std * std_dev
        data["BB_Middle"] = sma
        data["BB_Width"] = data["BB_Upper"] - data["BB_Lower"]
        data["BB_Position"] = (data["Close"] - data["BB_Lower"]) / (data["BB_Upper"] - data["BB_Lower"])
        return data

    def add_stochastic(self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Add Stochastic %K and %D."""
        low = data["Low"].rolling(k_period).min()
        high = data["High"].rolling(k_period).max()
        data["Stoch_K"] = 100 * (data["Close"] - low) / (high - low)
        data["Stoch_D"] = data["Stoch_K"].rolling(d_period).mean()
        return data

    def add_atr(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average True Range (ATR)."""
        hl = data["High"] - data["Low"]
        hc = (data["High"] - data["Close"].shift()).abs()
        lc = (data["Low"] - data["Close"].shift()).abs()
        true_range = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        data["ATR"] = true_range.rolling(period).mean()
        return data

    def add_williams_r(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Williams %R."""
        high = data["High"].rolling(period).max()
        low = data["Low"].rolling(period).min()
        data["Williams_R"] = -100 * (high - data["Close"]) / (high - low)
        return data

    def add_cci(self, data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Add Commodity Channel Index (CCI)."""
        tp = (data["High"] + data["Low"] + data["Close"]) / 3
        sma_tp = tp.rolling(period).mean()
        mean_dev = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
        data["CCI"] = (tp - sma_tp) / (0.015 * mean_dev)
        return data

    def add_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add OBV, VPT, and volume moving averages."""
        data["Volume_SMA_10"] = data["Volume"].rolling(10).mean()
        data["Volume_SMA_20"] = data["Volume"].rolling(20).mean()
        data["Volume_ROC"] = data["Volume"].pct_change(periods=10) * 100
        data["OBV"] = (np.sign(data["Close"].diff()) * data["Volume"]).fillna(0).cumsum()
        data["VPT"] = (data["Volume"] * data["Close"].pct_change()).fillna(0).cumsum()
        return data
