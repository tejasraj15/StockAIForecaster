import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


class StockPredictor:
    """Trains ML models and generates stock price predictions."""

    def __init__(self):
        self.scaler = StandardScaler()

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Select and clean the feature columns from *data*.
        Returns an empty DataFrame if no suitable features exist.
        """
        feature_columns = [
            "Open", "High", "Low", "Volume",
            "Price_Change_Pct", "HL_Spread_Pct", "OC_Spread_Pct",
            "Range_Pct", "Intraday_Return",
        ]
        feature_columns += [c for c in data.columns if "SMA_" in c or "EMA_" in c][:10]
        for col in ["RSI", "MACD", "MACD_Signal", "BB_Position",
                    "Stoch_K", "ATR", "Williams_R", "CCI"]:
            if col in data.columns:
                feature_columns.append(col)

        available = [c for c in feature_columns if c in data.columns]
        if not available:
            return pd.DataFrame()

        features = data[available].copy()
        features = features.ffill().bfill()
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
        return features

    def create_features_and_targets(self, data: pd.DataFrame, prediction_days: int = 1):
        """
        Return (X, y) where y is the Close price *prediction_days* into the future.
        """
        if data.empty:
            return np.array([]), np.array([])

        X = data.values

        if "Close" in data.columns:
            close_idx = list(data.columns).index("Close")
            y = np.roll(X[:, close_idx], -prediction_days)
        else:
            return np.array([]), np.array([])

        return X[:-prediction_days], y[:-prediction_days]

    # ------------------------------------------------------------------
    # Model training
    # ------------------------------------------------------------------

    def train_linear_regression(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fit a scaled LinearRegression and return the model, or None on error."""
        try:
            X_scaled = self.scaler.fit_transform(X_train)
            model = LinearRegression()
            model.fit(X_scaled, y_train)
            return model
        except Exception:
            return None

    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fit a RandomForestRegressor and return the model, or None on error."""
        try:
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
            )
            model.fit(X_train, y_train)
            return model
        except Exception:
            return None

    def train_lstm(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   sequence_length: int = 60):
        """Build and train a two-layer LSTM. Returns the model or None."""
        try:
            seq_len = min(sequence_length, max(1, len(X_train) // 2))

            X_seq, y_seq = [], []
            for i in range(seq_len, len(X_train)):
                X_seq.append(X_train[i - seq_len:i])
                y_seq.append(y_train[i])

            if not X_seq:
                return None

            X_seq = np.array(X_seq)
            y_seq = np.array(y_seq)

            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(seq_len, X_train.shape[1])),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1),
            ])
            model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
            model.fit(X_seq, y_seq, batch_size=32, epochs=50,
                      verbose=0, validation_split=0.1)
            return model
        except Exception:
            return None

    def train_ensemble(self, X_train, y_train, X_val, y_val) -> dict:
        """Train LR, RF, and LSTM (if enough data). Returns a dict of models."""
        models = {}
        lr = self.train_linear_regression(X_train, y_train)
        if lr:
            models["LinearRegression"] = lr
        rf = self.train_random_forest(X_train, y_train)
        if rf:
            models["RandomForest"] = rf
        if len(X_train) > 60:
            lstm = self.train_lstm(X_train, y_train, X_val, y_val)
            if lstm:
                models["LSTM"] = lstm
        return models

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, model, X_test: np.ndarray) -> np.ndarray:
        """Run inference. Applies scaler transform automatically for LinearRegression."""
        if model is None:
            return np.array([])
        try:
            if isinstance(model, LinearRegression):
                preds = model.predict(self.scaler.transform(X_test))
            else:
                preds = model.predict(X_test)
            return preds.flatten() if preds.ndim > 1 else preds
        except Exception:
            return np.array([])

    def predict_lstm(self, model, X_test: np.ndarray, sequence_length: int = 60) -> np.ndarray:
        """Slide a window over X_test and return LSTM predictions."""
        if model is None or len(X_test) < sequence_length:
            return np.array([])
        try:
            X_seq = [X_test[i - sequence_length:i]
                     for i in range(sequence_length, len(X_test))]
            if not X_seq:
                return np.array([])
            return model.predict(np.array(X_seq), verbose=0).flatten()
        except Exception:
            return np.array([])

    def predict_ensemble(self, models: dict, X_test: np.ndarray, weights: dict = None) -> np.ndarray:
        """Weighted average of all model predictions. Defaults to equal weights."""
        if not models:
            return np.array([])

        raw_preds, names = [], []
        for name, model in models.items():
            pred = self.predict_lstm(model, X_test) if name == "LSTM" else self.predict(model, X_test)
            if len(pred) > 0:
                raw_preds.append(pred)
                names.append(name)

        if not raw_preds:
            return np.array([])

        min_len = min(len(p) for p in raw_preds)
        aligned = [p[-min_len:] for p in raw_preds]

        if weights is None:
            weights = {n: 1.0 / len(names) for n in names}

        total = sum(weights.get(n, 0) for n in names)
        weights = {n: weights.get(n, 0) / total for n in names} if total > 0 else {n: 1.0 / len(names) for n in names}

        result = np.zeros(min_len)
        for i, name in enumerate(names):
            result += aligned[i] * weights[name]
        return result

    def optimize_ensemble_weights(self, models: dict, X_val: np.ndarray, y_val: np.ndarray) -> dict:
        """
        Find weights that minimise RMSE on the validation set via SLSQP.
        Falls back to equal weights if optimisation fails.
        """
        from scipy.optimize import minimize

        names = list(models.keys())
        n = len(names)
        if n == 0:
            return {}

        preds = []
        for name in names:
            p = self.predict_lstm(models[name], X_val) if name == "LSTM" else self.predict(models[name], X_val)
            preds.append(p if len(p) > 0 else np.zeros(len(y_val)))

        min_len = min(len(p) for p in preds + [y_val])
        preds = [p[-min_len:] for p in preds]
        y_val = y_val[-min_len:]

        def objective(w):
            return np.sqrt(mean_squared_error(y_val, sum(preds[i] * w[i] for i in range(n))))

        result = minimize(
            objective,
            np.ones(n) / n,
            method="SLSQP",
            bounds=[(0, 1)] * n,
            constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},
        )
        if result.success:
            return {name: float(w) for name, w in zip(names, result.x)}
        return {name: 1.0 / n for name in names}

    # ------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------

    def random_walk_baseline(self, y_test: np.ndarray) -> np.ndarray:
        """Predict tomorrow = today (naïve random-walk baseline)."""
        if len(y_test) == 0:
            return np.array([])
        preds = np.roll(y_test, 1)
        preds[0] = y_test[0]
        return preds

    def moving_average_baseline(self, prices: pd.Series, n_predictions: int, window: int = 5) -> np.ndarray:
        """Predict each future value as the trailing *window*-day mean."""
        window = max(1, min(window, len(prices)))
        last = prices.tail(window + n_predictions)
        preds = []
        for i in range(n_predictions):
            tail = last.iloc[-(window - i):] if i > 0 else last.iloc[-window:]
            preds.append(tail.mean())
        return np.array(preds)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Return RMSE, MAE, R², and directional accuracy (%)."""
        _empty = {"RMSE": float("inf"), "MAE": float("inf"),
                  "R2": -float("inf"), "Directional_Accuracy": 0.0}
        if len(y_true) == 0 or len(y_pred) == 0:
            return _empty

        n = min(len(y_true), len(y_pred))
        y_true, y_pred = y_true[:n], y_pred[:n]

        dir_acc = 0.0
        if n > 1:
            dir_acc = float(np.mean((np.diff(y_true) > 0) == (np.diff(y_pred) > 0))) * 100

        return {
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "R2": float(r2_score(y_true, y_pred)),
            "Directional_Accuracy": dir_acc,
        }

    # ------------------------------------------------------------------
    # Walk-forward validation
    # ------------------------------------------------------------------

    def walk_forward_validate(self, X: np.ndarray, y: np.ndarray,
                               n_splits: int = 5,
                               model_type: str = "linear_regression") -> pd.DataFrame:
        """
        Walk-forward time-series cross-validation.

        Each fold trains only on past data and tests on the immediately
        following window — a realistic simulation of out-of-sample performance,
        unlike k-fold which allows future data to leak into training.

        Args:
            X: Feature matrix.
            y: Target vector.
            n_splits: Number of folds.
            model_type: ``'linear_regression'`` or ``'random_forest'``.

        Returns:
            DataFrame with columns [Fold, Train_Size, Test_Size, RMSE, MAE, R2,
            Directional_Accuracy].
        """
        n = len(X)
        min_train = max(20, n // (n_splits + 1))
        fold_size = max(5, (n - min_train) // n_splits)

        records = []
        for fold in range(n_splits):
            train_end = min_train + fold * fold_size
            test_end = min(train_end + fold_size, n)

            if train_end >= n or train_end >= test_end:
                break

            X_tr, y_tr = X[:train_end], y[:train_end]
            X_te, y_te = X[train_end:test_end], y[train_end:test_end]

            model = (self.train_random_forest(X_tr, y_tr)
                     if model_type == "random_forest"
                     else self.train_linear_regression(X_tr, y_tr))

            metrics = self.calculate_metrics(y_te, self.predict(model, X_te))
            metrics.update({"Fold": fold + 1, "Train_Size": train_end, "Test_Size": len(y_te)})
            records.append(metrics)

        cols = ["Fold", "Train_Size", "Test_Size", "RMSE", "MAE", "R2", "Directional_Accuracy"]
        return pd.DataFrame(records)[cols]
