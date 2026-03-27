import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    """Class to handle machine learning models for stock prediction"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def prepare_features(self, data):
        """
        Prepare features for machine learning models
        
        Args:
            data (pd.DataFrame): Processed stock data
            
        Returns:
            pd.DataFrame: Feature matrix
        """
        try:
            # Select relevant features for prediction
            feature_columns = [
                'Open', 'High', 'Low', 'Volume',
                'Price_Change_Pct', 'HL_Spread_Pct', 'OC_Spread_Pct',
                'Range_Pct', 'Intraday_Return'
            ]
            
            # Add moving averages if available
            ma_columns = [col for col in data.columns if 'SMA_' in col or 'EMA_' in col]
            feature_columns.extend(ma_columns[:10])  # Limit to first 10 MA columns
            
            # Add technical indicators if available
            tech_columns = ['RSI', 'MACD', 'MACD_Signal', 'BB_Position', 'Stoch_K', 'ATR', 'Williams_R', 'CCI']
            for col in tech_columns:
                if col in data.columns:
                    feature_columns.append(col)
            
            # Select available features
            available_features = [col for col in feature_columns if col in data.columns]
            
            if not available_features:
                st.error("No suitable features found for modeling")
                return pd.DataFrame()
            
            features = data[available_features].copy()
            
            # Handle missing values
            features = features.ffill().bfill()
            
            # Remove infinite values
            features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            return features
            
        except Exception as e:
            st.error(f"Error preparing features: {str(e)}")
            return pd.DataFrame()
    
    def create_features_and_targets(self, data, prediction_days=1):
        """
        Create feature matrix and target vector for prediction
        
        Args:
            data (pd.DataFrame): Feature data
            prediction_days (int): Number of days to predict ahead
            
        Returns:
            tuple: (X, y) feature matrix and target vector
        """
        try:
            if data.empty:
                return np.array([]), np.array([])
            
            # Use all available features
            X = data.values
            
            # Create target variable (future close prices)
            if 'Close' in data.columns:
                close_prices = data.index.get_indexer(data.index)
                # Use the close price from the original data
                close_col_idx = list(data.columns).index('Close') if 'Close' in data.columns else 0
                y = np.roll(X[:, close_col_idx], -prediction_days)
            else:
                # If Close is not in features, try to get it from session state
                if 'processed_data' in st.session_state:
                    close_prices = st.session_state.processed_data['Close'].values
                    y = np.roll(close_prices, -prediction_days)
                else:
                    st.error("Cannot create target variable: Close prices not found")
                    return np.array([]), np.array([])
            
            # Remove the last prediction_days rows as they don't have targets
            X = X[:-prediction_days]
            y = y[:-prediction_days]
            
            return X, y
            
        except Exception as e:
            st.error(f"Error creating features and targets: {str(e)}")
            return np.array([]), np.array([])
    
    def train_linear_regression(self, X_train, y_train):
        """
        Train Linear Regression model
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training targets
            
        Returns:
            LinearRegression: Trained model
        """
        try:
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            
            return model
            
        except Exception as e:
            st.error(f"Error training Linear Regression: {str(e)}")
            return None
    
    def train_random_forest(self, X_train, y_train):
        """
        Train Random Forest model
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training targets
            
        Returns:
            RandomForestRegressor: Trained model
        """
        try:
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            return model
            
        except Exception as e:
            st.error(f"Error training Random Forest: {str(e)}")
            return None
    
    def train_lstm(self, X_train, y_train, X_test, y_test, sequence_length=60):
        """
        Train LSTM model
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training targets
            X_test (np.array): Test features
            y_test (np.array): Test targets
            sequence_length (int): Length of input sequences
            
        Returns:
            keras.Model: Trained LSTM model
        """
        try:
            # Reshape data for LSTM (samples, timesteps, features)
            if len(X_train) < sequence_length:
                sequence_length = max(1, len(X_train) // 2)
            
            X_train_lstm = []
            y_train_lstm = []
            
            for i in range(sequence_length, len(X_train)):
                X_train_lstm.append(X_train[i-sequence_length:i])
                y_train_lstm.append(y_train[i])
            
            if not X_train_lstm:
                st.error("Not enough data for LSTM training")
                return None
            
            X_train_lstm = np.array(X_train_lstm)
            y_train_lstm = np.array(y_train_lstm)
            
            # Build LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(sequence_length, X_train.shape[1])),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            # Train model
            model.fit(
                X_train_lstm, y_train_lstm,
                batch_size=32,
                epochs=50,
                verbose=0,
                validation_split=0.1
            )
            
            return model
            
        except Exception as e:
            st.error(f"Error training LSTM: {str(e)}")
            return None
    
    def predict(self, model, X_test):
        """
        Make predictions using trained model
        
        Args:
            model: Trained model
            X_test (np.array): Test features
            
        Returns:
            np.array: Predictions
        """
        try:
            if model is None:
                return np.array([])
            
            if hasattr(model, 'predict'):
                if isinstance(model, LinearRegression):
                    X_test_scaled = self.scaler.transform(X_test)
                    predictions = model.predict(X_test_scaled)
                else:
                    predictions = model.predict(X_test)
                
                return predictions.flatten() if predictions.ndim > 1 else predictions
            else:
                return np.array([])
                
        except Exception as e:
            st.error(f"Error making predictions: {str(e)}")
            return np.array([])
    
    def predict_lstm(self, model, X_test, sequence_length=60):
        """
        Make predictions using LSTM model
        
        Args:
            model: Trained LSTM model
            X_test (np.array): Test features
            sequence_length (int): Length of input sequences
            
        Returns:
            np.array: Predictions
        """
        try:
            if model is None or len(X_test) < sequence_length:
                return np.array([])
            
            X_test_lstm = []
            for i in range(sequence_length, len(X_test)):
                X_test_lstm.append(X_test[i-sequence_length:i])
            
            if not X_test_lstm:
                return np.array([])
            
            X_test_lstm = np.array(X_test_lstm)
            predictions = model.predict(X_test_lstm, verbose=0)
            
            return predictions.flatten()
            
        except Exception as e:
            st.error(f"Error making LSTM predictions: {str(e)}")
            return np.array([])
    
    def random_walk_baseline(self, y_test):
        """
        Random walk baseline model
        
        Args:
            y_test (np.array): Test targets
            
        Returns:
            np.array: Baseline predictions
        """
        # Random walk: tomorrow's price = today's price
        if len(y_test) > 0:
            predictions = np.roll(y_test, 1)
            predictions[0] = y_test[0]  # First prediction is same as first actual
            return predictions
        return np.array([])
    
    def moving_average_baseline(self, prices, n_predictions, window=5):
        """
        Simple moving average baseline
        
        Args:
            prices (pd.Series): Price series
            n_predictions (int): Number of predictions to make
            window (int): Moving average window
            
        Returns:
            np.array: Baseline predictions
        """
        try:
            if len(prices) < window:
                window = max(1, len(prices))
            
            # Use last 'window' prices to predict
            last_prices = prices.tail(window + n_predictions)
            predictions = []
            
            for i in range(n_predictions):
                if i == 0:
                    ma = last_prices.iloc[-(window):].mean()
                else:
                    ma = last_prices.iloc[-(window-i):].mean()
                predictions.append(ma)
            
            return np.array(predictions)
            
        except Exception as e:
            st.error(f"Error in moving average baseline: {str(e)}")
            return np.array([])
    
    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate performance metrics
        
        Args:
            y_true (np.array): True values
            y_pred (np.array): Predicted values
            
        Returns:
            dict: Performance metrics
        """
        try:
            if len(y_true) == 0 or len(y_pred) == 0:
                return {
                    'RMSE': float('inf'),
                    'MAE': float('inf'),
                    'R2': -float('inf'),
                    'Directional_Accuracy': 0.0
                }
            
            # Ensure arrays have the same length
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
            
            # Basic metrics
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            # Directional accuracy
            if len(y_true) > 1:
                true_direction = np.diff(y_true) > 0
                pred_direction = np.diff(y_pred) > 0
                directional_accuracy = np.mean(true_direction == pred_direction) * 100
            else:
                directional_accuracy = 0.0
            
            return {
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'Directional_Accuracy': directional_accuracy
            }
            
        except Exception as e:
            st.error(f"Error calculating metrics: {str(e)}")
            return {
                'RMSE': float('inf'),
                'MAE': float('inf'),
                'R2': -float('inf'),
                'Directional_Accuracy': 0.0
            }
    
    def train_ensemble(self, X_train, y_train, X_test, y_test):
        """
        Train ensemble model combining multiple algorithms
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training targets
            X_test (np.array): Test features
            y_test (np.array): Test targets
            
        Returns:
            dict: Dictionary containing all trained models
        """
        try:
            models = {}
            
            # Train Linear Regression
            lr_model = self.train_linear_regression(X_train, y_train)
            if lr_model:
                models['LinearRegression'] = lr_model
            
            # Train Random Forest
            rf_model = self.train_random_forest(X_train, y_train)
            if rf_model:
                models['RandomForest'] = rf_model
            
            # Train LSTM (if enough data)
            if len(X_train) > 60:
                lstm_model = self.train_lstm(X_train, y_train, X_test, y_test)
                if lstm_model:
                    models['LSTM'] = lstm_model
            
            return models
            
        except Exception as e:
            st.error(f"Error training ensemble: {str(e)}")
            return {}
    
    def predict_ensemble(self, models, X_test, weights=None):
        """
        Make ensemble predictions using weighted averaging
        
        Args:
            models (dict): Dictionary of trained models
            X_test (np.array): Test features
            weights (dict): Optional weights for each model (must sum to 1)
            
        Returns:
            np.array: Ensemble predictions
        """
        try:
            if not models:
                return np.array([])
            
            predictions = []
            model_names = []
            
            # Get predictions from each model
            for name, model in models.items():
                if name == 'LSTM':
                    pred = self.predict_lstm(model, X_test)
                else:
                    pred = self.predict(model, X_test)
                
                if len(pred) > 0:
                    predictions.append(pred)
                    model_names.append(name)
            
            if not predictions:
                return np.array([])
            
            # Align all predictions to same length (use tail end to match LSTM alignment)
            min_len = min(len(p) for p in predictions)
            # Take the last min_len elements from each prediction to align timestamps
            predictions = [p[-min_len:] if len(p) > min_len else p for p in predictions]
            
            # Apply weights
            if weights is None:
                # Equal weights if not specified
                weights = {name: 1.0 / len(model_names) for name in model_names}
            
            # Ensure weights sum to 1
            total_weight = sum(weights.get(name, 0) for name in model_names)
            if total_weight > 0:
                weights = {name: weights.get(name, 0) / total_weight for name in model_names}
            else:
                weights = {name: 1.0 / len(model_names) for name in model_names}
            
            # Calculate weighted average
            ensemble_pred = np.zeros(min_len)
            for i, name in enumerate(model_names):
                ensemble_pred += predictions[i] * weights.get(name, 0)
            
            return ensemble_pred
            
        except Exception as e:
            st.error(f"Error making ensemble predictions: {str(e)}")
            return np.array([])
    
    def walk_forward_validate(self, X, y, n_splits=5, model_type='linear_regression'):
        """
        Walk-forward time-series cross-validation.

        Unlike k-fold, each fold trains only on past data and tests on the
        immediately following window — matching how a real model would be used.

        Args:
            X (np.array): Feature matrix
            y (np.array): Target vector
            n_splits (int): Number of folds
            model_type (str): 'linear_regression' or 'random_forest'

        Returns:
            pd.DataFrame: Per-fold metrics (RMSE, MAE, R2, Directional_Accuracy)
        """
        n = len(X)
        # Each fold uses at least 20 samples for training
        min_train = max(20, n // (n_splits + 1))
        fold_size = max(5, (n - min_train) // n_splits)

        records = []
        for fold in range(n_splits):
            train_end = min_train + fold * fold_size
            test_end = min(train_end + fold_size, n)

            if train_end >= n or test_end > n or train_end >= test_end:
                break

            X_tr, y_tr = X[:train_end], y[:train_end]
            X_te, y_te = X[train_end:test_end], y[train_end:test_end]

            if model_type == 'random_forest':
                model = self.train_random_forest(X_tr, y_tr)
                preds = self.predict(model, X_te)
            else:
                model = self.train_linear_regression(X_tr, y_tr)
                preds = self.predict(model, X_te)

            metrics = self.calculate_metrics(y_te, preds)
            metrics['Fold'] = fold + 1
            metrics['Train_Size'] = train_end
            metrics['Test_Size'] = len(y_te)
            records.append(metrics)

        return pd.DataFrame(records)[['Fold', 'Train_Size', 'Test_Size',
                                       'RMSE', 'MAE', 'R2', 'Directional_Accuracy']]

    def optimize_ensemble_weights(self, models, X_val, y_val):
        """
        Optimize ensemble weights using validation set
        
        Args:
            models (dict): Dictionary of trained models
            X_val (np.array): Validation features
            y_val (np.array): Validation targets
            
        Returns:
            dict: Optimized weights for each model
        """
        try:
            from scipy.optimize import minimize
            
            model_names = list(models.keys())
            n_models = len(model_names)
            
            if n_models == 0:
                return {}
            
            # Get predictions from each model
            predictions = []
            for name in model_names:
                if name == 'LSTM':
                    pred = self.predict_lstm(models[name], X_val)
                else:
                    pred = self.predict(models[name], X_val)
                
                if len(pred) > 0:
                    predictions.append(pred)
                else:
                    predictions.append(np.zeros(len(y_val)))
            
            # Align lengths (use tail end to match LSTM alignment)
            min_len = min(len(p) for p in predictions + [y_val])
            predictions = [p[-min_len:] if len(p) > min_len else p for p in predictions]
            y_val = y_val[-min_len:] if len(y_val) > min_len else y_val
            
            # Objective function: minimize RMSE
            def objective(weights):
                ensemble_pred = np.zeros(min_len)
                for i in range(n_models):
                    ensemble_pred += predictions[i] * weights[i]
                return np.sqrt(mean_squared_error(y_val, ensemble_pred))
            
            # Constraints: weights must sum to 1
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            
            # Bounds: each weight between 0 and 1
            bounds = [(0, 1) for _ in range(n_models)]
            
            # Initial guess: equal weights
            initial_weights = np.ones(n_models) / n_models
            
            # Optimize
            result = minimize(objective, initial_weights, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                optimized_weights = {name: float(weight) 
                                   for name, weight in zip(model_names, result.x)}
                return optimized_weights
            else:
                # Return equal weights if optimization fails
                return {name: 1.0 / n_models for name in model_names}
            
        except Exception as e:
            st.warning(f"Error optimizing ensemble weights: {str(e)}")
            # Return equal weights
            return {name: 1.0 / len(models) for name in models.keys()}
