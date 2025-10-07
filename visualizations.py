import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st

class StockVisualizer:
    """Class to handle all visualization tasks"""
    
    def __init__(self):
        # Color palette
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
    
    def plot_price_with_indicators(self, data, ticker):
        """
        Create comprehensive price chart with technical indicators
        
        Args:
            data (pd.DataFrame): Stock data with indicators
            ticker (str): Stock ticker symbol
            
        Returns:
            plotly.graph_objects.Figure: Price chart with indicators
        """
        try:
            # Create subplots
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.5, 0.15, 0.2, 0.15],
                subplot_titles=['Price & Moving Averages', 'Volume', 'RSI', 'MACD']
            )
            
            # Price and moving averages
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Price'
                ),
                row=1, col=1
            )
            
            # Add moving averages if available
            ma_columns = [(col, col.replace('_', ' ')) for col in data.columns if 'SMA_' in col or 'EMA_' in col]
            colors_ma = ['orange', 'red', 'green', 'purple', 'brown']
            
            for i, (col, label) in enumerate(ma_columns[:5]):
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data[col],
                        mode='lines',
                        name=label,
                        line=dict(color=colors_ma[i % len(colors_ma)], width=1)
                    ),
                    row=1, col=1
                )
            
            # Bollinger Bands if available
            if all(col in data.columns for col in ['BB_Upper', 'BB_Lower']):
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['BB_Upper'],
                        mode='lines',
                        name='BB Upper',
                        line=dict(color='gray', width=1, dash='dash'),
                        showlegend=False
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['BB_Lower'],
                        mode='lines',
                        name='BB Lower',
                        line=dict(color='gray', width=1, dash='dash'),
                        fill='tonexty',
                        fillcolor='rgba(128,128,128,0.1)'
                    ),
                    row=1, col=1
                )
            
            # Volume
            colors_volume = ['green' if row['Close'] >= row['Open'] else 'red' 
                           for _, row in data.iterrows()]
            
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name='Volume',
                    marker_color=colors_volume
                ),
                row=2, col=1
            )
            
            # RSI
            if 'RSI' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['RSI'],
                        mode='lines',
                        name='RSI',
                        line=dict(color='purple')
                    ),
                    row=3, col=1
                )
                
                # RSI levels
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
                fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)
            
            # MACD
            if all(col in data.columns for col in ['MACD', 'MACD_Signal']):
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['MACD'],
                        mode='lines',
                        name='MACD',
                        line=dict(color='blue')
                    ),
                    row=4, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['MACD_Signal'],
                        mode='lines',
                        name='Signal',
                        line=dict(color='red')
                    ),
                    row=4, col=1
                )
                
                if 'MACD_Histogram' in data.columns:
                    colors_hist = ['green' if x >= 0 else 'red' for x in data['MACD_Histogram']]
                    fig.add_trace(
                        go.Bar(
                            x=data.index,
                            y=data['MACD_Histogram'],
                            name='Histogram',
                            marker_color=colors_hist
                        ),
                        row=4, col=1
                    )
            
            # Update layout
            fig.update_layout(
                title=f'{ticker} - Comprehensive Technical Analysis',
                xaxis_rangeslider_visible=False,
                height=800,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating price chart: {str(e)}")
            return go.Figure()
    
    def plot_volume_analysis(self, data):
        """
        Create volume analysis chart
        
        Args:
            data (pd.DataFrame): Stock data
            
        Returns:
            plotly.graph_objects.Figure: Volume analysis chart
        """
        try:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=['Volume vs Price', 'Volume Moving Averages']
            )
            
            # Volume vs Price
            colors_volume = ['green' if row['Close'] >= row['Open'] else 'red' 
                           for _, row in data.iterrows()]
            
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name='Volume',
                    marker_color=colors_volume
                ),
                row=1, col=1
            )
            
            # Volume moving averages
            if 'Volume_SMA_10' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['Volume_SMA_10'],
                        mode='lines',
                        name='Volume SMA 10',
                        line=dict(color='orange')
                    ),
                    row=2, col=1
                )
            
            if 'Volume_SMA_20' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['Volume_SMA_20'],
                        mode='lines',
                        name='Volume SMA 20',
                        line=dict(color='red')
                    ),
                    row=2, col=1
                )
            
            fig.update_layout(
                title='Volume Analysis',
                height=400,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating volume chart: {str(e)}")
            return go.Figure()
    
    def plot_volatility_analysis(self, data):
        """
        Create volatility analysis chart
        
        Args:
            data (pd.DataFrame): Stock data
            
        Returns:
            plotly.graph_objects.Figure: Volatility chart
        """
        try:
            # Calculate rolling volatility
            data['Volatility_10'] = data['Close'].pct_change().rolling(10).std() * np.sqrt(252) * 100
            data['Volatility_30'] = data['Close'].pct_change().rolling(30).std() * np.sqrt(252) * 100
            
            fig = go.Figure()
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['Volatility_10'],
                    mode='lines',
                    name='10-day Volatility',
                    line=dict(color='red')
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['Volatility_30'],
                    mode='lines',
                    name='30-day Volatility',
                    line=dict(color='blue')
                )
            )
            
            if 'ATR' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['ATR'] / data['Close'] * 100,
                        mode='lines',
                        name='ATR (%)',
                        line=dict(color='green')
                    )
                )
            
            fig.update_layout(
                title='Volatility Analysis',
                yaxis_title='Volatility (%)',
                height=400
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating volatility chart: {str(e)}")
            return go.Figure()
    
    def plot_correlation_heatmap(self, data):
        """
        Create correlation heatmap
        
        Args:
            data (pd.DataFrame): Stock data
            
        Returns:
            plotly.graph_objects.Figure: Correlation heatmap
        """
        try:
            # Select numeric columns for correlation
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Limit to most relevant columns
            relevant_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Add technical indicators
            tech_cols = [col for col in numeric_cols if any(indicator in col for indicator in 
                        ['SMA', 'EMA', 'RSI', 'MACD', 'BB', 'Stoch', 'ATR', 'Williams', 'CCI'])]
            
            relevant_cols.extend([col for col in tech_cols if col in data.columns][:10])
            available_cols = [col for col in relevant_cols if col in data.columns]
            
            if len(available_cols) < 2:
                st.warning("Not enough numeric columns for correlation analysis")
                return go.Figure()
            
            corr_matrix = data[available_cols].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 8}
            ))
            
            fig.update_layout(
                title='Feature Correlation Matrix',
                height=500
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating correlation heatmap: {str(e)}")
            return go.Figure()
    
    def plot_predictions_comparison(self, results, test_dates):
        """
        Plot actual vs predicted values for all models
        
        Args:
            results (dict): Prediction results for all models
            test_dates (pd.Index): Test period dates
            
        Returns:
            plotly.graph_objects.Figure: Predictions comparison chart
        """
        try:
            fig = go.Figure()
            
            # Get actual values (assume they're the same for all models)
            actual_values = None
            for result in results.values():
                if len(result['actual']) > 0:
                    actual_values = result['actual']
                    break
            
            if actual_values is None:
                st.error("No actual values found for plotting")
                return go.Figure()
            
            # Ensure test_dates matches the length of actual values
            min_len = min(len(actual_values), len(test_dates))
            actual_values = actual_values[:min_len]
            plot_dates = test_dates[:min_len]
            
            # Plot actual values
            fig.add_trace(
                go.Scatter(
                    x=plot_dates,
                    y=actual_values,
                    mode='lines',
                    name='Actual',
                    line=dict(color='black', width=2)
                )
            )
            
            # Plot predictions for each model
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
            
            for i, (model_name, result) in enumerate(results.items()):
                if len(result['predictions']) > 0:
                    predictions = result['predictions'][:min_len]
                    fig.add_trace(
                        go.Scatter(
                            x=plot_dates,
                            y=predictions,
                            mode='lines',
                            name=f'{model_name} Prediction',
                            line=dict(color=colors[i % len(colors)], width=1, dash='dash')
                        )
                    )
            
            fig.update_layout(
                title='Actual vs Predicted Stock Prices',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                height=500,
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating predictions comparison: {str(e)}")
            return go.Figure()
    
    def plot_metrics_comparison(self, metrics, metric_name):
        """
        Plot metrics comparison across models
        
        Args:
            metrics (dict): Metrics for all models
            metric_name (str): Name of metric to plot
            
        Returns:
            plotly.graph_objects.Figure: Metrics comparison chart
        """
        try:
            models = list(metrics.keys())
            values = [metrics[model].get(metric_name, 0) for model in models]
            
            # Color code: lower is better for RMSE/MAE, higher is better for accuracy/R2
            if metric_name in ['RMSE', 'MAE']:
                colors = ['green' if v == min(values) else 'red' for v in values]
            else:
                colors = ['green' if v == max(values) else 'red' for v in values]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=models,
                    y=values,
                    marker_color=colors,
                    text=[f'{v:.4f}' for v in values],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title=f'{metric_name} Comparison Across Models',
                yaxis_title=metric_name,
                height=400
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating metrics comparison: {str(e)}")
            return go.Figure()
    
    def plot_feature_importance(self, model):
        """
        Plot feature importance for tree-based models
        
        Args:
            model: Trained model with feature_importances_ attribute
            
        Returns:
            plotly.graph_objects.Figure: Feature importance chart
        """
        try:
            if not hasattr(model, 'feature_importances_'):
                st.warning("Model doesn't have feature importance information")
                return go.Figure()
            
            # Get feature names from session state
            if 'processed_data' in st.session_state:
                data = st.session_state.processed_data
                feature_columns = [
                    'Open', 'High', 'Low', 'Volume',
                    'Price_Change_Pct', 'HL_Spread_Pct', 'OC_Spread_Pct',
                    'Range_Pct', 'Intraday_Return'
                ]
                
                # Add available MA and technical indicators
                ma_columns = [col for col in data.columns if 'SMA_' in col or 'EMA_' in col]
                feature_columns.extend(ma_columns[:10])
                
                tech_columns = ['RSI', 'MACD', 'MACD_Signal', 'BB_Position', 'Stoch_K', 'ATR', 'Williams_R', 'CCI']
                for col in tech_columns:
                    if col in data.columns:
                        feature_columns.append(col)
                
                available_features = [col for col in feature_columns if col in data.columns]
                feature_names = available_features[:len(model.feature_importances_)]
            else:
                feature_names = [f'Feature_{i}' for i in range(len(model.feature_importances_))]
            
            importances = model.feature_importances_
            
            # Sort by importance
            indices = np.argsort(importances)[::-1]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=[feature_names[i] for i in indices[:20]],  # Top 20 features
                    y=[importances[i] for i in indices[:20]],
                    marker_color='steelblue'
                )
            ])
            
            fig.update_layout(
                title='Feature Importance (Random Forest)',
                xaxis_title='Features',
                yaxis_title='Importance',
                height=400,
                xaxis_tickangle=-45
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating feature importance plot: {str(e)}")
            return go.Figure()
