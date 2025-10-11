"""
Chart Components for Trading Dashboard

This module provides various chart components for visualizing
trading data, performance metrics, and market analysis.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

from ..core.data_models import OHLCV, Trade, Position


class ChartComponents:
    """Chart components for the trading dashboard."""
    
    def __init__(self):
        self.color_scheme = {
            'bullish': '#00ff88',
            'bearish': '#ff4444',
            'neutral': '#888888',
            'background': '#1e1e1e',
            'grid': '#333333'
        }
    
    def render_candlestick_chart(self, 
                                data: List[OHLCV], 
                                symbol: str, 
                                timeframe: str,
                                show_volume: bool = True,
                                show_indicators: bool = True):
        """
        Render candlestick chart with optional volume and indicators.
        
        Args:
            data: List of OHLCV data points
            symbol: Trading symbol
            timeframe: Chart timeframe
            show_volume: Whether to show volume subplot
            show_indicators: Whether to show technical indicators
        """
        if not data:
            st.warning("No data available for chart")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'timestamp': candle.timestamp,
            'open': candle.open,
            'high': candle.high,
            'low': candle.low,
            'close': candle.close,
            'volume': candle.volume
        } for candle in data])
        
        # Create subplots
        if show_volume:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=(f'{symbol} - {timeframe}', 'Volume'),
                row_heights=[0.7, 0.3]
            )
        else:
            fig = make_subplots(
                rows=1, cols=1,
                subplot_titles=(f'{symbol} - {timeframe}',)
            )
        
        # Candlestick chart
        candlestick = go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=symbol,
            increasing_line_color=self.color_scheme['bullish'],
            decreasing_line_color=self.color_scheme['bearish']
        )
        
        fig.add_trace(candlestick, row=1, col=1)
        
        # Add technical indicators if requested
        if show_indicators:
            self._add_technical_indicators(fig, df, row=1, col=1)
        
        # Volume chart
        if show_volume:
            volume_colors = ['red' if close < open else 'green' 
                           for close, open in zip(df['close'], df['open'])]
            
            volume_bar = go.Bar(
                x=df['timestamp'],
                y=df['volume'],
                name='Volume',
                marker_color=volume_colors,
                opacity=0.7
            )
            
            fig.add_trace(volume_bar, row=2, col=1)
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} Price Chart - {timeframe}',
            xaxis_title='Time',
            yaxis_title='Price',
            template='plotly_dark',
            height=600,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        # Update volume subplot
        if show_volume:
            fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_equity_curve(self, 
                           equity_data: List[Tuple[datetime, float]], 
                           title: str = "Equity Curve"):
        """
        Render equity curve chart.
        
        Args:
            equity_data: List of (timestamp, equity) tuples
            title: Chart title
        """
        if not equity_data:
            st.warning("No equity data available")
            return
        
        df = pd.DataFrame(equity_data, columns=['timestamp', 'equity'])
        
        fig = go.Figure()
        
        # Equity curve
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['equity'],
            mode='lines',
            name='Equity',
            line=dict(color=self.color_scheme['bullish'], width=2),
            hovertemplate='<b>%{x}</b><br>Equity: $%{y:,.2f}<extra></extra>'
        ))
        
        # Add running maximum line
        df['running_max'] = df['equity'].expanding().max()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['running_max'],
            mode='lines',
            name='Running Max',
            line=dict(color=self.color_scheme['neutral'], width=1, dash='dash'),
            opacity=0.7
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Equity ($)',
            template='plotly_dark',
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_drawdown_chart(self, 
                             equity_data: List[Tuple[datetime, float]], 
                             title: str = "Drawdown"):
        """
        Render drawdown chart.
        
        Args:
            equity_data: List of (timestamp, equity) tuples
            title: Chart title
        """
        if not equity_data:
            st.warning("No equity data available")
            return
        
        df = pd.DataFrame(equity_data, columns=['timestamp', 'equity'])
        df['running_max'] = df['equity'].expanding().max()
        df['drawdown'] = (df['equity'] - df['running_max']) / df['running_max'] * 100
        
        fig = go.Figure()
        
        # Drawdown area
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['drawdown'],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(255, 68, 68, 0.3)',
            line=dict(color=self.color_scheme['bearish'], width=1),
            name='Drawdown',
            hovertemplate='<b>%{x}</b><br>Drawdown: %{y:.2f}%<extra></extra>'
        ))
        
        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
        
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Drawdown (%)',
            template='plotly_dark',
            height=300,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_returns_distribution(self, 
                                  trades: List[Trade], 
                                  title: str = "Returns Distribution"):
        """
        Render returns distribution histogram.
        
        Args:
            trades: List of Trade objects
            title: Chart title
        """
        if not trades:
            st.warning("No trades available")
            return
        
        pnls = [trade.pnl for trade in trades if trade.pnl is not None]
        
        if not pnls:
            st.warning("No PnL data available")
            return
        
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=pnls,
            nbinsx=30,
            name='Returns',
            marker_color=self.color_scheme['neutral'],
            opacity=0.7
        ))
        
        # Mean line
        mean_pnl = np.mean(pnls)
        fig.add_vline(
            x=mean_pnl,
            line_dash="dash",
            line_color=self.color_scheme['bullish'],
            annotation_text=f"Mean: ${mean_pnl:.2f}"
        )
        
        fig.update_layout(
            title=title,
            xaxis_title='PnL ($)',
            yaxis_title='Frequency',
            template='plotly_dark',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_monthly_returns_heatmap(self, 
                                      equity_data: List[Tuple[datetime, float]], 
                                      title: str = "Monthly Returns Heatmap"):
        """
        Render monthly returns heatmap.
        
        Args:
            equity_data: List of (timestamp, equity) tuples
            title: Chart title
        """
        if not equity_data:
            st.warning("No equity data available")
            return
        
        df = pd.DataFrame(equity_data, columns=['timestamp', 'equity'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Calculate monthly returns
        monthly_equity = df.resample('M').last()
        monthly_returns = monthly_equity['equity'].pct_change().dropna() * 100
        
        # Create year-month matrix
        returns_matrix = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).first()
        returns_matrix = returns_matrix.unstack(level=1)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=returns_matrix.values,
            x=returns_matrix.columns,
            y=returns_matrix.index,
            colorscale='RdYlGn',
            zmid=0,
            text=np.round(returns_matrix.values, 2),
            texttemplate="%{text}%",
            textfont={"size": 10},
            hovertemplate='<b>%{y}-%{x:02d}</b><br>Return: %{z:.2f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Month',
            yaxis_title='Year',
            template='plotly_dark',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_correlation_matrix(self, 
                                strategies_data: Dict[str, List[Tuple[datetime, float]]], 
                                title: str = "Strategy Correlation Matrix"):
        """
        Render correlation matrix for multiple strategies.
        
        Args:
            strategies_data: Dictionary mapping strategy names to equity data
            title: Chart title
        """
        if not strategies_data:
            st.warning("No strategy data available")
            return
        
        # Calculate returns for each strategy
        returns_data = {}
        for strategy, equity_data in strategies_data.items():
            df = pd.DataFrame(equity_data, columns=['timestamp', 'equity'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Calculate daily returns
            daily_returns = df['equity'].pct_change().dropna()
            returns_data[strategy] = daily_returns
        
        # Create correlation matrix
        returns_df = pd.DataFrame(returns_data)
        correlation_matrix = returns_df.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=np.round(correlation_matrix.values, 3),
            texttemplate="%{text}",
            textfont={"size": 12},
            hovertemplate='<b>%{y} vs %{x}</b><br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            template='plotly_dark',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_risk_return_scatter(self, 
                                 strategies_metrics: Dict[str, Dict[str, float]], 
                                 title: str = "Risk-Return Analysis"):
        """
        Render risk-return scatter plot.
        
        Args:
            strategies_metrics: Dictionary mapping strategy names to metrics
            title: Chart title
        """
        if not strategies_metrics:
            st.warning("No strategy metrics available")
            return
        
        # Extract data
        strategies = list(strategies_metrics.keys())
        returns = [strategies_metrics[s].get('total_return', 0) * 100 for s in strategies]
        risks = [strategies_metrics[s].get('volatility', 0) * 100 for s in strategies]
        sharpe_ratios = [strategies_metrics[s].get('sharpe_ratio', 0) for s in strategies]
        
        fig = go.Figure()
        
        # Scatter plot
        fig.add_trace(go.Scatter(
            x=risks,
            y=returns,
            mode='markers+text',
            text=strategies,
            textposition='top center',
            marker=dict(
                size=[abs(sr) * 20 for sr in sharpe_ratios],  # Size by Sharpe ratio
                color=sharpe_ratios,
                colorscale='RdYlGn',
                colorbar=dict(title="Sharpe Ratio"),
                line=dict(width=2, color='white')
            ),
            hovertemplate='<b>%{text}</b><br>Return: %{y:.2f}%<br>Risk: %{x:.2f}%<br>Sharpe: %{marker.color:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Risk (Volatility %)',
            yaxis_title='Return (%)',
            template='plotly_dark',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_performance_comparison(self, 
                                    strategies_metrics: Dict[str, Dict[str, float]], 
                                    title: str = "Performance Comparison"):
        """
        Render performance comparison bar chart.
        
        Args:
            strategies_metrics: Dictionary mapping strategy names to metrics
            title: Chart title
        """
        if not strategies_metrics:
            st.warning("No strategy metrics available")
            return
        
        # Create subplots for different metrics
        metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'profit_factor']
        metric_titles = ['Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Profit Factor']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=metric_titles,
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        strategies = list(strategies_metrics.keys())
        colors = px.colors.qualitative.Set3[:len(strategies)]
        
        for i, (metric, metric_title) in enumerate(zip(metrics, metric_titles)):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            values = []
            for strategy in strategies:
                value = strategies_metrics[strategy].get(metric, 0)
                if metric in ['total_return', 'max_drawdown']:
                    value *= 100  # Convert to percentage
                values.append(value)
            
            fig.add_trace(go.Bar(
                x=strategies,
                y=values,
                name=metric_title,
                marker_color=colors,
                showlegend=False
            ), row=row, col=col)
        
        fig.update_layout(
            title=title,
            template='plotly_dark',
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _add_technical_indicators(self, fig, df: pd.DataFrame, row: int = 1, col: int = 1):
        """
        Add technical indicators to the chart.
        
        Args:
            fig: Plotly figure object
            df: DataFrame with OHLCV data
            row: Subplot row
            col: Subplot column
        """
        # Simple Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Add SMA lines
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['sma_20'],
            mode='lines',
            name='SMA 20',
            line=dict(color='orange', width=1),
            opacity=0.8
        ), row=row, col=col)
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['sma_50'],
            mode='lines',
            name='SMA 50',
            line=dict(color='blue', width=1),
            opacity=0.8
        ), row=row, col=col)
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Add Bollinger Bands
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['bb_upper'],
            mode='lines',
            name='BB Upper',
            line=dict(color='purple', width=1),
            opacity=0.6,
            showlegend=False
        ), row=row, col=col)
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['bb_lower'],
            mode='lines',
            name='BB Lower',
            line=dict(color='purple', width=1),
            opacity=0.6,
            fill='tonexty',
            fillcolor='rgba(128, 0, 128, 0.1)',
            showlegend=False
        ), row=row, col=col)
    
    def render_optimization_progress(self, 
                                   optimization_history: List[Dict[str, Any]], 
                                   title: str = "Optimization Progress"):
        """
        Render optimization progress chart.
        
        Args:
            optimization_history: List of optimization history records
            title: Chart title
        """
        if not optimization_history:
            st.warning("No optimization history available")
            return
        
        df = pd.DataFrame(optimization_history)
        
        fig = go.Figure()
        
        # Current scores
        fig.add_trace(go.Scatter(
            x=df['evaluation'],
            y=df['score'],
            mode='markers',
            name='Current Score',
            marker=dict(color='blue', size=4),
            opacity=0.6
        ))
        
        # Best scores
        fig.add_trace(go.Scatter(
            x=df['evaluation'],
            y=df['best_score'],
            mode='lines',
            name='Best Score',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Evaluation',
            yaxis_title='Score',
            template='plotly_dark',
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
