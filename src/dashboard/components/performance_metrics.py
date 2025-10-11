"""
Performance Metrics Component for Trading Dashboard

This module provides components for displaying trading performance metrics,
risk analysis, and performance comparisons.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import math

from ...core.data_models import Trade, Position, OHLCV
from ...backtesting.performance_metrics import PerformanceMetrics


class PerformanceMetricsComponent:
    """Component for displaying performance metrics and analysis."""
    
    def __init__(self):
        self.color_scheme = {
            'profit': '#00ff88',
            'loss': '#ff4444',
            'neutral': '#888888',
            'warning': '#ffaa00',
            'success': '#00ff88'
        }
    
    def render_performance_metrics(self, 
                                 trades: List[Trade] = None,
                                 equity_curve: List[Tuple[datetime, float]] = None,
                                 positions: List[Position] = None):
        """
        Render comprehensive performance metrics.
        
        Args:
            trades: List of Trade objects
            equity_curve: List of (timestamp, equity) tuples
            positions: List of Position objects
        """
        st.header("📊 Performance Metrics")
        
        # Calculate metrics
        metrics = self._calculate_performance_metrics(trades, equity_curve, positions)
        
        # Display key metrics
        self._render_key_metrics(metrics)
        
        # Display detailed metrics
        self._render_detailed_metrics(metrics)
        
        # Display risk metrics
        self._render_risk_metrics(metrics)
        
        # Display performance charts
        self._render_performance_charts(metrics)
    
    def render_strategy_comparison(self, 
                                strategies_metrics: Dict[str, Dict[str, float]]):
        """
        Render strategy comparison metrics.
        
        Args:
            strategies_metrics: Dictionary mapping strategy names to metrics
        """
        st.header("📈 Strategy Comparison")
        
        if not strategies_metrics:
            st.warning("No strategy metrics available")
            return
        
        # Create comparison table
        self._render_comparison_table(strategies_metrics)
        
        # Create comparison charts
        self._render_comparison_charts(strategies_metrics)
        
        # Create ranking
        self._render_strategy_ranking(strategies_metrics)
    
    def render_risk_analysis(self, 
                           trades: List[Trade] = None,
                           equity_curve: List[Tuple[datetime, float]] = None):
        """
        Render risk analysis metrics.
        
        Args:
            trades: List of Trade objects
            equity_curve: List of (timestamp, equity) tuples
        """
        st.header("⚠️ Risk Analysis")
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(trades, equity_curve)
        
        # Display risk metrics
        self._render_risk_metrics_detailed(risk_metrics)
        
        # Display risk charts
        self._render_risk_charts(risk_metrics)
        
        # Display risk warnings
        self._render_risk_warnings(risk_metrics)
    
    def render_monthly_performance(self, 
                                equity_curve: List[Tuple[datetime, float]]):
        """
        Render monthly performance analysis.
        
        Args:
            equity_curve: List of (timestamp, equity) tuples
        """
        st.header("📅 Monthly Performance")
        
        if not equity_curve:
            st.warning("No equity data available")
            return
        
        # Calculate monthly metrics
        monthly_metrics = self._calculate_monthly_metrics(equity_curve)
        
        # Display monthly table
        self._render_monthly_table(monthly_metrics)
        
        # Display monthly charts
        self._render_monthly_charts(monthly_metrics)
    
    def _calculate_performance_metrics(self, 
                                    trades: List[Trade] = None,
                                    equity_curve: List[Tuple[datetime, float]] = None,
                                    positions: List[Position] = None) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        metrics = {}
        
        # Basic metrics
        if trades:
            metrics.update(self._calculate_trade_metrics(trades))
        
        if equity_curve:
            metrics.update(self._calculate_equity_metrics(equity_curve))
        
        if positions:
            metrics.update(self._calculate_position_metrics(positions))
        
        return metrics
    
    def _calculate_trade_metrics(self, trades: List[Trade]) -> Dict[str, Any]:
        """Calculate metrics from trades."""
        if not trades:
            return {}
        
        # Filter trades with PnL
        trades_with_pnl = [t for t in trades if t.pnl is not None]
        
        if not trades_with_pnl:
            return {}
        
        pnls = [t.pnl for t in trades_with_pnl]
        
        # Basic metrics
        total_trades = len(trades_with_pnl)
        winning_trades = len([p for p in pnls if p > 0])
        losing_trades = len([p for p in pnls if p < 0])
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # PnL metrics
        total_pnl = sum(pnls)
        avg_pnl = np.mean(pnls)
        median_pnl = np.median(pnls)
        
        # Win/Loss metrics
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        largest_win = max(wins) if wins else 0
        largest_loss = min(losses) if losses else 0
        
        # Profit factor
        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'median_pnl': median_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'profit_factor': profit_factor
        }
    
    def _calculate_equity_metrics(self, equity_curve: List[Tuple[datetime, float]]) -> Dict[str, Any]:
        """Calculate metrics from equity curve."""
        if not equity_curve:
            return {}
        
        df = pd.DataFrame(equity_curve, columns=['timestamp', 'equity'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Basic equity metrics
        initial_equity = df['equity'].iloc[0]
        final_equity = df['equity'].iloc[-1]
        total_return = (final_equity - initial_equity) / initial_equity * 100
        
        # Drawdown metrics
        df['running_max'] = df['equity'].expanding().max()
        df['drawdown'] = (df['equity'] - df['running_max']) / df['running_max'] * 100
        max_drawdown = df['drawdown'].min()
        
        # Volatility metrics
        df['returns'] = df['equity'].pct_change().dropna()
        volatility = df['returns'].std() * np.sqrt(252) * 100  # Annualized
        
        # Sharpe ratio (assuming risk-free rate of 2%)
        risk_free_rate = 0.02
        excess_returns = df['returns'].mean() * 252 - risk_free_rate
        sharpe_ratio = excess_returns / (df['returns'].std() * np.sqrt(252)) if df['returns'].std() > 0 else 0
        
        # Sortino ratio
        downside_returns = df[df['returns'] < 0]['returns']
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = excess_returns / downside_volatility if downside_volatility > 0 else 0
        
        return {
            'initial_equity': initial_equity,
            'final_equity': final_equity,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio
        }
    
    def _calculate_position_metrics(self, positions: List[Position]) -> Dict[str, Any]:
        """Calculate metrics from positions."""
        if not positions:
            return {}
        
        # Position size metrics
        position_sizes = [p.quantity * p.entry_price for p in positions]
        total_exposure = sum(position_sizes)
        avg_position_size = np.mean(position_sizes)
        max_position_size = max(position_sizes)
        
        # Position concentration
        position_concentration = (max_position_size / total_exposure * 100) if total_exposure > 0 else 0
        
        # Time in position metrics
        time_in_positions = [(datetime.now() - p.timestamp).total_seconds() / 3600 for p in positions]
        avg_time_in_position = np.mean(time_in_positions)
        
        return {
            'total_exposure': total_exposure,
            'avg_position_size': avg_position_size,
            'max_position_size': max_position_size,
            'position_concentration': position_concentration,
            'avg_time_in_position': avg_time_in_position
        }
    
    def _calculate_risk_metrics(self, 
                             trades: List[Trade] = None,
                             equity_curve: List[Tuple[datetime, float]] = None) -> Dict[str, Any]:
        """Calculate risk metrics."""
        risk_metrics = {}
        
        if trades:
            trades_with_pnl = [t for t in trades if t.pnl is not None]
            if trades_with_pnl:
                pnls = [t.pnl for t in trades_with_pnl]
                
                # VaR and CVaR
                var_95 = np.percentile(pnls, 5)
                var_99 = np.percentile(pnls, 1)
                
                cvar_95 = np.mean([p for p in pnls if p <= var_95])
                cvar_99 = np.mean([p for p in pnls if p <= var_99])
                
                risk_metrics.update({
                    'var_95': var_95,
                    'var_99': var_99,
                    'cvar_95': cvar_95,
                    'cvar_99': cvar_99
                })
        
        if equity_curve:
            df = pd.DataFrame(equity_curve, columns=['timestamp', 'equity'])
            df['returns'] = df['equity'].pct_change().dropna()
            
            if len(df['returns']) > 0:
                # Skewness and Kurtosis
                skewness = df['returns'].skew()
                kurtosis = df['returns'].kurtosis()
                
                # Maximum consecutive losses
                returns = df['returns'].values
                max_consecutive_losses = self._calculate_max_consecutive_losses(returns)
                
                risk_metrics.update({
                    'skewness': skewness,
                    'kurtosis': kurtosis,
                    'max_consecutive_losses': max_consecutive_losses
                })
        
        return risk_metrics
    
    def _calculate_max_consecutive_losses(self, returns: np.ndarray) -> int:
        """Calculate maximum consecutive losses."""
        max_consecutive = 0
        current_consecutive = 0
        
        for ret in returns:
            if ret < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_monthly_metrics(self, equity_curve: List[Tuple[datetime, float]]) -> Dict[str, Any]:
        """Calculate monthly performance metrics."""
        df = pd.DataFrame(equity_curve, columns=['timestamp', 'equity'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Calculate monthly returns
        monthly_equity = df.resample('M').last()
        monthly_returns = monthly_equity['equity'].pct_change().dropna() * 100
        
        # Create monthly data
        monthly_data = []
        for date, return_pct in monthly_returns.items():
            monthly_data.append({
                'Month': date.strftime('%Y-%m'),
                'Return (%)': round(return_pct, 2),
                'Year': date.year,
                'Month_Name': date.strftime('%B')
            })
        
        return {
            'monthly_data': monthly_data,
            'avg_monthly_return': monthly_returns.mean(),
            'monthly_volatility': monthly_returns.std(),
            'best_month': monthly_returns.max(),
            'worst_month': monthly_returns.min(),
            'positive_months': len(monthly_returns[monthly_returns > 0]),
            'negative_months': len(monthly_returns[monthly_returns < 0])
        }
    
    def _render_key_metrics(self, metrics: Dict[str, Any]):
        """Render key performance metrics."""
        st.subheader("Key Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_return = metrics.get('total_return', 0)
            st.metric(
                "Total Return", 
                f"{total_return:.2f}%",
                delta=f"{total_return:.2f}%" if total_return != 0 else None
            )
        
        with col2:
            sharpe_ratio = metrics.get('sharpe_ratio', 0)
            st.metric(
                "Sharpe Ratio", 
                f"{sharpe_ratio:.2f}",
                delta=f"{sharpe_ratio:.2f}" if sharpe_ratio != 0 else None
            )
        
        with col3:
            max_drawdown = metrics.get('max_drawdown', 0)
            st.metric(
                "Max Drawdown", 
                f"{max_drawdown:.2f}%",
                delta=f"{max_drawdown:.2f}%" if max_drawdown != 0 else None
            )
        
        with col4:
            win_rate = metrics.get('win_rate', 0)
            st.metric(
                "Win Rate", 
                f"{win_rate:.1f}%",
                delta=f"{win_rate:.1f}%" if win_rate != 0 else None
            )
    
    def _render_detailed_metrics(self, metrics: Dict[str, Any]):
        """Render detailed performance metrics."""
        st.subheader("Detailed Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Trading Metrics**")
            st.write(f"Total Trades: {metrics.get('total_trades', 0)}")
            st.write(f"Winning Trades: {metrics.get('winning_trades', 0)}")
            st.write(f"Losing Trades: {metrics.get('losing_trades', 0)}")
            st.write(f"Average PnL: ${metrics.get('avg_pnl', 0):.2f}")
            st.write(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        
        with col2:
            st.write("**Risk Metrics**")
            st.write(f"Volatility: {metrics.get('volatility', 0):.2f}%")
            st.write(f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}")
            st.write(f"VaR (95%): ${metrics.get('var_95', 0):.2f}")
            st.write(f"CVaR (95%): ${metrics.get('cvar_95', 0):.2f}")
            st.write(f"Skewness: {metrics.get('skewness', 0):.2f}")
    
    def _render_risk_metrics(self, metrics: Dict[str, Any]):
        """Render risk metrics."""
        st.subheader("Risk Analysis")
        
        # Risk level assessment
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        max_drawdown = abs(metrics.get('max_drawdown', 0))
        volatility = metrics.get('volatility', 0)
        
        risk_level = "Low"
        if sharpe_ratio < 1 or max_drawdown > 20 or volatility > 30:
            risk_level = "High"
        elif sharpe_ratio < 1.5 or max_drawdown > 15 or volatility > 20:
            risk_level = "Medium"
        
        st.info(f"**Risk Level:** {risk_level}")
        
        # Risk warnings
        if max_drawdown > 20:
            st.warning("⚠️ High maximum drawdown detected!")
        
        if volatility > 30:
            st.warning("⚠️ High volatility detected!")
        
        if sharpe_ratio < 1:
            st.warning("⚠️ Low Sharpe ratio - consider improving risk-adjusted returns!")
    
    def _render_performance_charts(self, metrics: Dict[str, Any]):
        """Render performance charts."""
        st.subheader("Performance Charts")
        
        # This would normally render charts based on the metrics
        # For now, show placeholder
        st.info("Performance charts will be rendered here")
    
    def _render_comparison_table(self, strategies_metrics: Dict[str, Dict[str, float]]):
        """Render strategy comparison table."""
        st.subheader("Strategy Comparison Table")
        
        # Create comparison DataFrame
        comparison_data = []
        for strategy, metrics in strategies_metrics.items():
            comparison_data.append({
                'Strategy': strategy,
                'Total Return (%)': f"{metrics.get('total_return', 0) * 100:.2f}",
                'Sharpe Ratio': f"{metrics.get('sharpe_ratio', 0):.2f}",
                'Max Drawdown (%)': f"{metrics.get('max_drawdown', 0) * 100:.2f}",
                'Volatility (%)': f"{metrics.get('volatility', 0) * 100:.2f}",
                'Win Rate (%)': f"{metrics.get('win_rate', 0):.1f}",
                'Profit Factor': f"{metrics.get('profit_factor', 0):.2f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
    
    def _render_comparison_charts(self, strategies_metrics: Dict[str, Dict[str, float]]):
        """Render strategy comparison charts."""
        st.subheader("Strategy Comparison Charts")
        
        # Risk-Return scatter plot
        strategies = list(strategies_metrics.keys())
        returns = [strategies_metrics[s].get('total_return', 0) * 100 for s in strategies]
        risks = [strategies_metrics[s].get('volatility', 0) * 100 for s in strategies]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=risks,
            y=returns,
            mode='markers+text',
            text=strategies,
            textposition='top center',
            marker=dict(size=20, color=returns, colorscale='RdYlGn'),
            hovertemplate='<b>%{text}</b><br>Return: %{y:.2f}%<br>Risk: %{x:.2f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title="Risk-Return Analysis",
            xaxis_title="Risk (Volatility %)",
            yaxis_title="Return (%)",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_strategy_ranking(self, strategies_metrics: Dict[str, Dict[str, float]]):
        """Render strategy ranking."""
        st.subheader("Strategy Ranking")
        
        # Calculate composite score (weighted average of key metrics)
        rankings = []
        for strategy, metrics in strategies_metrics.items():
            # Weighted score: 40% Sharpe, 30% Return, 20% Drawdown, 10% Win Rate
            sharpe = metrics.get('sharpe_ratio', 0)
            total_return = metrics.get('total_return', 0) * 100
            max_drawdown = abs(metrics.get('max_drawdown', 0)) * 100
            win_rate = metrics.get('win_rate', 0)
            
            # Normalize and weight
            composite_score = (
                sharpe * 0.4 +
                total_return * 0.3 +
                (100 - max_drawdown) * 0.2 +  # Lower drawdown is better
                win_rate * 0.1
            )
            
            rankings.append({
                'Strategy': strategy,
                'Composite Score': f"{composite_score:.2f}",
                'Sharpe Ratio': f"{sharpe:.2f}",
                'Total Return (%)': f"{total_return:.2f}",
                'Max Drawdown (%)': f"{max_drawdown:.2f}",
                'Win Rate (%)': f"{win_rate:.1f}"
            })
        
        # Sort by composite score
        rankings.sort(key=lambda x: float(x['Composite Score']), reverse=True)
        
        ranking_df = pd.DataFrame(rankings)
        ranking_df.index = range(1, len(ranking_df) + 1)  # Start ranking from 1
        
        st.dataframe(ranking_df, use_container_width=True)
    
    def _render_risk_metrics_detailed(self, risk_metrics: Dict[str, Any]):
        """Render detailed risk metrics."""
        st.subheader("Risk Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("VaR (95%)", f"${risk_metrics.get('var_95', 0):.2f}")
        with col2:
            st.metric("CVaR (95%)", f"${risk_metrics.get('cvar_95', 0):.2f}")
        with col3:
            st.metric("Skewness", f"{risk_metrics.get('skewness', 0):.2f}")
        with col4:
            st.metric("Kurtosis", f"{risk_metrics.get('kurtosis', 0):.2f}")
    
    def _render_risk_charts(self, risk_metrics: Dict[str, Any]):
        """Render risk charts."""
        st.subheader("Risk Charts")
        
        # This would normally render risk-related charts
        st.info("Risk charts will be rendered here")
    
    def _render_risk_warnings(self, risk_metrics: Dict[str, Any]):
        """Render risk warnings."""
        st.subheader("Risk Warnings")
        
        warnings = []
        
        var_95 = risk_metrics.get('var_95', 0)
        if var_95 < -1000:  # Example threshold
            warnings.append("⚠️ High Value at Risk detected")
        
        skewness = risk_metrics.get('skewness', 0)
        if skewness < -1:
            warnings.append("⚠️ Negative skewness indicates tail risk")
        
        kurtosis = risk_metrics.get('kurtosis', 0)
        if kurtosis > 3:
            warnings.append("⚠️ High kurtosis indicates fat tails")
        
        if warnings:
            for warning in warnings:
                st.warning(warning)
        else:
            st.success("✅ No significant risk warnings")
    
    def _render_monthly_table(self, monthly_metrics: Dict[str, Any]):
        """Render monthly performance table."""
        st.subheader("Monthly Performance")
        
        monthly_data = monthly_metrics.get('monthly_data', [])
        if monthly_data:
            df = pd.DataFrame(monthly_data)
            st.dataframe(df, use_container_width=True)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Avg Monthly Return", f"{monthly_metrics.get('avg_monthly_return', 0):.2f}%")
        with col2:
            st.metric("Monthly Volatility", f"{monthly_metrics.get('monthly_volatility', 0):.2f}%")
        with col3:
            st.metric("Best Month", f"{monthly_metrics.get('best_month', 0):.2f}%")
        with col4:
            st.metric("Worst Month", f"{monthly_metrics.get('worst_month', 0):.2f}%")
    
    def _render_monthly_charts(self, monthly_metrics: Dict[str, Any]):
        """Render monthly performance charts."""
        st.subheader("Monthly Performance Charts")
        
        monthly_data = monthly_metrics.get('monthly_data', [])
        if not monthly_data:
            st.info("No monthly data available")
            return
        
        df = pd.DataFrame(monthly_data)
        
        # Monthly returns bar chart
        fig = go.Figure()
        
        colors = ['green' if x > 0 else 'red' for x in df['Return (%)']]
        
        fig.add_trace(go.Bar(
            x=df['Month'],
            y=df['Return (%)'],
            marker_color=colors,
            name='Monthly Return'
        ))
        
        fig.update_layout(
            title="Monthly Returns",
            xaxis_title="Month",
            yaxis_title="Return (%)",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
