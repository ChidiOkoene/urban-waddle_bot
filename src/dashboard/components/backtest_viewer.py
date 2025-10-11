"""
Backtest Viewer Component for Trading Dashboard

This module provides components for viewing and analyzing backtest results,
including performance metrics, charts, and detailed trade analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json

from ...core.data_models import Trade, Position, OHLCV
from ...backtesting.backtest_engine import BacktestResult
from ...backtesting.performance_metrics import PerformanceMetrics
from ...backtesting.report_generator import ReportGenerator


class BacktestViewer:
    """Component for viewing and analyzing backtest results."""
    
    def __init__(self):
        self.color_scheme = {
            'profit': '#00ff88',
            'loss': '#ff4444',
            'neutral': '#888888',
            'warning': '#ffaa00',
            'success': '#00ff88'
        }
    
    def render_backtest_results(self, results: BacktestResult):
        """
        Render comprehensive backtest results.
        
        Args:
            results: BacktestResult object containing backtest data
        """
        st.header("📈 Backtest Results")
        
        # Calculate performance metrics
        metrics_calculator = PerformanceMetrics()
        performance = metrics_calculator.calculate_metrics(results)
        
        # Display key metrics
        self._render_key_metrics(performance)
        
        # Display detailed metrics
        self._render_detailed_metrics(performance)
        
        # Display charts
        self._render_backtest_charts(results)
        
        # Display trade analysis
        self._render_trade_analysis(results.trades)
        
        # Display position analysis
        self._render_position_analysis(results.positions)
        
        # Display risk analysis
        self._render_risk_analysis(results)
        
        # Export options
        self._render_export_options(results, performance)
    
    def render_backtest_comparison(self, results_list: List[Tuple[str, BacktestResult]]):
        """
        Render comparison of multiple backtest results.
        
        Args:
            results_list: List of (strategy_name, BacktestResult) tuples
        """
        st.header("📊 Backtest Comparison")
        
        if len(results_list) < 2:
            st.warning("Need at least 2 backtest results to compare")
            return
        
        # Calculate metrics for all strategies
        metrics_calculator = PerformanceMetrics()
        all_metrics = {}
        
        for strategy_name, results in results_list:
            metrics = metrics_calculator.calculate_metrics(results)
            all_metrics[strategy_name] = metrics
        
        # Display comparison table
        self._render_comparison_table(all_metrics)
        
        # Display comparison charts
        self._render_comparison_charts(all_metrics)
        
        # Display ranking
        self._render_strategy_ranking(all_metrics)
    
    def render_backtest_optimization_results(self, optimization_results: List[Dict[str, Any]]):
        """
        Render optimization results.
        
        Args:
            optimization_results: List of optimization result dictionaries
        """
        st.header("🔧 Optimization Results")
        
        if not optimization_results:
            st.warning("No optimization results available")
            return
        
        # Display optimization summary
        self._render_optimization_summary(optimization_results)
        
        # Display parameter analysis
        self._render_parameter_analysis(optimization_results)
        
        # Display optimization charts
        self._render_optimization_charts(optimization_results)
    
    def _render_key_metrics(self, performance: Dict[str, Any]):
        """Render key performance metrics."""
        st.subheader("Key Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_return = performance.get('total_return', 0) * 100
            st.metric(
                "Total Return", 
                f"{total_return:.2f}%",
                delta=f"{total_return:.2f}%" if total_return != 0 else None
            )
        
        with col2:
            sharpe_ratio = performance.get('sharpe_ratio', 0)
            st.metric(
                "Sharpe Ratio", 
                f"{sharpe_ratio:.2f}",
                delta=f"{sharpe_ratio:.2f}" if sharpe_ratio != 0 else None
            )
        
        with col3:
            max_drawdown = performance.get('max_drawdown', 0) * 100
            st.metric(
                "Max Drawdown", 
                f"{max_drawdown:.2f}%",
                delta=f"{max_drawdown:.2f}%" if max_drawdown != 0 else None
            )
        
        with col4:
            win_rate = performance.get('win_rate', 0) * 100
            st.metric(
                "Win Rate", 
                f"{win_rate:.1f}%",
                delta=f"{win_rate:.1f}%" if win_rate != 0 else None
            )
    
    def _render_detailed_metrics(self, performance: Dict[str, Any]):
        """Render detailed performance metrics."""
        st.subheader("Detailed Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Trading Metrics**")
            st.write(f"Total Trades: {performance.get('total_trades', 0)}")
            st.write(f"Winning Trades: {performance.get('winning_trades', 0)}")
            st.write(f"Losing Trades: {performance.get('losing_trades', 0)}")
            st.write(f"Average PnL: ${performance.get('avg_pnl', 0):.2f}")
            st.write(f"Profit Factor: {performance.get('profit_factor', 0):.2f}")
        
        with col2:
            st.write("**Risk Metrics**")
            st.write(f"Volatility: {performance.get('volatility', 0) * 100:.2f}%")
            st.write(f"Sortino Ratio: {performance.get('sortino_ratio', 0):.2f}")
            st.write(f"VaR (95%): ${performance.get('var_95', 0):.2f}")
            st.write(f"CVaR (95%): ${performance.get('cvar_95', 0):.2f}")
            st.write(f"Skewness: {performance.get('skewness', 0):.2f}")
    
    def _render_backtest_charts(self, results: BacktestResult):
        """Render backtest charts."""
        st.subheader("Performance Charts")
        
        if results.equity_curve:
            # Equity curve
            self._render_equity_curve(results.equity_curve)
            
            # Drawdown chart
            self._render_drawdown_chart(results.equity_curve)
            
            # Monthly returns heatmap
            self._render_monthly_returns_heatmap(results.equity_curve)
        
        if results.trades:
            # Returns distribution
            self._render_returns_distribution(results.trades)
    
    def _render_equity_curve(self, equity_curve: List[Tuple[datetime, float]]):
        """Render equity curve chart."""
        df = pd.DataFrame(equity_curve, columns=['timestamp', 'equity'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig = go.Figure()
        
        # Equity curve
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['equity'],
            mode='lines',
            name='Equity',
            line=dict(color=self.color_scheme['profit'], width=2),
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
            title="Equity Curve",
            xaxis_title="Time",
            yaxis_title="Equity ($)",
            template="plotly_dark",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_drawdown_chart(self, equity_curve: List[Tuple[datetime, float]]):
        """Render drawdown chart."""
        df = pd.DataFrame(equity_curve, columns=['timestamp', 'equity'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
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
            line=dict(color=self.color_scheme['loss'], width=1),
            name='Drawdown',
            hovertemplate='<b>%{x}</b><br>Drawdown: %{y:.2f}%<extra></extra>'
        ))
        
        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
        
        fig.update_layout(
            title="Drawdown",
            xaxis_title="Time",
            yaxis_title="Drawdown (%)",
            template="plotly_dark",
            height=300,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_monthly_returns_heatmap(self, equity_curve: List[Tuple[datetime, float]]):
        """Render monthly returns heatmap."""
        df = pd.DataFrame(equity_curve, columns=['timestamp', 'equity'])
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
            title="Monthly Returns Heatmap",
            xaxis_title="Month",
            yaxis_title="Year",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_returns_distribution(self, trades: List[Trade]):
        """Render returns distribution histogram."""
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
            line_color=self.color_scheme['profit'],
            annotation_text=f"Mean: ${mean_pnl:.2f}"
        )
        
        fig.update_layout(
            title="Returns Distribution",
            xaxis_title="PnL ($)",
            yaxis_title="Frequency",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_trade_analysis(self, trades: List[Trade]):
        """Render trade analysis."""
        st.subheader("Trade Analysis")
        
        if not trades:
            st.info("No trades to analyze")
            return
        
        # Convert trades to DataFrame
        trades_data = []
        for trade in trades:
            trades_data.append({
                'Timestamp': trade.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'Symbol': trade.symbol,
                'Side': trade.side.value,
                'Quantity': f"{trade.quantity:.6f}",
                'Price': f"${trade.price:.2f}",
                'PnL': f"${trade.pnl:.2f}" if trade.pnl else "N/A"
            })
        
        trades_df = pd.DataFrame(trades_data)
        
        # Display trades table
        st.dataframe(trades_df, use_container_width=True)
        
        # Trade statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", len(trades))
        with col2:
            winning_trades = len([t for t in trades if t.pnl and t.pnl > 0])
            st.metric("Winning Trades", winning_trades)
        with col3:
            win_rate = (winning_trades / len(trades) * 100) if trades else 0
            st.metric("Win Rate", f"{win_rate:.1f}%")
        with col4:
            total_pnl = sum([t.pnl for t in trades if t.pnl])
            st.metric("Total PnL", f"${total_pnl:.2f}")
    
    def _render_position_analysis(self, positions: List[Position]):
        """Render position analysis."""
        st.subheader("Position Analysis")
        
        if not positions:
            st.info("No positions to analyze")
            return
        
        # Position statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Positions", len(positions))
        with col2:
            open_positions = len([p for p in positions if p.status.value == 'open'])
            st.metric("Open Positions", open_positions)
        with col3:
            avg_position_size = np.mean([p.quantity for p in positions])
            st.metric("Avg Position Size", f"{avg_position_size:.6f}")
        with col4:
            avg_time_in_position = np.mean([(datetime.now() - p.timestamp).total_seconds() / 3600 
                                          for p in positions])
            st.metric("Avg Time (hours)", f"{avg_time_in_position:.1f}")
    
    def _render_risk_analysis(self, results: BacktestResult):
        """Render risk analysis."""
        st.subheader("Risk Analysis")
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(results)
        
        # Display risk metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("VaR (95%)", f"${risk_metrics.get('var_95', 0):.2f}")
        with col2:
            st.metric("CVaR (95%)", f"${risk_metrics.get('cvar_95', 0):.2f}")
        with col3:
            st.metric("Skewness", f"{risk_metrics.get('skewness', 0):.2f}")
        with col4:
            st.metric("Kurtosis", f"{risk_metrics.get('kurtosis', 0):.2f}")
        
        # Risk warnings
        self._render_risk_warnings(risk_metrics)
    
    def _calculate_risk_metrics(self, results: BacktestResult) -> Dict[str, Any]:
        """Calculate risk metrics from backtest results."""
        risk_metrics = {}
        
        # Calculate VaR and CVaR from trades
        if results.trades:
            trades_with_pnl = [t for t in results.trades if t.pnl is not None]
            if trades_with_pnl:
                pnls = [t.pnl for t in trades_with_pnl]
                
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
        
        # Calculate skewness and kurtosis from equity curve
        if results.equity_curve:
            df = pd.DataFrame(results.equity_curve, columns=['timestamp', 'equity'])
            df['returns'] = df['equity'].pct_change().dropna()
            
            if len(df['returns']) > 0:
                risk_metrics.update({
                    'skewness': df['returns'].skew(),
                    'kurtosis': df['returns'].kurtosis()
                })
        
        return risk_metrics
    
    def _render_risk_warnings(self, risk_metrics: Dict[str, Any]):
        """Render risk warnings."""
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
            st.subheader("Risk Warnings")
            for warning in warnings:
                st.warning(warning)
        else:
            st.success("✅ No significant risk warnings")
    
    def _render_export_options(self, results: BacktestResult, performance: Dict[str, Any]):
        """Render export options."""
        st.subheader("Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export to CSV"):
                self._export_to_csv(results)
        
        with col2:
            if st.button("📈 Export Charts"):
                self._export_charts(results)
        
        with col3:
            if st.button("📋 Generate Report"):
                self._generate_report(results, performance)
    
    def _export_to_csv(self, results: BacktestResult):
        """Export results to CSV."""
        # Export trades
        if results.trades:
            trades_data = []
            for trade in results.trades:
                trades_data.append({
                    'timestamp': trade.timestamp,
                    'symbol': trade.symbol,
                    'side': trade.side.value,
                    'quantity': trade.quantity,
                    'price': trade.price,
                    'pnl': trade.pnl
                })
            
            trades_df = pd.DataFrame(trades_data)
            csv = trades_df.to_csv(index=False)
            
            st.download_button(
                label="Download Trades CSV",
                data=csv,
                file_name="trades.csv",
                mime="text/csv"
            )
        
        # Export equity curve
        if results.equity_curve:
            equity_df = pd.DataFrame(results.equity_curve, columns=['timestamp', 'equity'])
            csv = equity_df.to_csv(index=False)
            
            st.download_button(
                label="Download Equity Curve CSV",
                data=csv,
                file_name="equity_curve.csv",
                mime="text/csv"
            )
    
    def _export_charts(self, results: BacktestResult):
        """Export charts as images."""
        st.info("Chart export functionality would be implemented here")
    
    def _generate_report(self, results: BacktestResult, performance: Dict[str, Any]):
        """Generate comprehensive report."""
        st.info("Report generation functionality would be implemented here")
    
    def _render_comparison_table(self, all_metrics: Dict[str, Dict[str, Any]]):
        """Render comparison table."""
        st.subheader("Strategy Comparison")
        
        comparison_data = []
        for strategy, metrics in all_metrics.items():
            comparison_data.append({
                'Strategy': strategy,
                'Total Return (%)': f"{metrics.get('total_return', 0) * 100:.2f}",
                'Sharpe Ratio': f"{metrics.get('sharpe_ratio', 0):.2f}",
                'Max Drawdown (%)': f"{metrics.get('max_drawdown', 0) * 100:.2f}",
                'Win Rate (%)': f"{metrics.get('win_rate', 0) * 100:.1f}",
                'Profit Factor': f"{metrics.get('profit_factor', 0):.2f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
    
    def _render_comparison_charts(self, all_metrics: Dict[str, Dict[str, Any]]):
        """Render comparison charts."""
        st.subheader("Comparison Charts")
        
        # Risk-Return scatter plot
        strategies = list(all_metrics.keys())
        returns = [all_metrics[s].get('total_return', 0) * 100 for s in strategies]
        risks = [all_metrics[s].get('volatility', 0) * 100 for s in strategies]
        
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
    
    def _render_strategy_ranking(self, all_metrics: Dict[str, Dict[str, Any]]):
        """Render strategy ranking."""
        st.subheader("Strategy Ranking")
        
        # Calculate composite score
        rankings = []
        for strategy, metrics in all_metrics.items():
            sharpe = metrics.get('sharpe_ratio', 0)
            total_return = metrics.get('total_return', 0) * 100
            max_drawdown = abs(metrics.get('max_drawdown', 0)) * 100
            win_rate = metrics.get('win_rate', 0) * 100
            
            # Weighted composite score
            composite_score = (
                sharpe * 0.4 +
                total_return * 0.3 +
                (100 - max_drawdown) * 0.2 +
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
        ranking_df.index = range(1, len(ranking_df) + 1)
        
        st.dataframe(ranking_df, use_container_width=True)
    
    def _render_optimization_summary(self, optimization_results: List[Dict[str, Any]]):
        """Render optimization summary."""
        st.subheader("Optimization Summary")
        
        # Display best results
        best_result = max(optimization_results, key=lambda x: x.get('best_score', 0))
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Best Score", f"{best_result.get('best_score', 0):.4f}")
        with col2:
            st.metric("Method", best_result.get('method', 'Unknown'))
        with col3:
            st.metric("Evaluations", best_result.get('total_evaluations', 0))
        with col4:
            st.metric("Time (s)", f"{best_result.get('optimization_time', 0):.2f}")
        
        # Display best parameters
        st.subheader("Best Parameters")
        best_params = best_result.get('best_params', {})
        for param, value in best_params.items():
            st.write(f"**{param}**: {value}")
    
    def _render_parameter_analysis(self, optimization_results: List[Dict[str, Any]]):
        """Render parameter analysis."""
        st.subheader("Parameter Analysis")
        
        # This would normally analyze parameter sensitivity
        st.info("Parameter sensitivity analysis would be displayed here")
    
    def _render_optimization_charts(self, optimization_results: List[Dict[str, Any]]):
        """Render optimization charts."""
        st.subheader("Optimization Charts")
        
        # Plot optimization progress
        for result in optimization_results:
            if 'optimization_history' in result:
                history = result['optimization_history']
                if history:
                    df = pd.DataFrame(history)
                    
                    fig = go.Figure()
                    
                    # Current scores
                    fig.add_trace(go.Scatter(
                        x=df['evaluation'],
                        y=df['score'],
                        mode='markers',
                        name='Current Score',
                        marker=dict(size=4, color='blue'),
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
                        title=f"Optimization Progress - {result.get('method', 'Unknown')}",
                        xaxis_title="Evaluation",
                        yaxis_title="Score",
                        template="plotly_dark",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
