"""
Report Generator for Backtesting Results

This module generates comprehensive reports for backtesting results,
including performance metrics, charts, and optimization summaries.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

from .backtest_engine import BacktestResult
from .performance_metrics import PerformanceMetrics
from .strategy_comparison import ComparisonResult
from .optimization_engine import OptimizationResult
from ..core.data_models import OHLCV, Trade, Position


class ReportGenerator:
    """Generates comprehensive backtesting reports."""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style for plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def generate_backtest_report(self, 
                               result: BacktestResult,
                               strategy_name: str,
                               save_plots: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive backtest report.
        
        Args:
            result: BacktestResult from backtesting
            strategy_name: Name of the strategy
            save_plots: Whether to save plots to files
        
        Returns:
            Dictionary containing report data
        """
        # Calculate performance metrics
        metrics_calculator = PerformanceMetrics()
        performance = metrics_calculator.calculate_metrics(result)
        
        # Generate report data
        report = {
            'strategy_name': strategy_name,
            'generation_time': datetime.now().isoformat(),
            'backtest_period': {
                'start': result.start_date.isoformat() if result.start_date else None,
                'end': result.end_date.isoformat() if result.end_date else None,
                'duration_days': (result.end_date - result.start_date).days if result.start_date and result.end_date else None
            },
            'performance_metrics': performance,
            'trades_summary': self._generate_trades_summary(result.trades),
            'positions_summary': self._generate_positions_summary(result.positions),
            'equity_curve': self._generate_equity_curve_data(result.equity_curve),
            'drawdown_analysis': self._generate_drawdown_analysis(result.equity_curve),
            'monthly_returns': self._generate_monthly_returns(result.equity_curve),
            'risk_metrics': self._generate_risk_metrics(result)
        }
        
        # Generate plots if requested
        if save_plots:
            plot_files = self._generate_plots(result, strategy_name)
            report['plot_files'] = plot_files
        
        # Save report to JSON
        report_file = self.output_dir / f"{strategy_name}_backtest_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report
    
    def generate_optimization_report(self,
                                   result: OptimizationResult,
                                   strategy_name: str,
                                   param_space: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Generate optimization report.
        
        Args:
            result: OptimizationResult from optimization
            strategy_name: Name of the strategy
            param_space: Parameter space that was optimized
        
        Returns:
            Dictionary containing optimization report
        """
        report = {
            'strategy_name': strategy_name,
            'generation_time': datetime.now().isoformat(),
            'optimization_method': result.method,
            'parameter_space': param_space,
            'best_parameters': result.best_params,
            'best_score': result.best_score,
            'total_evaluations': result.total_evaluations,
            'optimization_time': result.optimization_time,
            'convergence_analysis': self._analyze_optimization_convergence(result),
            'parameter_importance': self._analyze_parameter_importance(result, param_space),
            'optimization_history': result.optimization_history
        }
        
        # Generate optimization plots
        plot_files = self._generate_optimization_plots(result, strategy_name)
        report['plot_files'] = plot_files
        
        # Save report to JSON
        report_file = self.output_dir / f"{strategy_name}_optimization_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report
    
    def generate_comparison_report(self,
                                 comparison: ComparisonResult,
                                 strategy_names: List[str]) -> Dict[str, Any]:
        """
        Generate strategy comparison report.
        
        Args:
            comparison: ComparisonResult from strategy comparison
            strategy_names: List of strategy names
        
        Returns:
            Dictionary containing comparison report
        """
        report = {
            'comparison_time': datetime.now().isoformat(),
            'strategies': strategy_names,
            'comparison_metrics': comparison.metrics_comparison,
            'ranking': comparison.ranking,
            'statistical_tests': comparison.statistical_tests,
            'correlation_analysis': comparison.correlation_analysis,
            'risk_return_analysis': comparison.risk_return_analysis
        }
        
        # Generate comparison plots
        plot_files = self._generate_comparison_plots(comparison, strategy_names)
        report['plot_files'] = plot_files
        
        # Save report to JSON
        report_file = self.output_dir / f"strategy_comparison_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report
    
    def generate_html_report(self, 
                           backtest_reports: List[Dict[str, Any]],
                           optimization_reports: List[Dict[str, Any]] = None,
                           comparison_report: Dict[str, Any] = None) -> str:
        """
        Generate HTML report combining all reports.
        
        Args:
            backtest_reports: List of backtest reports
            optimization_reports: List of optimization reports
            comparison_report: Strategy comparison report
        
        Returns:
            Path to generated HTML file
        """
        html_content = self._generate_html_template(
            backtest_reports, optimization_reports, comparison_report
        )
        
        html_file = self.output_dir / "comprehensive_report.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(html_file)
    
    def _generate_trades_summary(self, trades: List[Trade]) -> Dict[str, Any]:
        """Generate trades summary statistics."""
        if not trades:
            return {'total_trades': 0}
        
        df = pd.DataFrame([{
            'symbol': trade.symbol,
            'side': trade.side.value,
            'quantity': trade.quantity,
            'price': trade.price,
            'timestamp': trade.timestamp,
            'pnl': trade.pnl or 0
        } for trade in trades])
        
        return {
            'total_trades': len(trades),
            'winning_trades': len(df[df['pnl'] > 0]),
            'losing_trades': len(df[df['pnl'] < 0]),
            'win_rate': len(df[df['pnl'] > 0]) / len(df) if len(df) > 0 else 0,
            'avg_win': df[df['pnl'] > 0]['pnl'].mean() if len(df[df['pnl'] > 0]) > 0 else 0,
            'avg_loss': df[df['pnl'] < 0]['pnl'].mean() if len(df[df['pnl'] < 0]) > 0 else 0,
            'largest_win': df['pnl'].max(),
            'largest_loss': df['pnl'].min(),
            'total_pnl': df['pnl'].sum()
        }
    
    def _generate_positions_summary(self, positions: List[Position]) -> Dict[str, Any]:
        """Generate positions summary statistics."""
        if not positions:
            return {'total_positions': 0}
        
        return {
            'total_positions': len(positions),
            'open_positions': len([p for p in positions if p.status == 'open']),
            'closed_positions': len([p for p in positions if p.status == 'closed']),
            'avg_position_size': np.mean([p.quantity for p in positions]),
            'max_position_size': max([p.quantity for p in positions]),
            'min_position_size': min([p.quantity for p in positions])
        }
    
    def _generate_equity_curve_data(self, equity_curve: List[Tuple[datetime, float]]) -> Dict[str, Any]:
        """Generate equity curve data."""
        if not equity_curve:
            return {'data': [], 'statistics': {}}
        
        df = pd.DataFrame(equity_curve, columns=['timestamp', 'equity'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return {
            'data': df.to_dict('records'),
            'statistics': {
                'initial_equity': df['equity'].iloc[0],
                'final_equity': df['equity'].iloc[-1],
                'max_equity': df['equity'].max(),
                'min_equity': df['equity'].min(),
                'equity_volatility': df['equity'].std()
            }
        }
    
    def _generate_drawdown_analysis(self, equity_curve: List[Tuple[datetime, float]]) -> Dict[str, Any]:
        """Generate drawdown analysis."""
        if not equity_curve:
            return {'max_drawdown': 0, 'drawdown_periods': []}
        
        df = pd.DataFrame(equity_curve, columns=['timestamp', 'equity'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate running maximum
        df['running_max'] = df['equity'].expanding().max()
        
        # Calculate drawdown
        df['drawdown'] = (df['equity'] - df['running_max']) / df['running_max']
        
        # Find drawdown periods
        drawdown_periods = []
        in_drawdown = False
        start_date = None
        
        for i, row in df.iterrows():
            if row['drawdown'] < 0 and not in_drawdown:
                in_drawdown = True
                start_date = row['timestamp']
            elif row['drawdown'] >= 0 and in_drawdown:
                in_drawdown = False
                if start_date:
                    drawdown_periods.append({
                        'start': start_date.isoformat(),
                        'end': row['timestamp'].isoformat(),
                        'duration_days': (row['timestamp'] - start_date).days,
                        'max_drawdown': df.loc[start_date:row['timestamp'], 'drawdown'].min()
                    })
        
        return {
            'max_drawdown': df['drawdown'].min(),
            'avg_drawdown': df[df['drawdown'] < 0]['drawdown'].mean(),
            'drawdown_periods': drawdown_periods,
            'drawdown_data': df[['timestamp', 'drawdown']].to_dict('records')
        }
    
    def _generate_monthly_returns(self, equity_curve: List[Tuple[datetime, float]]) -> Dict[str, Any]:
        """Generate monthly returns analysis."""
        if not equity_curve:
            return {'monthly_returns': [], 'statistics': {}}
        
        df = pd.DataFrame(equity_curve, columns=['timestamp', 'equity'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Calculate monthly returns
        monthly_equity = df.resample('M').last()
        monthly_returns = monthly_equity['equity'].pct_change().dropna()
        
        return {
            'monthly_returns': monthly_returns.to_dict(),
            'statistics': {
                'avg_monthly_return': monthly_returns.mean(),
                'monthly_volatility': monthly_returns.std(),
                'best_month': monthly_returns.max(),
                'worst_month': monthly_returns.min(),
                'positive_months': len(monthly_returns[monthly_returns > 0]),
                'negative_months': len(monthly_returns[monthly_returns < 0])
            }
        }
    
    def _generate_risk_metrics(self, result: BacktestResult) -> Dict[str, Any]:
        """Generate risk metrics."""
        if not result.equity_curve:
            return {}
        
        df = pd.DataFrame(result.equity_curve, columns=['timestamp', 'equity'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['returns'] = df['equity'].pct_change().dropna()
        
        # Calculate VaR and CVaR
        returns = df['returns'].dropna()
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'sharpe_ratio': returns.mean() / returns.std() if returns.std() > 0 else 0
        }
    
    def _generate_plots(self, result: BacktestResult, strategy_name: str) -> Dict[str, str]:
        """Generate plots for backtest results."""
        plot_files = {}
        
        # Equity curve plot
        if result.equity_curve:
            plot_files['equity_curve'] = self._plot_equity_curve(result.equity_curve, strategy_name)
        
        # Drawdown plot
        if result.equity_curve:
            plot_files['drawdown'] = self._plot_drawdown(result.equity_curve, strategy_name)
        
        # Returns distribution
        if result.trades:
            plot_files['returns_distribution'] = self._plot_returns_distribution(result.trades, strategy_name)
        
        # Monthly returns heatmap
        if result.equity_curve:
            plot_files['monthly_returns'] = self._plot_monthly_returns(result.equity_curve, strategy_name)
        
        return plot_files
    
    def _plot_equity_curve(self, equity_curve: List[Tuple[datetime, float]], strategy_name: str) -> str:
        """Plot equity curve."""
        df = pd.DataFrame(equity_curve, columns=['timestamp', 'equity'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        plt.figure(figsize=(12, 6))
        plt.plot(df['timestamp'], df['equity'], linewidth=2)
        plt.title(f'{strategy_name} - Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = self.output_dir / f"{strategy_name}_equity_curve.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filename)
    
    def _plot_drawdown(self, equity_curve: List[Tuple[datetime, float]], strategy_name: str) -> str:
        """Plot drawdown."""
        df = pd.DataFrame(equity_curve, columns=['timestamp', 'equity'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['running_max'] = df['equity'].expanding().max()
        df['drawdown'] = (df['equity'] - df['running_max']) / df['running_max']
        
        plt.figure(figsize=(12, 6))
        plt.fill_between(df['timestamp'], df['drawdown'], 0, alpha=0.7, color='red')
        plt.plot(df['timestamp'], df['drawdown'], color='darkred', linewidth=1)
        plt.title(f'{strategy_name} - Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = self.output_dir / f"{strategy_name}_drawdown.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filename)
    
    def _plot_returns_distribution(self, trades: List[Trade], strategy_name: str) -> str:
        """Plot returns distribution."""
        pnls = [trade.pnl for trade in trades if trade.pnl is not None]
        
        plt.figure(figsize=(10, 6))
        plt.hist(pnls, bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(pnls), color='red', linestyle='--', label=f'Mean: {np.mean(pnls):.2f}')
        plt.title(f'{strategy_name} - Returns Distribution')
        plt.xlabel('PnL')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = self.output_dir / f"{strategy_name}_returns_distribution.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filename)
    
    def _plot_monthly_returns(self, equity_curve: List[Tuple[datetime, float]], strategy_name: str) -> str:
        """Plot monthly returns heatmap."""
        df = pd.DataFrame(equity_curve, columns=['timestamp', 'equity'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Calculate monthly returns
        monthly_equity = df.resample('M').last()
        monthly_returns = monthly_equity['equity'].pct_change().dropna()
        
        # Create year-month matrix
        returns_matrix = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).first()
        returns_matrix = returns_matrix.unstack(level=1)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(returns_matrix, annot=True, fmt='.2%', cmap='RdYlGn', center=0)
        plt.title(f'{strategy_name} - Monthly Returns Heatmap')
        plt.xlabel('Month')
        plt.ylabel('Year')
        plt.tight_layout()
        
        filename = self.output_dir / f"{strategy_name}_monthly_returns.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filename)
    
    def _generate_optimization_plots(self, result: OptimizationResult, strategy_name: str) -> Dict[str, str]:
        """Generate optimization plots."""
        plot_files = {}
        
        # Optimization progress
        plot_files['optimization_progress'] = self._plot_optimization_progress(result, strategy_name)
        
        # Parameter importance
        plot_files['parameter_importance'] = self._plot_parameter_importance(result, strategy_name)
        
        return plot_files
    
    def _plot_optimization_progress(self, result: OptimizationResult, strategy_name: str) -> str:
        """Plot optimization progress."""
        history = result.optimization_history
        evaluations = [h['evaluation'] for h in history]
        scores = [h['score'] for h in history]
        best_scores = [h['best_score'] for h in history]
        
        plt.figure(figsize=(12, 6))
        plt.plot(evaluations, scores, alpha=0.3, color='blue', label='Current Score')
        plt.plot(evaluations, best_scores, color='red', linewidth=2, label='Best Score')
        plt.title(f'{strategy_name} - Optimization Progress ({result.method})')
        plt.xlabel('Evaluation')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = self.output_dir / f"{strategy_name}_optimization_progress.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filename)
    
    def _plot_parameter_importance(self, result: OptimizationResult, strategy_name: str) -> str:
        """Plot parameter importance."""
        # This is a simplified version - in practice, you'd use more sophisticated methods
        param_scores = {}
        for h in result.optimization_history:
            for param, value in h['params'].items():
                if param not in param_scores:
                    param_scores[param] = []
                param_scores[param].append(h['score'])
        
        # Calculate parameter importance (correlation with score)
        importance = {}
        for param, scores in param_scores.items():
            if len(scores) > 1:
                importance[param] = abs(np.corrcoef(scores, range(len(scores)))[0, 1])
        
        if importance:
            params = list(importance.keys())
            values = list(importance.values())
            
            plt.figure(figsize=(10, 6))
            plt.bar(params, values)
            plt.title(f'{strategy_name} - Parameter Importance')
            plt.xlabel('Parameter')
            plt.ylabel('Importance')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            filename = self.output_dir / f"{strategy_name}_parameter_importance.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(filename)
        
        return ""
    
    def _generate_comparison_plots(self, comparison: ComparisonResult, strategy_names: List[str]) -> Dict[str, str]:
        """Generate comparison plots."""
        plot_files = {}
        
        # Risk-return scatter plot
        plot_files['risk_return'] = self._plot_risk_return_comparison(comparison, strategy_names)
        
        # Performance comparison
        plot_files['performance_comparison'] = self._plot_performance_comparison(comparison, strategy_names)
        
        return plot_files
    
    def _plot_risk_return_comparison(self, comparison: ComparisonResult, strategy_names: List[str]) -> str:
        """Plot risk-return comparison."""
        plt.figure(figsize=(10, 8))
        
        for i, strategy in enumerate(strategy_names):
            if strategy in comparison.metrics_comparison:
                metrics = comparison.metrics_comparison[strategy]
                plt.scatter(metrics.get('volatility', 0), metrics.get('total_return', 0), 
                           label=strategy, s=100)
        
        plt.xlabel('Volatility')
        plt.ylabel('Total Return')
        plt.title('Risk-Return Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = self.output_dir / "risk_return_comparison.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filename)
    
    def _plot_performance_comparison(self, comparison: ComparisonResult, strategy_names: List[str]) -> str:
        """Plot performance metrics comparison."""
        metrics = ['sharpe_ratio', 'total_return', 'max_drawdown', 'profit_factor']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = []
            labels = []
            
            for strategy in strategy_names:
                if strategy in comparison.metrics_comparison:
                    value = comparison.metrics_comparison[strategy].get(metric, 0)
                    values.append(value)
                    labels.append(strategy)
            
            if values:
                axes[i].bar(labels, values)
                axes[i].set_title(metric.replace('_', ' ').title())
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        filename = self.output_dir / "performance_comparison.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filename)
    
    def _analyze_optimization_convergence(self, result: OptimizationResult) -> Dict[str, Any]:
        """Analyze optimization convergence."""
        history = result.optimization_history
        if len(history) < 2:
            return {'converged': False, 'convergence_rate': 0}
        
        scores = [h['score'] for h in history]
        best_scores = [h['best_score'] for h in history]
        
        # Calculate convergence metrics
        improvement = best_scores[-1] - best_scores[0]
        convergence_rate = improvement / len(history) if len(history) > 0 else 0
        
        # Check convergence
        last_20_percent = int(len(history) * 0.2)
        if last_20_percent > 0:
            recent_best = best_scores[-last_20_percent:]
            converged = max(recent_best) - min(recent_best) < 0.01
        else:
            converged = False
        
        return {
            'converged': converged,
            'convergence_rate': convergence_rate,
            'final_improvement': improvement,
            'score_volatility': np.std(scores),
            'best_score_volatility': np.std(best_scores)
        }
    
    def _analyze_parameter_importance(self, result: OptimizationResult, param_space: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Analyze parameter importance."""
        param_scores = {}
        
        for h in result.optimization_history:
            for param, value in h['params'].items():
                if param not in param_scores:
                    param_scores[param] = []
                param_scores[param].append(h['score'])
        
        importance = {}
        for param, scores in param_scores.items():
            if len(scores) > 1:
                # Calculate correlation between parameter values and scores
                importance[param] = abs(np.corrcoef(scores, range(len(scores)))[0, 1])
        
        return importance
    
    def _generate_html_template(self, 
                               backtest_reports: List[Dict[str, Any]],
                               optimization_reports: List[Dict[str, Any]] = None,
                               comparison_report: Dict[str, Any] = None) -> str:
        """Generate HTML template for comprehensive report."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trading Strategy Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                .section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
                .metric { display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }
                .plot { text-align: center; margin: 20px 0; }
                table { width: 100%; border-collapse: collapse; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Trading Strategy Analysis Report</h1>
                <p>Generated on: {generation_time}</p>
            </div>
        """
        
        # Add backtest reports
        for report in backtest_reports:
            html += f"""
            <div class="section">
                <h2>{report['strategy_name']} - Backtest Results</h2>
                <div class="metric">Total Return: {report['performance_metrics'].get('total_return', 0):.2%}</div>
                <div class="metric">Sharpe Ratio: {report['performance_metrics'].get('sharpe_ratio', 0):.2f}</div>
                <div class="metric">Max Drawdown: {report['performance_metrics'].get('max_drawdown', 0):.2%}</div>
                <div class="metric">Win Rate: {report['trades_summary'].get('win_rate', 0):.2%}</div>
            </div>
            """
        
        # Add optimization reports
        if optimization_reports:
            for report in optimization_reports:
                html += f"""
                <div class="section">
                    <h2>{report['strategy_name']} - Optimization Results</h2>
                    <p>Method: {report['optimization_method']}</p>
                    <p>Best Score: {report['best_score']:.4f}</p>
                    <p>Total Evaluations: {report['total_evaluations']}</p>
                    <p>Optimization Time: {report['optimization_time']:.2f} seconds</p>
                </div>
                """
        
        # Add comparison report
        if comparison_report:
            html += """
            <div class="section">
                <h2>Strategy Comparison</h2>
                <p>Comparison results and rankings...</p>
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        
        return html.format(generation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
