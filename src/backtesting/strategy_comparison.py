"""
Strategy comparison and ranking.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from .backtest_engine import BacktestResult
from .performance_metrics import PerformanceMetrics, PerformanceCalculator


class ComparisonMethod(str, Enum):
    """Comparison methods."""
    SHARPE_RATIO = "sharpe_ratio"
    TOTAL_RETURN = "total_return"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    CALMAR_RATIO = "calmar_ratio"
    COMPOSITE_SCORE = "composite_score"


@dataclass
class StrategyComparison:
    """Strategy comparison result."""
    strategy_name: str
    performance_metrics: PerformanceMetrics
    backtest_result: BacktestResult
    ranking_score: float
    rank: int


class StrategyComparator:
    """Compare and rank multiple strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize strategy comparator.
        
        Args:
            config: Comparison configuration
        """
        self.config = config
        
        # Comparison parameters
        self.primary_metric = config.get('primary_metric', ComparisonMethod.COMPOSITE_SCORE)
        self.weighted_metrics = config.get('weighted_metrics', {
            'sharpe_ratio': 0.3,
            'total_return': 0.25,
            'max_drawdown': 0.2,
            'win_rate': 0.15,
            'profit_factor': 0.1
        })
        
        # Performance calculator
        self.performance_calculator = PerformanceCalculator(
            risk_free_rate=config.get('risk_free_rate', 0.02)
        )
        
        # Results storage
        self.comparison_results = {}
        self.rankings = []
    
    def compare_strategies(self, strategy_results: Dict[str, BacktestResult]) -> List[StrategyComparison]:
        """
        Compare multiple strategies.
        
        Args:
            strategy_results: Dictionary of strategy name -> BacktestResult
            
        Returns:
            List of StrategyComparison objects ranked by performance
        """
        comparisons = []
        
        # Calculate performance metrics for each strategy
        for strategy_name, backtest_result in strategy_results.items():
            performance_metrics = self.performance_calculator.calculate_metrics(backtest_result)
            
            comparison = StrategyComparison(
                strategy_name=strategy_name,
                performance_metrics=performance_metrics,
                backtest_result=backtest_result,
                ranking_score=0.0,  # Will be calculated
                rank=0  # Will be set after ranking
            )
            
            comparisons.append(comparison)
        
        # Calculate ranking scores
        self._calculate_ranking_scores(comparisons)
        
        # Rank strategies
        comparisons.sort(key=lambda x: x.ranking_score, reverse=True)
        
        # Set ranks
        for i, comparison in enumerate(comparisons):
            comparison.rank = i + 1
        
        # Store results
        self.comparison_results = {comp.strategy_name: comp for comp in comparisons}
        self.rankings = comparisons
        
        return comparisons
    
    def _calculate_ranking_scores(self, comparisons: List[StrategyComparison]):
        """Calculate ranking scores for strategies."""
        if not comparisons:
            return
        
        # Extract metrics for normalization
        metrics_data = {}
        for metric_name in self.weighted_metrics.keys():
            values = []
            for comp in comparisons:
                value = getattr(comp.performance_metrics, metric_name, 0)
                values.append(value)
            metrics_data[metric_name] = values
        
        # Normalize metrics (0-1 scale)
        normalized_metrics = {}
        for metric_name, values in metrics_data.items():
            if not values:
                normalized_metrics[metric_name] = [0] * len(comparisons)
                continue
            
            min_val = min(values)
            max_val = max(values)
            
            if max_val == min_val:
                normalized_metrics[metric_name] = [0.5] * len(comparisons)
            else:
                normalized_values = [(v - min_val) / (max_val - min_val) for v in values]
                normalized_metrics[metric_name] = normalized_values
        
        # Calculate composite scores
        for i, comparison in enumerate(comparisons):
            composite_score = 0.0
            
            for metric_name, weight in self.weighted_metrics.items():
                normalized_value = normalized_metrics[metric_name][i]
                
                # For max_drawdown, invert the score (lower is better)
                if metric_name == 'max_drawdown':
                    normalized_value = 1.0 - normalized_value
                
                composite_score += weight * normalized_value
            
            comparison.ranking_score = composite_score
    
    def get_top_strategies(self, n: int = 5) -> List[StrategyComparison]:
        """Get top N strategies."""
        return self.rankings[:n]
    
    def get_strategy_ranking(self, strategy_name: str) -> Optional[StrategyComparison]:
        """Get ranking for a specific strategy."""
        return self.comparison_results.get(strategy_name)
    
    def calculate_statistical_significance(self, strategy1: str, strategy2: str) -> Dict[str, Any]:
        """Calculate statistical significance between two strategies."""
        if strategy1 not in self.comparison_results or strategy2 not in self.comparison_results:
            return {'error': 'Strategy not found'}
        
        comp1 = self.comparison_results[strategy1]
        comp2 = self.comparison_results[strategy2]
        
        # Extract returns
        returns1 = self._extract_returns(comp1.backtest_result)
        returns2 = self._extract_returns(comp2.backtest_result)
        
        if len(returns1) < 30 or len(returns2) < 30:
            return {'error': 'Insufficient data for statistical test'}
        
        # Perform t-test
        from scipy import stats
        
        t_stat, p_value = stats.ttest_ind(returns1, returns2)
        
        # Calculate correlation
        min_length = min(len(returns1), len(returns2))
        returns1_truncated = returns1[:min_length]
        returns2_truncated = returns2[:min_length]
        
        correlation = np.corrcoef(returns1_truncated, returns2_truncated)[0, 1]
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'correlation': correlation,
            'significant': p_value < 0.05,
            'strategy1_mean': np.mean(returns1),
            'strategy2_mean': np.mean(returns2),
            'strategy1_std': np.std(returns1),
            'strategy2_std': np.std(returns2)
        }
    
    def _extract_returns(self, backtest_result: BacktestResult) -> List[float]:
        """Extract returns from backtest result."""
        equity_curve = [float(eq) for eq in backtest_result.equity_curve]
        returns = []
        
        for i in range(1, len(equity_curve)):
            if equity_curve[i-1] != 0:
                ret = (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
                returns.append(ret)
        
        return returns
    
    def generate_comparison_report(self) -> str:
        """Generate comprehensive comparison report."""
        if not self.rankings:
            return "No comparison results available"
        
        report = f"""
# Strategy Comparison Report

## Summary
- **Total Strategies**: {len(self.rankings)}
- **Primary Metric**: {self.primary_metric.value}
- **Comparison Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Top 5 Strategies

"""
        
        for i, comparison in enumerate(self.rankings[:5]):
            metrics = comparison.performance_metrics
            report += f"""
### {i+1}. {comparison.strategy_name}
- **Ranking Score**: {comparison.ranking_score:.3f}
- **Total Return**: {metrics.total_return:.2%}
- **Annualized Return**: {metrics.annualized_return:.2%}
- **Sharpe Ratio**: {metrics.sharpe_ratio:.2f}
- **Max Drawdown**: {metrics.max_drawdown:.2%}
- **Win Rate**: {metrics.win_rate:.2%}
- **Profit Factor**: {metrics.profit_factor:.2f}
- **Total Trades**: {metrics.total_trades}

"""
        
        # Performance comparison table
        report += """
## Performance Comparison Table

| Strategy | Total Return | Sharpe | Max DD | Win Rate | Profit Factor | Rank |
|----------|--------------|--------|--------|----------|---------------|------|
"""
        
        for comparison in self.rankings:
            metrics = comparison.performance_metrics
            report += f"| {comparison.strategy_name} | {metrics.total_return:.2%} | {metrics.sharpe_ratio:.2f} | {metrics.max_drawdown:.2%} | {metrics.win_rate:.2%} | {metrics.profit_factor:.2f} | {comparison.rank} |\n"
        
        # Risk-return analysis
        report += """
## Risk-Return Analysis

"""
        
        # Calculate risk-return metrics
        returns = [comp.performance_metrics.annualized_return for comp in self.rankings]
        risks = [comp.performance_metrics.volatility for comp in self.rankings]
        
        if returns and risks:
            avg_return = np.mean(returns)
            avg_risk = np.mean(risks)
            
            report += f"""
- **Average Annualized Return**: {avg_return:.2%}
- **Average Volatility**: {avg_risk:.2%}
- **Best Risk-Adjusted Return**: {self.rankings[0].strategy_name} (Sharpe: {self.rankings[0].performance_metrics.sharpe_ratio:.2f})
- **Lowest Drawdown**: {min(self.rankings, key=lambda x: x.performance_metrics.max_drawdown).strategy_name} ({min(self.rankings, key=lambda x: x.performance_metrics.max_drawdown).performance_metrics.max_drawdown:.2%})
"""
        
        # Recommendations
        report += """
## Recommendations

"""
        
        best_strategy = self.rankings[0]
        best_metrics = best_strategy.performance_metrics
        
        if best_metrics.sharpe_ratio > 1.0:
            report += f"- **Best Overall Strategy**: {best_strategy.strategy_name} shows excellent risk-adjusted returns\n"
        elif best_metrics.sharpe_ratio > 0.5:
            report += f"- **Best Overall Strategy**: {best_strategy.strategy_name} shows good risk-adjusted returns\n"
        else:
            report += f"- **Best Overall Strategy**: {best_strategy.strategy_name} shows moderate risk-adjusted returns\n"
        
        if best_metrics.max_drawdown < 0.1:
            report += "- **Low Risk**: All top strategies show low maximum drawdown\n"
        elif best_metrics.max_drawdown < 0.2:
            report += "- **Moderate Risk**: Top strategies show moderate maximum drawdown\n"
        else:
            report += "- **High Risk**: Some strategies show high maximum drawdown\n"
        
        if best_metrics.win_rate > 0.6:
            report += "- **High Win Rate**: Top strategies show high win rates\n"
        elif best_metrics.win_rate > 0.5:
            report += "- **Moderate Win Rate**: Top strategies show moderate win rates\n"
        else:
            report += "- **Low Win Rate**: Strategies rely on high profit factors\n"
        
        return report
    
    def get_comparison_summary(self) -> Dict[str, Any]:
        """Get comparison summary."""
        if not self.rankings:
            return {}
        
        # Calculate summary statistics
        total_returns = [comp.performance_metrics.total_return for comp in self.rankings]
        sharpe_ratios = [comp.performance_metrics.sharpe_ratio for comp in self.rankings]
        max_drawdowns = [comp.performance_metrics.max_drawdown for comp in self.rankings]
        win_rates = [comp.performance_metrics.win_rate for comp in self.rankings]
        
        return {
            'total_strategies': len(self.rankings),
            'primary_metric': self.primary_metric.value,
            'weighted_metrics': self.weighted_metrics,
            'summary_stats': {
                'total_return': {
                    'mean': np.mean(total_returns),
                    'std': np.std(total_returns),
                    'min': np.min(total_returns),
                    'max': np.max(total_returns)
                },
                'sharpe_ratio': {
                    'mean': np.mean(sharpe_ratios),
                    'std': np.std(sharpe_ratios),
                    'min': np.min(sharpe_ratios),
                    'max': np.max(sharpe_ratios)
                },
                'max_drawdown': {
                    'mean': np.mean(max_drawdowns),
                    'std': np.std(max_drawdowns),
                    'min': np.min(max_drawdowns),
                    'max': np.max(max_drawdowns)
                },
                'win_rate': {
                    'mean': np.mean(win_rates),
                    'std': np.std(win_rates),
                    'min': np.min(win_rates),
                    'max': np.max(win_rates)
                }
            },
            'top_strategy': {
                'name': self.rankings[0].strategy_name,
                'ranking_score': self.rankings[0].ranking_score,
                'total_return': self.rankings[0].performance_metrics.total_return,
                'sharpe_ratio': self.rankings[0].performance_metrics.sharpe_ratio
            }
        }
    
    def export_comparison_data(self, format: str = 'csv') -> str:
        """Export comparison data."""
        if not self.rankings:
            return ""
        
        # Create DataFrame
        data = []
        for comparison in self.rankings:
            metrics = comparison.performance_metrics
            data.append({
                'strategy_name': comparison.strategy_name,
                'rank': comparison.rank,
                'ranking_score': comparison.ranking_score,
                'total_return': metrics.total_return,
                'annualized_return': metrics.annualized_return,
                'sharpe_ratio': metrics.sharpe_ratio,
                'sortino_ratio': metrics.sortino_ratio,
                'calmar_ratio': metrics.calmar_ratio,
                'max_drawdown': metrics.max_drawdown,
                'volatility': metrics.volatility,
                'win_rate': metrics.win_rate,
                'profit_factor': metrics.profit_factor,
                'total_trades': metrics.total_trades,
                'winning_trades': metrics.winning_trades,
                'losing_trades': metrics.losing_trades,
                'avg_win': metrics.avg_win,
                'avg_loss': metrics.avg_loss
            })
        
        df = pd.DataFrame(data)
        
        if format == 'csv':
            return df.to_csv(index=False)
        elif format == 'json':
            return df.to_json(orient='records', indent=2)
        else:
            return df.to_string(index=False)
