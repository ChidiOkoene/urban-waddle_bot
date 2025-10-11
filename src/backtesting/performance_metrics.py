"""
Performance metrics for backtesting results.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
from datetime import datetime, timedelta
from dataclasses import dataclass

from .backtest_engine import BacktestResult


@dataclass
class PerformanceMetrics:
    """Performance metrics container."""
    # Return metrics
    total_return: float
    annualized_return: float
    monthly_return: float
    daily_return: float
    
    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    volatility: float
    
    # Trade metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    
    # Advanced metrics
    var_95: float
    cvar_95: float
    skewness: float
    kurtosis: float
    information_ratio: float
    treynor_ratio: float
    jensen_alpha: float
    
    # Time-based metrics
    start_date: datetime
    end_date: datetime
    trading_days: int
    avg_trades_per_day: float
    avg_trades_per_month: float


class PerformanceCalculator:
    """Calculate comprehensive performance metrics."""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize performance calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_metrics(self, backtest_result: BacktestResult) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            backtest_result: Backtesting result
            
        Returns:
            PerformanceMetrics
        """
        # Extract data
        equity_curve = [float(eq) for eq in backtest_result.equity_curve]
        trades = backtest_result.trades
        
        # Calculate returns
        returns = self._calculate_returns(equity_curve)
        
        # Calculate time-based metrics
        trading_days = (backtest_result.end_date - backtest_result.start_date).days
        trading_months = trading_days / 30.44  # Average days per month
        
        # Return metrics
        total_return = float(backtest_result.total_return)
        annualized_return = float(backtest_result.annualized_return)
        monthly_return = (1 + total_return) ** (1 / trading_months) - 1 if trading_months > 0 else 0
        daily_return = (1 + total_return) ** (1 / trading_days) - 1 if trading_days > 0 else 0
        
        # Risk metrics
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        calmar_ratio = self._calculate_calmar_ratio(annualized_return, backtest_result.max_drawdown)
        max_drawdown = float(backtest_result.max_drawdown)
        max_drawdown_duration = self._calculate_max_drawdown_duration(equity_curve)
        volatility = self._calculate_volatility(returns)
        
        # Trade metrics
        total_trades = backtest_result.total_trades
        winning_trades = backtest_result.winning_trades
        losing_trades = backtest_result.losing_trades
        win_rate = backtest_result.win_rate
        profit_factor = backtest_result.profit_factor
        avg_win = float(backtest_result.avg_win)
        avg_loss = float(backtest_result.avg_loss)
        largest_win, largest_loss = self._calculate_largest_trades(trades)
        
        # Advanced metrics
        var_95 = self._calculate_var(returns, 0.95)
        cvar_95 = self._calculate_cvar(returns, 0.95)
        skewness = self._calculate_skewness(returns)
        kurtosis = self._calculate_kurtosis(returns)
        information_ratio = self._calculate_information_ratio(returns)
        treynor_ratio = self._calculate_treynor_ratio(annualized_return, returns)
        jensen_alpha = self._calculate_jensen_alpha(annualized_return, returns)
        
        # Time-based metrics
        avg_trades_per_day = total_trades / trading_days if trading_days > 0 else 0
        avg_trades_per_month = total_trades / trading_months if trading_months > 0 else 0
        
        return PerformanceMetrics(
            # Return metrics
            total_return=total_return,
            annualized_return=annualized_return,
            monthly_return=monthly_return,
            daily_return=daily_return,
            
            # Risk metrics
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            volatility=volatility,
            
            # Trade metrics
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            
            # Advanced metrics
            var_95=var_95,
            cvar_95=cvar_95,
            skewness=skewness,
            kurtosis=kurtosis,
            information_ratio=information_ratio,
            treynor_ratio=treynor_ratio,
            jensen_alpha=jensen_alpha,
            
            # Time-based metrics
            start_date=backtest_result.start_date,
            end_date=backtest_result.end_date,
            trading_days=trading_days,
            avg_trades_per_day=avg_trades_per_day,
            avg_trades_per_month=avg_trades_per_month
        )
    
    def _calculate_returns(self, equity_curve: List[float]) -> np.ndarray:
        """Calculate returns from equity curve."""
        if len(equity_curve) < 2:
            return np.array([])
        
        returns = []
        for i in range(1, len(equity_curve)):
            if equity_curve[i-1] != 0:
                ret = (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
                returns.append(ret)
        
        return np.array(returns)
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualize
        annualized_return = mean_return * 252  # 252 trading days
        annualized_std = std_return * np.sqrt(252)
        
        return (annualized_return - self.risk_free_rate) / annualized_std
    
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio."""
        if len(returns) == 0:
            return 0.0
        
        mean_return = np.mean(returns)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf') if mean_return > 0 else 0.0
        
        downside_std = np.std(downside_returns)
        
        if downside_std == 0:
            return 0.0
        
        # Annualize
        annualized_return = mean_return * 252
        annualized_downside_std = downside_std * np.sqrt(252)
        
        return (annualized_return - self.risk_free_rate) / annualized_downside_std
    
    def _calculate_calmar_ratio(self, annualized_return: float, max_drawdown: float) -> float:
        """Calculate Calmar ratio."""
        if max_drawdown == 0:
            return float('inf') if annualized_return > 0 else 0.0
        
        return annualized_return / max_drawdown
    
    def _calculate_max_drawdown_duration(self, equity_curve: List[float]) -> int:
        """Calculate maximum drawdown duration in days."""
        if len(equity_curve) < 2:
            return 0
        
        max_equity = equity_curve[0]
        max_drawdown_duration = 0
        current_drawdown_duration = 0
        
        for equity in equity_curve:
            if equity > max_equity:
                max_equity = equity
                current_drawdown_duration = 0
            else:
                current_drawdown_duration += 1
                max_drawdown_duration = max(max_drawdown_duration, current_drawdown_duration)
        
        return max_drawdown_duration
    
    def _calculate_volatility(self, returns: np.ndarray) -> float:
        """Calculate annualized volatility."""
        if len(returns) == 0:
            return 0.0
        
        return np.std(returns) * np.sqrt(252)
    
    def _calculate_largest_trades(self, trades) -> Tuple[float, float]:
        """Calculate largest win and loss."""
        if not trades:
            return 0.0, 0.0
        
        pnls = [float(trade.pnl) for trade in trades]
        
        largest_win = max(pnls) if pnls else 0.0
        largest_loss = min(pnls) if pnls else 0.0
        
        return largest_win, largest_loss
    
    def _calculate_var(self, returns: np.ndarray, confidence_level: float) -> float:
        """Calculate Value at Risk."""
        if len(returns) == 0:
            return 0.0
        
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    def _calculate_cvar(self, returns: np.ndarray, confidence_level: float) -> float:
        """Calculate Conditional Value at Risk."""
        if len(returns) == 0:
            return 0.0
        
        var = self._calculate_var(returns, confidence_level)
        tail_returns = returns[returns <= var]
        
        if len(tail_returns) == 0:
            return var
        
        return np.mean(tail_returns)
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness."""
        if len(returns) < 3:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        skewness = np.mean(((returns - mean_return) / std_return) ** 3)
        return skewness
    
    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate kurtosis."""
        if len(returns) < 4:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        kurtosis = np.mean(((returns - mean_return) / std_return) ** 4) - 3
        return kurtosis
    
    def _calculate_information_ratio(self, returns: np.ndarray) -> float:
        """Calculate information ratio."""
        if len(returns) == 0:
            return 0.0
        
        # Assume benchmark return of 0 (risk-free rate)
        excess_returns = returns - self.risk_free_rate / 252
        
        mean_excess_return = np.mean(excess_returns)
        tracking_error = np.std(excess_returns)
        
        if tracking_error == 0:
            return 0.0
        
        return mean_excess_return / tracking_error * np.sqrt(252)
    
    def _calculate_treynor_ratio(self, annualized_return: float, returns: np.ndarray) -> float:
        """Calculate Treynor ratio."""
        if len(returns) == 0:
            return 0.0
        
        # Assume beta of 1 (market beta)
        beta = 1.0
        
        if beta == 0:
            return 0.0
        
        return (annualized_return - self.risk_free_rate) / beta
    
    def _calculate_jensen_alpha(self, annualized_return: float, returns: np.ndarray) -> float:
        """Calculate Jensen's alpha."""
        if len(returns) == 0:
            return 0.0
        
        # Assume market return equals risk-free rate
        market_return = self.risk_free_rate
        beta = 1.0
        
        expected_return = self.risk_free_rate + beta * (market_return - self.risk_free_rate)
        alpha = annualized_return - expected_return
        
        return alpha
    
    def calculate_rolling_metrics(self, equity_curve: List[float], window: int = 252) -> Dict[str, List[float]]:
        """Calculate rolling performance metrics."""
        if len(equity_curve) < window:
            return {}
        
        returns = self._calculate_returns(equity_curve)
        
        rolling_metrics = {
            'rolling_sharpe': [],
            'rolling_volatility': [],
            'rolling_max_drawdown': [],
            'rolling_return': []
        }
        
        for i in range(window, len(returns)):
            window_returns = returns[i-window:i]
            
            # Rolling Sharpe ratio
            mean_return = np.mean(window_returns)
            std_return = np.std(window_returns)
            sharpe = (mean_return * 252 - self.risk_free_rate) / (std_return * np.sqrt(252)) if std_return > 0 else 0
            rolling_metrics['rolling_sharpe'].append(sharpe)
            
            # Rolling volatility
            volatility = std_return * np.sqrt(252)
            rolling_metrics['rolling_volatility'].append(volatility)
            
            # Rolling max drawdown
            window_equity = equity_curve[i-window:i+1]
            max_equity = max(window_equity)
            min_equity = min(window_equity)
            max_dd = (max_equity - min_equity) / max_equity if max_equity > 0 else 0
            rolling_metrics['rolling_max_drawdown'].append(max_dd)
            
            # Rolling return
            total_return = (window_equity[-1] - window_equity[0]) / window_equity[0] if window_equity[0] > 0 else 0
            rolling_metrics['rolling_return'].append(total_return)
        
        return rolling_metrics
    
    def calculate_monthly_returns(self, equity_curve: List[float], dates: List[datetime]) -> Dict[str, float]:
        """Calculate monthly returns."""
        if len(equity_curve) != len(dates):
            return {}
        
        # Group by month
        monthly_data = {}
        for i, date in enumerate(dates):
            month_key = f"{date.year}-{date.month:02d}"
            if month_key not in monthly_data:
                monthly_data[month_key] = []
            monthly_data[month_key].append(equity_curve[i])
        
        monthly_returns = {}
        for month, values in monthly_data.items():
            if len(values) > 1:
                monthly_return = (values[-1] - values[0]) / values[0] if values[0] > 0 else 0
                monthly_returns[month] = monthly_return
        
        return monthly_returns
    
    def calculate_underwater_curve(self, equity_curve: List[float]) -> List[float]:
        """Calculate underwater curve (drawdown over time)."""
        if not equity_curve:
            return []
        
        underwater = []
        max_equity = equity_curve[0]
        
        for equity in equity_curve:
            if equity > max_equity:
                max_equity = equity
                underwater.append(0.0)
            else:
                drawdown = (max_equity - equity) / max_equity if max_equity > 0 else 0
                underwater.append(-drawdown)
        
        return underwater
    
    def get_performance_summary(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            'returns': {
                'total_return': f"{metrics.total_return:.2%}",
                'annualized_return': f"{metrics.annualized_return:.2%}",
                'monthly_return': f"{metrics.monthly_return:.2%}",
                'daily_return': f"{metrics.daily_return:.2%}"
            },
            'risk': {
                'sharpe_ratio': f"{metrics.sharpe_ratio:.2f}",
                'sortino_ratio': f"{metrics.sortino_ratio:.2f}",
                'calmar_ratio': f"{metrics.calmar_ratio:.2f}",
                'max_drawdown': f"{metrics.max_drawdown:.2%}",
                'volatility': f"{metrics.volatility:.2%}"
            },
            'trades': {
                'total_trades': metrics.total_trades,
                'win_rate': f"{metrics.win_rate:.2%}",
                'profit_factor': f"{metrics.profit_factor:.2f}",
                'avg_win': f"{metrics.avg_win:.2f}",
                'avg_loss': f"{metrics.avg_loss:.2f}"
            },
            'advanced': {
                'var_95': f"{metrics.var_95:.2%}",
                'cvar_95': f"{metrics.cvar_95:.2%}",
                'skewness': f"{metrics.skewness:.2f}",
                'kurtosis': f"{metrics.kurtosis:.2f}"
            }
        }
