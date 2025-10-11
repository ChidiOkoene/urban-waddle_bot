"""
Backtesting engine for strategy evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from decimal import Decimal
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from ..core.data_models import OHLCV, OrderSide, OrderType, OrderStatus, Position, Trade
from ..strategies.base_strategy import BaseStrategy
from ..risk.position_sizer import PositionSizer, PositionSizingMethod
from ..risk.stop_loss_manager import StopLossManager


class BacktestMode(str, Enum):
    """Backtesting modes."""
    PAPER = "paper"
    SIMULATION = "simulation"
    HISTORICAL = "historical"


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    initial_capital: Decimal
    commission: Decimal
    slippage: Decimal
    start_date: datetime
    end_date: datetime
    mode: BacktestMode = BacktestMode.SIMULATION
    position_sizing_method: PositionSizingMethod = PositionSizingMethod.FIXED_FRACTIONAL
    risk_per_trade: float = 0.02
    max_positions: int = 5
    enable_stop_loss: bool = True
    enable_take_profit: bool = True
    enable_trailing_stop: bool = True


@dataclass
class BacktestResult:
    """Backtesting result."""
    total_return: Decimal
    annualized_return: Decimal
    sharpe_ratio: float
    max_drawdown: Decimal
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: Decimal
    avg_loss: Decimal
    equity_curve: List[Decimal]
    trades: List[Trade]
    positions: List[Position]
    start_date: datetime
    end_date: datetime
    initial_capital: Decimal
    final_capital: Decimal


class BacktestEngine:
    """Backtesting engine for strategy evaluation."""
    
    def __init__(self, config: BacktestConfig):
        """
        Initialize backtesting engine.
        
        Args:
            config: Backtesting configuration
        """
        self.config = config
        
        # Initialize components
        self.position_sizer = PositionSizer(config.position_sizing_method)
        self.position_sizer.set_risk_per_trade(config.risk_per_trade)
        
        self.stop_loss_manager = StopLossManager({
            'stop_loss_type': 'percentage',
            'stop_loss_percentage': 0.02,
            'trailing_stop_activation': 0.015,
            'trailing_stop_distance': 0.01
        })
        
        # Backtesting state
        self.current_capital = config.initial_capital
        self.initial_capital = config.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = [config.initial_capital]
        self.daily_returns = []
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = Decimal("0")
        self.max_equity = config.initial_capital
        self.max_drawdown = Decimal("0")
        
        # Current state
        self.current_date = None
        self.current_prices = {}
        self.current_indicators = {}
        self.current_patterns = {}
    
    def run_backtest(self, strategy: BaseStrategy, ohlcv_data: List[OHLCV]) -> BacktestResult:
        """
        Run backtest for a strategy.
        
        Args:
            strategy: Trading strategy to backtest
            ohlcv_data: Historical OHLCV data
            
        Returns:
            BacktestResult
        """
        # Filter data by date range
        filtered_data = self._filter_data_by_date(ohlcv_data)
        
        if len(filtered_data) < 100:
            raise ValueError("Insufficient data for backtesting")
        
        # Reset state
        self._reset_state()
        
        # Run backtest
        for i in range(len(filtered_data)):
            self.current_date = filtered_data[i].timestamp
            current_candle = filtered_data[i]
            
            # Update current prices
            self.current_prices[current_candle.symbol] = current_candle.close
            
            # Get historical data up to current point
            historical_data = filtered_data[:i+1]
            
            # Calculate indicators and patterns
            self.current_indicators = strategy.calculate_indicators(historical_data)
            self.current_patterns = strategy.detect_patterns(historical_data)
            
            # Generate signal
            signal = strategy.analyze(historical_data)
            
            # Process signal
            if signal:
                self._process_signal(signal, current_candle)
            
            # Update existing positions
            self._update_positions(current_candle)
            
            # Update equity curve
            self._update_equity_curve()
        
        # Close all remaining positions
        self._close_all_positions(filtered_data[-1])
        
        # Calculate final results
        return self._calculate_results(strategy)
    
    def _filter_data_by_date(self, ohlcv_data: List[OHLCV]) -> List[OHLCV]:
        """Filter data by date range."""
        filtered = []
        for candle in ohlcv_data:
            if self.config.start_date <= candle.timestamp <= self.config.end_date:
                filtered.append(candle)
        return filtered
    
    def _reset_state(self):
        """Reset backtesting state."""
        self.current_capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.daily_returns = []
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = Decimal("0")
        self.max_equity = self.initial_capital
        self.max_drawdown = Decimal("0")
    
    def _process_signal(self, signal, current_candle: OHLCV):
        """Process trading signal."""
        symbol = signal.symbol
        
        # Check if we can open a new position
        if len(self.positions) >= self.config.max_positions:
            return
        
        # Check if position already exists for this symbol
        if symbol in self.positions:
            return
        
        # Calculate position size
        position_size = self.position_sizer.calculate_position_size(
            account_balance=self.current_capital,
            current_price=signal.price,
            stop_loss_price=self._calculate_stop_loss(signal),
            signal_strength=signal.strength
        )
        
        if position_size <= 0:
            return
        
        # Calculate position value
        position_value = position_size * signal.price
        
        # Check if we have enough capital
        if position_value > self.current_capital:
            return
        
        # Create position
        position = Position(
            id=f"{symbol}_{self.current_date.isoformat()}",
            symbol=symbol,
            side=signal.signal,
            amount=position_size,
            entry_price=signal.price,
            current_price=signal.price,
            unrealized_pnl=Decimal("0"),
            entry_time=self.current_date,
            strategy=signal.strategy
        )
        
        # Set stop-loss and take-profit
        if self.config.enable_stop_loss:
            stop_loss_price = self._calculate_stop_loss(signal)
            position.stop_loss = stop_loss_price
        
        if self.config.enable_take_profit:
            take_profit_price = self._calculate_take_profit(signal)
            position.take_profit = take_profit_price
        
        # Add position
        self.positions[symbol] = position
        
        # Update capital
        self.current_capital -= position_value
        
        # Record trade
        trade = Trade(
            id=f"trade_{len(self.trades)}",
            symbol=symbol,
            side=signal.signal,
            amount=position_size,
            price=signal.price,
            timestamp=self.current_date,
            strategy=signal.strategy,
            pnl=Decimal("0"),
            commission=self._calculate_commission(position_value),
            status=OrderStatus.FILLED
        )
        
        self.trades.append(trade)
    
    def _update_positions(self, current_candle: OHLCV):
        """Update existing positions."""
        symbol = current_candle.symbol
        
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        # Update position
        position.current_price = current_candle.close
        position.unrealized_pnl = self._calculate_unrealized_pnl(position)
        
        # Check stop-loss
        if self.config.enable_stop_loss and position.stop_loss:
            if self._check_stop_loss(position):
                self._close_position(position, current_candle, "stop_loss")
                return
        
        # Check take-profit
        if self.config.enable_take_profit and position.take_profit:
            if self._check_take_profit(position):
                self._close_position(position, current_candle, "take_profit")
                return
        
        # Update trailing stop
        if self.config.enable_trailing_stop:
            self.stop_loss_manager.update_stop_loss(position, current_candle.close, self.current_indicators)
    
    def _close_position(self, position: Position, current_candle: OHLCV, reason: str):
        """Close a position."""
        symbol = position.symbol
        
        # Calculate final PnL
        final_pnl = self._calculate_final_pnl(position)
        
        # Calculate commission
        position_value = position.amount * current_candle.close
        commission = self._calculate_commission(position_value)
        
        # Net PnL after commission
        net_pnl = final_pnl - commission
        
        # Update capital
        self.current_capital += position_value + net_pnl
        
        # Update statistics
        self.total_trades += 1
        self.total_pnl += net_pnl
        
        if net_pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Record trade
        trade = Trade(
            id=f"trade_{len(self.trades)}",
            symbol=symbol,
            side=OrderSide.SELL if position.side == OrderSide.LONG else OrderSide.BUY,
            amount=position.amount,
            price=current_candle.close,
            timestamp=self.current_date,
            strategy=position.strategy,
            pnl=net_pnl,
            commission=commission,
            status=OrderStatus.FILLED
        )
        
        self.trades.append(trade)
        
        # Remove position
        del self.positions[symbol]
    
    def _close_all_positions(self, final_candle: OHLCV):
        """Close all remaining positions."""
        for symbol in list(self.positions.keys()):
            position = self.positions[symbol]
            self._close_position(position, final_candle, "end_of_backtest")
    
    def _calculate_stop_loss(self, signal) -> Decimal:
        """Calculate stop-loss price."""
        if signal.signal == OrderSide.LONG:
            return signal.price * (1 - Decimal(str(self.config.risk_per_trade)))
        else:
            return signal.price * (1 + Decimal(str(self.config.risk_per_trade)))
    
    def _calculate_take_profit(self, signal) -> Decimal:
        """Calculate take-profit price."""
        risk_reward_ratio = 2.0  # 2:1 risk-reward
        
        if signal.signal == OrderSide.LONG:
            stop_loss = self._calculate_stop_loss(signal)
            risk = signal.price - stop_loss
            return signal.price + (risk * Decimal(str(risk_reward_ratio)))
        else:
            stop_loss = self._calculate_stop_loss(signal)
            risk = stop_loss - signal.price
            return signal.price - (risk * Decimal(str(risk_reward_ratio)))
    
    def _calculate_unrealized_pnl(self, position: Position) -> Decimal:
        """Calculate unrealized PnL."""
        if position.side == OrderSide.LONG:
            return position.amount * (position.current_price - position.entry_price)
        else:
            return position.amount * (position.entry_price - position.current_price)
    
    def _calculate_final_pnl(self, position: Position) -> Decimal:
        """Calculate final PnL for closing position."""
        return self._calculate_unrealized_pnl(position)
    
    def _calculate_commission(self, position_value: Decimal) -> Decimal:
        """Calculate commission."""
        return position_value * self.config.commission
    
    def _check_stop_loss(self, position: Position) -> bool:
        """Check if stop-loss should be triggered."""
        if position.side == OrderSide.LONG:
            return position.current_price <= position.stop_loss
        else:
            return position.current_price >= position.stop_loss
    
    def _check_take_profit(self, position: Position) -> bool:
        """Check if take-profit should be triggered."""
        if position.side == OrderSide.LONG:
            return position.current_price >= position.take_profit
        else:
            return position.current_price <= position.take_profit
    
    def _update_equity_curve(self):
        """Update equity curve."""
        # Calculate current equity
        current_equity = self.current_capital
        
        # Add unrealized PnL from open positions
        for position in self.positions.values():
            current_equity += self._calculate_unrealized_pnl(position)
        
        self.equity_curve.append(current_equity)
        
        # Update maximum equity and drawdown
        if current_equity > self.max_equity:
            self.max_equity = current_equity
        
        current_drawdown = (self.max_equity - current_equity) / self.max_equity
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
        
        # Calculate daily return
        if len(self.equity_curve) > 1:
            daily_return = (current_equity - self.equity_curve[-2]) / self.equity_curve[-2]
            self.daily_returns.append(float(daily_return))
    
    def _calculate_results(self, strategy: BaseStrategy) -> BacktestResult:
        """Calculate backtesting results."""
        # Calculate returns
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital
        
        # Calculate annualized return
        days = (self.config.end_date - self.config.start_date).days
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else Decimal("0")
        
        # Calculate Sharpe ratio
        if self.daily_returns:
            sharpe_ratio = self._calculate_sharpe_ratio()
        else:
            sharpe_ratio = 0.0
        
        # Calculate win rate
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0
        
        # Calculate profit factor
        profit_factor = self._calculate_profit_factor()
        
        # Calculate average win/loss
        avg_win, avg_loss = self._calculate_avg_win_loss()
        
        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=self.max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=self.total_trades,
            winning_trades=self.winning_trades,
            losing_trades=self.losing_trades,
            avg_win=avg_win,
            avg_loss=avg_loss,
            equity_curve=self.equity_curve,
            trades=self.trades,
            positions=list(self.positions.values()),
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            initial_capital=self.initial_capital,
            final_capital=self.current_capital
        )
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio."""
        if not self.daily_returns or len(self.daily_returns) < 2:
            return 0.0
        
        returns = np.array(self.daily_returns)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Assume risk-free rate of 0
        return mean_return / std_return
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor."""
        if not self.trades:
            return 0.0
        
        total_profit = sum(float(trade.pnl) for trade in self.trades if trade.pnl > 0)
        total_loss = abs(sum(float(trade.pnl) for trade in self.trades if trade.pnl < 0))
        
        if total_loss == 0:
            return float('inf') if total_profit > 0 else 0.0
        
        return total_profit / total_loss
    
    def _calculate_avg_win_loss(self) -> Tuple[Decimal, Decimal]:
        """Calculate average win and loss."""
        if not self.trades:
            return Decimal("0"), Decimal("0")
        
        wins = [trade.pnl for trade in self.trades if trade.pnl > 0]
        losses = [trade.pnl for trade in self.trades if trade.pnl < 0]
        
        avg_win = sum(wins) / len(wins) if wins else Decimal("0")
        avg_loss = sum(losses) / len(losses) if losses else Decimal("0")
        
        return avg_win, avg_loss
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current backtesting state."""
        return {
            'current_capital': float(self.current_capital),
            'initial_capital': float(self.initial_capital),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'total_pnl': float(self.total_pnl),
            'max_drawdown': float(self.max_drawdown),
            'open_positions': len(self.positions),
            'current_date': self.current_date.isoformat() if self.current_date else None,
            'equity_curve_length': len(self.equity_curve)
        }
