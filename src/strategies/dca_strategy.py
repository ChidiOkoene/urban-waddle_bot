"""
Dollar Cost Averaging (DCA) Strategy implementation.
"""

import pandas as pd
from typing import Dict, List, Optional, Any
from decimal import Decimal
from datetime import datetime, timedelta

from ..core.data_models import OHLCV, StrategySignal, OrderSide, TimeFrame
from .base_strategy import BaseStrategy


class DCAStrategy(BaseStrategy):
    """Dollar Cost Averaging strategy with smart entry timing."""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize DCA strategy.
        
        Args:
            parameters: Strategy parameters
        """
        default_params = {
            'dca_interval': 24,  # hours
            'dca_amount': 100,  # USD
            'dca_max_orders': 5,
            'dca_increase_factor': 1.5,  # Increase amount by 50% on dips
            'dip_threshold': 0.05,  # 5% dip to trigger increased DCA
            'rsi_oversold': 30,
            'rsi_period': 14,
            'min_signal_strength': 0.6,
            'stop_loss_type': 'percentage',
            'stop_loss_percentage': 0.15,  # 15% stop loss for DCA
            'take_profit_ratio': 3.0,
            'smart_entry': True,
            'volume_confirmation': True
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__("DCA", default_params)
        
        # DCA state
        self.last_dca_time = None
        self.dca_count = 0
        self.total_dca_amount = Decimal("0")
        self.average_entry_price = Decimal("0")
        self.dca_history = []
    
    def generate_signal(self, ohlcv_data: List[OHLCV], indicators: Dict[str, List[float]], 
                       patterns: Dict[str, Any]) -> Optional[StrategySignal]:
        """Generate DCA trading signal."""
        if len(ohlcv_data) < self.parameters['rsi_period']:
            return None
        
        current_price = ohlcv_data[-1].close
        current_time = ohlcv_data[-1].timestamp
        
        # Check if it's time for DCA
        if not self._should_execute_dca(current_time):
            return None
        
        # Get RSI for smart entry
        rsi_values = indicators.get('rsi', [])
        if not rsi_values:
            return None
        
        current_rsi = rsi_values[-1]
        
        # Check for NaN values
        if pd.isna(current_rsi):
            return None
        
        signal_strength = 0.0
        signal_type = OrderSide.BUY  # DCA is always buying
        dca_amount = Decimal(str(self.parameters['dca_amount']))
        
        # Check if we've reached max DCA orders
        if self.dca_count >= self.parameters['dca_max_orders']:
            return None
        
        # Base signal strength
        signal_strength = 0.5
        
        # Smart entry logic
        if self.parameters['smart_entry']:
            # Increase strength if RSI is oversold
            if current_rsi < self.parameters['rsi_oversold']:
                signal_strength += 0.2
                # Increase DCA amount on dips
                if current_rsi < 20:
                    dca_amount *= Decimal(str(self.parameters['dca_increase_factor']))
                    signal_strength += 0.1
            
            # Check for dip from recent high
            if self._check_dip_condition(ohlcv_data):
                dca_amount *= Decimal(str(self.parameters['dca_increase_factor']))
                signal_strength += 0.1
            
            # Check for volume confirmation
            if self.parameters['volume_confirmation'] and self._check_volume_confirmation(ohlcv_data):
                signal_strength += 0.1
            
            # Check for trend confirmation
            if self._check_trend_confirmation(indicators):
                signal_strength += 0.1
        
        # Only generate signal if strength meets minimum threshold
        if signal_strength >= self.parameters['min_signal_strength']:
            # Update DCA state
            self.last_dca_time = current_time
            self.dca_count += 1
            self.total_dca_amount += dca_amount
            
            # Update average entry price
            if self.average_entry_price == 0:
                self.average_entry_price = current_price
            else:
                # Calculate weighted average
                total_cost = self.average_entry_price * (self.dca_count - 1) + current_price
                self.average_entry_price = total_cost / self.dca_count
            
            # Record DCA history
            self.dca_history.append({
                'timestamp': current_time,
                'price': float(current_price),
                'amount': float(dca_amount),
                'rsi': current_rsi,
                'strength': signal_strength
            })
            
            return StrategySignal(
                symbol=ohlcv_data[-1].symbol,
                signal=signal_type,
                strength=min(signal_strength, 1.0),
                price=current_price,
                strategy=self.name,
                timeframe=ohlcv_data[-1].timeframe,
                indicators={
                    'rsi': current_rsi,
                    'dca_count': self.dca_count,
                    'dca_amount': float(dca_amount),
                    'average_entry_price': float(self.average_entry_price),
                    'total_dca_amount': float(self.total_dca_amount)
                },
                metadata={
                    'dca_interval': self.parameters['dca_interval'],
                    'dca_max_orders': self.parameters['dca_max_orders'],
                    'dca_increase_factor': self.parameters['dca_increase_factor'],
                    'dca_count': self.dca_count,
                    'dca_amount': float(dca_amount)
                }
            )
        
        return None
    
    def _should_execute_dca(self, current_time: datetime) -> bool:
        """Check if it's time to execute DCA."""
        if not self.last_dca_time:
            return True  # First DCA
        
        # Check if enough time has passed
        time_diff = current_time - self.last_dca_time
        interval_hours = self.parameters['dca_interval']
        
        return time_diff >= timedelta(hours=interval_hours)
    
    def _check_dip_condition(self, ohlcv_data: List[OHLCV]) -> bool:
        """Check if price has dipped significantly."""
        if len(ohlcv_data) < 20:
            return False
        
        # Get recent high (last 20 candles)
        recent_highs = [float(candle.high) for candle in ohlcv_data[-20:]]
        recent_high = max(recent_highs)
        current_price = float(ohlcv_data[-1].close)
        
        # Check if current price is below recent high by dip threshold
        dip_percentage = (recent_high - current_price) / recent_high
        return dip_percentage >= self.parameters['dip_threshold']
    
    def _check_volume_confirmation(self, ohlcv_data: List[OHLCV]) -> bool:
        """Check for volume confirmation."""
        if len(ohlcv_data) < 5:
            return True  # No volume filter if not enough data
        
        # Calculate average volume
        recent_volumes = [float(candle.volume) for candle in ohlcv_data[-5:]]
        avg_volume = sum(recent_volumes) / len(recent_volumes)
        current_volume = float(ohlcv_data[-1].volume)
        
        # Volume should be reasonable (not too low)
        return current_volume > avg_volume * 0.8
    
    def _check_trend_confirmation(self, indicators: Dict[str, List[float]]) -> bool:
        """Check for trend confirmation using moving averages."""
        # Get EMA values
        ema_20 = indicators.get('sma_20', [])
        ema_50 = indicators.get('sma_50', [])
        
        if not ema_20 or not ema_50:
            return True  # No trend filter if indicators not available
        
        current_ema_20 = ema_20[-1]
        current_ema_50 = ema_50[-1]
        
        # For DCA, prefer when price is not in a strong downtrend
        return current_ema_20 > current_ema_50 * Decimal("0.95")  # Within 5% of EMA 50
    
    def get_dca_status(self) -> Dict[str, Any]:
        """Get current DCA status."""
        return {
            'dca_count': self.dca_count,
            'max_orders': self.parameters['dca_max_orders'],
            'total_dca_amount': float(self.total_dca_amount),
            'average_entry_price': float(self.average_entry_price),
            'last_dca_time': self.last_dca_time,
            'next_dca_time': self.last_dca_time + timedelta(hours=self.parameters['dca_interval']) if self.last_dca_time else None,
            'dca_history': self.dca_history[-10:] if self.dca_history else []  # Last 10 DCA entries
        }
    
    def calculate_unrealized_pnl(self, current_price: Decimal) -> Decimal:
        """Calculate unrealized PnL for DCA positions."""
        if self.average_entry_price == 0:
            return Decimal("0")
        
        total_quantity = self.total_dca_amount / self.average_entry_price
        current_value = total_quantity * current_price
        return current_value - self.total_dca_amount
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        return self.parameters.copy()
    
    def set_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Set strategy parameters."""
        if self.validate_parameters(parameters):
            self.parameters.update(parameters)
            return True
        return False
    
    def get_required_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get required parameters for the strategy."""
        return {
            'dca_interval': {'type': int, 'min': 1, 'max': 168, 'default': 24},  # 1 hour to 1 week
            'dca_amount': {'type': float, 'min': 10, 'max': 10000, 'default': 100},
            'dca_max_orders': {'type': int, 'min': 1, 'max': 20, 'default': 5},
            'dca_increase_factor': {'type': float, 'min': 1.0, 'max': 3.0, 'default': 1.5},
            'dip_threshold': {'type': float, 'min': 0.01, 'max': 0.2, 'default': 0.05},
            'rsi_oversold': {'type': int, 'min': 10, 'max': 40, 'default': 30},
            'rsi_period': {'type': int, 'min': 5, 'max': 30, 'default': 14},
            'min_signal_strength': {'type': float, 'min': 0.1, 'max': 1.0, 'default': 0.6},
            'stop_loss_type': {'type': str, 'options': ['percentage', 'atr'], 'default': 'percentage'},
            'stop_loss_percentage': {'type': float, 'min': 0.05, 'max': 0.5, 'default': 0.15},
            'take_profit_ratio': {'type': float, 'min': 1.0, 'max': 10.0, 'default': 3.0},
            'smart_entry': {'type': bool, 'default': True},
            'volume_confirmation': {'type': bool, 'default': True}
        }
    
    def get_description(self) -> str:
        """Get strategy description."""
        return ("Dollar Cost Averaging Strategy: Accumulates positions at regular intervals with smart entry timing. "
                "Increases DCA amount on dips and uses RSI for optimal entry timing. "
                "Suitable for long-term accumulation strategies.")
    
    def reset_dca(self):
        """Reset DCA state."""
        self.last_dca_time = None
        self.dca_count = 0
        self.total_dca_amount = Decimal("0")
        self.average_entry_price = Decimal("0")
        self.dca_history = []
