"""
Grid Bot Strategy implementation.
"""

import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
from datetime import datetime

from ..core.data_models import OHLCV, StrategySignal, OrderSide, TimeFrame, GridLevel, OrderStatus
from .base_strategy import BaseStrategy


class GridBotStrategy(BaseStrategy):
    """Dynamic grid trading strategy with profit taking."""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize Grid Bot strategy.
        
        Args:
            parameters: Strategy parameters
        """
        default_params = {
            'grid_levels': 10,
            'grid_spacing': 0.01,  # 1%
            'grid_profit_target': 0.005,  # 0.5%
            'base_price': None,  # Will be set dynamically
            'max_grid_levels': 20,
            'min_grid_levels': 5,
            'trailing_grid': True,
            'grid_rebalance_threshold': 0.05,  # 5%
            'volume_per_level': 0.1,  # 10% of position per level
            'stop_loss_type': 'percentage',
            'stop_loss_percentage': 0.1,  # 10% stop loss for grid
            'take_profit_ratio': 1.0,
            'min_signal_strength': 0.5
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__("Grid_Bot", default_params)
        
        # Grid state
        self.grid_levels: List[GridLevel] = []
        self.base_price = None
        self.last_rebalance_time = None
        self.total_grid_profit = Decimal("0")
        self.active_orders = {}
    
    def generate_signal(self, ohlcv_data: List[OHLCV], indicators: Dict[str, List[float]], 
                       patterns: Dict[str, Any]) -> Optional[StrategySignal]:
        """Generate grid trading signal."""
        if len(ohlcv_data) < 2:
            return None
        
        current_price = ohlcv_data[-1].close
        
        # Initialize grid if not set
        if not self.grid_levels:
            self._initialize_grid(current_price)
            return None
        
        # Check for grid rebalancing
        if self._should_rebalance_grid(current_price):
            self._rebalance_grid(current_price)
            return None
        
        # Check for grid level triggers
        signal = self._check_grid_levels(current_price, ohlcv_data)
        
        if signal:
            self.last_signal = signal
            self.last_signal_time = datetime.utcnow()
        
        return signal
    
    def _initialize_grid(self, current_price: Decimal):
        """Initialize grid levels around current price."""
        self.base_price = current_price
        self.grid_levels = []
        
        grid_spacing = Decimal(str(self.parameters['grid_spacing']))
        num_levels = self.parameters['grid_levels']
        
        # Create buy levels below current price
        for i in range(1, num_levels // 2 + 1):
            price = current_price * (1 - grid_spacing * i)
            level = GridLevel(
                level=i,
                price=price,
                side=OrderSide.BUY,
                amount=Decimal(str(self.parameters['volume_per_level'])),
                profit_target=price * (1 + Decimal(str(self.parameters['grid_profit_target'])))
            )
            self.grid_levels.append(level)
        
        # Create sell levels above current price
        for i in range(1, num_levels // 2 + 1):
            price = current_price * (1 + grid_spacing * i)
            level = GridLevel(
                level=i + num_levels // 2,
                price=price,
                side=OrderSide.SELL,
                amount=Decimal(str(self.parameters['volume_per_level'])),
                profit_target=price * (1 - Decimal(str(self.parameters['grid_profit_target'])))
            )
            self.grid_levels.append(level)
        
        # Sort levels by price
        self.grid_levels.sort(key=lambda x: x.price)
    
    def _should_rebalance_grid(self, current_price: Decimal) -> bool:
        """Check if grid should be rebalanced."""
        if not self.base_price:
            return False
        
        # Check price deviation from base price
        price_deviation = abs(current_price - self.base_price) / self.base_price
        
        # Rebalance if price moved significantly
        if price_deviation > self.parameters['grid_rebalance_threshold']:
            return True
        
        # Rebalance if trailing grid is enabled and price moved favorably
        if (self.parameters['trailing_grid'] and 
            current_price > self.base_price * Decimal("1.02")):  # 2% above base
            return True
        
        return False
    
    def _rebalance_grid(self, current_price: Decimal):
        """Rebalance grid around new price."""
        # Update base price
        self.base_price = current_price
        
        # Clear existing levels
        self.grid_levels = []
        
        # Reinitialize grid
        self._initialize_grid(current_price)
        
        # Update rebalance time
        self.last_rebalance_time = datetime.utcnow()
    
    def _check_grid_levels(self, current_price: Decimal, ohlcv_data: List[OHLCV]) -> Optional[StrategySignal]:
        """Check if any grid level is triggered."""
        signal_strength = 0.0
        signal_type = None
        triggered_level = None
        
        # Check each grid level
        for level in self.grid_levels:
            if level.status != OrderStatus.PENDING:
                continue
            
            # Check if price reached the level
            if self._is_price_at_level(current_price, level.price):
                triggered_level = level
                
                if level.side == OrderSide.BUY:
                    signal_type = OrderSide.BUY
                    signal_strength = 0.8
                else:
                    signal_type = OrderSide.SELL
                    signal_strength = 0.8
                
                # Mark level as triggered
                level.status = OrderStatus.OPEN
                
                break
        
        if signal_type and triggered_level:
            # Calculate additional signal strength based on grid position
            grid_position = self._calculate_grid_position(current_price)
            signal_strength += grid_position * 0.2
            
            # Check for volume confirmation
            if self._check_volume_confirmation(ohlcv_data, signal_type):
                signal_strength += 0.1
            
            return StrategySignal(
                symbol=ohlcv_data[-1].symbol,
                signal=signal_type,
                strength=min(signal_strength, 1.0),
                price=current_price,
                strategy=self.name,
                timeframe=ohlcv_data[-1].timeframe,
                indicators={
                    'grid_level': triggered_level.level,
                    'grid_price': float(triggered_level.price),
                    'grid_position': grid_position,
                    'base_price': float(self.base_price),
                    'total_levels': len(self.grid_levels)
                },
                metadata={
                    'grid_levels': self.parameters['grid_levels'],
                    'grid_spacing': self.parameters['grid_spacing'],
                    'grid_profit_target': self.parameters['grid_profit_target'],
                    'triggered_level': triggered_level.level
                }
            )
        
        return None
    
    def _is_price_at_level(self, current_price: Decimal, level_price: Decimal) -> bool:
        """Check if current price is at grid level."""
        price_diff = abs(current_price - level_price) / level_price
        return price_diff <= Decimal("0.005")  # Within 0.5%
    
    def _calculate_grid_position(self, current_price: Decimal) -> float:
        """Calculate position within grid (0-1)."""
        if not self.grid_levels:
            return 0.5
        
        # Find closest levels
        below_levels = [level for level in self.grid_levels if level.price < current_price]
        above_levels = [level for level in self.grid_levels if level.price > current_price]
        
        if not below_levels or not above_levels:
            return 0.5
        
        lowest_above = min(above_levels, key=lambda x: x.price)
        highest_below = max(below_levels, key=lambda x: x.price)
        
        if lowest_above.price == highest_below.price:
            return 0.5
        
        # Calculate position between levels
        position = (current_price - highest_below.price) / (lowest_above.price - highest_below.price)
        return float(position)
    
    def _check_volume_confirmation(self, ohlcv_data: List[OHLCV], signal_type: OrderSide) -> bool:
        """Check for volume confirmation."""
        if len(ohlcv_data) < 5:
            return True  # No volume filter if not enough data
        
        # Calculate average volume
        recent_volumes = [float(candle.volume) for candle in ohlcv_data[-5:]]
        avg_volume = sum(recent_volumes) / len(recent_volumes)
        current_volume = float(ohlcv_data[-1].volume)
        
        # Volume should be reasonable (not too low)
        return current_volume > avg_volume * 0.8
    
    def update_grid_level(self, level: int, order_id: str, status: OrderStatus):
        """Update grid level with order information."""
        for grid_level in self.grid_levels:
            if grid_level.level == level:
                grid_level.order_id = order_id
                grid_level.status = status
                break
    
    def close_grid_level(self, level: int, profit: Decimal):
        """Close grid level and update profit."""
        for grid_level in self.grid_levels:
            if grid_level.level == level:
                grid_level.status = OrderStatus.FILLED
                self.total_grid_profit += profit
                break
    
    def get_grid_status(self) -> Dict[str, Any]:
        """Get current grid status."""
        active_levels = [level for level in self.grid_levels if level.status == OrderStatus.OPEN]
        filled_levels = [level for level in self.grid_levels if level.status == OrderStatus.FILLED]
        
        return {
            'total_levels': len(self.grid_levels),
            'active_levels': len(active_levels),
            'filled_levels': len(filled_levels),
            'base_price': float(self.base_price) if self.base_price else None,
            'total_profit': float(self.total_grid_profit),
            'last_rebalance': self.last_rebalance_time,
            'grid_levels': [
                {
                    'level': level.level,
                    'price': float(level.price),
                    'side': level.side.value,
                    'amount': float(level.amount),
                    'status': level.status.value,
                    'profit_target': float(level.profit_target)
                }
                for level in self.grid_levels
            ]
        }
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        return self.parameters.copy()
    
    def set_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Set strategy parameters."""
        if self.validate_parameters(parameters):
            self.parameters.update(parameters)
            # Reinitialize grid if key parameters changed
            if any(param in parameters for param in ['grid_levels', 'grid_spacing', 'base_price']):
                if self.base_price:
                    self._initialize_grid(self.base_price)
            return True
        return False
    
    def get_required_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get required parameters for the strategy."""
        return {
            'grid_levels': {'type': int, 'min': 5, 'max': 50, 'default': 10},
            'grid_spacing': {'type': float, 'min': 0.001, 'max': 0.05, 'default': 0.01},
            'grid_profit_target': {'type': float, 'min': 0.001, 'max': 0.02, 'default': 0.005},
            'max_grid_levels': {'type': int, 'min': 10, 'max': 100, 'default': 20},
            'min_grid_levels': {'type': int, 'min': 3, 'max': 20, 'default': 5},
            'trailing_grid': {'type': bool, 'default': True},
            'grid_rebalance_threshold': {'type': float, 'min': 0.01, 'max': 0.2, 'default': 0.05},
            'volume_per_level': {'type': float, 'min': 0.01, 'max': 0.5, 'default': 0.1},
            'stop_loss_type': {'type': str, 'options': ['percentage', 'atr'], 'default': 'percentage'},
            'stop_loss_percentage': {'type': float, 'min': 0.01, 'max': 0.3, 'default': 0.1},
            'take_profit_ratio': {'type': float, 'min': 0.5, 'max': 3.0, 'default': 1.0},
            'min_signal_strength': {'type': float, 'min': 0.1, 'max': 1.0, 'default': 0.5}
        }
    
    def get_description(self) -> str:
        """Get strategy description."""
        return ("Grid Bot Strategy: Places buy/sell orders at regular intervals around current price. "
                "Profits from price oscillations within the grid range. "
                "Supports dynamic rebalancing and trailing grid functionality.")
    
    def reset_grid(self):
        """Reset grid state."""
        self.grid_levels = []
        self.base_price = None
        self.last_rebalance_time = None
        self.total_grid_profit = Decimal("0")
        self.active_orders = {}
