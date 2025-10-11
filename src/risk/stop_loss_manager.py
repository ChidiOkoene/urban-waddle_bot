"""
Stop-loss management with multiple algorithms.
"""

from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
from datetime import datetime
from enum import Enum

from ..core.data_models import Position, OrderSide, OrderType


class StopLossType(str, Enum):
    """Stop-loss types."""
    FIXED = "fixed"
    PERCENTAGE = "percentage"
    ATR = "atr"
    TRAILING = "trailing"
    PARABOLIC_SAR = "parabolic_sar"
    SUPPORT_RESISTANCE = "support_resistance"


class StopLossManager:
    """Stop-loss management with multiple algorithms."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize stop-loss manager.
        
        Args:
            config: Stop-loss configuration
        """
        self.config = config
        
        # Stop-loss parameters
        self.default_stop_loss_type = config.get('stop_loss_type', StopLossType.PERCENTAGE)
        self.default_stop_loss_percentage = Decimal(str(config.get('stop_loss_percentage', 0.02)))  # 2%
        self.atr_multiplier = config.get('atr_multiplier', 2.0)
        self.atr_period = config.get('atr_period', 14)
        
        # Trailing stop parameters
        self.trailing_activation = Decimal(str(config.get('trailing_stop_activation', 0.015)))  # 1.5%
        self.trailing_distance = Decimal(str(config.get('trailing_stop_distance', 0.01)))  # 1%
        
        # Parabolic SAR parameters
        self.parabolic_acceleration = config.get('parabolic_acceleration', 0.02)
        self.parabolic_maximum = config.get('parabolic_maximum', 0.2)
        
        # Active stop-losses
        self.active_stops = {}  # position_id -> stop_loss_info
    
    def calculate_stop_loss(self, 
                           position: Position,
                           current_price: Decimal,
                           indicators: Optional[Dict[str, List[float]]] = None,
                           support_resistance: Optional[Dict[str, List[float]]] = None) -> Decimal:
        """
        Calculate stop-loss price for a position.
        
        Args:
            position: Position object
            current_price: Current price
            indicators: Technical indicators
            support_resistance: Support/resistance levels
            
        Returns:
            Stop-loss price
        """
        stop_loss_type = self._get_stop_loss_type(position)
        
        if stop_loss_type == StopLossType.FIXED:
            return self._calculate_fixed_stop_loss(position, current_price)
        
        elif stop_loss_type == StopLossType.PERCENTAGE:
            return self._calculate_percentage_stop_loss(position, current_price)
        
        elif stop_loss_type == StopLossType.ATR:
            return self._calculate_atr_stop_loss(position, current_price, indicators)
        
        elif stop_loss_type == StopLossType.TRAILING:
            return self._calculate_trailing_stop_loss(position, current_price)
        
        elif stop_loss_type == StopLossType.PARABOLIC_SAR:
            return self._calculate_parabolic_sar_stop_loss(position, current_price, indicators)
        
        elif stop_loss_type == StopLossType.SUPPORT_RESISTANCE:
            return self._calculate_support_resistance_stop_loss(position, current_price, support_resistance)
        
        else:
            # Default to percentage
            return self._calculate_percentage_stop_loss(position, current_price)
    
    def update_stop_loss(self, position: Position, current_price: Decimal, 
                        indicators: Optional[Dict[str, List[float]]] = None) -> Optional[Decimal]:
        """
        Update stop-loss for a position.
        
        Args:
            position: Position object
            current_price: Current price
            indicators: Technical indicators
            
        Returns:
            New stop-loss price if updated, None otherwise
        """
        position_id = position.id
        
        # Get current stop-loss info
        stop_info = self.active_stops.get(position_id)
        if not stop_info:
            # Initialize stop-loss
            stop_loss_price = self.calculate_stop_loss(position, current_price, indicators)
            self.active_stops[position_id] = {
                'stop_loss_price': stop_loss_price,
                'stop_loss_type': self._get_stop_loss_type(position),
                'entry_price': position.entry_price,
                'highest_price': current_price if position.side == OrderSide.LONG else position.entry_price,
                'lowest_price': current_price if position.side == OrderSide.SHORT else position.entry_price,
                'last_update': datetime.utcnow()
            }
            return stop_loss_price
        
        # Update highest/lowest prices
        if position.side == OrderSide.LONG:
            if current_price > stop_info['highest_price']:
                stop_info['highest_price'] = current_price
        else:  # SHORT
            if current_price < stop_info['lowest_price']:
                stop_info['lowest_price'] = current_price
        
        # Check if stop-loss should be updated
        stop_loss_type = stop_info['stop_loss_type']
        new_stop_loss = None
        
        if stop_loss_type == StopLossType.TRAILING:
            new_stop_loss = self._update_trailing_stop_loss(position, current_price, stop_info)
        
        elif stop_loss_type == StopLossType.PARABOLIC_SAR:
            new_stop_loss = self._update_parabolic_sar_stop_loss(position, current_price, stop_info, indicators)
        
        elif stop_loss_type == StopLossType.ATR:
            new_stop_loss = self._update_atr_stop_loss(position, current_price, stop_info, indicators)
        
        # Update stop-loss if changed
        if new_stop_loss and new_stop_loss != stop_info['stop_loss_price']:
            stop_info['stop_loss_price'] = new_stop_loss
            stop_info['last_update'] = datetime.utcnow()
            return new_stop_loss
        
        return None
    
    def check_stop_loss_trigger(self, position: Position, current_price: Decimal) -> bool:
        """
        Check if stop-loss should be triggered.
        
        Args:
            position: Position object
            current_price: Current price
            
        Returns:
            True if stop-loss should be triggered
        """
        stop_info = self.active_stops.get(position.id)
        if not stop_info:
            return False
        
        stop_loss_price = stop_info['stop_loss_price']
        
        if position.side == OrderSide.LONG:
            return current_price <= stop_loss_price
        else:  # SHORT
            return current_price >= stop_loss_price
    
    def remove_stop_loss(self, position_id: str):
        """Remove stop-loss for a position."""
        if position_id in self.active_stops:
            del self.active_stops[position_id]
    
    def _get_stop_loss_type(self, position: Position) -> StopLossType:
        """Get stop-loss type for a position."""
        # Check if position has specific stop-loss type
        if hasattr(position, 'stop_loss_type') and position.stop_loss_type:
            return StopLossType(position.stop_loss_type)
        
        # Use default
        return self.default_stop_loss_type
    
    def _calculate_fixed_stop_loss(self, position: Position, current_price: Decimal) -> Decimal:
        """Calculate fixed stop-loss."""
        stop_loss_amount = Decimal(str(self.config.get('fixed_stop_loss_amount', 100)))
        
        if position.side == OrderSide.LONG:
            return current_price - stop_loss_amount
        else:  # SHORT
            return current_price + stop_loss_amount
    
    def _calculate_percentage_stop_loss(self, position: Position, current_price: Decimal) -> Decimal:
        """Calculate percentage-based stop-loss."""
        stop_loss_percentage = self.default_stop_loss_percentage
        
        if position.side == OrderSide.LONG:
            return current_price * (1 - stop_loss_percentage)
        else:  # SHORT
            return current_price * (1 + stop_loss_percentage)
    
    def _calculate_atr_stop_loss(self, position: Position, current_price: Decimal, 
                                indicators: Optional[Dict[str, List[float]]]) -> Decimal:
        """Calculate ATR-based stop-loss."""
        if not indicators or 'atr' not in indicators:
            # Fallback to percentage
            return self._calculate_percentage_stop_loss(position, current_price)
        
        atr_values = indicators['atr']
        if not atr_values:
            return self._calculate_percentage_stop_loss(position, current_price)
        
        current_atr = atr_values[-1]
        atr_stop_distance = Decimal(str(current_atr * self.atr_multiplier))
        
        if position.side == OrderSide.LONG:
            return current_price - atr_stop_distance
        else:  # SHORT
            return current_price + atr_stop_distance
    
    def _calculate_trailing_stop_loss(self, position: Position, current_price: Decimal) -> Decimal:
        """Calculate trailing stop-loss."""
        if position.side == OrderSide.LONG:
            # For long positions, trail below the price
            return current_price * (1 - self.trailing_distance)
        else:  # SHORT
            # For short positions, trail above the price
            return current_price * (1 + self.trailing_distance)
    
    def _calculate_parabolic_sar_stop_loss(self, position: Position, current_price: Decimal, 
                                         indicators: Optional[Dict[str, List[float]]]) -> Decimal:
        """Calculate Parabolic SAR stop-loss."""
        if not indicators or 'parabolic_sar' not in indicators:
            return self._calculate_percentage_stop_loss(position, current_price)
        
        sar_values = indicators['parabolic_sar']
        if not sar_values:
            return self._calculate_percentage_stop_loss(position, current_price)
        
        current_sar = sar_values[-1]
        return Decimal(str(current_sar))
    
    def _calculate_support_resistance_stop_loss(self, position: Position, current_price: Decimal, 
                                              support_resistance: Optional[Dict[str, List[float]]]) -> Decimal:
        """Calculate support/resistance stop-loss."""
        if not support_resistance:
            return self._calculate_percentage_stop_loss(position, current_price)
        
        if position.side == OrderSide.LONG:
            # For long positions, use support levels
            support_levels = support_resistance.get('support_levels', [])
            if support_levels:
                # Find closest support level below current price
                valid_supports = [s for s in support_levels if s < float(current_price)]
                if valid_supports:
                    closest_support = max(valid_supports)
                    return Decimal(str(closest_support))
        else:  # SHORT
            # For short positions, use resistance levels
            resistance_levels = support_resistance.get('resistance_levels', [])
            if resistance_levels:
                # Find closest resistance level above current price
                valid_resistances = [r for r in resistance_levels if r > float(current_price)]
                if valid_resistances:
                    closest_resistance = min(valid_resistances)
                    return Decimal(str(closest_resistance))
        
        # Fallback to percentage
        return self._calculate_percentage_stop_loss(position, current_price)
    
    def _update_trailing_stop_loss(self, position: Position, current_price: Decimal, 
                                  stop_info: Dict[str, Any]) -> Optional[Decimal]:
        """Update trailing stop-loss."""
        if position.side == OrderSide.LONG:
            # Check if price has moved favorably enough to update stop-loss
            price_increase = (current_price - stop_info['entry_price']) / stop_info['entry_price']
            
            if price_increase >= self.trailing_activation:
                # Update trailing stop-loss
                new_stop_loss = current_price * (1 - self.trailing_distance)
                
                # Only update if new stop-loss is better (higher for long positions)
                if new_stop_loss > stop_info['stop_loss_price']:
                    return new_stop_loss
        
        else:  # SHORT
            # Check if price has moved favorably enough to update stop-loss
            price_decrease = (stop_info['entry_price'] - current_price) / stop_info['entry_price']
            
            if price_decrease >= self.trailing_activation:
                # Update trailing stop-loss
                new_stop_loss = current_price * (1 + self.trailing_distance)
                
                # Only update if new stop-loss is better (lower for short positions)
                if new_stop_loss < stop_info['stop_loss_price']:
                    return new_stop_loss
        
        return None
    
    def _update_parabolic_sar_stop_loss(self, position: Position, current_price: Decimal, 
                                      stop_info: Dict[str, Any], 
                                      indicators: Optional[Dict[str, List[float]]]) -> Optional[Decimal]:
        """Update Parabolic SAR stop-loss."""
        if not indicators or 'parabolic_sar' not in indicators:
            return None
        
        sar_values = indicators['parabolic_sar']
        if not sar_values:
            return None
        
        current_sar = sar_values[-1]
        new_stop_loss = Decimal(str(current_sar))
        
        # Only update if SAR has moved favorably
        if position.side == OrderSide.LONG:
            if new_stop_loss > stop_info['stop_loss_price']:
                return new_stop_loss
        else:  # SHORT
            if new_stop_loss < stop_info['stop_loss_price']:
                return new_stop_loss
        
        return None
    
    def _update_atr_stop_loss(self, position: Position, current_price: Decimal, 
                             stop_info: Dict[str, Any], 
                             indicators: Optional[Dict[str, List[float]]]) -> Optional[Decimal]:
        """Update ATR-based stop-loss."""
        if not indicators or 'atr' not in indicators:
            return None
        
        atr_values = indicators['atr']
        if not atr_values:
            return None
        
        current_atr = atr_values[-1]
        atr_stop_distance = Decimal(str(current_atr * self.atr_multiplier))
        
        if position.side == OrderSide.LONG:
            new_stop_loss = current_price - atr_stop_distance
            # Only update if new stop-loss is better (higher)
            if new_stop_loss > stop_info['stop_loss_price']:
                return new_stop_loss
        else:  # SHORT
            new_stop_loss = current_price + atr_stop_distance
            # Only update if new stop-loss is better (lower)
            if new_stop_loss < stop_info['stop_loss_price']:
                return new_stop_loss
        
        return None
    
    def get_stop_loss_info(self, position_id: str) -> Optional[Dict[str, Any]]:
        """Get stop-loss information for a position."""
        return self.active_stops.get(position_id)
    
    def get_all_stop_losses(self) -> Dict[str, Dict[str, Any]]:
        """Get all active stop-losses."""
        return self.active_stops.copy()
    
    def set_stop_loss_parameters(self, parameters: Dict[str, Any]):
        """Update stop-loss parameters."""
        if 'stop_loss_type' in parameters:
            self.default_stop_loss_type = StopLossType(parameters['stop_loss_type'])
        
        if 'stop_loss_percentage' in parameters:
            self.default_stop_loss_percentage = Decimal(str(parameters['stop_loss_percentage']))
        
        if 'atr_multiplier' in parameters:
            self.atr_multiplier = parameters['atr_multiplier']
        
        if 'atr_period' in parameters:
            self.atr_period = parameters['atr_period']
        
        if 'trailing_stop_activation' in parameters:
            self.trailing_activation = Decimal(str(parameters['trailing_stop_activation']))
        
        if 'trailing_stop_distance' in parameters:
            self.trailing_distance = Decimal(str(parameters['trailing_stop_distance']))
        
        if 'parabolic_acceleration' in parameters:
            self.parabolic_acceleration = parameters['parabolic_acceleration']
        
        if 'parabolic_maximum' in parameters:
            self.parabolic_maximum = parameters['parabolic_maximum']
    
    def get_stop_loss_status(self) -> Dict[str, Any]:
        """Get stop-loss manager status."""
        return {
            'active_stops': len(self.active_stops),
            'default_stop_loss_type': self.default_stop_loss_type.value,
            'default_stop_loss_percentage': float(self.default_stop_loss_percentage),
            'atr_multiplier': self.atr_multiplier,
            'atr_period': self.atr_period,
            'trailing_activation': float(self.trailing_activation),
            'trailing_distance': float(self.trailing_distance),
            'parabolic_acceleration': self.parabolic_acceleration,
            'parabolic_maximum': self.parabolic_maximum,
            'stop_losses': {
                pos_id: {
                    'stop_loss_price': float(info['stop_loss_price']),
                    'stop_loss_type': info['stop_loss_type'].value,
                    'entry_price': float(info['entry_price']),
                    'last_update': info['last_update'].isoformat()
                }
                for pos_id, info in self.active_stops.items()
            }
        }
