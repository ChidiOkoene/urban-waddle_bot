"""
Ichimoku Cloud Strategy implementation.
"""

import pandas as pd
from typing import Dict, List, Optional, Any
from decimal import Decimal

from ..core.data_models import OHLCV, StrategySignal, OrderSide, TimeFrame
from .base_strategy import BaseStrategy


class IchimokuStrategy(BaseStrategy):
    """Complete Ichimoku Cloud trading system."""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize Ichimoku strategy.
        
        Args:
            parameters: Strategy parameters
        """
        default_params = {
            'ichimoku_conversion': 9,
            'ichimoku_base': 26,
            'ichimoku_span_b': 52,
            'ichimoku_displacement': 26,
            'min_signal_strength': 0.6,
            'stop_loss_type': 'percentage',
            'stop_loss_percentage': 0.02,
            'take_profit_ratio': 2.0,
            'cloud_filter': True,
            'tk_cross_filter': True,
            'lagging_span_filter': True,
            'volume_confirmation': True
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__("Ichimoku", default_params)
    
    def generate_signal(self, ohlcv_data: List[OHLCV], indicators: Dict[str, List[float]], 
                       patterns: Dict[str, Any]) -> Optional[StrategySignal]:
        """Generate Ichimoku trading signal."""
        if len(ohlcv_data) < self.parameters['ichimoku_span_b']:
            return None
        
        current_price = ohlcv_data[-1].close
        
        # Get Ichimoku components
        tenkan_sen = indicators.get('ichimoku_tenkan', [])
        kijun_sen = indicators.get('ichimoku_kijun', [])
        span_a = indicators.get('ichimoku_span_a', [])
        span_b = indicators.get('ichimoku_span_b', [])
        chikou_span = indicators.get('ichimoku_chikou', [])
        
        if not all([tenkan_sen, kijun_sen, span_a, span_b]):
            return None
        
        if len(tenkan_sen) < 2:
            return None
        
        # Get current values
        current_tenkan = tenkan_sen[-1]
        current_kijun = kijun_sen[-1]
        current_span_a = span_a[-1]
        current_span_b = span_b[-1]
        
        # Check for NaN values
        if any(pd.isna([current_tenkan, current_kijun, current_span_a, current_span_b])):
            return None
        
        signal_strength = 0.0
        signal_type = None
        
        # Check for TK cross (Tenkan-sen crosses Kijun-sen)
        tk_cross_signal = self._check_tk_cross(tenkan_sen, kijun_sen)
        
        if tk_cross_signal:
            signal_type = tk_cross_signal
            signal_strength = 0.6
            
            # Check cloud position
            cloud_position = self._check_cloud_position(current_price, current_span_a, current_span_b)
            
            if self.parameters['cloud_filter']:
                if signal_type == OrderSide.BUY and cloud_position == 'above':
                    signal_strength += 0.2
                elif signal_type == OrderSide.SELL and cloud_position == 'below':
                    signal_strength += 0.2
                elif cloud_position == 'inside':
                    signal_strength -= 0.1  # Reduce strength if inside cloud
            
            # Check lagging span
            if self.parameters['lagging_span_filter'] and chikou_span:
                lagging_signal = self._check_lagging_span(chikou_span, ohlcv_data)
                if lagging_signal == signal_type:
                    signal_strength += 0.1
                elif lagging_signal and lagging_signal != signal_type:
                    signal_strength -= 0.1  # Conflicting signal
        
        # Additional confirmation signals
        if signal_type:
            # Check for volume confirmation
            if self.parameters['volume_confirmation'] and self._check_volume_confirmation(ohlcv_data, signal_type):
                signal_strength += 0.1
            
            # Check for cloud thickness
            cloud_thickness = self._calculate_cloud_thickness(current_span_a, current_span_b)
            if cloud_thickness > 0.02:  # Thick cloud
                signal_strength += 0.1
            
            # Check for price distance from cloud
            price_distance = self._calculate_price_distance_from_cloud(current_price, current_span_a, current_span_b)
            if price_distance > 0.01:  # Price is well away from cloud
                signal_strength += 0.1
            
            # Check for trend strength
            if self._check_trend_strength(tenkan_sen, kijun_sen, current_price):
                signal_strength += 0.1
        
        # Only generate signal if strength meets minimum threshold
        if signal_type and signal_strength >= self.parameters['min_signal_strength']:
            return StrategySignal(
                symbol=ohlcv_data[-1].symbol,
                signal=signal_type,
                strength=min(signal_strength, 1.0),
                price=current_price,
                strategy=self.name,
                timeframe=ohlcv_data[-1].timeframe,
                indicators={
                    'tenkan_sen': current_tenkan,
                    'kijun_sen': current_kijun,
                    'span_a': current_span_a,
                    'span_b': current_span_b,
                    'chikou_span': chikou_span[-1] if chikou_span else 0,
                    'cloud_position': cloud_position,
                    'cloud_thickness': cloud_thickness,
                    'price_distance_from_cloud': price_distance
                },
                metadata={
                    'ichimoku_conversion': self.parameters['ichimoku_conversion'],
                    'ichimoku_base': self.parameters['ichimoku_base'],
                    'ichimoku_span_b': self.parameters['ichimoku_span_b'],
                    'ichimoku_displacement': self.parameters['ichimoku_displacement']
                }
            )
        
        return None
    
    def _check_tk_cross(self, tenkan_sen: List[float], kijun_sen: List[float]) -> Optional[OrderSide]:
        """Check for Tenkan-sen/Kijun-sen cross."""
        if len(tenkan_sen) < 2 or len(kijun_sen) < 2:
            return None
        
        current_tenkan = tenkan_sen[-1]
        prev_tenkan = tenkan_sen[-2]
        current_kijun = kijun_sen[-1]
        prev_kijun = kijun_sen[-2]
        
        # Bullish TK cross: Tenkan-sen crosses above Kijun-sen
        if prev_tenkan <= prev_kijun and current_tenkan > current_kijun:
            return OrderSide.BUY
        
        # Bearish TK cross: Tenkan-sen crosses below Kijun-sen
        elif prev_tenkan >= prev_kijun and current_tenkan < current_kijun:
            return OrderSide.SELL
        
        return None
    
    def _check_cloud_position(self, current_price: Decimal, span_a: float, span_b: float) -> str:
        """Check position relative to cloud."""
        current_price_float = float(current_price)
        
        # Determine cloud boundaries
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        
        if current_price_float > cloud_top:
            return 'above'
        elif current_price_float < cloud_bottom:
            return 'below'
        else:
            return 'inside'
    
    def _check_lagging_span(self, chikou_span: List[float], ohlcv_data: List[OHLCV]) -> Optional[OrderSide]:
        """Check lagging span signal."""
        if not chikou_span or len(chikou_span) < 1:
            return None
        
        current_chikou = chikou_span[-1]
        current_price = float(ohlcv_data[-1].close)
        
        # Lagging span above price = bullish
        if current_chikou > current_price:
            return OrderSide.BUY
        # Lagging span below price = bearish
        elif current_chikou < current_price:
            return OrderSide.SELL
        
        return None
    
    def _check_volume_confirmation(self, ohlcv_data: List[OHLCV], signal_type: OrderSide) -> bool:
        """Check for volume confirmation."""
        if len(ohlcv_data) < 5:
            return True  # No volume filter if not enough data
        
        # Calculate average volume
        recent_volumes = [float(candle.volume) for candle in ohlcv_data[-5:]]
        avg_volume = sum(recent_volumes) / len(recent_volumes)
        current_volume = float(ohlcv_data[-1].volume)
        
        # Volume should be above average for signal confirmation
        return current_volume > avg_volume * 1.1
    
    def _calculate_cloud_thickness(self, span_a: float, span_b: float) -> float:
        """Calculate cloud thickness as percentage."""
        thickness = abs(span_a - span_b)
        avg_span = (span_a + span_b) / 2
        
        if avg_span == 0:
            return 0.0
        
        return thickness / avg_span
    
    def _calculate_price_distance_from_cloud(self, current_price: Decimal, span_a: float, span_b: float) -> float:
        """Calculate price distance from cloud."""
        current_price_float = float(current_price)
        
        # Determine cloud boundaries
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        
        if current_price_float > cloud_top:
            return (current_price_float - cloud_top) / cloud_top
        elif current_price_float < cloud_bottom:
            return (cloud_bottom - current_price_float) / cloud_bottom
        else:
            return 0.0
    
    def _check_trend_strength(self, tenkan_sen: List[float], kijun_sen: List[float], current_price: Decimal) -> bool:
        """Check trend strength using Ichimoku components."""
        if len(tenkan_sen) < 1 or len(kijun_sen) < 1:
            return False
        
        current_tenkan = tenkan_sen[-1]
        current_kijun = kijun_sen[-1]
        current_price_float = float(current_price)
        
        # Strong trend: price, tenkan, and kijun are aligned
        if current_price_float > current_tenkan > current_kijun:
            return True  # Strong uptrend
        elif current_price_float < current_tenkan < current_kijun:
            return True  # Strong downtrend
        
        return False
    
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
            'ichimoku_conversion': {'type': int, 'min': 5, 'max': 20, 'default': 9},
            'ichimoku_base': {'type': int, 'min': 15, 'max': 35, 'default': 26},
            'ichimoku_span_b': {'type': int, 'min': 30, 'max': 100, 'default': 52},
            'ichimoku_displacement': {'type': int, 'min': 15, 'max': 35, 'default': 26},
            'min_signal_strength': {'type': float, 'min': 0.1, 'max': 1.0, 'default': 0.6},
            'stop_loss_type': {'type': str, 'options': ['percentage', 'atr'], 'default': 'percentage'},
            'stop_loss_percentage': {'type': float, 'min': 0.005, 'max': 0.1, 'default': 0.02},
            'take_profit_ratio': {'type': float, 'min': 1.0, 'max': 5.0, 'default': 2.0},
            'cloud_filter': {'type': bool, 'default': True},
            'tk_cross_filter': {'type': bool, 'default': True},
            'lagging_span_filter': {'type': bool, 'default': True},
            'volume_confirmation': {'type': bool, 'default': True}
        }
    
    def get_description(self) -> str:
        """Get strategy description."""
        return ("Ichimoku Cloud Strategy: Complete Ichimoku trading system using all components. "
                "Buy on bullish TK cross above cloud with lagging span confirmation. "
                "Sell on bearish TK cross below cloud with lagging span confirmation.")
