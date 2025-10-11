"""
EMA Crossover Strategy implementation.
"""

import pandas as pd
from typing import Dict, List, Optional, Any
from decimal import Decimal

from ..core.data_models import OHLCV, StrategySignal, OrderSide, TimeFrame
from .base_strategy import BaseStrategy


class EMACrossoverStrategy(BaseStrategy):
    """EMA Crossover strategy with trend filter."""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize EMA Crossover strategy.
        
        Args:
            parameters: Strategy parameters
        """
        default_params = {
            'ema_fast': 12,
            'ema_medium': 26,
            'ema_slow': 50,
            'adx_period': 14,
            'adx_threshold': 25,
            'min_signal_strength': 0.6,
            'stop_loss_type': 'percentage',
            'stop_loss_percentage': 0.02,
            'take_profit_ratio': 2.0,
            'volume_confirmation': True,
            'trend_strength_filter': True
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__("EMA_Crossover", default_params)
    
    def generate_signal(self, ohlcv_data: List[OHLCV], indicators: Dict[str, List[float]], 
                       patterns: Dict[str, Any]) -> Optional[StrategySignal]:
        """Generate EMA crossover trading signal."""
        if len(ohlcv_data) < self.parameters['ema_slow']:
            return None
        
        # Get latest values
        latest_idx = -1
        current_price = ohlcv_data[latest_idx].close
        
        # Get EMA values
        ema_fast = indicators.get('ema_12', [])  # Using ema_12 as fast EMA
        ema_medium = indicators.get('ema_26', [])  # Using ema_26 as medium EMA
        ema_slow = indicators.get('sma_50', [])  # Using sma_50 as slow EMA
        
        if not all([ema_fast, ema_medium, ema_slow]):
            return None
        
        if len(ema_fast) < 2 or len(ema_medium) < 2 or len(ema_slow) < 2:
            return None
        
        current_ema_fast = ema_fast[latest_idx]
        prev_ema_fast = ema_fast[latest_idx - 1]
        current_ema_medium = ema_medium[latest_idx]
        prev_ema_medium = ema_medium[latest_idx - 1]
        current_ema_slow = ema_slow[latest_idx]
        prev_ema_slow = ema_slow[latest_idx - 1]
        
        # Get ADX for trend strength
        adx_values = indicators.get('adx', [])
        current_adx = adx_values[latest_idx] if adx_values else None
        
        # Check for NaN values
        if any(pd.isna([current_ema_fast, prev_ema_fast, current_ema_medium, 
                       prev_ema_medium, current_ema_slow, prev_ema_slow])):
            return None
        
        signal_strength = 0.0
        signal_type = None
        
        # Check for bullish crossover: Fast EMA crosses above Medium EMA
        if (prev_ema_fast <= prev_ema_medium and 
            current_ema_fast > current_ema_medium):
            
            signal_type = OrderSide.BUY
            signal_strength = 0.6
            
            # Increase strength if all EMAs are aligned (Fast > Medium > Slow)
            if current_ema_fast > current_ema_medium > current_ema_slow:
                signal_strength += 0.2
            
            # Increase strength if price is above all EMAs
            if current_price > current_ema_fast:
                signal_strength += 0.1
            
            # Increase strength if trend is strong (ADX > threshold)
            if (self.parameters['trend_strength_filter'] and 
                current_adx and current_adx > self.parameters['adx_threshold']):
                signal_strength += 0.1
            
            # Increase strength if crossover is strong
            crossover_strength = (current_ema_fast - current_ema_medium) / current_ema_medium
            if crossover_strength > 0.01:  # 1% crossover
                signal_strength += 0.1
        
        # Check for bearish crossover: Fast EMA crosses below Medium EMA
        elif (prev_ema_fast >= prev_ema_medium and 
              current_ema_fast < current_ema_medium):
            
            signal_type = OrderSide.SELL
            signal_strength = 0.6
            
            # Increase strength if all EMAs are aligned (Fast < Medium < Slow)
            if current_ema_fast < current_ema_medium < current_ema_slow:
                signal_strength += 0.2
            
            # Increase strength if price is below all EMAs
            if current_price < current_ema_fast:
                signal_strength += 0.1
            
            # Increase strength if trend is strong (ADX > threshold)
            if (self.parameters['trend_strength_filter'] and 
                current_adx and current_adx > self.parameters['adx_threshold']):
                signal_strength += 0.1
            
            # Increase strength if crossover is strong
            crossover_strength = (current_ema_medium - current_ema_fast) / current_ema_medium
            if crossover_strength > 0.01:  # 1% crossover
                signal_strength += 0.1
        
        # Additional confirmation signals
        if signal_type:
            # Check for volume confirmation
            if self.parameters['volume_confirmation'] and self._check_volume_confirmation(ohlcv_data, signal_type):
                signal_strength += 0.1
            
            # Check for momentum confirmation
            if self._check_momentum_confirmation(indicators, signal_type):
                signal_strength += 0.1
            
            # Check for support/resistance levels
            if self._check_support_resistance(ohlcv_data, current_price, signal_type):
                signal_strength += 0.1
            
            # Check for multiple timeframe alignment
            if self._check_timeframe_alignment(ohlcv_data, signal_type):
                signal_strength += 0.1
        
        # Only generate signal if strength meets minimum threshold
        if signal_type and signal_strength >= self.parameters['min_signal_strength']:
            return StrategySignal(
                symbol=ohlcv_data[latest_idx].symbol,
                signal=signal_type,
                strength=min(signal_strength, 1.0),
                price=current_price,
                strategy=self.name,
                timeframe=ohlcv_data[latest_idx].timeframe,
                indicators={
                    'ema_fast': current_ema_fast,
                    'ema_medium': current_ema_medium,
                    'ema_slow': current_ema_slow,
                    'adx': current_adx,
                    'ema_alignment': self._calculate_ema_alignment(current_ema_fast, current_ema_medium, current_ema_slow),
                    'price_vs_ema': (current_price - current_ema_fast) / current_ema_fast
                },
                metadata={
                    'ema_fast': self.parameters['ema_fast'],
                    'ema_medium': self.parameters['ema_medium'],
                    'ema_slow': self.parameters['ema_slow'],
                    'adx_period': self.parameters['adx_period']
                }
            )
        
        return None
    
    def _check_volume_confirmation(self, ohlcv_data: List[OHLCV], signal_type: OrderSide) -> bool:
        """Check for volume confirmation."""
        if len(ohlcv_data) < 5:
            return False
        
        # Calculate average volume
        recent_volumes = [float(candle.volume) for candle in ohlcv_data[-5:]]
        avg_volume = sum(recent_volumes) / len(recent_volumes)
        current_volume = float(ohlcv_data[-1].volume)
        
        # Volume should be above average for signal confirmation
        return current_volume > avg_volume * 1.2
    
    def _check_momentum_confirmation(self, indicators: Dict[str, List[float]], signal_type: OrderSide) -> bool:
        """Check for momentum confirmation using RSI."""
        rsi_values = indicators.get('rsi', [])
        if not rsi_values:
            return True  # No momentum filter if RSI not available
        
        current_rsi = rsi_values[-1]
        
        if signal_type == OrderSide.BUY:
            # For buy signals, prefer RSI not overbought
            return current_rsi < 70
        else:  # SELL
            # For sell signals, prefer RSI not oversold
            return current_rsi > 30
    
    def _check_support_resistance(self, ohlcv_data: List[OHLCV], current_price: Decimal, signal_type: OrderSide) -> bool:
        """Check for support/resistance levels."""
        if len(ohlcv_data) < 20:
            return True  # No support/resistance filter if not enough data
        
        # Get recent highs and lows
        recent_highs = [float(candle.high) for candle in ohlcv_data[-20:]]
        recent_lows = [float(candle.low) for candle in ohlcv_data[-20:]]
        
        current_price_float = float(current_price)
        
        if signal_type == OrderSide.BUY:
            # For buy signals, check if price is near support
            support_levels = [min(recent_lows[i:i+5]) for i in range(0, len(recent_lows)-4, 5)]
            for support in support_levels:
                if abs(current_price_float - support) / support < 0.02:  # Within 2%
                    return True
        else:  # SELL
            # For sell signals, check if price is near resistance
            resistance_levels = [max(recent_highs[i:i+5]) for i in range(0, len(recent_highs)-4, 5)]
            for resistance in resistance_levels:
                if abs(current_price_float - resistance) / resistance < 0.02:  # Within 2%
                    return True
        
        return False
    
    def _check_timeframe_alignment(self, ohlcv_data: List[OHLCV], signal_type: OrderSide) -> bool:
        """Check for multiple timeframe alignment."""
        # This is a simplified check - in a real implementation, you'd check higher timeframes
        if len(ohlcv_data) < 50:
            return True  # No timeframe filter if not enough data
        
        # Check if the trend is consistent over a longer period
        recent_prices = [float(candle.close) for candle in ohlcv_data[-50:]]
        price_trend = recent_prices[-1] - recent_prices[0]
        
        if signal_type == OrderSide.BUY:
            return price_trend > 0  # Uptrend over longer period
        else:  # SELL
            return price_trend < 0  # Downtrend over longer period
    
    def _calculate_ema_alignment(self, ema_fast: float, ema_medium: float, ema_slow: float) -> str:
        """Calculate EMA alignment."""
        if ema_fast > ema_medium > ema_slow:
            return "bullish"
        elif ema_fast < ema_medium < ema_slow:
            return "bearish"
        else:
            return "mixed"
    
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
            'ema_fast': {'type': int, 'min': 5, 'max': 20, 'default': 12},
            'ema_medium': {'type': int, 'min': 15, 'max': 35, 'default': 26},
            'ema_slow': {'type': int, 'min': 30, 'max': 100, 'default': 50},
            'adx_period': {'type': int, 'min': 10, 'max': 30, 'default': 14},
            'adx_threshold': {'type': int, 'min': 15, 'max': 40, 'default': 25},
            'min_signal_strength': {'type': float, 'min': 0.1, 'max': 1.0, 'default': 0.6},
            'stop_loss_type': {'type': str, 'options': ['percentage', 'atr'], 'default': 'percentage'},
            'stop_loss_percentage': {'type': float, 'min': 0.005, 'max': 0.1, 'default': 0.02},
            'take_profit_ratio': {'type': float, 'min': 1.0, 'max': 5.0, 'default': 2.0},
            'volume_confirmation': {'type': bool, 'default': True},
            'trend_strength_filter': {'type': bool, 'default': True}
        }
    
    def get_description(self) -> str:
        """Get strategy description."""
        return ("EMA Crossover Strategy: Uses multiple EMA crossovers with trend strength filter. "
                "Buy when fast EMA crosses above medium EMA with all EMAs aligned. "
                "Sell when fast EMA crosses below medium EMA with all EMAs aligned.")
