"""
Breakout Strategy implementation.
"""

import pandas as pd
from typing import Dict, List, Optional, Any
from decimal import Decimal

from ..core.data_models import OHLCV, StrategySignal, OrderSide, TimeFrame
from .base_strategy import BaseStrategy


class BreakoutStrategy(BaseStrategy):
    """Support/resistance breakout strategy with volume confirmation."""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize Breakout strategy.
        
        Args:
            parameters: Strategy parameters
        """
        default_params = {
            'breakout_lookback': 20,
            'breakout_threshold': 0.02,  # 2%
            'volume_threshold': 1.5,  # 1.5x average volume
            'atr_period': 14,
            'atr_multiplier': 2.0,
            'min_signal_strength': 0.6,
            'stop_loss_type': 'atr',
            'stop_loss_atr_multiplier': 2.0,
            'take_profit_ratio': 2.0,
            'volume_confirmation': True,
            'trend_filter': True,
            'false_breakout_filter': True
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__("Breakout", default_params)
    
    def generate_signal(self, ohlcv_data: List[OHLCV], indicators: Dict[str, List[float]], 
                       patterns: Dict[str, Any]) -> Optional[StrategySignal]:
        """Generate breakout trading signal."""
        if len(ohlcv_data) < self.parameters['breakout_lookback'] + 5:
            return None
        
        current_price = ohlcv_data[-1].close
        current_high = ohlcv_data[-1].high
        current_low = ohlcv_data[-1].low
        
        # Get ATR for volatility measurement
        atr_values = indicators.get('atr', [])
        if not atr_values:
            return None
        
        current_atr = atr_values[-1]
        
        # Check for NaN values
        if pd.isna(current_atr):
            return None
        
        signal_strength = 0.0
        signal_type = None
        
        # Check for resistance breakout
        resistance_breakout = self._check_resistance_breakout(ohlcv_data, current_high)
        if resistance_breakout:
            signal_type = OrderSide.BUY
            signal_strength = 0.7
            
            # Increase strength based on breakout magnitude
            breakout_magnitude = resistance_breakout['magnitude']
            if breakout_magnitude > self.parameters['breakout_threshold'] * 2:
                signal_strength += 0.1
            
            # Increase strength if volume confirms
            if self._check_volume_confirmation(ohlcv_data, signal_type):
                signal_strength += 0.1
            
            # Increase strength if trend confirms
            if self._check_trend_confirmation(indicators, signal_type):
                signal_strength += 0.1
        
        # Check for support breakdown
        support_breakdown = self._check_support_breakdown(ohlcv_data, current_low)
        if support_breakdown:
            signal_type = OrderSide.SELL
            signal_strength = 0.7
            
            # Increase strength based on breakdown magnitude
            breakdown_magnitude = support_breakdown['magnitude']
            if breakdown_magnitude > self.parameters['breakout_threshold'] * 2:
                signal_strength += 0.1
            
            # Increase strength if volume confirms
            if self._check_volume_confirmation(ohlcv_data, signal_type):
                signal_strength += 0.1
            
            # Increase strength if trend confirms
            if self._check_trend_confirmation(indicators, signal_type):
                signal_strength += 0.1
        
        # Additional confirmation signals
        if signal_type:
            # Check for false breakout filter
            if self.parameters['false_breakout_filter'] and self._check_false_breakout(ohlcv_data, signal_type):
                signal_strength += 0.1
            
            # Check for momentum confirmation
            if self._check_momentum_confirmation(indicators, signal_type):
                signal_strength += 0.1
            
            # Check for pattern confirmation
            if self._check_pattern_confirmation(patterns, signal_type):
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
                    'atr': current_atr,
                    'breakout_level': resistance_breakout['level'] if signal_type == OrderSide.BUY else support_breakdown['level'],
                    'breakout_magnitude': resistance_breakout['magnitude'] if signal_type == OrderSide.BUY else support_breakdown['magnitude'],
                    'volume_ratio': self._calculate_volume_ratio(ohlcv_data)
                },
                metadata={
                    'breakout_lookback': self.parameters['breakout_lookback'],
                    'breakout_threshold': self.parameters['breakout_threshold'],
                    'volume_threshold': self.parameters['volume_threshold'],
                    'atr_period': self.parameters['atr_period']
                }
            )
        
        return None
    
    def _check_resistance_breakout(self, ohlcv_data: List[OHLCV], current_high: Decimal) -> Optional[Dict[str, Any]]:
        """Check for resistance breakout."""
        lookback = self.parameters['breakout_lookback']
        
        # Get recent highs
        recent_highs = [float(candle.high) for candle in ohlcv_data[-lookback-1:-1]]
        if not recent_highs:
            return None
        
        resistance_level = max(recent_highs)
        current_high_float = float(current_high)
        
        # Check if current high breaks above resistance
        if current_high_float > resistance_level:
            breakout_magnitude = (current_high_float - resistance_level) / resistance_level
            
            if breakout_magnitude >= self.parameters['breakout_threshold']:
                return {
                    'level': resistance_level,
                    'magnitude': breakout_magnitude,
                    'breakout_price': current_high_float
                }
        
        return None
    
    def _check_support_breakdown(self, ohlcv_data: List[OHLCV], current_low: Decimal) -> Optional[Dict[str, Any]]:
        """Check for support breakdown."""
        lookback = self.parameters['breakout_lookback']
        
        # Get recent lows
        recent_lows = [float(candle.low) for candle in ohlcv_data[-lookback-1:-1]]
        if not recent_lows:
            return None
        
        support_level = min(recent_lows)
        current_low_float = float(current_low)
        
        # Check if current low breaks below support
        if current_low_float < support_level:
            breakdown_magnitude = (support_level - current_low_float) / support_level
            
            if breakdown_magnitude >= self.parameters['breakout_threshold']:
                return {
                    'level': support_level,
                    'magnitude': breakdown_magnitude,
                    'breakdown_price': current_low_float
                }
        
        return None
    
    def _check_volume_confirmation(self, ohlcv_data: List[OHLCV], signal_type: OrderSide) -> bool:
        """Check for volume confirmation."""
        if not self.parameters['volume_confirmation']:
            return True
        
        if len(ohlcv_data) < 10:
            return False
        
        # Calculate average volume
        recent_volumes = [float(candle.volume) for candle in ohlcv_data[-10:]]
        avg_volume = sum(recent_volumes) / len(recent_volumes)
        current_volume = float(ohlcv_data[-1].volume)
        
        # Volume should be above threshold
        volume_ratio = current_volume / avg_volume
        return volume_ratio >= self.parameters['volume_threshold']
    
    def _check_trend_confirmation(self, indicators: Dict[str, List[float]], signal_type: OrderSide) -> bool:
        """Check for trend confirmation using moving averages."""
        if not self.parameters['trend_filter']:
            return True
        
        # Get EMA values
        ema_20 = indicators.get('sma_20', [])
        ema_50 = indicators.get('sma_50', [])
        
        if not ema_20 or not ema_50:
            return True  # No trend filter if indicators not available
        
        current_ema_20 = ema_20[-1]
        current_ema_50 = ema_50[-1]
        
        if signal_type == OrderSide.BUY:
            # For buy signals, prefer uptrend
            return current_ema_20 > current_ema_50
        else:  # SELL
            # For sell signals, prefer downtrend
            return current_ema_20 < current_ema_50
    
    def _check_false_breakout(self, ohlcv_data: List[OHLCV], signal_type: OrderSide) -> bool:
        """Check for false breakout patterns."""
        if len(ohlcv_data) < 5:
            return True  # No false breakout filter if not enough data
        
        # Check if price quickly reverses after breakout
        recent_closes = [float(candle.close) for candle in ohlcv_data[-5:]]
        
        if signal_type == OrderSide.BUY:
            # For buy signals, check if price doesn't quickly fall back
            return recent_closes[-1] > recent_closes[-2]
        else:  # SELL
            # For sell signals, check if price doesn't quickly rise back
            return recent_closes[-1] < recent_closes[-2]
    
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
    
    def _check_pattern_confirmation(self, patterns: Dict[str, Any], signal_type: OrderSide) -> bool:
        """Check for pattern confirmation."""
        # Check for candlestick patterns
        candlestick_patterns = patterns.get('candlestick_patterns', [])
        
        if not candlestick_patterns:
            return True  # No pattern filter if no patterns detected
        
        # Get the most recent pattern
        latest_pattern = candlestick_patterns[-1] if candlestick_patterns else None
        
        if not latest_pattern:
            return True
        
        pattern_signal = latest_pattern.get('signal')
        
        if signal_type == OrderSide.BUY:
            return pattern_signal in ['bullish', 'neutral']
        else:  # SELL
            return pattern_signal in ['bearish', 'neutral']
    
    def _calculate_volume_ratio(self, ohlcv_data: List[OHLCV]) -> float:
        """Calculate current volume ratio to average."""
        if len(ohlcv_data) < 10:
            return 1.0
        
        recent_volumes = [float(candle.volume) for candle in ohlcv_data[-10:]]
        avg_volume = sum(recent_volumes) / len(recent_volumes)
        current_volume = float(ohlcv_data[-1].volume)
        
        return current_volume / avg_volume if avg_volume > 0 else 1.0
    
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
            'breakout_lookback': {'type': int, 'min': 10, 'max': 50, 'default': 20},
            'breakout_threshold': {'type': float, 'min': 0.005, 'max': 0.05, 'default': 0.02},
            'volume_threshold': {'type': float, 'min': 1.0, 'max': 3.0, 'default': 1.5},
            'atr_period': {'type': int, 'min': 10, 'max': 30, 'default': 14},
            'atr_multiplier': {'type': float, 'min': 1.0, 'max': 5.0, 'default': 2.0},
            'min_signal_strength': {'type': float, 'min': 0.1, 'max': 1.0, 'default': 0.6},
            'stop_loss_type': {'type': str, 'options': ['percentage', 'atr'], 'default': 'atr'},
            'stop_loss_atr_multiplier': {'type': float, 'min': 1.0, 'max': 5.0, 'default': 2.0},
            'take_profit_ratio': {'type': float, 'min': 1.0, 'max': 5.0, 'default': 2.0},
            'volume_confirmation': {'type': bool, 'default': True},
            'trend_filter': {'type': bool, 'default': True},
            'false_breakout_filter': {'type': bool, 'default': True}
        }
    
    def get_description(self) -> str:
        """Get strategy description."""
        return ("Breakout Strategy: Trades breakouts of support/resistance levels with volume confirmation. "
                "Buy when price breaks above resistance with high volume. "
                "Sell when price breaks below support with high volume.")
