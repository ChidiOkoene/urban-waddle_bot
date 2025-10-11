"""
Bollinger Bands Mean Reversion Strategy implementation.
"""

import pandas as pd
from typing import Dict, List, Optional, Any
from decimal import Decimal

from ..core.data_models import OHLCV, StrategySignal, OrderSide, TimeFrame
from .base_strategy import BaseStrategy


class BollingerMeanReversionStrategy(BaseStrategy):
    """Bollinger Bands mean reversion strategy."""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize Bollinger Bands strategy.
        
        Args:
            parameters: Strategy parameters
        """
        default_params = {
            'bb_period': 20,
            'bb_std': 2.0,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'min_signal_strength': 0.6,
            'stop_loss_type': 'percentage',
            'stop_loss_percentage': 0.02,
            'take_profit_ratio': 2.0,
            'volume_confirmation': True,
            'trend_filter': True
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__("Bollinger_Mean_Reversion", default_params)
    
    def generate_signal(self, ohlcv_data: List[OHLCV], indicators: Dict[str, List[float]], 
                       patterns: Dict[str, Any]) -> Optional[StrategySignal]:
        """Generate Bollinger Bands mean reversion signal."""
        if len(ohlcv_data) < self.parameters['bb_period']:
            return None
        
        # Get latest values
        latest_idx = -1
        current_price = ohlcv_data[latest_idx].close
        
        # Get Bollinger Bands values
        bb_upper = indicators.get('bb_upper', [])
        bb_middle = indicators.get('bb_middle', [])
        bb_lower = indicators.get('bb_lower', [])
        
        if not all([bb_upper, bb_middle, bb_lower]):
            return None
        
        current_bb_upper = bb_upper[latest_idx]
        current_bb_middle = bb_middle[latest_idx]
        current_bb_lower = bb_lower[latest_idx]
        
        # Get RSI values
        rsi_values = indicators.get('rsi', [])
        if not rsi_values:
            return None
        
        current_rsi = rsi_values[latest_idx]
        
        # Check for NaN values
        if any(pd.isna([current_bb_upper, current_bb_middle, current_bb_lower, current_rsi])):
            return None
        
        signal_strength = 0.0
        signal_type = None
        
        # Calculate Bollinger Band position (0 = lower band, 1 = upper band)
        bb_position = (current_price - current_bb_lower) / (current_bb_upper - current_bb_lower)
        
        # Buy signal: Price touches lower band + RSI oversold
        if (bb_position <= 0.1 and current_rsi < self.parameters['rsi_oversold']):
            signal_type = OrderSide.BUY
            signal_strength = 0.7
            
            # Increase strength if price is well below lower band
            if bb_position <= 0.05:
                signal_strength += 0.1
            
            # Increase strength if RSI is very oversold
            if current_rsi < 20:
                signal_strength += 0.1
            
            # Increase strength if price is moving away from lower band
            if len(ohlcv_data) > 1:
                prev_price = ohlcv_data[latest_idx - 1].close
                if current_price > prev_price:
                    signal_strength += 0.1
        
        # Sell signal: Price touches upper band + RSI overbought
        elif (bb_position >= 0.9 and current_rsi > self.parameters['rsi_overbought']):
            signal_type = OrderSide.SELL
            signal_strength = 0.7
            
            # Increase strength if price is well above upper band
            if bb_position >= 0.95:
                signal_strength += 0.1
            
            # Increase strength if RSI is very overbought
            if current_rsi > 80:
                signal_strength += 0.1
            
            # Increase strength if price is moving away from upper band
            if len(ohlcv_data) > 1:
                prev_price = ohlcv_data[latest_idx - 1].close
                if current_price < prev_price:
                    signal_strength += 0.1
        
        # Additional confirmation signals
        if signal_type:
            # Check for volume confirmation
            if self.parameters['volume_confirmation'] and self._check_volume_confirmation(ohlcv_data, signal_type):
                signal_strength += 0.1
            
            # Check for trend filter
            if self.parameters['trend_filter'] and self._check_trend_filter(indicators, signal_type):
                signal_strength += 0.1
            
            # Check for squeeze pattern
            if self._check_squeeze_pattern(bb_upper, bb_middle, bb_lower, latest_idx):
                signal_strength += 0.1
            
            # Check for divergence
            if self._check_divergence(ohlcv_data, bb_position, current_rsi, signal_type):
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
                    'bb_upper': current_bb_upper,
                    'bb_middle': current_bb_middle,
                    'bb_lower': current_bb_lower,
                    'bb_position': bb_position,
                    'rsi': current_rsi,
                    'bb_width': (current_bb_upper - current_bb_lower) / current_bb_middle
                },
                metadata={
                    'bb_period': self.parameters['bb_period'],
                    'bb_std': self.parameters['bb_std'],
                    'rsi_period': self.parameters['rsi_period']
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
    
    def _check_trend_filter(self, indicators: Dict[str, List[float]], signal_type: OrderSide) -> bool:
        """Check trend filter using moving averages."""
        # Get EMA values
        ema_20 = indicators.get('sma_20', [])
        ema_50 = indicators.get('sma_50', [])
        
        if not ema_20 or not ema_50:
            return True  # No trend filter if indicators not available
        
        current_ema_20 = ema_20[-1]
        current_ema_50 = ema_50[-1]
        
        if signal_type == OrderSide.BUY:
            # For buy signals, prefer when price is above both EMAs (uptrend)
            return current_ema_20 > current_ema_50
        else:  # SELL
            # For sell signals, prefer when price is below both EMAs (downtrend)
            return current_ema_20 < current_ema_50
    
    def _check_squeeze_pattern(self, bb_upper: List[float], bb_middle: List[float], 
                             bb_lower: List[float], current_idx: int) -> bool:
        """Check for Bollinger Band squeeze pattern."""
        if current_idx < 10:
            return False
        
        # Calculate BB width for recent periods
        recent_widths = []
        for i in range(current_idx - 10, current_idx + 1):
            if i >= 0 and i < len(bb_upper):
                width = (bb_upper[i] - bb_lower[i]) / bb_middle[i]
                recent_widths.append(width)
        
        if len(recent_widths) < 10:
            return False
        
        # Check if BB width is contracting (squeeze)
        current_width = recent_widths[-1]
        avg_width = sum(recent_widths[:-1]) / len(recent_widths[:-1])
        
        # Squeeze if current width is significantly smaller than average
        return current_width < avg_width * 0.8
    
    def _check_divergence(self, ohlcv_data: List[OHLCV], bb_position: float, 
                         current_rsi: float, signal_type: OrderSide) -> bool:
        """Check for divergence between price and indicators."""
        if len(ohlcv_data) < 10:
            return False
        
        # Get recent price and RSI values
        recent_prices = [float(candle.close) for candle in ohlcv_data[-10:]]
        recent_rsi = [current_rsi]  # Would need RSI history for full divergence check
        
        if signal_type == OrderSide.BUY:
            # Bullish divergence: price makes lower lows, RSI makes higher lows
            price_trend = recent_prices[-1] - recent_prices[0]
            return price_trend < 0  # Simplified check
        
        else:  # SELL
            # Bearish divergence: price makes higher highs, RSI makes lower highs
            price_trend = recent_prices[-1] - recent_prices[0]
            return price_trend > 0  # Simplified check
    
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
            'bb_period': {'type': int, 'min': 10, 'max': 50, 'default': 20},
            'bb_std': {'type': float, 'min': 1.0, 'max': 3.0, 'default': 2.0},
            'rsi_period': {'type': int, 'min': 5, 'max': 30, 'default': 14},
            'rsi_oversold': {'type': int, 'min': 10, 'max': 40, 'default': 30},
            'rsi_overbought': {'type': int, 'min': 60, 'max': 90, 'default': 70},
            'min_signal_strength': {'type': float, 'min': 0.1, 'max': 1.0, 'default': 0.6},
            'stop_loss_type': {'type': str, 'options': ['percentage', 'atr'], 'default': 'percentage'},
            'stop_loss_percentage': {'type': float, 'min': 0.005, 'max': 0.1, 'default': 0.02},
            'take_profit_ratio': {'type': float, 'min': 1.0, 'max': 5.0, 'default': 2.0},
            'volume_confirmation': {'type': bool, 'default': True},
            'trend_filter': {'type': bool, 'default': True}
        }
    
    def get_description(self) -> str:
        """Get strategy description."""
        return ("Bollinger Bands Mean Reversion Strategy: Trades bounces off Bollinger Bands with RSI confirmation. "
                "Buy when price touches lower band and RSI < 30. "
                "Sell when price touches upper band and RSI > 70.")
