"""
RSI + MACD Strategy implementation.
"""

import pandas as pd
from typing import Dict, List, Optional, Any
from decimal import Decimal

from ..core.data_models import OHLCV, StrategySignal, OrderSide, TimeFrame
from .base_strategy import BaseStrategy


class RSIMACDStrategy(BaseStrategy):
    """RSI + MACD combination strategy."""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize RSI + MACD strategy.
        
        Args:
            parameters: Strategy parameters
        """
        default_params = {
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'min_signal_strength': 0.6,
            'stop_loss_type': 'percentage',
            'stop_loss_percentage': 0.02,
            'take_profit_ratio': 2.0
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__("RSI_MACD", default_params)
    
    def generate_signal(self, ohlcv_data: List[OHLCV], indicators: Dict[str, List[float]], 
                       patterns: Dict[str, Any]) -> Optional[StrategySignal]:
        """Generate RSI + MACD trading signal."""
        if len(ohlcv_data) < max(self.parameters['rsi_period'], self.parameters['macd_slow']):
            return None
        
        # Get latest values
        latest_idx = -1
        current_price = ohlcv_data[latest_idx].close
        
        # Get RSI values
        rsi_values = indicators.get('rsi', [])
        if not rsi_values or len(rsi_values) < 2:
            return None
        
        current_rsi = rsi_values[latest_idx]
        prev_rsi = rsi_values[latest_idx - 1]
        
        # Get MACD values
        macd_values = indicators.get('macd', [])
        macd_signal_values = indicators.get('macd_signal', [])
        
        if not macd_values or not macd_signal_values or len(macd_values) < 2:
            return None
        
        current_macd = macd_values[latest_idx]
        prev_macd = macd_values[latest_idx - 1]
        current_macd_signal = macd_signal_values[latest_idx]
        prev_macd_signal = macd_signal_values[latest_idx - 1]
        
        # Check for NaN values
        if any(pd.isna([current_rsi, prev_rsi, current_macd, prev_macd, current_macd_signal, prev_macd_signal])):
            return None
        
        signal_strength = 0.0
        signal_type = None
        
        # Buy signal: RSI oversold + MACD bullish crossover
        if (current_rsi < self.parameters['rsi_oversold'] and 
            prev_macd <= prev_macd_signal and 
            current_macd > current_macd_signal):
            
            signal_type = OrderSide.BUY
            signal_strength = 0.8
            
            # Increase strength if RSI is very oversold
            if current_rsi < 20:
                signal_strength += 0.1
            
            # Increase strength if MACD crossover is strong
            macd_crossover_strength = (current_macd - current_macd_signal) / abs(current_macd_signal) if current_macd_signal != 0 else 0
            if macd_crossover_strength > 0.1:
                signal_strength += 0.1
        
        # Sell signal: RSI overbought + MACD bearish crossover
        elif (current_rsi > self.parameters['rsi_overbought'] and 
              prev_macd >= prev_macd_signal and 
              current_macd < current_macd_signal):
            
            signal_type = OrderSide.SELL
            signal_strength = 0.8
            
            # Increase strength if RSI is very overbought
            if current_rsi > 80:
                signal_strength += 0.1
            
            # Increase strength if MACD crossover is strong
            macd_crossover_strength = (current_macd_signal - current_macd) / abs(current_macd_signal) if current_macd_signal != 0 else 0
            if macd_crossover_strength > 0.1:
                signal_strength += 0.1
        
        # Additional confirmation signals
        if signal_type:
            # Check for divergence
            if self._check_divergence(ohlcv_data, rsi_values, macd_values, signal_type):
                signal_strength += 0.1
            
            # Check for volume confirmation
            if self._check_volume_confirmation(ohlcv_data, signal_type):
                signal_strength += 0.1
            
            # Check for trend confirmation
            if self._check_trend_confirmation(indicators, signal_type):
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
                    'rsi': current_rsi,
                    'macd': current_macd,
                    'macd_signal': current_macd_signal,
                    'macd_histogram': current_macd - current_macd_signal
                },
                metadata={
                    'rsi_period': self.parameters['rsi_period'],
                    'macd_fast': self.parameters['macd_fast'],
                    'macd_slow': self.parameters['macd_slow'],
                    'macd_signal': self.parameters['macd_signal']
                }
            )
        
        return None
    
    def _check_divergence(self, ohlcv_data: List[OHLCV], rsi_values: List[float], 
                         macd_values: List[float], signal_type: OrderSide) -> bool:
        """Check for divergence between price and indicators."""
        if len(ohlcv_data) < 10 or len(rsi_values) < 10 or len(macd_values) < 10:
            return False
        
        # Get recent price and indicator values
        recent_prices = [float(candle.close) for candle in ohlcv_data[-10:]]
        recent_rsi = rsi_values[-10:]
        recent_macd = macd_values[-10:]
        
        if signal_type == OrderSide.BUY:
            # Bullish divergence: price makes lower lows, RSI makes higher lows
            price_trend = recent_prices[-1] - recent_prices[0]
            rsi_trend = recent_rsi[-1] - recent_rsi[0]
            macd_trend = recent_macd[-1] - recent_macd[0]
            
            return price_trend < 0 and rsi_trend > 0 and macd_trend > 0
        
        else:  # SELL
            # Bearish divergence: price makes higher highs, RSI makes lower highs
            price_trend = recent_prices[-1] - recent_prices[0]
            rsi_trend = recent_rsi[-1] - recent_rsi[0]
            macd_trend = recent_macd[-1] - recent_macd[0]
            
            return price_trend > 0 and rsi_trend < 0 and macd_trend < 0
    
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
    
    def _check_trend_confirmation(self, indicators: Dict[str, List[float]], signal_type: OrderSide) -> bool:
        """Check for trend confirmation using moving averages."""
        # Get EMA values
        ema_12 = indicators.get('ema_12', [])
        ema_26 = indicators.get('ema_26', [])
        
        if not ema_12 or not ema_26 or len(ema_12) < 2 or len(ema_26) < 2:
            return False
        
        current_ema_12 = ema_12[-1]
        current_ema_26 = ema_26[-1]
        
        if signal_type == OrderSide.BUY:
            # Bullish trend: EMA 12 > EMA 26
            return current_ema_12 > current_ema_26
        else:  # SELL
            # Bearish trend: EMA 12 < EMA 26
            return current_ema_12 < current_ema_26
    
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
            'rsi_period': {'type': int, 'min': 5, 'max': 50, 'default': 14},
            'rsi_oversold': {'type': int, 'min': 10, 'max': 40, 'default': 30},
            'rsi_overbought': {'type': int, 'min': 60, 'max': 90, 'default': 70},
            'macd_fast': {'type': int, 'min': 5, 'max': 20, 'default': 12},
            'macd_slow': {'type': int, 'min': 20, 'max': 50, 'default': 26},
            'macd_signal': {'type': int, 'min': 5, 'max': 20, 'default': 9},
            'min_signal_strength': {'type': float, 'min': 0.1, 'max': 1.0, 'default': 0.6},
            'stop_loss_type': {'type': str, 'options': ['percentage', 'atr'], 'default': 'percentage'},
            'stop_loss_percentage': {'type': float, 'min': 0.005, 'max': 0.1, 'default': 0.02},
            'take_profit_ratio': {'type': float, 'min': 1.0, 'max': 5.0, 'default': 2.0}
        }
    
    def get_description(self) -> str:
        """Get strategy description."""
        return ("RSI + MACD Strategy: Combines RSI oversold/overbought signals with MACD crossovers. "
                "Buy when RSI < 30 and MACD crosses above signal line. "
                "Sell when RSI > 70 and MACD crosses below signal line.")
