"""
Momentum Strategy implementation.
"""

import pandas as pd
from typing import Dict, List, Optional, Any
from decimal import Decimal

from ..core.data_models import OHLCV, StrategySignal, OrderSide, TimeFrame
from .base_strategy import BaseStrategy


class MomentumStrategy(BaseStrategy):
    """Multi-timeframe momentum strategy."""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize Momentum strategy.
        
        Args:
            parameters: Strategy parameters
        """
        default_params = {
            'momentum_timeframes': ['1h', '4h', '1d'],
            'momentum_period': 14,
            'momentum_threshold': 0.6,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'min_signal_strength': 0.6,
            'stop_loss_type': 'percentage',
            'stop_loss_percentage': 0.02,
            'take_profit_ratio': 2.0,
            'trend_filter': True,
            'volume_confirmation': True
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__("Momentum", default_params)
    
    def generate_signal(self, ohlcv_data: List[OHLCV], indicators: Dict[str, List[float]], 
                       patterns: Dict[str, Any]) -> Optional[StrategySignal]:
        """Generate momentum trading signal."""
        if len(ohlcv_data) < self.parameters['momentum_period'] + 5:
            return None
        
        current_price = ohlcv_data[-1].close
        
        # Calculate momentum indicators
        momentum_score = self._calculate_momentum_score(ohlcv_data, indicators)
        
        if momentum_score is None:
            return None
        
        signal_strength = 0.0
        signal_type = None
        
        # Buy signal: Strong positive momentum
        if momentum_score > self.parameters['momentum_threshold']:
            signal_type = OrderSide.BUY
            signal_strength = 0.6
            
            # Increase strength based on momentum magnitude
            if momentum_score > 0.8:
                signal_strength += 0.2
            
            # Check for RSI confirmation
            rsi_values = indicators.get('rsi', [])
            if rsi_values and not pd.isna(rsi_values[-1]):
                current_rsi = rsi_values[-1]
                if current_rsi < self.parameters['rsi_overbought']:
                    signal_strength += 0.1
        
        # Sell signal: Strong negative momentum
        elif momentum_score < -self.parameters['momentum_threshold']:
            signal_type = OrderSide.SELL
            signal_strength = 0.6
            
            # Increase strength based on momentum magnitude
            if momentum_score < -0.8:
                signal_strength += 0.2
            
            # Check for RSI confirmation
            rsi_values = indicators.get('rsi', [])
            if rsi_values and not pd.isna(rsi_values[-1]):
                current_rsi = rsi_values[-1]
                if current_rsi > self.parameters['rsi_oversold']:
                    signal_strength += 0.1
        
        # Additional confirmation signals
        if signal_type:
            # Check for MACD confirmation
            if self._check_macd_confirmation(indicators, signal_type):
                signal_strength += 0.1
            
            # Check for trend confirmation
            if self.parameters['trend_filter'] and self._check_trend_confirmation(indicators, signal_type):
                signal_strength += 0.1
            
            # Check for volume confirmation
            if self.parameters['volume_confirmation'] and self._check_volume_confirmation(ohlcv_data, signal_type):
                signal_strength += 0.1
            
            # Check for momentum divergence
            if self._check_momentum_divergence(ohlcv_data, indicators, signal_type):
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
                    'momentum_score': momentum_score,
                    'rsi': indicators.get('rsi', [0])[-1] if indicators.get('rsi') else 0,
                    'macd': indicators.get('macd', [0])[-1] if indicators.get('macd') else 0,
                    'macd_signal': indicators.get('macd_signal', [0])[-1] if indicators.get('macd_signal') else 0,
                    'price_momentum': self._calculate_price_momentum(ohlcv_data)
                },
                metadata={
                    'momentum_period': self.parameters['momentum_period'],
                    'momentum_threshold': self.parameters['momentum_threshold'],
                    'momentum_timeframes': self.parameters['momentum_timeframes']
                }
            )
        
        return None
    
    def _calculate_momentum_score(self, ohlcv_data: List[OHLCV], indicators: Dict[str, List[float]]) -> Optional[float]:
        """Calculate overall momentum score."""
        momentum_period = self.parameters['momentum_period']
        
        if len(ohlcv_data) < momentum_period:
            return None
        
        # Calculate price momentum
        price_momentum = self._calculate_price_momentum(ohlcv_data)
        
        # Calculate RSI momentum
        rsi_momentum = self._calculate_rsi_momentum(indicators)
        
        # Calculate MACD momentum
        macd_momentum = self._calculate_macd_momentum(indicators)
        
        # Calculate volume momentum
        volume_momentum = self._calculate_volume_momentum(ohlcv_data)
        
        # Combine momentum scores
        momentum_score = (
            price_momentum * 0.4 +
            rsi_momentum * 0.3 +
            macd_momentum * 0.2 +
            volume_momentum * 0.1
        )
        
        return momentum_score
    
    def _calculate_price_momentum(self, ohlcv_data: List[OHLCV]) -> float:
        """Calculate price momentum."""
        momentum_period = self.parameters['momentum_period']
        
        if len(ohlcv_data) < momentum_period:
            return 0.0
        
        current_price = float(ohlcv_data[-1].close)
        past_price = float(ohlcv_data[-momentum_period].close)
        
        if past_price == 0:
            return 0.0
        
        # Calculate rate of change
        roc = (current_price - past_price) / past_price
        
        # Normalize to -1 to 1 range
        return max(-1.0, min(1.0, roc * 10))  # Scale factor of 10
    
    def _calculate_rsi_momentum(self, indicators: Dict[str, List[float]]) -> float:
        """Calculate RSI momentum."""
        rsi_values = indicators.get('rsi', [])
        
        if not rsi_values or len(rsi_values) < 2:
            return 0.0
        
        current_rsi = rsi_values[-1]
        prev_rsi = rsi_values[-2]
        
        if pd.isna(current_rsi) or pd.isna(prev_rsi):
            return 0.0
        
        # RSI momentum: positive if RSI is rising and not overbought
        rsi_change = current_rsi - prev_rsi
        
        # Normalize based on RSI position
        if current_rsi > 70:  # Overbought
            return -0.5
        elif current_rsi < 30:  # Oversold
            return 0.5
        else:
            return rsi_change / 10  # Scale factor
    
    def _calculate_macd_momentum(self, indicators: Dict[str, List[float]]) -> float:
        """Calculate MACD momentum."""
        macd_values = indicators.get('macd', [])
        macd_signal_values = indicators.get('macd_signal', [])
        
        if not macd_values or not macd_signal_values or len(macd_values) < 2:
            return 0.0
        
        current_macd = macd_values[-1]
        current_signal = macd_signal_values[-1]
        
        if pd.isna(current_macd) or pd.isna(current_signal):
            return 0.0
        
        # MACD momentum: positive if MACD is above signal and rising
        macd_histogram = current_macd - current_signal
        
        # Normalize to -1 to 1 range
        return max(-1.0, min(1.0, macd_histogram * 100))  # Scale factor
    
    def _calculate_volume_momentum(self, ohlcv_data: List[OHLCV]) -> float:
        """Calculate volume momentum."""
        if len(ohlcv_data) < 5:
            return 0.0
        
        # Calculate average volume
        recent_volumes = [float(candle.volume) for candle in ohlcv_data[-5:]]
        avg_volume = sum(recent_volumes) / len(recent_volumes)
        current_volume = float(ohlcv_data[-1].volume)
        
        # Volume momentum: positive if current volume is above average
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Normalize to -1 to 1 range
        return max(-1.0, min(1.0, (volume_ratio - 1) * 2))
    
    def _check_macd_confirmation(self, indicators: Dict[str, List[float]], signal_type: OrderSide) -> bool:
        """Check for MACD confirmation."""
        macd_values = indicators.get('macd', [])
        macd_signal_values = indicators.get('macd_signal', [])
        
        if not macd_values or not macd_signal_values:
            return True  # No MACD filter if not available
        
        current_macd = macd_values[-1]
        current_signal = macd_signal_values[-1]
        
        if pd.isna(current_macd) or pd.isna(current_signal):
            return True
        
        if signal_type == OrderSide.BUY:
            return current_macd > current_signal
        else:  # SELL
            return current_macd < current_signal
    
    def _check_trend_confirmation(self, indicators: Dict[str, List[float]], signal_type: OrderSide) -> bool:
        """Check for trend confirmation using moving averages."""
        # Get EMA values
        ema_20 = indicators.get('sma_20', [])
        ema_50 = indicators.get('sma_50', [])
        
        if not ema_20 or not ema_50:
            return True  # No trend filter if indicators not available
        
        current_ema_20 = ema_20[-1]
        current_ema_50 = ema_50[-1]
        
        if signal_type == OrderSide.BUY:
            return current_ema_20 > current_ema_50
        else:  # SELL
            return current_ema_20 < current_ema_50
    
    def _check_volume_confirmation(self, ohlcv_data: List[OHLCV], signal_type: OrderSide) -> bool:
        """Check for volume confirmation."""
        if len(ohlcv_data) < 5:
            return True  # No volume filter if not enough data
        
        # Calculate average volume
        recent_volumes = [float(candle.volume) for candle in ohlcv_data[-5:]]
        avg_volume = sum(recent_volumes) / len(recent_volumes)
        current_volume = float(ohlcv_data[-1].volume)
        
        # Volume should be above average for momentum confirmation
        return current_volume > avg_volume * 1.1
    
    def _check_momentum_divergence(self, ohlcv_data: List[OHLCV], indicators: Dict[str, List[float]], 
                                 signal_type: OrderSide) -> bool:
        """Check for momentum divergence."""
        if len(ohlcv_data) < 10:
            return False
        
        # Get recent price and momentum data
        recent_prices = [float(candle.close) for candle in ohlcv_data[-10:]]
        recent_momentum = self._calculate_price_momentum(ohlcv_data[-10:])
        
        if signal_type == OrderSide.BUY:
            # Bullish divergence: price makes lower lows, momentum makes higher lows
            price_trend = recent_prices[-1] - recent_prices[0]
            return price_trend < 0 and recent_momentum > 0
        else:  # SELL
            # Bearish divergence: price makes higher highs, momentum makes lower highs
            price_trend = recent_prices[-1] - recent_prices[0]
            return price_trend > 0 and recent_momentum < 0
    
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
            'momentum_timeframes': {'type': list, 'default': ['1h', '4h', '1d']},
            'momentum_period': {'type': int, 'min': 5, 'max': 30, 'default': 14},
            'momentum_threshold': {'type': float, 'min': 0.1, 'max': 1.0, 'default': 0.6},
            'rsi_period': {'type': int, 'min': 5, 'max': 30, 'default': 14},
            'rsi_oversold': {'type': int, 'min': 10, 'max': 40, 'default': 30},
            'rsi_overbought': {'type': int, 'min': 60, 'max': 90, 'default': 70},
            'macd_fast': {'type': int, 'min': 5, 'max': 20, 'default': 12},
            'macd_slow': {'type': int, 'min': 20, 'max': 50, 'default': 26},
            'macd_signal': {'type': int, 'min': 5, 'max': 20, 'default': 9},
            'min_signal_strength': {'type': float, 'min': 0.1, 'max': 1.0, 'default': 0.6},
            'stop_loss_type': {'type': str, 'options': ['percentage', 'atr'], 'default': 'percentage'},
            'stop_loss_percentage': {'type': float, 'min': 0.005, 'max': 0.1, 'default': 0.02},
            'take_profit_ratio': {'type': float, 'min': 1.0, 'max': 5.0, 'default': 2.0},
            'trend_filter': {'type': bool, 'default': True},
            'volume_confirmation': {'type': bool, 'default': True}
        }
    
    def get_description(self) -> str:
        """Get strategy description."""
        return ("Momentum Strategy: Uses multiple momentum indicators across different timeframes. "
                "Buy when momentum is strongly positive across price, RSI, MACD, and volume. "
                "Sell when momentum is strongly negative across all indicators.")
