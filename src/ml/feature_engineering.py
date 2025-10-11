"""
Feature engineering for machine learning models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

from ..core.data_models import OHLCV
from ..indicators.technical_indicators import TechnicalIndicators
from ..indicators.pattern_recognition import PatternRecognition


class FeatureEngineer:
    """Feature engineering for ML models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize feature engineer.
        
        Args:
            config: Feature engineering configuration
        """
        self.config = config
        
        # Feature parameters
        self.lookback_periods = config.get('lookback_periods', [5, 10, 20, 50])
        self.feature_scaling = config.get('feature_scaling', 'standard')  # standard, minmax, robust
        self.feature_selection = config.get('feature_selection', True)
        self.n_features = config.get('n_features', 50)
        
        # Feature categories
        self.price_features = config.get('price_features', True)
        self.volume_features = config.get('volume_features', True)
        self.technical_features = config.get('technical_features', True)
        self.pattern_features = config.get('pattern_features', True)
        self.time_features = config.get('time_features', True)
        self.cross_features = config.get('cross_features', True)
        
        # Scalers
        self.scaler = self._get_scaler()
        self.feature_selector = None
        self.feature_names = []
        self.is_fitted = False
        
        # Feature cache
        self.feature_cache = {}
        self.cache_size = config.get('cache_size', 1000)
    
    def create_features(self, ohlcv_data: List[OHLCV], indicators: Optional[Dict] = None, 
                       patterns: Optional[Dict] = None) -> np.ndarray:
        """
        Create features from OHLCV data.
        
        Args:
            ohlcv_data: List of OHLCV candles
            indicators: Technical indicators
            patterns: Pattern recognition results
            
        Returns:
            Feature matrix
        """
        if len(ohlcv_data) < max(self.lookback_periods):
            return np.array([])
        
        features = []
        
        # Price features
        if self.price_features:
            price_features = self._create_price_features(ohlcv_data)
            features.extend(price_features)
        
        # Volume features
        if self.volume_features:
            volume_features = self._create_volume_features(ohlcv_data)
            features.extend(volume_features)
        
        # Technical indicator features
        if self.technical_features and indicators:
            technical_features = self._create_technical_features(ohlcv_data, indicators)
            features.extend(technical_features)
        
        # Pattern features
        if self.pattern_features and patterns:
            pattern_features = self._create_pattern_features(ohlcv_data, patterns)
            features.extend(pattern_features)
        
        # Time features
        if self.time_features:
            time_features = self._create_time_features(ohlcv_data)
            features.extend(time_features)
        
        # Cross features
        if self.cross_features:
            cross_features = self._create_cross_features(ohlcv_data, indicators)
            features.extend(cross_features)
        
        # Convert to numpy array
        feature_array = np.array(features)
        
        # Handle NaN values
        feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return feature_array
    
    def _create_price_features(self, ohlcv_data: List[OHLCV]) -> List[float]:
        """Create price-based features."""
        features = []
        
        # Extract price data
        prices = [float(candle.close) for candle in ohlcv_data]
        highs = [float(candle.high) for candle in ohlcv_data]
        lows = [float(candle.low) for candle in ohlcv_data]
        opens = [float(candle.open) for candle in ohlcv_data]
        
        # Price ratios
        current_price = prices[-1]
        features.append(current_price)
        
        # Price changes
        for period in self.lookback_periods:
            if len(prices) > period:
                price_change = (current_price - prices[-period-1]) / prices[-period-1]
                features.append(price_change)
        
        # High-Low ratios
        for period in self.lookback_periods:
            if len(highs) > period:
                period_high = max(highs[-period:])
                period_low = min(lows[-period:])
                if period_low > 0:
                    hl_ratio = (period_high - period_low) / period_low
                    features.append(hl_ratio)
        
        # Price volatility
        for period in self.lookback_periods:
            if len(prices) > period:
                period_prices = prices[-period:]
                volatility = np.std(period_prices) / np.mean(period_prices) if np.mean(period_prices) > 0 else 0
                features.append(volatility)
        
        # Price momentum
        for period in self.lookback_periods:
            if len(prices) > period * 2:
                recent_avg = np.mean(prices[-period:])
                previous_avg = np.mean(prices[-period*2:-period])
                momentum = (recent_avg - previous_avg) / previous_avg if previous_avg > 0 else 0
                features.append(momentum)
        
        # Support/Resistance levels
        if len(prices) > 20:
            recent_highs = highs[-20:]
            recent_lows = lows[-20:]
            
            resistance_level = max(recent_highs)
            support_level = min(recent_lows)
            
            resistance_distance = (resistance_level - current_price) / current_price
            support_distance = (current_price - support_level) / current_price
            
            features.extend([resistance_distance, support_distance])
        
        return features
    
    def _create_volume_features(self, ohlcv_data: List[OHLCV]) -> List[float]:
        """Create volume-based features."""
        features = []
        
        volumes = [float(candle.volume) for candle in ohlcv_data]
        current_volume = volumes[-1]
        
        # Volume ratios
        for period in self.lookback_periods:
            if len(volumes) > period:
                avg_volume = np.mean(volumes[-period:])
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                features.append(volume_ratio)
        
        # Volume trend
        for period in self.lookback_periods:
            if len(volumes) > period * 2:
                recent_avg = np.mean(volumes[-period:])
                previous_avg = np.mean(volumes[-period*2:-period])
                volume_trend = (recent_avg - previous_avg) / previous_avg if previous_avg > 0 else 0
                features.append(volume_trend)
        
        # Volume volatility
        for period in self.lookback_periods:
            if len(volumes) > period:
                period_volumes = volumes[-period:]
                volume_volatility = np.std(period_volumes) / np.mean(period_volumes) if np.mean(period_volumes) > 0 else 0
                features.append(volume_volatility)
        
        # Price-Volume correlation
        if len(ohlcv_data) > 10:
            prices = [float(candle.close) for candle in ohlcv_data[-10:]]
            volumes = [float(candle.volume) for candle in ohlcv_data[-10:]]
            
            if len(prices) == len(volumes) and len(prices) > 1:
                correlation = np.corrcoef(prices, volumes)[0, 1]
                features.append(correlation if not np.isnan(correlation) else 0.0)
        
        return features
    
    def _create_technical_features(self, ohlcv_data: List[OHLCV], indicators: Dict) -> List[float]:
        """Create technical indicator features."""
        features = []
        
        # RSI features
        if 'rsi' in indicators and indicators['rsi']:
            rsi_values = indicators['rsi']
            if len(rsi_values) > 0 and not np.isnan(rsi_values[-1]):
                current_rsi = rsi_values[-1]
                features.append(current_rsi / 100.0)  # Normalize to 0-1
                
                # RSI momentum
                if len(rsi_values) > 1:
                    rsi_momentum = rsi_values[-1] - rsi_values[-2]
                    features.append(rsi_momentum / 100.0)
        
        # MACD features
        if 'macd' in indicators and 'macd_signal' in indicators:
            macd_values = indicators['macd']
            macd_signal_values = indicators['macd_signal']
            
            if (macd_values and macd_signal_values and 
                len(macd_values) > 0 and len(macd_signal_values) > 0):
                
                current_macd = macd_values[-1]
                current_signal = macd_signal_values[-1]
                
                if not np.isnan(current_macd) and not np.isnan(current_signal):
                    features.append(current_macd)
                    features.append(current_signal)
                    features.append(current_macd - current_signal)  # Histogram
        
        # Bollinger Bands features
        if all(key in indicators for key in ['bb_upper', 'bb_middle', 'bb_lower']):
            bb_upper = indicators['bb_upper']
            bb_middle = indicators['bb_middle']
            bb_lower = indicators['bb_lower']
            
            if (bb_upper and bb_middle and bb_lower and 
                len(bb_upper) > 0 and len(bb_middle) > 0 and len(bb_lower) > 0):
                
                current_price = float(ohlcv_data[-1].close)
                current_upper = bb_upper[-1]
                current_middle = bb_middle[-1]
                current_lower = bb_lower[-1]
                
                if not any(np.isnan([current_upper, current_middle, current_lower])):
                    # BB position
                    bb_position = (current_price - current_lower) / (current_upper - current_lower)
                    features.append(bb_position)
                    
                    # BB width
                    bb_width = (current_upper - current_lower) / current_middle
                    features.append(bb_width)
        
        # Moving average features
        for ma_type in ['sma_20', 'sma_50', 'ema_12', 'ema_26']:
            if ma_type in indicators and indicators[ma_type]:
                ma_values = indicators[ma_type]
                if len(ma_values) > 0 and not np.isnan(ma_values[-1]):
                    current_price = float(ohlcv_data[-1].close)
                    current_ma = ma_values[-1]
                    
                    # Price vs MA ratio
                    ma_ratio = current_price / current_ma
                    features.append(ma_ratio)
                    
                    # MA slope
                    if len(ma_values) > 1:
                        ma_slope = (ma_values[-1] - ma_values[-2]) / ma_values[-2]
                        features.append(ma_slope)
        
        # ATR features
        if 'atr' in indicators and indicators['atr']:
            atr_values = indicators['atr']
            if len(atr_values) > 0 and not np.isnan(atr_values[-1]):
                current_price = float(ohlcv_data[-1].close)
                current_atr = atr_values[-1]
                
                # ATR ratio
                atr_ratio = current_atr / current_price
                features.append(atr_ratio)
        
        return features
    
    def _create_pattern_features(self, ohlcv_data: List[OHLCV], patterns: Dict) -> List[float]:
        """Create pattern-based features."""
        features = []
        
        # Candlestick patterns
        candlestick_patterns = patterns.get('candlestick_patterns', [])
        if candlestick_patterns:
            # Count patterns by type
            pattern_counts = {}
            for pattern in candlestick_patterns:
                pattern_type = pattern.get('type', 'unknown')
                pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
            
            # Add pattern features
            for pattern_type in ['hammer', 'shooting_star', 'doji', 'bullish_engulfing', 'bearish_engulfing']:
                count = pattern_counts.get(pattern_type, 0)
                features.append(count)
        
        # Support/Resistance levels
        support_levels = patterns.get('support_levels', [])
        resistance_levels = patterns.get('resistance_levels', [])
        
        if support_levels and resistance_levels:
            current_price = float(ohlcv_data[-1].close)
            
            # Distance to nearest support/resistance
            if support_levels:
                nearest_support = max([s for s in support_levels if s < current_price], default=0)
                support_distance = (current_price - nearest_support) / current_price if nearest_support > 0 else 1.0
                features.append(support_distance)
            
            if resistance_levels:
                nearest_resistance = min([r for r in resistance_levels if r > current_price], default=current_price * 2)
                resistance_distance = (nearest_resistance - current_price) / current_price
                features.append(resistance_distance)
        
        return features
    
    def _create_time_features(self, ohlcv_data: List[OHLCV]) -> List[float]:
        """Create time-based features."""
        features = []
        
        if not ohlcv_data:
            return features
        
        current_time = ohlcv_data[-1].timestamp
        
        # Hour of day (for intraday patterns)
        hour = current_time.hour
        features.append(hour / 24.0)  # Normalize to 0-1
        
        # Day of week
        weekday = current_time.weekday()
        features.append(weekday / 6.0)  # Normalize to 0-1
        
        # Day of month
        day_of_month = current_time.day
        features.append(day_of_month / 31.0)  # Normalize to 0-1
        
        # Month
        month = current_time.month
        features.append(month / 12.0)  # Normalize to 0-1
        
        # Time since market open (simplified)
        market_open_hour = 9  # Assume 9 AM market open
        hours_since_open = (hour - market_open_hour) % 24
        features.append(hours_since_open / 24.0)
        
        return features
    
    def _create_cross_features(self, ohlcv_data: List[OHLCV], indicators: Optional[Dict]) -> List[float]:
        """Create cross-features (interactions between indicators)."""
        features = []
        
        if not indicators:
            return features
        
        # RSI vs MACD
        if ('rsi' in indicators and 'macd' in indicators and 
            indicators['rsi'] and indicators['macd']):
            
            rsi_values = indicators['rsi']
            macd_values = indicators['macd']
            
            if (len(rsi_values) > 0 and len(macd_values) > 0 and 
                not np.isnan(rsi_values[-1]) and not np.isnan(macd_values[-1])):
                
                rsi_macd_interaction = (rsi_values[-1] / 100.0) * macd_values[-1]
                features.append(rsi_macd_interaction)
        
        # Volume vs Price momentum
        if len(ohlcv_data) > 10:
            prices = [float(candle.close) for candle in ohlcv_data[-10:]]
            volumes = [float(candle.volume) for candle in ohlcv_data[-10:]]
            
            price_momentum = (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0
            volume_momentum = (volumes[-1] - volumes[0]) / volumes[0] if volumes[0] > 0 else 0
            
            momentum_interaction = price_momentum * volume_momentum
            features.append(momentum_interaction)
        
        # Bollinger Bands vs RSI
        if (all(key in indicators for key in ['bb_upper', 'bb_middle', 'bb_lower', 'rsi']) and
            all(indicators[key] for key in ['bb_upper', 'bb_middle', 'bb_lower', 'rsi'])):
            
            bb_upper = indicators['bb_upper'][-1]
            bb_middle = indicators['bb_middle'][-1]
            bb_lower = indicators['bb_lower'][-1]
            rsi = indicators['rsi'][-1]
            
            if not any(np.isnan([bb_upper, bb_middle, bb_lower, rsi])):
                current_price = float(ohlcv_data[-1].close)
                bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
                rsi_normalized = rsi / 100.0
                
                bb_rsi_interaction = bb_position * rsi_normalized
                features.append(bb_rsi_interaction)
        
        return features
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit scaler and transform features."""
        if X.size == 0:
            return X
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Feature selection
        if self.feature_selection and y is not None and len(y) > 0:
            self.feature_selector = SelectKBest(
                score_func=f_regression, 
                k=min(self.n_features, X_scaled.shape[1])
            )
            X_scaled = self.feature_selector.fit_transform(X_scaled, y)
        
        self.is_fitted = True
        return X_scaled
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using fitted scaler."""
        if not self.is_fitted:
            raise ValueError("Feature engineer must be fitted before transform")
        
        if X.size == 0:
            return X
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Apply feature selection
        if self.feature_selector is not None:
            X_scaled = self.feature_selector.transform(X_scaled)
        
        return X_scaled
    
    def _get_scaler(self):
        """Get scaler based on configuration."""
        if self.feature_scaling == 'standard':
            return StandardScaler()
        elif self.feature_scaling == 'minmax':
            return MinMaxScaler()
        elif self.feature_scaling == 'robust':
            return RobustScaler()
        else:
            return StandardScaler()
    
    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        if not self.feature_names:
            self.feature_names = self._generate_feature_names()
        
        if self.feature_selector is not None:
            selected_indices = self.feature_selector.get_support(indices=True)
            return [self.feature_names[i] for i in selected_indices]
        
        return self.feature_names
    
    def _generate_feature_names(self) -> List[str]:
        """Generate feature names."""
        names = []
        
        if self.price_features:
            names.extend(['current_price'])
            for period in self.lookback_periods:
                names.extend([
                    f'price_change_{period}',
                    f'hl_ratio_{period}',
                    f'price_volatility_{period}',
                    f'price_momentum_{period}'
                ])
            names.extend(['resistance_distance', 'support_distance'])
        
        if self.volume_features:
            for period in self.lookback_periods:
                names.extend([
                    f'volume_ratio_{period}',
                    f'volume_trend_{period}',
                    f'volume_volatility_{period}'
                ])
            names.append('price_volume_correlation')
        
        if self.technical_features:
            names.extend([
                'rsi', 'rsi_momentum',
                'macd', 'macd_signal', 'macd_histogram',
                'bb_position', 'bb_width'
            ])
            for ma_type in ['sma_20', 'sma_50', 'ema_12', 'ema_26']:
                names.extend([f'{ma_type}_ratio', f'{ma_type}_slope'])
            names.append('atr_ratio')
        
        if self.pattern_features:
            names.extend([
                'hammer_count', 'shooting_star_count', 'doji_count',
                'bullish_engulfing_count', 'bearish_engulfing_count',
                'support_distance', 'resistance_distance'
            ])
        
        if self.time_features:
            names.extend([
                'hour', 'weekday', 'day_of_month', 'month', 'hours_since_open'
            ])
        
        if self.cross_features:
            names.extend([
                'rsi_macd_interaction', 'momentum_interaction', 'bb_rsi_interaction'
            ])
        
        return names
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance scores."""
        if self.feature_selector is not None:
            return self.feature_selector.scores_
        return None
