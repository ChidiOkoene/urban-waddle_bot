"""
Pattern recognition for chart patterns and candlestick patterns.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from decimal import Decimal

from ..core.data_models import OHLCV


class PatternRecognition:
    """Pattern recognition for trading signals."""
    
    @staticmethod
    def find_support_resistance(high: List[float], low: List[float], close: List[float], 
                               window: int = 20, min_touches: int = 2) -> Tuple[List[float], List[float]]:
        """Find support and resistance levels."""
        if len(high) < window:
            return [], []
        
        support_levels = []
        resistance_levels = []
        
        for i in range(window, len(high) - window):
            # Check for resistance (local maximum)
            is_resistance = True
            for j in range(i - window, i + window + 1):
                if j != i and high[j] >= high[i]:
                    is_resistance = False
                    break
            
            if is_resistance:
                # Count touches
                touches = 0
                for j in range(len(high)):
                    if abs(high[j] - high[i]) / high[i] < 0.01:  # Within 1%
                        touches += 1
                
                if touches >= min_touches:
                    resistance_levels.append(high[i])
            
            # Check for support (local minimum)
            is_support = True
            for j in range(i - window, i + window + 1):
                if j != i and low[j] <= low[i]:
                    is_support = False
                    break
            
            if is_support:
                # Count touches
                touches = 0
                for j in range(len(low)):
                    if abs(low[j] - low[i]) / low[i] < 0.01:  # Within 1%
                        touches += 1
                
                if touches >= min_touches:
                    support_levels.append(low[i])
        
        return support_levels, resistance_levels
    
    @staticmethod
    def detect_trend_lines(high: List[float], low: List[float], close: List[float], 
                          min_points: int = 3) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """Detect trend lines."""
        uptrend_lines = []
        downtrend_lines = []
        
        # Find uptrend lines (connecting lows)
        for i in range(min_points, len(low)):
            for j in range(i + min_points, len(low)):
                # Calculate slope
                slope = (low[j] - low[i]) / (j - i)
                
                # Check if other points align with this trend line
                aligned_points = 0
                for k in range(i, j + 1):
                    expected_y = low[i] + slope * (k - i)
                    if abs(low[k] - expected_y) / low[k] < 0.02:  # Within 2%
                        aligned_points += 1
                
                if aligned_points >= min_points:
                    uptrend_lines.append((slope, low[i] - slope * i))
        
        # Find downtrend lines (connecting highs)
        for i in range(min_points, len(high)):
            for j in range(i + min_points, len(high)):
                # Calculate slope
                slope = (high[j] - high[i]) / (j - i)
                
                # Check if other points align with this trend line
                aligned_points = 0
                for k in range(i, j + 1):
                    expected_y = high[i] + slope * (k - i)
                    if abs(high[k] - expected_y) / high[k] < 0.02:  # Within 2%
                        aligned_points += 1
                
                if aligned_points >= min_points:
                    downtrend_lines.append((slope, high[i] - slope * i))
        
        return uptrend_lines, downtrend_lines
    
    @staticmethod
    def detect_head_and_shoulders(high: List[float], low: List[float], close: List[float], 
                                 window: int = 20) -> List[Dict]:
        """Detect Head and Shoulders pattern."""
        patterns = []
        
        if len(high) < window * 3:
            return patterns
        
        for i in range(window, len(high) - window * 2):
            # Find three peaks
            left_shoulder_idx = i
            head_idx = i + window
            right_shoulder_idx = i + window * 2
            
            if head_idx >= len(high) or right_shoulder_idx >= len(high):
                continue
            
            # Check if head is higher than shoulders
            if (high[head_idx] > high[left_shoulder_idx] and 
                high[head_idx] > high[right_shoulder_idx]):
                
                # Check if shoulders are roughly equal height
                shoulder_diff = abs(high[left_shoulder_idx] - high[right_shoulder_idx]) / high[left_shoulder_idx]
                
                if shoulder_diff < 0.05:  # Within 5%
                    # Check for neckline
                    neckline = min(low[left_shoulder_idx], low[right_shoulder_idx])
                    
                    patterns.append({
                        'type': 'head_and_shoulders',
                        'left_shoulder': left_shoulder_idx,
                        'head': head_idx,
                        'right_shoulder': right_shoulder_idx,
                        'neckline': neckline,
                        'target': neckline - (high[head_idx] - neckline)
                    })
        
        return patterns
    
    @staticmethod
    def detect_double_top_bottom(high: List[float], low: List[float], close: List[float], 
                                window: int = 20, tolerance: float = 0.02) -> Tuple[List[Dict], List[Dict]]:
        """Detect Double Top and Double Bottom patterns."""
        double_tops = []
        double_bottoms = []
        
        if len(high) < window * 2:
            return double_tops, double_bottoms
        
        # Find double tops
        for i in range(window, len(high) - window):
            for j in range(i + window, len(high) - window):
                # Check if peaks are similar height
                height_diff = abs(high[i] - high[j]) / high[i]
                
                if height_diff < tolerance:
                    # Check if there's a valley between peaks
                    valley_idx = i + (j - i) // 2
                    valley_low = min(low[i:j+1])
                    
                    if valley_low < high[i] * 0.95:  # Valley is at least 5% below peaks
                        double_tops.append({
                            'type': 'double_top',
                            'first_peak': i,
                            'second_peak': j,
                            'valley': valley_low,
                            'target': valley_low - (high[i] - valley_low)
                        })
        
        # Find double bottoms
        for i in range(window, len(low) - window):
            for j in range(i + window, len(low) - window):
                # Check if troughs are similar depth
                depth_diff = abs(low[i] - low[j]) / low[i]
                
                if depth_diff < tolerance:
                    # Check if there's a peak between troughs
                    peak_idx = i + (j - i) // 2
                    peak_high = max(high[i:j+1])
                    
                    if peak_high > low[i] * 1.05:  # Peak is at least 5% above troughs
                        double_bottoms.append({
                            'type': 'double_bottom',
                            'first_trough': i,
                            'second_trough': j,
                            'peak': peak_high,
                            'target': peak_high + (peak_high - low[i])
                        })
        
        return double_tops, double_bottoms
    
    @staticmethod
    def detect_candlestick_patterns(ohlcv_data: List[OHLCV]) -> List[Dict]:
        """Detect candlestick patterns."""
        patterns = []
        
        if len(ohlcv_data) < 3:
            return patterns
        
        for i in range(2, len(ohlcv_data)):
            current = ohlcv_data[i]
            previous = ohlcv_data[i - 1]
            prev_prev = ohlcv_data[i - 2]
            
            # Hammer
            if PatternRecognition._is_hammer(current):
                patterns.append({
                    'type': 'hammer',
                    'index': i,
                    'signal': 'bullish',
                    'strength': PatternRecognition._calculate_pattern_strength(current)
                })
            
            # Shooting Star
            if PatternRecognition._is_shooting_star(current):
                patterns.append({
                    'type': 'shooting_star',
                    'index': i,
                    'signal': 'bearish',
                    'strength': PatternRecognition._calculate_pattern_strength(current)
                })
            
            # Doji
            if PatternRecognition._is_doji(current):
                patterns.append({
                    'type': 'doji',
                    'index': i,
                    'signal': 'neutral',
                    'strength': 0.5
                })
            
            # Engulfing patterns
            if PatternRecognition._is_bullish_engulfing(previous, current):
                patterns.append({
                    'type': 'bullish_engulfing',
                    'index': i,
                    'signal': 'bullish',
                    'strength': PatternRecognition._calculate_engulfing_strength(previous, current)
                })
            
            if PatternRecognition._is_bearish_engulfing(previous, current):
                patterns.append({
                    'type': 'bearish_engulfing',
                    'index': i,
                    'signal': 'bearish',
                    'strength': PatternRecognition._calculate_engulfing_strength(previous, current)
                })
            
            # Three pattern sequences
            if i >= 2:
                # Morning Star
                if PatternRecognition._is_morning_star(prev_prev, previous, current):
                    patterns.append({
                        'type': 'morning_star',
                        'index': i,
                        'signal': 'bullish',
                        'strength': 0.8
                    })
                
                # Evening Star
                if PatternRecognition._is_evening_star(prev_prev, previous, current):
                    patterns.append({
                        'type': 'evening_star',
                        'index': i,
                        'signal': 'bearish',
                        'strength': 0.8
                    })
        
        return patterns
    
    @staticmethod
    def _is_hammer(candle: OHLCV) -> bool:
        """Check if candle is a hammer."""
        body_size = abs(float(candle.close) - float(candle.open))
        total_range = float(candle.high) - float(candle.low)
        
        if total_range == 0:
            return False
        
        # Small body, long lower shadow, short upper shadow
        body_ratio = body_size / total_range
        lower_shadow = float(candle.open) - float(candle.low) if float(candle.close) > float(candle.open) else float(candle.close) - float(candle.low)
        upper_shadow = float(candle.high) - float(candle.close) if float(candle.close) > float(candle.open) else float(candle.high) - float(candle.open)
        
        lower_ratio = lower_shadow / total_range
        upper_ratio = upper_shadow / total_range
        
        return (body_ratio < 0.3 and lower_ratio > 0.6 and upper_ratio < 0.1)
    
    @staticmethod
    def _is_shooting_star(candle: OHLCV) -> bool:
        """Check if candle is a shooting star."""
        body_size = abs(float(candle.close) - float(candle.open))
        total_range = float(candle.high) - float(candle.low)
        
        if total_range == 0:
            return False
        
        # Small body, long upper shadow, short lower shadow
        body_ratio = body_size / total_range
        lower_shadow = float(candle.open) - float(candle.low) if float(candle.close) > float(candle.open) else float(candle.close) - float(candle.low)
        upper_shadow = float(candle.high) - float(candle.close) if float(candle.close) > float(candle.open) else float(candle.high) - float(candle.open)
        
        lower_ratio = lower_shadow / total_range
        upper_ratio = upper_shadow / total_range
        
        return (body_ratio < 0.3 and upper_ratio > 0.6 and lower_ratio < 0.1)
    
    @staticmethod
    def _is_doji(candle: OHLCV) -> bool:
        """Check if candle is a doji."""
        body_size = abs(float(candle.close) - float(candle.open))
        total_range = float(candle.high) - float(candle.low)
        
        if total_range == 0:
            return False
        
        body_ratio = body_size / total_range
        return body_ratio < 0.1
    
    @staticmethod
    def _is_bullish_engulfing(prev: OHLCV, curr: OHLCV) -> bool:
        """Check if current candle is bullish engulfing."""
        prev_body = abs(float(prev.close) - float(prev.open))
        curr_body = abs(float(curr.close) - float(curr.open))
        
        # Previous candle is bearish, current is bullish
        is_bearish_prev = float(prev.close) < float(prev.open)
        is_bullish_curr = float(curr.close) > float(curr.open)
        
        # Current candle engulfs previous
        engulfs = (float(curr.open) < float(prev.close) and 
                  float(curr.close) > float(prev.open))
        
        return is_bearish_prev and is_bullish_curr and engulfs
    
    @staticmethod
    def _is_bearish_engulfing(prev: OHLCV, curr: OHLCV) -> bool:
        """Check if current candle is bearish engulfing."""
        prev_body = abs(float(prev.close) - float(prev.open))
        curr_body = abs(float(curr.close) - float(curr.open))
        
        # Previous candle is bullish, current is bearish
        is_bullish_prev = float(prev.close) > float(prev.open)
        is_bearish_curr = float(curr.close) < float(curr.open)
        
        # Current candle engulfs previous
        engulfs = (float(curr.open) > float(prev.close) and 
                  float(curr.close) < float(prev.open))
        
        return is_bullish_prev and is_bearish_curr and engulfs
    
    @staticmethod
    def _is_morning_star(prev_prev: OHLCV, prev: OHLCV, curr: OHLCV) -> bool:
        """Check if three candles form a morning star pattern."""
        # First candle is bearish
        is_bearish_first = float(prev_prev.close) < float(prev_prev.open)
        
        # Second candle has small body (star)
        second_body = abs(float(prev.close) - float(prev.open))
        second_range = float(prev.high) - float(prev.low)
        is_star = second_body / second_range < 0.3 if second_range > 0 else False
        
        # Third candle is bullish and closes above first candle's midpoint
        is_bullish_third = float(curr.close) > float(curr.open)
        closes_above_midpoint = float(curr.close) > (float(prev_prev.open) + float(prev_prev.close)) / 2
        
        return is_bearish_first and is_star and is_bullish_third and closes_above_midpoint
    
    @staticmethod
    def _is_evening_star(prev_prev: OHLCV, prev: OHLCV, curr: OHLCV) -> bool:
        """Check if three candles form an evening star pattern."""
        # First candle is bullish
        is_bullish_first = float(prev_prev.close) > float(prev_prev.open)
        
        # Second candle has small body (star)
        second_body = abs(float(prev.close) - float(prev.open))
        second_range = float(prev.high) - float(prev.low)
        is_star = second_body / second_range < 0.3 if second_range > 0 else False
        
        # Third candle is bearish and closes below first candle's midpoint
        is_bearish_third = float(curr.close) < float(curr.open)
        closes_below_midpoint = float(curr.close) < (float(prev_prev.open) + float(prev_prev.close)) / 2
        
        return is_bullish_first and is_star and is_bearish_third and closes_below_midpoint
    
    @staticmethod
    def _calculate_pattern_strength(candle: OHLCV) -> float:
        """Calculate pattern strength (0-1)."""
        body_size = abs(float(candle.close) - float(candle.open))
        total_range = float(candle.high) - float(candle.low)
        
        if total_range == 0:
            return 0.0
        
        # Stronger patterns have smaller bodies relative to range
        strength = 1.0 - (body_size / total_range)
        return min(max(strength, 0.0), 1.0)
    
    @staticmethod
    def _calculate_engulfing_strength(prev: OHLCV, curr: OHLCV) -> float:
        """Calculate engulfing pattern strength."""
        prev_body = abs(float(prev.close) - float(prev.open))
        curr_body = abs(float(curr.close) - float(curr.open))
        
        if prev_body == 0:
            return 0.0
        
        # Strength based on how much the current candle engulfs the previous
        engulfing_ratio = curr_body / prev_body
        return min(engulfing_ratio, 2.0) / 2.0  # Normalize to 0-1
    
    @staticmethod
    def detect_all_patterns(ohlcv_data: List[OHLCV]) -> Dict:
        """Detect all patterns for OHLCV data."""
        if not ohlcv_data:
            return {}
        
        # Extract data
        high = [float(candle.high) for candle in ohlcv_data]
        low = [float(candle.low) for candle in ohlcv_data]
        close = [float(candle.close) for candle in ohlcv_data]
        
        patterns = {}
        
        # Chart patterns
        support_levels, resistance_levels = PatternRecognition.find_support_resistance(high, low, close)
        patterns['support_levels'] = support_levels
        patterns['resistance_levels'] = resistance_levels
        
        uptrend_lines, downtrend_lines = PatternRecognition.detect_trend_lines(high, low, close)
        patterns['uptrend_lines'] = uptrend_lines
        patterns['downtrend_lines'] = downtrend_lines
        
        head_shoulders = PatternRecognition.detect_head_and_shoulders(high, low, close)
        patterns['head_and_shoulders'] = head_shoulders
        
        double_tops, double_bottoms = PatternRecognition.detect_double_top_bottom(high, low, close)
        patterns['double_tops'] = double_tops
        patterns['double_bottoms'] = double_bottoms
        
        # Candlestick patterns
        candlestick_patterns = PatternRecognition.detect_candlestick_patterns(ohlcv_data)
        patterns['candlestick_patterns'] = candlestick_patterns
        
        return patterns
