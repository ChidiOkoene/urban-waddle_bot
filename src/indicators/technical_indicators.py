"""
Technical indicators implementation using pure Python (no TA-Lib dependency).
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Union
from decimal import Decimal

from ..core.data_models import OHLCV


class TechnicalIndicators:
    """Technical indicators calculator."""
    
    @staticmethod
    def sma(data: List[float], period: int) -> List[float]:
        """Simple Moving Average."""
        if len(data) < period:
            return [np.nan] * len(data)
        
        result = []
        for i in range(len(data)):
            if i < period - 1:
                result.append(np.nan)
            else:
                result.append(np.mean(data[i - period + 1:i + 1]))
        
        return result
    
    @staticmethod
    def ema(data: List[float], period: int) -> List[float]:
        """Exponential Moving Average."""
        if len(data) < period:
            return [np.nan] * len(data)
        
        alpha = 2.0 / (period + 1)
        result = [np.nan] * len(data)
        
        # Initialize with SMA
        sma_value = np.mean(data[:period])
        result[period - 1] = sma_value
        
        # Calculate EMA
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
        
        return result
    
    @staticmethod
    def macd(data: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[List[float], List[float], List[float]]:
        """MACD (Moving Average Convergence Divergence)."""
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        
        macd_line = []
        for i in range(len(data)):
            if pd.isna(ema_fast[i]) or pd.isna(ema_slow[i]):
                macd_line.append(np.nan)
            else:
                macd_line.append(ema_fast[i] - ema_slow[i])
        
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        
        histogram = []
        for i in range(len(data)):
            if pd.isna(macd_line[i]) or pd.isna(signal_line[i]):
                histogram.append(np.nan)
            else:
                histogram.append(macd_line[i] - signal_line[i])
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def rsi(data: List[float], period: int = 14) -> List[float]:
        """Relative Strength Index."""
        if len(data) < period + 1:
            return [np.nan] * len(data)
        
        deltas = [data[i] - data[i - 1] for i in range(1, len(data))]
        gains = [delta if delta > 0 else 0 for delta in deltas]
        losses = [-delta if delta < 0 else 0 for delta in deltas]
        
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        result = [np.nan] * len(data)
        
        for i in range(period, len(data)):
            if avg_loss == 0:
                result[i] = 100
            else:
                rs = avg_gain / avg_loss
                result[i] = 100 - (100 / (1 + rs))
            
            # Update averages
            if i < len(data) - 1:
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        return result
    
    @staticmethod
    def bollinger_bands(data: List[float], period: int = 20, std_dev: float = 2.0) -> Tuple[List[float], List[float], List[float]]:
        """Bollinger Bands."""
        sma_values = TechnicalIndicators.sma(data, period)
        
        upper_band = []
        lower_band = []
        
        for i in range(len(data)):
            if pd.isna(sma_values[i]):
                upper_band.append(np.nan)
                lower_band.append(np.nan)
            else:
                if i < period - 1:
                    std = np.std(data[:i + 1])
                else:
                    std = np.std(data[i - period + 1:i + 1])
                
                upper_band.append(sma_values[i] + (std_dev * std))
                lower_band.append(sma_values[i] - (std_dev * std))
        
        return upper_band, sma_values, lower_band
    
    @staticmethod
    def atr(high: List[float], low: List[float], close: List[float], period: int = 14) -> List[float]:
        """Average True Range."""
        if len(high) < period + 1:
            return [np.nan] * len(high)
        
        true_ranges = []
        for i in range(1, len(high)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i - 1])
            tr3 = abs(low[i] - close[i - 1])
            true_ranges.append(max(tr1, tr2, tr3))
        
        return TechnicalIndicators.sma(true_ranges, period)
    
    @staticmethod
    def stochastic(high: List[float], low: List[float], close: List[float], k_period: int = 14, d_period: int = 3) -> Tuple[List[float], List[float]]:
        """Stochastic Oscillator."""
        if len(high) < k_period:
            return [np.nan] * len(high), [np.nan] * len(high)
        
        k_values = []
        for i in range(len(high)):
            if i < k_period - 1:
                k_values.append(np.nan)
            else:
                highest_high = max(high[i - k_period + 1:i + 1])
                lowest_low = min(low[i - k_period + 1:i + 1])
                
                if highest_high == lowest_low:
                    k_values.append(50)
                else:
                    k_values.append(100 * (close[i] - lowest_low) / (highest_high - lowest_low))
        
        d_values = TechnicalIndicators.sma(k_values, d_period)
        
        return k_values, d_values
    
    @staticmethod
    def adx(high: List[float], low: List[float], close: List[float], period: int = 14) -> Tuple[List[float], List[float], List[float]]:
        """Average Directional Index."""
        if len(high) < period + 1:
            return [np.nan] * len(high), [np.nan] * len(high), [np.nan] * len(high)
        
        # Calculate True Range
        tr = []
        for i in range(1, len(high)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i - 1])
            tr3 = abs(low[i] - close[i - 1])
            tr.append(max(tr1, tr2, tr3))
        
        # Calculate Directional Movement
        dm_plus = []
        dm_minus = []
        
        for i in range(1, len(high)):
            high_diff = high[i] - high[i - 1]
            low_diff = low[i - 1] - low[i]
            
            if high_diff > low_diff and high_diff > 0:
                dm_plus.append(high_diff)
            else:
                dm_plus.append(0)
            
            if low_diff > high_diff and low_diff > 0:
                dm_minus.append(low_diff)
            else:
                dm_minus.append(0)
        
        # Calculate smoothed values
        atr_values = TechnicalIndicators.sma(tr, period)
        di_plus = []
        di_minus = []
        
        for i in range(len(tr)):
            if i < period - 1:
                di_plus.append(np.nan)
                di_minus.append(np.nan)
            else:
                sma_dm_plus = np.mean(dm_plus[i - period + 1:i + 1])
                sma_dm_minus = np.mean(dm_minus[i - period + 1:i + 1])
                
                if atr_values[i] == 0:
                    di_plus.append(0)
                    di_minus.append(0)
                else:
                    di_plus.append(100 * sma_dm_plus / atr_values[i])
                    di_minus.append(100 * sma_dm_minus / atr_values[i])
        
        # Calculate ADX
        adx_values = []
        for i in range(len(di_plus)):
            if pd.isna(di_plus[i]) or pd.isna(di_minus[i]):
                adx_values.append(np.nan)
            else:
                dx = abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i]) * 100
                adx_values.append(dx)
        
        adx_smoothed = TechnicalIndicators.sma(adx_values, period)
        
        return di_plus, di_minus, adx_smoothed
    
    @staticmethod
    def parabolic_sar(high: List[float], low: List[float], close: List[float], acceleration: float = 0.02, maximum: float = 0.2) -> List[float]:
        """Parabolic SAR."""
        if len(high) < 2:
            return [np.nan] * len(high)
        
        sar = [np.nan] * len(high)
        trend = 1  # 1 for uptrend, -1 for downtrend
        af = acceleration
        ep = high[0] if trend == 1 else low[0]
        
        sar[0] = low[0] if trend == 1 else high[0]
        
        for i in range(1, len(high)):
            prev_sar = sar[i - 1]
            
            if trend == 1:
                sar[i] = prev_sar + af * (ep - prev_sar)
                
                # Check for trend reversal
                if low[i] <= sar[i]:
                    trend = -1
                    sar[i] = ep
                    ep = low[i]
                    af = acceleration
                else:
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + acceleration, maximum)
                    
                    # Ensure SAR doesn't go above previous lows
                    if i > 0:
                        sar[i] = min(sar[i], low[i - 1])
                    if i > 1:
                        sar[i] = min(sar[i], low[i - 2])
            else:
                sar[i] = prev_sar + af * (ep - prev_sar)
                
                # Check for trend reversal
                if high[i] >= sar[i]:
                    trend = 1
                    sar[i] = ep
                    ep = high[i]
                    af = acceleration
                else:
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + acceleration, maximum)
                    
                    # Ensure SAR doesn't go below previous highs
                    if i > 0:
                        sar[i] = max(sar[i], high[i - 1])
                    if i > 1:
                        sar[i] = max(sar[i], high[i - 2])
        
        return sar
    
    @staticmethod
    def ichimoku(high: List[float], low: List[float], close: List[float], 
                 conversion: int = 9, base: int = 26, span_b: int = 52, displacement: int = 26) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
        """Ichimoku Cloud."""
        if len(high) < span_b:
            return [np.nan] * len(high), [np.nan] * len(high), [np.nan] * len(high), [np.nan] * len(high), [np.nan] * len(high)
        
        # Tenkan-sen (Conversion Line)
        tenkan_sen = []
        for i in range(len(high)):
            if i < conversion - 1:
                tenkan_sen.append(np.nan)
            else:
                highest = max(high[i - conversion + 1:i + 1])
                lowest = min(low[i - conversion + 1:i + 1])
                tenkan_sen.append((highest + lowest) / 2)
        
        # Kijun-sen (Base Line)
        kijun_sen = []
        for i in range(len(high)):
            if i < base - 1:
                kijun_sen.append(np.nan)
            else:
                highest = max(high[i - base + 1:i + 1])
                lowest = min(low[i - base + 1:i + 1])
                kijun_sen.append((highest + lowest) / 2)
        
        # Senkou Span A (Leading Span A)
        senkou_span_a = []
        for i in range(len(high)):
            if pd.isna(tenkan_sen[i]) or pd.isna(kijun_sen[i]):
                senkou_span_a.append(np.nan)
            else:
                senkou_span_a.append((tenkan_sen[i] + kijun_sen[i]) / 2)
        
        # Senkou Span B (Leading Span B)
        senkou_span_b = []
        for i in range(len(high)):
            if i < span_b - 1:
                senkou_span_b.append(np.nan)
            else:
                highest = max(high[i - span_b + 1:i + 1])
                lowest = min(low[i - span_b + 1:i + 1])
                senkou_span_b.append((highest + lowest) / 2)
        
        # Chikou Span (Lagging Span)
        chikou_span = []
        for i in range(len(close)):
            if i < displacement:
                chikou_span.append(np.nan)
            else:
                chikou_span.append(close[i - displacement])
        
        return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span
    
    @staticmethod
    def williams_r(high: List[float], low: List[float], close: List[float], period: int = 14) -> List[float]:
        """Williams %R."""
        if len(high) < period:
            return [np.nan] * len(high)
        
        result = []
        for i in range(len(high)):
            if i < period - 1:
                result.append(np.nan)
            else:
                highest_high = max(high[i - period + 1:i + 1])
                lowest_low = min(low[i - period + 1:i + 1])
                
                if highest_high == lowest_low:
                    result.append(-50)
                else:
                    result.append(-100 * (highest_high - close[i]) / (highest_high - lowest_low))
        
        return result
    
    @staticmethod
    def cci(high: List[float], low: List[float], close: List[float], period: int = 20) -> List[float]:
        """Commodity Channel Index."""
        if len(high) < period:
            return [np.nan] * len(high)
        
        result = []
        for i in range(len(high)):
            if i < period - 1:
                result.append(np.nan)
            else:
                typical_price = [(high[j] + low[j] + close[j]) / 3 for j in range(i - period + 1, i + 1)]
                sma_tp = np.mean(typical_price)
                mean_deviation = np.mean([abs(tp - sma_tp) for tp in typical_price])
                
                if mean_deviation == 0:
                    result.append(0)
                else:
                    result.append((typical_price[-1] - sma_tp) / (0.015 * mean_deviation))
        
        return result
    
    @staticmethod
    def roc(data: List[float], period: int = 10) -> List[float]:
        """Rate of Change."""
        if len(data) < period + 1:
            return [np.nan] * len(data)
        
        result = []
        for i in range(len(data)):
            if i < period:
                result.append(np.nan)
            else:
                if data[i - period] == 0:
                    result.append(0)
                else:
                    result.append((data[i] - data[i - period]) / data[i - period] * 100)
        
        return result
    
    @staticmethod
    def obv(close: List[float], volume: List[float]) -> List[float]:
        """On-Balance Volume."""
        if len(close) != len(volume):
            return [np.nan] * len(close)
        
        result = [0]
        for i in range(1, len(close)):
            if close[i] > close[i - 1]:
                result.append(result[-1] + volume[i])
            elif close[i] < close[i - 1]:
                result.append(result[-1] - volume[i])
            else:
                result.append(result[-1])
        
        return result
    
    @staticmethod
    def vwap(high: List[float], low: List[float], close: List[float], volume: List[float]) -> List[float]:
        """Volume Weighted Average Price."""
        if len(high) != len(volume):
            return [np.nan] * len(high)
        
        result = []
        cumulative_volume = 0
        cumulative_pv = 0
        
        for i in range(len(high)):
            typical_price = (high[i] + low[i] + close[i]) / 3
            pv = typical_price * volume[i]
            
            cumulative_pv += pv
            cumulative_volume += volume[i]
            
            if cumulative_volume == 0:
                result.append(typical_price)
            else:
                result.append(cumulative_pv / cumulative_volume)
        
        return result
    
    @staticmethod
    def mfi(high: List[float], low: List[float], close: List[float], volume: List[float], period: int = 14) -> List[float]:
        """Money Flow Index."""
        if len(high) < period + 1:
            return [np.nan] * len(high)
        
        typical_prices = [(high[i] + low[i] + close[i]) / 3 for i in range(len(high))]
        
        positive_flow = []
        negative_flow = []
        
        for i in range(1, len(typical_prices)):
            if typical_prices[i] > typical_prices[i - 1]:
                positive_flow.append(typical_prices[i] * volume[i])
                negative_flow.append(0)
            elif typical_prices[i] < typical_prices[i - 1]:
                positive_flow.append(0)
                negative_flow.append(typical_prices[i] * volume[i])
            else:
                positive_flow.append(0)
                negative_flow.append(0)
        
        result = []
        for i in range(len(high)):
            if i < period:
                result.append(np.nan)
            else:
                sum_positive = sum(positive_flow[i - period:i])
                sum_negative = sum(negative_flow[i - period:i])
                
                if sum_negative == 0:
                    result.append(100)
                else:
                    mf_ratio = sum_positive / sum_negative
                    result.append(100 - (100 / (1 + mf_ratio)))
        
        return result
    
    @staticmethod
    def calculate_all_indicators(ohlcv_data: List[OHLCV]) -> dict:
        """Calculate all indicators for OHLCV data."""
        if not ohlcv_data:
            return {}
        
        # Extract data
        high = [float(candle.high) for candle in ohlcv_data]
        low = [float(candle.low) for candle in ohlcv_data]
        close = [float(candle.close) for candle in ohlcv_data]
        volume = [float(candle.volume) for candle in ohlcv_data]
        
        indicators = {}
        
        # Trend indicators
        indicators['sma_20'] = TechnicalIndicators.sma(close, 20)
        indicators['sma_50'] = TechnicalIndicators.sma(close, 50)
        indicators['ema_12'] = TechnicalIndicators.ema(close, 12)
        indicators['ema_26'] = TechnicalIndicators.ema(close, 26)
        
        macd_line, macd_signal, macd_histogram = TechnicalIndicators.macd(close)
        indicators['macd'] = macd_line
        indicators['macd_signal'] = macd_signal
        indicators['macd_histogram'] = macd_histogram
        
        # Momentum indicators
        indicators['rsi'] = TechnicalIndicators.rsi(close)
        k_values, d_values = TechnicalIndicators.stochastic(high, low, close)
        indicators['stoch_k'] = k_values
        indicators['stoch_d'] = d_values
        indicators['williams_r'] = TechnicalIndicators.williams_r(high, low, close)
        indicators['cci'] = TechnicalIndicators.cci(high, low, close)
        indicators['roc'] = TechnicalIndicators.roc(close)
        
        # Volatility indicators
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(close)
        indicators['bb_upper'] = bb_upper
        indicators['bb_middle'] = bb_middle
        indicators['bb_lower'] = bb_lower
        indicators['atr'] = TechnicalIndicators.atr(high, low, close)
        
        # Trend strength
        di_plus, di_minus, adx = TechnicalIndicators.adx(high, low, close)
        indicators['adx'] = adx
        indicators['di_plus'] = di_plus
        indicators['di_minus'] = di_minus
        
        # Other indicators
        indicators['parabolic_sar'] = TechnicalIndicators.parabolic_sar(high, low, close)
        
        # Ichimoku
        tenkan, kijun, span_a, span_b, chikou = TechnicalIndicators.ichimoku(high, low, close)
        indicators['ichimoku_tenkan'] = tenkan
        indicators['ichimoku_kijun'] = kijun
        indicators['ichimoku_span_a'] = span_a
        indicators['ichimoku_span_b'] = span_b
        indicators['ichimoku_chikou'] = chikou
        
        # Volume indicators
        indicators['obv'] = TechnicalIndicators.obv(close, volume)
        indicators['vwap'] = TechnicalIndicators.vwap(high, low, close, volume)
        indicators['mfi'] = TechnicalIndicators.mfi(high, low, close, volume)
        
        return indicators
