"""
Tests for technical indicators.
"""

import pytest
import numpy as np
from decimal import Decimal
from datetime import datetime, timezone

from src.indicators.technical_indicators import TechnicalIndicators
from src.indicators.pattern_recognition import PatternRecognition
from src.core.data_models import OHLCV, TimeFrame


class TestTechnicalIndicators:
    """Test technical indicators calculations."""
    
    def setup_method(self):
        """Setup test data."""
        # Create sample OHLCV data
        self.sample_data = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
        self.sample_high = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]
        self.sample_low = [99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        self.sample_close = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
        self.sample_volume = [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
        
        # Create OHLCV objects
        self.ohlcv_data = []
        for i in range(len(self.sample_data)):
            self.ohlcv_data.append(OHLCV(
                timestamp=datetime.now(timezone.utc),
                open=Decimal(str(self.sample_data[i])),
                high=Decimal(str(self.sample_high[i])),
                low=Decimal(str(self.sample_low[i])),
                close=Decimal(str(self.sample_close[i])),
                volume=Decimal(str(self.sample_volume[i])),
                symbol="BTC/USDT",
                timeframe=TimeFrame.H1
            ))
    
    def test_sma(self):
        """Test Simple Moving Average calculation."""
        sma_5 = TechnicalIndicators.sma(self.sample_data, 5)
        
        # Check that first 4 values are NaN
        assert all(np.isnan(sma_5[:4]))
        
        # Check that 5th value is correct
        expected_5th = np.mean(self.sample_data[:5])
        assert abs(sma_5[4] - expected_5th) < 1e-10
        
        # Check that all values after 5th are calculated
        assert not any(np.isnan(sma_5[4:]))
    
    def test_ema(self):
        """Test Exponential Moving Average calculation."""
        ema_5 = TechnicalIndicators.ema(self.sample_data, 5)
        
        # Check that first 4 values are NaN
        assert all(np.isnan(ema_5[:4]))
        
        # Check that 5th value equals SMA
        sma_5 = TechnicalIndicators.sma(self.sample_data, 5)
        assert abs(ema_5[4] - sma_5[4]) < 1e-10
        
        # Check that EMA is calculated for all subsequent values
        assert not any(np.isnan(ema_5[4:]))
    
    def test_macd(self):
        """Test MACD calculation."""
        macd_line, signal_line, histogram = TechnicalIndicators.macd(self.sample_data)
        
        # Check that all arrays have same length
        assert len(macd_line) == len(self.sample_data)
        assert len(signal_line) == len(self.sample_data)
        assert len(histogram) == len(self.sample_data)
        
        # Check that histogram equals macd - signal
        for i in range(len(histogram)):
            if not (np.isnan(histogram[i]) or np.isnan(macd_line[i]) or np.isnan(signal_line[i])):
                assert abs(histogram[i] - (macd_line[i] - signal_line[i])) < 1e-10
    
    def test_rsi(self):
        """Test RSI calculation."""
        rsi_values = TechnicalIndicators.rsi(self.sample_data)
        
        # Check that first 14 values are NaN
        assert all(np.isnan(rsi_values[:14]))
        
        # Check that RSI values are between 0 and 100
        for i in range(14, len(rsi_values)):
            if not np.isnan(rsi_values[i]):
                assert 0 <= rsi_values[i] <= 100
    
    def test_bollinger_bands(self):
        """Test Bollinger Bands calculation."""
        upper, middle, lower = TechnicalIndicators.bollinger_bands(self.sample_data)
        
        # Check that all arrays have same length
        assert len(upper) == len(self.sample_data)
        assert len(middle) == len(self.sample_data)
        assert len(lower) == len(self.sample_data)
        
        # Check that upper > middle > lower
        for i in range(len(self.sample_data)):
            if not (np.isnan(upper[i]) or np.isnan(middle[i]) or np.isnan(lower[i])):
                assert upper[i] > middle[i] > lower[i]
    
    def test_atr(self):
        """Test ATR calculation."""
        atr_values = TechnicalIndicators.atr(self.sample_high, self.sample_low, self.sample_close)
        
        # Check that first value is NaN
        assert np.isnan(atr_values[0])
        
        # Check that ATR values are positive
        for i in range(1, len(atr_values)):
            if not np.isnan(atr_values[i]):
                assert atr_values[i] > 0
    
    def test_stochastic(self):
        """Test Stochastic Oscillator calculation."""
        k_values, d_values = TechnicalIndicators.stochastic(
            self.sample_high, self.sample_low, self.sample_close
        )
        
        # Check that all arrays have same length
        assert len(k_values) == len(self.sample_data)
        assert len(d_values) == len(self.sample_data)
        
        # Check that K and D values are between 0 and 100
        for i in range(len(k_values)):
            if not np.isnan(k_values[i]):
                assert 0 <= k_values[i] <= 100
            if not np.isnan(d_values[i]):
                assert 0 <= d_values[i] <= 100
    
    def test_adx(self):
        """Test ADX calculation."""
        di_plus, di_minus, adx = TechnicalIndicators.adx(
            self.sample_high, self.sample_low, self.sample_close
        )
        
        # Check that all arrays have same length
        assert len(di_plus) == len(self.sample_data)
        assert len(di_minus) == len(self.sample_data)
        assert len(adx) == len(self.sample_data)
        
        # Check that ADX values are between 0 and 100
        for i in range(len(adx)):
            if not np.isnan(adx[i]):
                assert 0 <= adx[i] <= 100
    
    def test_parabolic_sar(self):
        """Test Parabolic SAR calculation."""
        sar_values = TechnicalIndicators.parabolic_sar(
            self.sample_high, self.sample_low, self.sample_close
        )
        
        # Check that all values are calculated
        assert len(sar_values) == len(self.sample_data)
        assert not any(np.isnan(sar_values))
        
        # Check that SAR values are within price range
        for i in range(len(sar_values)):
            assert self.sample_low[i] <= sar_values[i] <= self.sample_high[i]
    
    def test_ichimoku(self):
        """Test Ichimoku calculation."""
        tenkan, kijun, span_a, span_b, chikou = TechnicalIndicators.ichimoku(
            self.sample_high, self.sample_low, self.sample_close
        )
        
        # Check that all arrays have same length
        assert len(tenkan) == len(self.sample_data)
        assert len(kijun) == len(self.sample_data)
        assert len(span_a) == len(self.sample_data)
        assert len(span_b) == len(self.sample_data)
        assert len(chikou) == len(self.sample_data)
    
    def test_williams_r(self):
        """Test Williams %R calculation."""
        wr_values = TechnicalIndicators.williams_r(
            self.sample_high, self.sample_low, self.sample_close
        )
        
        # Check that all arrays have same length
        assert len(wr_values) == len(self.sample_data)
        
        # Check that Williams %R values are between -100 and 0
        for i in range(len(wr_values)):
            if not np.isnan(wr_values[i]):
                assert -100 <= wr_values[i] <= 0
    
    def test_cci(self):
        """Test CCI calculation."""
        cci_values = TechnicalIndicators.cci(
            self.sample_high, self.sample_low, self.sample_close
        )
        
        # Check that all arrays have same length
        assert len(cci_values) == len(self.sample_data)
    
    def test_roc(self):
        """Test ROC calculation."""
        roc_values = TechnicalIndicators.roc(self.sample_data, 5)
        
        # Check that first 5 values are NaN
        assert all(np.isnan(roc_values[:5]))
        
        # Check that all arrays have same length
        assert len(roc_values) == len(self.sample_data)
    
    def test_obv(self):
        """Test OBV calculation."""
        obv_values = TechnicalIndicators.obv(self.sample_close, self.sample_volume)
        
        # Check that all arrays have same length
        assert len(obv_values) == len(self.sample_data)
        
        # Check that first value is 0
        assert obv_values[0] == 0
        
        # Check that OBV is cumulative
        for i in range(1, len(obv_values)):
            if self.sample_close[i] > self.sample_close[i-1]:
                assert obv_values[i] > obv_values[i-1]
            elif self.sample_close[i] < self.sample_close[i-1]:
                assert obv_values[i] < obv_values[i-1]
            else:
                assert obv_values[i] == obv_values[i-1]
    
    def test_vwap(self):
        """Test VWAP calculation."""
        vwap_values = TechnicalIndicators.vwap(
            self.sample_high, self.sample_low, self.sample_close, self.sample_volume
        )
        
        # Check that all arrays have same length
        assert len(vwap_values) == len(self.sample_data)
        
        # Check that VWAP values are positive
        for i in range(len(vwap_values)):
            assert vwap_values[i] > 0
    
    def test_mfi(self):
        """Test MFI calculation."""
        mfi_values = TechnicalIndicators.mfi(
            self.sample_high, self.sample_low, self.sample_close, self.sample_volume
        )
        
        # Check that all arrays have same length
        assert len(mfi_values) == len(self.sample_data)
        
        # Check that MFI values are between 0 and 100
        for i in range(len(mfi_values)):
            if not np.isnan(mfi_values[i]):
                assert 0 <= mfi_values[i] <= 100
    
    def test_calculate_all_indicators(self):
        """Test calculate_all_indicators method."""
        indicators = TechnicalIndicators.calculate_all_indicators(self.ohlcv_data)
        
        # Check that all expected indicators are present
        expected_indicators = [
            'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'macd', 'macd_signal', 'macd_histogram',
            'rsi', 'stoch_k', 'stoch_d', 'williams_r', 'cci', 'roc',
            'bb_upper', 'bb_middle', 'bb_lower', 'atr',
            'adx', 'di_plus', 'di_minus',
            'parabolic_sar',
            'ichimoku_tenkan', 'ichimoku_kijun', 'ichimoku_span_a', 
            'ichimoku_span_b', 'ichimoku_chikou',
            'obv', 'vwap', 'mfi'
        ]
        
        for indicator in expected_indicators:
            assert indicator in indicators
            assert len(indicators[indicator]) == len(self.ohlcv_data)


class TestPatternRecognition:
    """Test pattern recognition functionality."""
    
    def setup_method(self):
        """Setup test data."""
        # Create sample OHLCV data with patterns
        self.sample_data = []
        for i in range(20):
            self.sample_data.append(OHLCV(
                timestamp=datetime.now(timezone.utc),
                open=Decimal(str(100 + i)),
                high=Decimal(str(101 + i)),
                low=Decimal(str(99 + i)),
                close=Decimal(str(100 + i)),
                volume=Decimal(str(1000 + i * 100)),
                symbol="BTC/USDT",
                timeframe=TimeFrame.H1
            ))
    
    def test_find_support_resistance(self):
        """Test support and resistance level detection."""
        high = [float(candle.high) for candle in self.sample_data]
        low = [float(candle.low) for candle in self.sample_data]
        close = [float(candle.close) for candle in self.sample_data]
        
        support, resistance = PatternRecognition.find_support_resistance(high, low, close)
        
        # Check that results are lists
        assert isinstance(support, list)
        assert isinstance(resistance, list)
        
        # Check that support levels are lower than resistance levels
        if support and resistance:
            assert max(support) < min(resistance)
    
    def test_detect_trend_lines(self):
        """Test trend line detection."""
        high = [float(candle.high) for candle in self.sample_data]
        low = [float(candle.low) for candle in self.sample_data]
        close = [float(candle.close) for candle in self.sample_data]
        
        uptrend_lines, downtrend_lines = PatternRecognition.detect_trend_lines(high, low, close)
        
        # Check that results are lists of tuples
        assert isinstance(uptrend_lines, list)
        assert isinstance(downtrend_lines, list)
        
        for line in uptrend_lines:
            assert isinstance(line, tuple)
            assert len(line) == 2
    
    def test_detect_head_and_shoulders(self):
        """Test head and shoulders pattern detection."""
        high = [float(candle.high) for candle in self.sample_data]
        low = [float(candle.low) for candle in self.sample_data]
        close = [float(candle.close) for candle in self.sample_data]
        
        patterns = PatternRecognition.detect_head_and_shoulders(high, low, close)
        
        # Check that result is a list
        assert isinstance(patterns, list)
        
        # Check pattern structure if any patterns found
        for pattern in patterns:
            assert 'type' in pattern
            assert pattern['type'] == 'head_and_shoulders'
            assert 'left_shoulder' in pattern
            assert 'head' in pattern
            assert 'right_shoulder' in pattern
    
    def test_detect_double_top_bottom(self):
        """Test double top and bottom pattern detection."""
        high = [float(candle.high) for candle in self.sample_data]
        low = [float(candle.low) for candle in self.sample_data]
        close = [float(candle.close) for candle in self.sample_data]
        
        double_tops, double_bottoms = PatternRecognition.detect_double_top_bottom(high, low, close)
        
        # Check that results are lists
        assert isinstance(double_tops, list)
        assert isinstance(double_bottoms, list)
        
        # Check pattern structure if any patterns found
        for pattern in double_tops:
            assert 'type' in pattern
            assert pattern['type'] == 'double_top'
        
        for pattern in double_bottoms:
            assert 'type' in pattern
            assert pattern['type'] == 'double_bottom'
    
    def test_detect_candlestick_patterns(self):
        """Test candlestick pattern detection."""
        patterns = PatternRecognition.detect_candlestick_patterns(self.sample_data)
        
        # Check that result is a list
        assert isinstance(patterns, list)
        
        # Check pattern structure if any patterns found
        for pattern in patterns:
            assert 'type' in pattern
            assert 'index' in pattern
            assert 'signal' in pattern
            assert 'strength' in pattern
            assert pattern['signal'] in ['bullish', 'bearish', 'neutral']
            assert 0 <= pattern['strength'] <= 1
    
    def test_detect_all_patterns(self):
        """Test detect_all_patterns method."""
        patterns = PatternRecognition.detect_all_patterns(self.sample_data)
        
        # Check that all expected pattern types are present
        expected_patterns = [
            'support_levels', 'resistance_levels',
            'uptrend_lines', 'downtrend_lines',
            'head_and_shoulders', 'double_tops', 'double_bottoms',
            'candlestick_patterns'
        ]
        
        for pattern_type in expected_patterns:
            assert pattern_type in patterns
    
    def test_hammer_pattern(self):
        """Test hammer pattern detection."""
        # Create a hammer candle
        hammer_candle = OHLCV(
            timestamp=datetime.now(timezone.utc),
            open=Decimal("100"),
            high=Decimal("101"),
            low=Decimal("95"),  # Long lower shadow
            close=Decimal("100.5"),
            volume=Decimal("1000"),
            symbol="BTC/USDT",
            timeframe=TimeFrame.H1
        )
        
        is_hammer = PatternRecognition._is_hammer(hammer_candle)
        assert is_hammer
    
    def test_shooting_star_pattern(self):
        """Test shooting star pattern detection."""
        # Create a shooting star candle
        shooting_star = OHLCV(
            timestamp=datetime.now(timezone.utc),
            open=Decimal("100"),
            high=Decimal("105"),  # Long upper shadow
            low=Decimal("99"),
            close=Decimal("99.5"),
            volume=Decimal("1000"),
            symbol="BTC/USDT",
            timeframe=TimeFrame.H1
        )
        
        is_shooting_star = PatternRecognition._is_shooting_star(shooting_star)
        assert is_shooting_star
    
    def test_doji_pattern(self):
        """Test doji pattern detection."""
        # Create a doji candle
        doji_candle = OHLCV(
            timestamp=datetime.now(timezone.utc),
            open=Decimal("100"),
            high=Decimal("101"),
            low=Decimal("99"),
            close=Decimal("100"),  # Small body
            volume=Decimal("1000"),
            symbol="BTC/USDT",
            timeframe=TimeFrame.H1
        )
        
        is_doji = PatternRecognition._is_doji(doji_candle)
        assert is_doji


if __name__ == "__main__":
    pytest.main([__file__])
