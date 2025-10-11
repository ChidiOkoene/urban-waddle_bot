"""
Tests for Trading Strategies

This module contains tests for all trading strategies,
including signal generation, parameter validation, and performance.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from src.strategies.base_strategy import BaseStrategy
from src.strategies.rsi_macd_strategy import RSIMACDStrategy
from src.strategies.bollinger_mean_reversion import BollingerMeanReversionStrategy
from src.strategies.ema_crossover_strategy import EMACrossoverStrategy
from src.strategies.grid_bot_strategy import GridBotStrategy
from src.strategies.dca_strategy import DCAStrategy
from src.core.data_models import OHLCV, OrderSide, TimeFrame
from tests.mocks.mock_exchange import MockExchange


class TestBaseStrategy:
    """Test cases for base strategy."""
    
    @pytest.fixture
    def base_strategy(self):
        """Create base strategy instance."""
        return BaseStrategy()
    
    def test_strategy_initialization(self, base_strategy):
        """Test strategy initialization."""
        assert base_strategy.name == "BaseStrategy"
        assert base_strategy.parameters == {}
        assert base_strategy.enabled is True
    
    def test_set_parameters(self, base_strategy):
        """Test parameter setting."""
        params = {'param1': 10, 'param2': 20}
        base_strategy.set_parameters(params)
        
        assert base_strategy.parameters == params
    
    def test_get_parameter(self, base_strategy):
        """Test parameter retrieval."""
        base_strategy.parameters = {'param1': 10, 'param2': 20}
        
        assert base_strategy.get_parameter('param1') == 10
        assert base_strategy.get_parameter('param2') == 20
        assert base_strategy.get_parameter('param3', default=30) == 30
    
    def test_enable_disable(self, base_strategy):
        """Test enable/disable functionality."""
        assert base_strategy.enabled is True
        
        base_strategy.disable()
        assert base_strategy.enabled is False
        
        base_strategy.enable()
        assert base_strategy.enabled is True


class TestRSIMACDStrategy:
    """Test cases for RSI + MACD strategy."""
    
    @pytest.fixture
    def rsi_macd_strategy(self):
        """Create RSI + MACD strategy instance."""
        return RSIMACDStrategy()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='H')
        prices = 100 + np.cumsum(np.random.randn(100) * 0.1)
        
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            ohlcv = OHLCV(
                symbol='BTC/USDT',
                timestamp=date,
                open=price,
                high=price * 1.01,
                low=price * 0.99,
                close=price,
                volume=1000
            )
            data.append(ohlcv)
        
        return data
    
    def test_strategy_initialization(self, rsi_macd_strategy):
        """Test strategy initialization."""
        assert rsi_macd_strategy.name == "RSIMACDStrategy"
        assert 'rsi_period' in rsi_macd_strategy.parameters
        assert 'rsi_overbought' in rsi_macd_strategy.parameters
        assert 'rsi_oversold' in rsi_macd_strategy.parameters
        assert 'macd_fast' in rsi_macd_strategy.parameters
        assert 'macd_slow' in rsi_macd_strategy.parameters
        assert 'macd_signal' in rsi_macd_strategy.parameters
    
    def test_default_parameters(self, rsi_macd_strategy):
        """Test default parameters."""
        assert rsi_macd_strategy.get_parameter('rsi_period') == 14
        assert rsi_macd_strategy.get_parameter('rsi_overbought') == 70
        assert rsi_macd_strategy.get_parameter('rsi_oversold') == 30
        assert rsi_macd_strategy.get_parameter('macd_fast') == 12
        assert rsi_macd_strategy.get_parameter('macd_slow') == 26
        assert rsi_macd_strategy.get_parameter('macd_signal') == 9
    
    @pytest.mark.asyncio
    async def test_generate_signals(self, rsi_macd_strategy, sample_data):
        """Test signal generation."""
        # Convert to DataFrame
        df = pd.DataFrame([{
            'timestamp': candle.timestamp,
            'open': candle.open,
            'high': candle.high,
            'low': candle.low,
            'close': candle.close,
            'volume': candle.volume
        } for candle in sample_data])
        
        # Mock indicators
        indicators = {
            'rsi': pd.Series([30, 35, 40, 45, 50, 55, 60, 65, 70, 75]),
            'macd': pd.Series([-0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]),
            'macd_signal': pd.Series([-0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
        }
        
        signals = await rsi_macd_strategy.generate_signals(
            symbol='BTC/USDT',
            timeframe='1h',
            data=df,
            indicators=indicators
        )
        
        assert isinstance(signals, list)
        # Should have some signals generated
        assert len(signals) > 0
    
    def test_parameter_validation(self, rsi_macd_strategy):
        """Test parameter validation."""
        # Valid parameters
        valid_params = {
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9
        }
        
        rsi_macd_strategy.set_parameters(valid_params)
        assert rsi_macd_strategy.parameters == valid_params
        
        # Invalid parameters should be handled gracefully
        invalid_params = {
            'rsi_period': -1,  # Invalid
            'rsi_overbought': 50,  # Should be > oversold
            'rsi_oversold': 80  # Should be < overbought
        }
        
        # Strategy should handle invalid parameters
        rsi_macd_strategy.set_parameters(invalid_params)
        # Implementation should either reject or correct invalid parameters


class TestBollingerMeanReversionStrategy:
    """Test cases for Bollinger Bands mean reversion strategy."""
    
    @pytest.fixture
    def bollinger_strategy(self):
        """Create Bollinger Bands strategy instance."""
        return BollingerMeanReversionStrategy()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='H')
        prices = 100 + np.cumsum(np.random.randn(100) * 0.1)
        
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            ohlcv = OHLCV(
                symbol='BTC/USDT',
                timestamp=date,
                open=price,
                high=price * 1.01,
                low=price * 0.99,
                close=price,
                volume=1000
            )
            data.append(ohlcv)
        
        return data
    
    def test_strategy_initialization(self, bollinger_strategy):
        """Test strategy initialization."""
        assert bollinger_strategy.name == "BollingerMeanReversionStrategy"
        assert 'period' in bollinger_strategy.parameters
        assert 'std_dev' in bollinger_strategy.parameters
        assert 'entry_threshold' in bollinger_strategy.parameters
    
    def test_default_parameters(self, bollinger_strategy):
        """Test default parameters."""
        assert bollinger_strategy.get_parameter('period') == 20
        assert bollinger_strategy.get_parameter('std_dev') == 2.0
        assert bollinger_strategy.get_parameter('entry_threshold') == 1.0
    
    @pytest.mark.asyncio
    async def test_generate_signals(self, bollinger_strategy, sample_data):
        """Test signal generation."""
        # Convert to DataFrame
        df = pd.DataFrame([{
            'timestamp': candle.timestamp,
            'open': candle.open,
            'high': candle.high,
            'low': candle.low,
            'close': candle.close,
            'volume': candle.volume
        } for candle in sample_data])
        
        # Mock indicators
        indicators = {
            'bollinger': {
                'upper': pd.Series([105, 106, 107, 108, 109, 110, 111, 112, 113, 114]),
                'middle': pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109]),
                'lower': pd.Series([95, 96, 97, 98, 99, 100, 101, 102, 103, 104])
            }
        }
        
        signals = await bollinger_strategy.generate_signals(
            symbol='BTC/USDT',
            timeframe='1h',
            data=df,
            indicators=indicators
        )
        
        assert isinstance(signals, list)
        # Should have some signals generated
        assert len(signals) > 0


class TestEMACrossoverStrategy:
    """Test cases for EMA crossover strategy."""
    
    @pytest.fixture
    def ema_strategy(self):
        """Create EMA crossover strategy instance."""
        return EMACrossoverStrategy()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='H')
        prices = 100 + np.cumsum(np.random.randn(100) * 0.1)
        
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            ohlcv = OHLCV(
                symbol='BTC/USDT',
                timestamp=date,
                open=price,
                high=price * 1.01,
                low=price * 0.99,
                close=price,
                volume=1000
            )
            data.append(ohlcv)
        
        return data
    
    def test_strategy_initialization(self, ema_strategy):
        """Test strategy initialization."""
        assert ema_strategy.name == "EMACrossoverStrategy"
        assert 'fast_period' in ema_strategy.parameters
        assert 'slow_period' in ema_strategy.parameters
        assert 'signal_threshold' in ema_strategy.parameters
    
    def test_default_parameters(self, ema_strategy):
        """Test default parameters."""
        assert ema_strategy.get_parameter('fast_period') == 12
        assert ema_strategy.get_parameter('slow_period') == 26
        assert ema_strategy.get_parameter('signal_threshold') == 0.005
    
    @pytest.mark.asyncio
    async def test_generate_signals(self, ema_strategy, sample_data):
        """Test signal generation."""
        # Convert to DataFrame
        df = pd.DataFrame([{
            'timestamp': candle.timestamp,
            'open': candle.open,
            'high': candle.high,
            'low': candle.low,
            'close': candle.close,
            'volume': candle.volume
        } for candle in sample_data])
        
        # Mock indicators
        indicators = {
            'ema_12': pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109]),
            'ema_26': pd.Series([99, 100, 101, 102, 103, 104, 105, 106, 107, 108])
        }
        
        signals = await ema_strategy.generate_signals(
            symbol='BTC/USDT',
            timeframe='1h',
            data=df,
            indicators=indicators
        )
        
        assert isinstance(signals, list)
        # Should have some signals generated
        assert len(signals) > 0


class TestGridBotStrategy:
    """Test cases for Grid Bot strategy."""
    
    @pytest.fixture
    def grid_strategy(self):
        """Create Grid Bot strategy instance."""
        return GridBotStrategy()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='H')
        prices = 100 + np.cumsum(np.random.randn(100) * 0.1)
        
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            ohlcv = OHLCV(
                symbol='BTC/USDT',
                timestamp=date,
                open=price,
                high=price * 1.01,
                low=price * 0.99,
                close=price,
                volume=1000
            )
            data.append(ohlcv)
        
        return data
    
    def test_strategy_initialization(self, grid_strategy):
        """Test strategy initialization."""
        assert grid_strategy.name == "GridBotStrategy"
        assert 'grid_levels' in grid_strategy.parameters
        assert 'grid_spacing' in grid_strategy.parameters
        assert 'max_position_size' in grid_strategy.parameters
    
    def test_default_parameters(self, grid_strategy):
        """Test default parameters."""
        assert grid_strategy.get_parameter('grid_levels') == 10
        assert grid_strategy.get_parameter('grid_spacing') == 0.01
        assert grid_strategy.get_parameter('max_position_size') == 0.1
    
    @pytest.mark.asyncio
    async def test_generate_signals(self, grid_strategy, sample_data):
        """Test signal generation."""
        # Convert to DataFrame
        df = pd.DataFrame([{
            'timestamp': candle.timestamp,
            'open': candle.open,
            'high': candle.high,
            'low': candle.low,
            'close': candle.close,
            'volume': candle.volume
        } for candle in sample_data])
        
        # Mock indicators
        indicators = {}
        
        signals = await grid_strategy.generate_signals(
            symbol='BTC/USDT',
            timeframe='1h',
            data=df,
            indicators=indicators
        )
        
        assert isinstance(signals, list)
        # Grid strategy should generate multiple signals
        assert len(signals) > 0


class TestDCAStrategy:
    """Test cases for DCA strategy."""
    
    @pytest.fixture
    def dca_strategy(self):
        """Create DCA strategy instance."""
        return DCAStrategy()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='H')
        prices = 100 + np.cumsum(np.random.randn(100) * 0.1)
        
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            ohlcv = OHLCV(
                symbol='BTC/USDT',
                timestamp=date,
                open=price,
                high=price * 1.01,
                low=price * 0.99,
                close=price,
                volume=1000
            )
            data.append(ohlcv)
        
        return data
    
    def test_strategy_initialization(self, dca_strategy):
        """Test strategy initialization."""
        assert dca_strategy.name == "DCAStrategy"
        assert 'interval_hours' in dca_strategy.parameters
        assert 'position_size' in dca_strategy.parameters
        assert 'max_positions' in dca_strategy.parameters
    
    def test_default_parameters(self, dca_strategy):
        """Test default parameters."""
        assert dca_strategy.get_parameter('interval_hours') == 24
        assert dca_strategy.get_parameter('position_size') == 0.05
        assert dca_strategy.get_parameter('max_positions') == 10
    
    @pytest.mark.asyncio
    async def test_generate_signals(self, dca_strategy, sample_data):
        """Test signal generation."""
        # Convert to DataFrame
        df = pd.DataFrame([{
            'timestamp': candle.timestamp,
            'open': candle.open,
            'high': candle.high,
            'low': candle.low,
            'close': candle.close,
            'volume': candle.volume
        } for candle in sample_data])
        
        # Mock indicators
        indicators = {}
        
        signals = await dca_strategy.generate_signals(
            symbol='BTC/USDT',
            timeframe='1h',
            data=df,
            indicators=indicators
        )
        
        assert isinstance(signals, list)
        # DCA strategy should generate signals based on time intervals
        assert len(signals) > 0


class TestStrategyIntegration:
    """Integration tests for strategies."""
    
    @pytest.fixture
    def mock_exchange(self):
        """Create mock exchange for testing."""
        return MockExchange(initial_balance=10000.0)
    
    @pytest.mark.asyncio
    async def test_strategy_with_mock_exchange(self, mock_exchange):
        """Test strategy integration with mock exchange."""
        await mock_exchange.connect()
        
        # Create strategy
        strategy = RSIMACDStrategy()
        
        # Get market data
        ohlcv_data = await mock_exchange.get_ohlcv('BTC/USDT', '1h', 50)
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'timestamp': candle.timestamp,
            'open': candle.open,
            'high': candle.high,
            'low': candle.low,
            'close': candle.close,
            'volume': candle.volume
        } for candle in ohlcv_data])
        
        # Mock indicators (in real implementation, these would be calculated)
        indicators = {
            'rsi': pd.Series([30, 35, 40, 45, 50, 55, 60, 65, 70, 75]),
            'macd': pd.Series([-0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]),
            'macd_signal': pd.Series([-0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
        }
        
        # Generate signals
        signals = await strategy.generate_signals(
            symbol='BTC/USDT',
            timeframe='1h',
            data=df,
            indicators=indicators
        )
        
        assert isinstance(signals, list)
        assert len(signals) > 0
        
        # Test signal execution
        for signal_data in signals:
            if signal_data.get('side') == 'buy':
                order_id = await mock_exchange.place_order(
                    symbol='BTC/USDT',
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=0.1
                )
                assert order_id is not None
    
    def test_strategy_parameter_validation(self):
        """Test parameter validation across strategies."""
        strategies = [
            RSIMACDStrategy(),
            BollingerMeanReversionStrategy(),
            EMACrossoverStrategy(),
            GridBotStrategy(),
            DCAStrategy()
        ]
        
        for strategy in strategies:
            # Test setting valid parameters
            valid_params = strategy.parameters.copy()
            strategy.set_parameters(valid_params)
            assert strategy.parameters == valid_params
            
            # Test getting parameters
            for param_name in valid_params:
                value = strategy.get_parameter(param_name)
                assert value is not None
    
    def test_strategy_enable_disable(self):
        """Test enable/disable functionality across strategies."""
        strategies = [
            RSIMACDStrategy(),
            BollingerMeanReversionStrategy(),
            EMACrossoverStrategy(),
            GridBotStrategy(),
            DCAStrategy()
        ]
        
        for strategy in strategies:
            # Test initial state
            assert strategy.enabled is True
            
            # Test disable
            strategy.disable()
            assert strategy.enabled is False
            
            # Test enable
            strategy.enable()
            assert strategy.enabled is True


if __name__ == "__main__":
    pytest.main([__file__])
