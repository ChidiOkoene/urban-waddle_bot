"""
Tests for Exchange Adapters

This module contains tests for the exchange adapters (MT5 and CCXT),
including connection, order placement, and data retrieval.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from src.adapters.mt5_adapter import MT5Adapter
from src.adapters.ccxt_adapter import CCXTAdapter
from src.core.data_models import OrderSide, OrderType, TimeFrame
from tests.mocks.mock_exchange import MockExchange


class TestMT5Adapter:
    """Test cases for MT5 adapter."""
    
    @pytest.fixture
    def mt5_adapter(self):
        """Create MT5 adapter instance."""
        return MT5Adapter()
    
    @pytest.mark.asyncio
    async def test_connect_success(self, mt5_adapter):
        """Test successful connection to MT5."""
        with patch('src.adapters.mt5_adapter.mt5.initialize') as mock_init:
            mock_init.return_value = True
            
            result = await mt5_adapter.connect()
            
            assert result is True
            assert mt5_adapter.connected is True
            mock_init.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_connect_failure(self, mt5_adapter):
        """Test failed connection to MT5."""
        with patch('src.adapters.mt5_adapter.mt5.initialize') as mock_init:
            mock_init.return_value = False
            
            result = await mt5_adapter.connect()
            
            assert result is False
            assert mt5_adapter.connected is False
    
    @pytest.mark.asyncio
    async def test_disconnect(self, mt5_adapter):
        """Test disconnection from MT5."""
        mt5_adapter.connected = True
        
        with patch('src.adapters.mt5_adapter.mt5.shutdown') as mock_shutdown:
            result = await mt5_adapter.disconnect()
            
            assert result is True
            assert mt5_adapter.connected is False
            mock_shutdown.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_balance_success(self, mt5_adapter):
        """Test successful balance retrieval."""
        mt5_adapter.connected = True
        
        mock_balance = {
            'USDT': 10000.0,
            'BTC': 0.5
        }
        
        with patch('src.adapters.mt5_adapter.mt5.account_info') as mock_account_info:
            mock_account_info.return_value = Mock(balance=10000.0, equity=10000.0)
            
            result = await mt5_adapter.get_balance()
            
            assert 'USDT' in result
            assert result['USDT'] == 10000.0
    
    @pytest.mark.asyncio
    async def test_get_balance_not_connected(self, mt5_adapter):
        """Test balance retrieval when not connected."""
        mt5_adapter.connected = False
        
        with pytest.raises(ConnectionError):
            await mt5_adapter.get_balance()
    
    @pytest.mark.asyncio
    async def test_get_ticker_success(self, mt5_adapter):
        """Test successful ticker retrieval."""
        mt5_adapter.connected = True
        
        mock_ticker = Mock()
        mock_ticker.bid = 50000.0
        mock_ticker.ask = 50010.0
        mock_ticker.last = 50005.0
        mock_ticker.volume = 1000000.0
        
        with patch('src.adapters.mt5_adapter.mt5.symbol_info_tick') as mock_tick:
            mock_tick.return_value = mock_ticker
            
            result = await mt5_adapter.get_ticker('BTCUSD')
            
            assert result['bid'] == 50000.0
            assert result['ask'] == 50010.0
            assert result['last'] == 50005.0
    
    @pytest.mark.asyncio
    async def test_get_ohlcv_success(self, mt5_adapter):
        """Test successful OHLCV data retrieval."""
        mt5_adapter.connected = True
        
        mock_rates = [
            Mock(time=datetime.now(), open=50000.0, high=50100.0, low=49900.0, close=50050.0, tick_volume=1000),
            Mock(time=datetime.now() - timedelta(hours=1), open=49900.0, high=50000.0, low=49800.0, close=50000.0, tick_volume=1200)
        ]
        
        with patch('src.adapters.mt5_adapter.mt5.copy_rates_from') as mock_copy_rates:
            mock_copy_rates.return_value = mock_rates
            
            result = await mt5_adapter.get_ohlcv('BTCUSD', '1h', 2)
            
            assert len(result) == 2
            assert result[0].open == 50000.0
            assert result[0].high == 50100.0
    
    @pytest.mark.asyncio
    async def test_place_order_success(self, mt5_adapter):
        """Test successful order placement."""
        mt5_adapter.connected = True
        
        mock_result = Mock(retcode=10009, order=12345)
        
        with patch('src.adapters.mt5_adapter.mt5.order_send') as mock_order_send:
            mock_order_send.return_value = mock_result
            
            result = await mt5_adapter.place_order(
                symbol='BTCUSD',
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=0.1
            )
            
            assert result == '12345'
            mock_order_send.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_place_order_failure(self, mt5_adapter):
        """Test failed order placement."""
        mt5_adapter.connected = True
        
        mock_result = Mock(retcode=10004, order=0)  # Invalid request
        
        with patch('src.adapters.mt5_adapter.mt5.order_send') as mock_order_send:
            mock_order_send.return_value = mock_result
            
            result = await mt5_adapter.place_order(
                symbol='BTCUSD',
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=0.1
            )
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_cancel_order_success(self, mt5_adapter):
        """Test successful order cancellation."""
        mt5_adapter.connected = True
        
        mock_result = Mock(retcode=10009)
        
        with patch('src.adapters.mt5_adapter.mt5.order_send') as mock_order_send:
            mock_order_send.return_value = mock_result
            
            result = await mt5_adapter.cancel_order('12345')
            
            assert result is True
            mock_order_send.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_order_status_success(self, mt5_adapter):
        """Test successful order status retrieval."""
        mt5_adapter.connected = True
        
        mock_order = Mock()
        mock_order.ticket = 12345
        mock_order.symbol = 'BTCUSD'
        mock_order.type = 0  # Buy
        mock_order.volume = 0.1
        mock_order.price_open = 50000.0
        mock_order.state = 0  # Filled
        
        with patch('src.adapters.mt5_adapter.mt5.orders_get') as mock_orders_get:
            mock_orders_get.return_value = [mock_order]
            
            result = await mt5_adapter.get_order_status('12345')
            
            assert result is not None
            assert result.symbol == 'BTCUSD'
            assert result.quantity == 0.1


class TestCCXTAdapter:
    """Test cases for CCXT adapter."""
    
    @pytest.fixture
    def ccxt_adapter(self):
        """Create CCXT adapter instance."""
        return CCXTAdapter('binance')
    
    @pytest.mark.asyncio
    async def test_connect_success(self, ccxt_adapter):
        """Test successful connection to exchange."""
        with patch.object(ccxt_adapter.exchange, 'load_markets') as mock_load_markets:
            mock_load_markets.return_value = {}
            
            result = await ccxt_adapter.connect()
            
            assert result is True
            assert ccxt_adapter.connected is True
    
    @pytest.mark.asyncio
    async def test_connect_failure(self, ccxt_adapter):
        """Test failed connection to exchange."""
        with patch.object(ccxt_adapter.exchange, 'load_markets') as mock_load_markets:
            mock_load_markets.side_effect = Exception("Connection failed")
            
            result = await ccxt_adapter.connect()
            
            assert result is False
            assert ccxt_adapter.connected is False
    
    @pytest.mark.asyncio
    async def test_disconnect(self, ccxt_adapter):
        """Test disconnection from exchange."""
        ccxt_adapter.connected = True
        
        with patch.object(ccxt_adapter.exchange, 'close') as mock_close:
            result = await ccxt_adapter.disconnect()
            
            assert result is True
            assert ccxt_adapter.connected is False
            mock_close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_balance_success(self, ccxt_adapter):
        """Test successful balance retrieval."""
        ccxt_adapter.connected = True
        
        mock_balance = {
            'USDT': {'free': 10000.0, 'used': 0.0, 'total': 10000.0},
            'BTC': {'free': 0.5, 'used': 0.0, 'total': 0.5}
        }
        
        with patch.object(ccxt_adapter.exchange, 'fetch_balance') as mock_fetch_balance:
            mock_fetch_balance.return_value = mock_balance
            
            result = await ccxt_adapter.get_balance()
            
            assert 'USDT' in result
            assert result['USDT'] == 10000.0
            assert 'BTC' in result
            assert result['BTC'] == 0.5
    
    @pytest.mark.asyncio
    async def test_get_ticker_success(self, ccxt_adapter):
        """Test successful ticker retrieval."""
        ccxt_adapter.connected = True
        
        mock_ticker = {
            'symbol': 'BTC/USDT',
            'bid': 50000.0,
            'ask': 50010.0,
            'last': 50005.0,
            'volume': 1000000.0,
            'timestamp': datetime.now().timestamp() * 1000
        }
        
        with patch.object(ccxt_adapter.exchange, 'fetch_ticker') as mock_fetch_ticker:
            mock_fetch_ticker.return_value = mock_ticker
            
            result = await ccxt_adapter.get_ticker('BTC/USDT')
            
            assert result['bid'] == 50000.0
            assert result['ask'] == 50010.0
            assert result['last'] == 50005.0
    
    @pytest.mark.asyncio
    async def test_get_ohlcv_success(self, ccxt_adapter):
        """Test successful OHLCV data retrieval."""
        ccxt_adapter.connected = True
        
        mock_ohlcv = [
            [datetime.now().timestamp() * 1000, 50000.0, 50100.0, 49900.0, 50050.0, 1000.0],
            [(datetime.now() - timedelta(hours=1)).timestamp() * 1000, 49900.0, 50000.0, 49800.0, 50000.0, 1200.0]
        ]
        
        with patch.object(ccxt_adapter.exchange, 'fetch_ohlcv') as mock_fetch_ohlcv:
            mock_fetch_ohlcv.return_value = mock_ohlcv
            
            result = await ccxt_adapter.get_ohlcv('BTC/USDT', '1h', 2)
            
            assert len(result) == 2
            assert result[0].open == 50000.0
            assert result[0].high == 50100.0
    
    @pytest.mark.asyncio
    async def test_place_order_success(self, ccxt_adapter):
        """Test successful order placement."""
        ccxt_adapter.connected = True
        
        mock_order = {
            'id': '12345',
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'type': 'market',
            'amount': 0.1,
            'price': 50000.0,
            'status': 'closed',
            'filled': 0.1
        }
        
        with patch.object(ccxt_adapter.exchange, 'create_order') as mock_create_order:
            mock_create_order.return_value = mock_order
            
            result = await ccxt_adapter.place_order(
                symbol='BTC/USDT',
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=0.1
            )
            
            assert result == '12345'
            mock_create_order.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cancel_order_success(self, ccxt_adapter):
        """Test successful order cancellation."""
        ccxt_adapter.connected = True
        
        mock_order = {
            'id': '12345',
            'status': 'canceled'
        }
        
        with patch.object(ccxt_adapter.exchange, 'cancel_order') as mock_cancel_order:
            mock_cancel_order.return_value = mock_order
            
            result = await ccxt_adapter.cancel_order('12345', 'BTC/USDT')
            
            assert result is True
            mock_cancel_order.assert_called_once_with('12345', 'BTC/USDT')
    
    @pytest.mark.asyncio
    async def test_get_order_status_success(self, ccxt_adapter):
        """Test successful order status retrieval."""
        ccxt_adapter.connected = True
        
        mock_order = {
            'id': '12345',
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'type': 'market',
            'amount': 0.1,
            'price': 50000.0,
            'status': 'closed',
            'filled': 0.1
        }
        
        with patch.object(ccxt_adapter.exchange, 'fetch_order') as mock_fetch_order:
            mock_fetch_order.return_value = mock_order
            
            result = await ccxt_adapter.get_order_status('12345', 'BTC/USDT')
            
            assert result is not None
            assert result.symbol == 'BTC/USDT'
            assert result.quantity == 0.1


class TestMockExchange:
    """Test cases for mock exchange."""
    
    @pytest.fixture
    def mock_exchange(self):
        """Create mock exchange instance."""
        return MockExchange(initial_balance=10000.0)
    
    @pytest.mark.asyncio
    async def test_connect_disconnect(self, mock_exchange):
        """Test connection and disconnection."""
        # Test connection
        result = await mock_exchange.connect()
        assert result is True
        assert mock_exchange.connected is True
        
        # Test disconnection
        result = await mock_exchange.disconnect()
        assert result is True
        assert mock_exchange.connected is False
    
    @pytest.mark.asyncio
    async def test_get_balance(self, mock_exchange):
        """Test balance retrieval."""
        await mock_exchange.connect()
        
        balance = await mock_exchange.get_balance()
        
        assert 'USDT' in balance
        assert balance['USDT'] == 10000.0
    
    @pytest.mark.asyncio
    async def test_get_ticker(self, mock_exchange):
        """Test ticker retrieval."""
        await mock_exchange.connect()
        
        ticker = await mock_exchange.get_ticker('BTC/USDT')
        
        assert 'last' in ticker
        assert 'bid' in ticker
        assert 'ask' in ticker
        assert ticker['last'] > 0
    
    @pytest.mark.asyncio
    async def test_get_ohlcv(self, mock_exchange):
        """Test OHLCV data retrieval."""
        await mock_exchange.connect()
        
        ohlcv_data = await mock_exchange.get_ohlcv('BTC/USDT', '1h', 10)
        
        assert len(ohlcv_data) == 10
        assert all(candle.symbol == 'BTC/USDT' for candle in ohlcv_data)
        assert all(candle.open > 0 for candle in ohlcv_data)
    
    @pytest.mark.asyncio
    async def test_place_order_success(self, mock_exchange):
        """Test successful order placement."""
        await mock_exchange.connect()
        
        order_id = await mock_exchange.place_order(
            symbol='BTC/USDT',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1
        )
        
        assert order_id is not None
        assert order_id in mock_exchange.orders
        
        # Check order was executed
        order = mock_exchange.orders[order_id]
        assert order.status.value == 'filled'
    
    @pytest.mark.asyncio
    async def test_place_order_insufficient_balance(self, mock_exchange):
        """Test order placement with insufficient balance."""
        await mock_exchange.connect()
        
        # Try to buy more than we can afford
        with pytest.raises(ValueError, match="Insufficient balance"):
            await mock_exchange.place_order(
                symbol='BTC/USDT',
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=1000.0  # Way more than we can afford
            )
    
    @pytest.mark.asyncio
    async def test_cancel_order(self, mock_exchange):
        """Test order cancellation."""
        await mock_exchange.connect()
        
        # Place a limit order (won't execute immediately)
        order_id = await mock_exchange.place_order(
            symbol='BTC/USDT',
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.1,
            price=1000.0  # Very low price, won't execute
        )
        
        # Cancel the order
        result = await mock_exchange.cancel_order(order_id)
        
        assert result is True
        order = mock_exchange.orders[order_id]
        assert order.status.value == 'cancelled'
    
    @pytest.mark.asyncio
    async def test_get_order_history(self, mock_exchange):
        """Test order history retrieval."""
        await mock_exchange.connect()
        
        # Place some orders
        await mock_exchange.place_order('BTC/USDT', OrderSide.BUY, OrderType.MARKET, 0.1)
        await mock_exchange.place_order('ETH/USDT', OrderSide.BUY, OrderType.MARKET, 1.0)
        
        # Get order history
        orders = await mock_exchange.get_order_history()
        
        assert len(orders) >= 2
        assert all(order.symbol in ['BTC/USDT', 'ETH/USDT'] for order in orders)
    
    @pytest.mark.asyncio
    async def test_get_trade_history(self, mock_exchange):
        """Test trade history retrieval."""
        await mock_exchange.connect()
        
        # Place some orders to generate trades
        await mock_exchange.place_order('BTC/USDT', OrderSide.BUY, OrderType.MARKET, 0.1)
        await mock_exchange.place_order('ETH/USDT', OrderSide.BUY, OrderType.MARKET, 1.0)
        
        # Get trade history
        trades = await mock_exchange.get_trade_history()
        
        assert len(trades) >= 2
        assert all(trade.symbol in ['BTC/USDT', 'ETH/USDT'] for trade in trades)
    
    def test_set_price(self, mock_exchange):
        """Test price setting for testing."""
        mock_exchange.set_price('BTC/USDT', 60000.0)
        
        assert mock_exchange.market_data['BTC/USDT']['price'] == 60000.0
    
    def test_set_balance(self, mock_exchange):
        """Test balance setting for testing."""
        mock_exchange.set_balance('BTC', 1.0)
        
        assert mock_exchange.balance['BTC'] == 1.0
    
    def test_enable_execution_errors(self, mock_exchange):
        """Test execution error simulation."""
        mock_exchange.enable_execution_errors(0.1)  # 10% error rate
        
        assert mock_exchange.execution_errors is True
        assert mock_exchange.execution_error_rate == 0.1
    
    def test_simulate_market_crash(self, mock_exchange):
        """Test market crash simulation."""
        initial_price = mock_exchange.market_data['BTC/USDT']['price']
        
        mock_exchange.simulate_market_crash(0.2)  # 20% crash
        
        new_price = mock_exchange.market_data['BTC/USDT']['price']
        assert new_price < initial_price
        assert abs(new_price - initial_price * 0.8) < 0.01
    
    def test_simulate_market_rally(self, mock_exchange):
        """Test market rally simulation."""
        initial_price = mock_exchange.market_data['BTC/USDT']['price']
        
        mock_exchange.simulate_market_rally(0.2)  # 20% rally
        
        new_price = mock_exchange.market_data['BTC/USDT']['price']
        assert new_price > initial_price
        assert abs(new_price - initial_price * 1.2) < 0.01
    
    def test_reset(self, mock_exchange):
        """Test exchange reset."""
        # Modify some state
        mock_exchange.set_balance('BTC', 1.0)
        mock_exchange.set_price('BTC/USDT', 60000.0)
        
        # Reset
        mock_exchange.reset()
        
        # Check state is reset
        assert mock_exchange.balance == {'USDT': 10000.0}
        assert mock_exchange.orders == {}
        assert mock_exchange.trades == {}
        assert mock_exchange.positions == {}
    
    def test_get_statistics(self, mock_exchange):
        """Test statistics retrieval."""
        stats = mock_exchange.get_statistics()
        
        assert 'connected' in stats
        assert 'balance' in stats
        assert 'orders_count' in stats
        assert 'trades_count' in stats
        assert 'symbols' in stats


if __name__ == "__main__":
    pytest.main([__file__])
