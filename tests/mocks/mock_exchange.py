"""
Mock Exchange for Testing

This module provides a mock exchange implementation for testing
trading strategies, risk management, and bot functionality without
requiring real exchange connections.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import random
import uuid
import numpy as np
import pandas as pd

from ..core.exchange_interface import ExchangeInterface
from ..core.data_models import (
    OHLCV, Order, Trade, Position, Balance, OrderSide, OrderType, 
    OrderStatus, PositionStatus, TimeFrame
)


class MockExchange(ExchangeInterface):
    """Mock exchange implementation for testing."""
    
    def __init__(self, 
                 initial_balance: float = 10000.0,
                 symbols: List[str] = None,
                 slippage: float = 0.001,
                 commission: float = 0.001):
        """
        Initialize the mock exchange.
        
        Args:
            initial_balance: Initial account balance
            symbols: List of supported symbols
            slippage: Slippage percentage for orders
            commission: Commission percentage for trades
        """
        super().__init__()
        self.logger = logging.getLogger("MockExchange")
        
        # Configuration
        self.initial_balance = initial_balance
        self.symbols = symbols or ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT']
        self.slippage = slippage
        self.commission = commission
        
        # State
        self.connected = False
        self.balance = {'USDT': initial_balance}
        self.orders = {}  # order_id -> Order
        self.trades = {}  # trade_id -> Trade
        self.positions = {}  # position_id -> Position
        
        # Market data
        self.market_data = {}
        self._initialize_market_data()
        
        # Order execution simulation
        self.order_execution_delay = 0.1  # seconds
        self.execution_errors = False
        self.execution_error_rate = 0.05  # 5% error rate
        
        # Price simulation
        self.price_volatility = 0.02  # 2% volatility
        self.trend_strength = 0.001  # Slight upward trend
    
    def _initialize_market_data(self):
        """Initialize market data for all symbols."""
        base_prices = {
            'BTC/USDT': 50000.0,
            'ETH/USDT': 3000.0,
            'BNB/USDT': 300.0,
            'ADA/USDT': 0.5
        }
        
        for symbol in self.symbols:
            base_price = base_prices.get(symbol, 100.0)
            self.market_data[symbol] = {
                'price': base_price,
                'volume': 1000000.0,
                'last_update': datetime.now()
            }
    
    async def connect(self) -> bool:
        """Connect to the mock exchange."""
        self.connected = True
        self.logger.info("Connected to mock exchange")
        return True
    
    async def disconnect(self) -> bool:
        """Disconnect from the mock exchange."""
        self.connected = False
        self.logger.info("Disconnected from mock exchange")
        return True
    
    async def get_balance(self) -> Dict[str, float]:
        """Get account balance."""
        if not self.connected:
            raise ConnectionError("Not connected to exchange")
        
        return self.balance.copy()
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get ticker information for a symbol."""
        if not self.connected:
            raise ConnectionError("Not connected to exchange")
        
        if symbol not in self.market_data:
            raise ValueError(f"Symbol {symbol} not supported")
        
        # Simulate price movement
        await self._update_price(symbol)
        
        price_data = self.market_data[symbol]
        
        return {
            'symbol': symbol,
            'last': price_data['price'],
            'bid': price_data['price'] * (1 - self.slippage),
            'ask': price_data['price'] * (1 + self.slippage),
            'volume': price_data['volume'],
            'timestamp': price_data['last_update']
        }
    
    async def get_ohlcv(self, 
                       symbol: str, 
                       timeframe: str = "1h", 
                       limit: int = 100) -> List[OHLCV]:
        """Get OHLCV data for a symbol."""
        if not self.connected:
            raise ConnectionError("Not connected to exchange")
        
        if symbol not in self.market_data:
            raise ValueError(f"Symbol {symbol} not supported")
        
        # Generate mock OHLCV data
        return await self._generate_ohlcv_data(symbol, timeframe, limit)
    
    async def place_order(self, 
                         symbol: str,
                         side: OrderSide,
                         order_type: OrderType,
                         quantity: float,
                         price: Optional[float] = None) -> str:
        """Place an order."""
        if not self.connected:
            raise ConnectionError("Not connected to exchange")
        
        if symbol not in self.market_data:
            raise ValueError(f"Symbol {symbol} not supported")
        
        # Check balance for buy orders
        if side == OrderSide.BUY:
            required_balance = quantity * (price or self.market_data[symbol]['price'])
            if self.balance.get('USDT', 0) < required_balance:
                raise ValueError("Insufficient balance")
        
        # Check available quantity for sell orders
        if side == OrderSide.SELL:
            available_quantity = self.balance.get(symbol.split('/')[0], 0)
            if available_quantity < quantity:
                raise ValueError("Insufficient quantity")
        
        # Create order
        order_id = str(uuid.uuid4())
        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            type=order_type,
            quantity=quantity,
            price=price,
            status=OrderStatus.PENDING,
            timestamp=datetime.now()
        )
        
        self.orders[order_id] = order
        
        # Simulate order execution
        await self._simulate_order_execution(order)
        
        return order_id
    
    async def cancel_order(self, order_id: str, symbol: str = None) -> bool:
        """Cancel an order."""
        if not self.connected:
            raise ConnectionError("Not connected to exchange")
        
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        if order.status != OrderStatus.PENDING:
            return False
        
        order.status = OrderStatus.CANCELLED
        order.cancelled_at = datetime.now()
        
        self.logger.info(f"Order {order_id} cancelled")
        return True
    
    async def get_order_status(self, order_id: str, symbol: str = None) -> Optional[Order]:
        """Get order status."""
        if not self.connected:
            raise ConnectionError("Not connected to exchange")
        
        return self.orders.get(order_id)
    
    async def get_open_orders(self, symbol: str = None) -> List[Order]:
        """Get open orders."""
        if not self.connected:
            raise ConnectionError("Not connected to exchange")
        
        open_orders = [
            order for order in self.orders.values()
            if order.status == OrderStatus.PENDING
        ]
        
        if symbol:
            open_orders = [order for order in open_orders if order.symbol == symbol]
        
        return open_orders
    
    async def get_order_history(self, symbol: str = None, limit: int = 100) -> List[Order]:
        """Get order history."""
        if not self.connected:
            raise ConnectionError("Not connected to exchange")
        
        orders = list(self.orders.values())
        
        if symbol:
            orders = [order for order in orders if order.symbol == symbol]
        
        # Sort by timestamp (newest first)
        orders.sort(key=lambda x: x.timestamp, reverse=True)
        
        return orders[:limit]
    
    async def get_trade_history(self, symbol: str = None, limit: int = 100) -> List[Trade]:
        """Get trade history."""
        if not self.connected:
            raise ConnectionError("Not connected to exchange")
        
        trades = list(self.trades.values())
        
        if symbol:
            trades = [trade for trade in trades if trade.symbol == symbol]
        
        # Sort by timestamp (newest first)
        trades.sort(key=lambda x: x.timestamp, reverse=True)
        
        return trades[:limit]
    
    # Mock-specific methods
    
    async def _update_price(self, symbol: str):
        """Update price for a symbol with random walk."""
        if symbol not in self.market_data:
            return
        
        price_data = self.market_data[symbol]
        current_price = price_data['price']
        
        # Random walk with slight upward trend
        change = np.random.normal(self.trend_strength, self.price_volatility)
        new_price = current_price * (1 + change)
        
        # Ensure price doesn't go negative
        new_price = max(new_price, current_price * 0.01)
        
        price_data['price'] = new_price
        price_data['last_update'] = datetime.now()
        
        # Update volume
        price_data['volume'] *= (1 + np.random.normal(0, 0.1))
        price_data['volume'] = max(price_data['volume'], 1000.0)
    
    async def _generate_ohlcv_data(self, symbol: str, timeframe: str, limit: int) -> List[OHLCV]:
        """Generate mock OHLCV data."""
        if symbol not in self.market_data:
            return []
        
        base_price = self.market_data[symbol]['price']
        timeframe_minutes = self._get_timeframe_minutes(timeframe)
        
        ohlcv_data = []
        current_time = datetime.now()
        
        for i in range(limit):
            # Generate price movement
            change = np.random.normal(0, self.price_volatility)
            price = base_price * (1 + change)
            
            # Generate OHLC
            open_price = price
            high_price = price * (1 + abs(np.random.normal(0, 0.005)))
            low_price = price * (1 - abs(np.random.normal(0, 0.005)))
            close_price = price * (1 + np.random.normal(0, 0.002))
            
            # Generate volume
            volume = np.random.uniform(1000, 10000)
            
            ohlcv = OHLCV(
                symbol=symbol,
                timestamp=current_time - timedelta(minutes=i * timeframe_minutes),
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume
            )
            
            ohlcv_data.append(ohlcv)
        
        return ohlcv_data
    
    def _get_timeframe_minutes(self, timeframe: str) -> int:
        """Get minutes for timeframe."""
        timeframe_map = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '4h': 240,
            '1d': 1440
        }
        return timeframe_map.get(timeframe, 60)
    
    async def _simulate_order_execution(self, order: Order):
        """Simulate order execution."""
        await asyncio.sleep(self.order_execution_delay)
        
        # Simulate execution errors
        if self.execution_errors and random.random() < self.execution_error_rate:
            order.status = OrderStatus.FAILED
            order.failed_at = datetime.now()
            self.logger.warning(f"Order {order.id} failed due to execution error")
            return
        
        # Get current market price
        ticker = await self.get_ticker(order.symbol)
        market_price = ticker['last']
        
        # Determine execution price
        if order.type == OrderType.MARKET:
            execution_price = market_price
        elif order.type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and order.price >= market_price:
                execution_price = market_price
            elif order.side == OrderSide.SELL and order.price <= market_price:
                execution_price = market_price
            else:
                # Order not filled, keep as pending
                return
        else:
            execution_price = market_price
        
        # Apply slippage
        if order.side == OrderSide.BUY:
            execution_price *= (1 + self.slippage)
        else:
            execution_price *= (1 - self.slippage)
        
        # Execute the order
        await self._execute_order(order, execution_price)
    
    async def _execute_order(self, order: Order, execution_price: float):
        """Execute an order."""
        # Calculate commission
        commission = order.quantity * execution_price * self.commission
        
        # Update balance
        if order.side == OrderSide.BUY:
            # Buy order: spend USDT, get base currency
            base_currency = order.symbol.split('/')[0]
            cost = order.quantity * execution_price + commission
            
            self.balance['USDT'] -= cost
            self.balance[base_currency] = self.balance.get(base_currency, 0) + order.quantity
            
        else:
            # Sell order: spend base currency, get USDT
            base_currency = order.symbol.split('/')[0]
            proceeds = order.quantity * execution_price - commission
            
            self.balance[base_currency] -= order.quantity
            self.balance['USDT'] += proceeds
        
        # Update order status
        order.status = OrderStatus.FILLED
        order.filled_at = datetime.now()
        order.price = execution_price
        
        # Create trade record
        trade_id = str(uuid.uuid4())
        trade = Trade(
            id=trade_id,
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=execution_price,
            timestamp=datetime.now(),
            commission=commission
        )
        
        self.trades[trade_id] = trade
        
        self.logger.info(f"Order {order.id} executed at {execution_price}")
    
    # Testing utilities
    
    def set_price(self, symbol: str, price: float):
        """Set price for a symbol (for testing)."""
        if symbol in self.market_data:
            self.market_data[symbol]['price'] = price
            self.market_data[symbol]['last_update'] = datetime.now()
    
    def set_balance(self, currency: str, amount: float):
        """Set balance for a currency (for testing)."""
        self.balance[currency] = amount
    
    def enable_execution_errors(self, error_rate: float = 0.05):
        """Enable execution errors for testing."""
        self.execution_errors = True
        self.execution_error_rate = error_rate
    
    def disable_execution_errors(self):
        """Disable execution errors."""
        self.execution_errors = False
    
    def set_slippage(self, slippage: float):
        """Set slippage for testing."""
        self.slippage = slippage
    
    def set_commission(self, commission: float):
        """Set commission for testing."""
        self.commission = commission
    
    def reset(self):
        """Reset the mock exchange to initial state."""
        self.balance = {'USDT': self.initial_balance}
        self.orders.clear()
        self.trades.clear()
        self.positions.clear()
        self._initialize_market_data()
        self.logger.info("Mock exchange reset")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get exchange statistics."""
        return {
            'connected': self.connected,
            'balance': self.balance.copy(),
            'orders_count': len(self.orders),
            'trades_count': len(self.trades),
            'positions_count': len(self.positions),
            'symbols': self.symbols.copy(),
            'slippage': self.slippage,
            'commission': self.commission,
            'execution_errors_enabled': self.execution_errors,
            'execution_error_rate': self.execution_error_rate
        }
    
    def get_market_data(self) -> Dict[str, Any]:
        """Get current market data."""
        return self.market_data.copy()
    
    def simulate_market_crash(self, crash_percentage: float = 0.2):
        """Simulate a market crash for testing."""
        for symbol in self.market_data:
            current_price = self.market_data[symbol]['price']
            new_price = current_price * (1 - crash_percentage)
            self.market_data[symbol]['price'] = new_price
            self.market_data[symbol]['last_update'] = datetime.now()
        
        self.logger.warning(f"Simulated market crash: {crash_percentage:.1%} drop")
    
    def simulate_market_rally(self, rally_percentage: float = 0.2):
        """Simulate a market rally for testing."""
        for symbol in self.market_data:
            current_price = self.market_data[symbol]['price']
            new_price = current_price * (1 + rally_percentage)
            self.market_data[symbol]['price'] = new_price
            self.market_data[symbol]['last_update'] = datetime.now()
        
        self.logger.info(f"Simulated market rally: {rally_percentage:.1%} gain")
    
    def simulate_high_volatility(self, volatility: float = 0.05):
        """Simulate high volatility for testing."""
        self.price_volatility = volatility
        self.logger.info(f"Simulated high volatility: {volatility:.1%}")
    
    def simulate_low_volatility(self, volatility: float = 0.005):
        """Simulate low volatility for testing."""
        self.price_volatility = volatility
        self.logger.info(f"Simulated low volatility: {volatility:.1%}")
    
    def simulate_network_delay(self, delay: float = 1.0):
        """Simulate network delay for testing."""
        self.order_execution_delay = delay
        self.logger.info(f"Simulated network delay: {delay}s")
    
    def simulate_connection_loss(self):
        """Simulate connection loss for testing."""
        self.connected = False
        self.logger.warning("Simulated connection loss")
    
    def simulate_connection_recovery(self):
        """Simulate connection recovery for testing."""
        self.connected = True
        self.logger.info("Simulated connection recovery")
