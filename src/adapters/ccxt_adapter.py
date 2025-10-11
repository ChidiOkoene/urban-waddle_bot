"""
CCXT adapter implementation for crypto exchanges.
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional

import ccxt
import ccxt.async_support as ccxt_async

from ..core.data_models import Balance, OHLCV, Order, OrderSide, OrderStatus, OrderType, Position, PositionSide, TimeFrame
from ..core.exchange_interface import ExchangeInterface


class CCXTAdapter(ExchangeInterface):
    """CCXT adapter implementation for crypto exchanges."""
    
    def __init__(self, credentials: Dict[str, str]):
        """
        Initialize CCXT adapter.
        
        Args:
            credentials: Dictionary containing 'exchange', 'api_key', 'secret', 'password', 'sandbox'
        """
        exchange_name = credentials.get('exchange', 'binance')
        super().__init__(exchange_name, credentials)
        
        self.api_key = credentials.get('api_key')
        self.secret = credentials.get('secret')
        self.password = credentials.get('password', '')
        self.sandbox = credentials.get('sandbox', False)
        
        # Initialize exchange
        exchange_class = getattr(ccxt_async, exchange_name)
        self.exchange = exchange_class({
            'apiKey': self.api_key,
            'secret': self.secret,
            'password': self.password,
            'sandbox': self.sandbox,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',  # or 'future', 'option'
            }
        })
        
        # Timeframe mapping
        self.timeframe_map = {
            TimeFrame.M1: '1m',
            TimeFrame.M5: '5m',
            TimeFrame.M15: '15m',
            TimeFrame.M30: '30m',
            TimeFrame.H1: '1h',
            TimeFrame.H4: '4h',
            TimeFrame.D1: '1d',
            TimeFrame.W1: '1w',
            TimeFrame.MN1: '1M',
        }
    
    async def connect(self) -> bool:
        """Connect to the exchange."""
        try:
            # Load markets
            await self.exchange.load_markets()
            self.is_connected = True
            print(f"Connected to {self.exchange_name}")
            return True
        except Exception as e:
            print(f"Error connecting to {self.exchange_name}: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from the exchange."""
        try:
            await self.exchange.close()
            self.is_connected = False
            return True
        except Exception as e:
            print(f"Error disconnecting from {self.exchange_name}: {e}")
            return False
    
    async def get_ohlcv(
        self, 
        symbol: str, 
        timeframe: TimeFrame, 
        limit: int = 1000,
        since: Optional[int] = None
    ) -> List[OHLCV]:
        """Get OHLCV data from the exchange."""
        try:
            # Convert timeframe
            ccxt_timeframe = self.timeframe_map.get(timeframe)
            if ccxt_timeframe is None:
                raise ValueError(f"Unsupported timeframe: {timeframe}")
            
            # Get OHLCV data
            ohlcv = await self.exchange.fetch_ohlcv(
                symbol, 
                ccxt_timeframe, 
                since=since, 
                limit=limit
            )
            
            # Convert to OHLCV objects
            ohlcv_data = []
            for candle in ohlcv:
                ohlcv_data.append(OHLCV(
                    timestamp=datetime.fromtimestamp(candle[0] / 1000, tz=timezone.utc),
                    open=Decimal(str(candle[1])),
                    high=Decimal(str(candle[2])),
                    low=Decimal(str(candle[3])),
                    close=Decimal(str(candle[4])),
                    volume=Decimal(str(candle[5])),
                    symbol=symbol,
                    timeframe=timeframe
                ))
            
            return ohlcv_data
            
        except Exception as e:
            print(f"Error getting OHLCV data: {e}")
            return []
    
    async def get_balance(self) -> Dict[str, Balance]:
        """Get account balance from the exchange."""
        try:
            balance = await self.exchange.fetch_balance()
            
            balance_dict = {}
            for currency, amounts in balance.items():
                if currency in ['info', 'free', 'used', 'total']:
                    continue
                
                if isinstance(amounts, dict) and 'free' in amounts:
                    balance_dict[currency] = Balance(
                        currency=currency,
                        free=Decimal(str(amounts['free'])),
                        used=Decimal(str(amounts['used'])),
                        total=Decimal(str(amounts['total']))
                    )
            
            return balance_dict
            
        except Exception as e:
            print(f"Error getting balance: {e}")
            return {}
    
    async def get_open_positions(self) -> List[Position]:
        """Get open positions from the exchange."""
        try:
            # For spot trading, positions are represented by non-zero balances
            balance = await self.get_balance()
            positions = []
            
            for currency, bal in balance.items():
                if bal.total > 0:
                    # This is a simplified approach for spot trading
                    # In reality, you'd need to track entry prices separately
                    position = Position(
                        symbol=f"{currency}/USDT",  # Assuming USDT as quote
                        side=PositionSide.LONG,
                        amount=bal.total,
                        entry_price=Decimal("0"),  # Would need to track this
                        current_price=Decimal("0"),  # Would need to fetch current price
                        unrealized_pnl=Decimal("0"),
                        exchange=self.exchange_name
                    )
                    positions.append(position)
            
            return positions
            
        except Exception as e:
            print(f"Error getting positions: {e}")
            return []
    
    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        **kwargs
    ) -> Order:
        """Place an order on the exchange."""
        try:
            # Map order types
            ccxt_order_type = 'market' if order_type.lower() == 'market' else 'limit'
            
            # Place order
            order = await self.exchange.create_order(
                symbol=symbol,
                type=ccxt_order_type,
                side=side.lower(),
                amount=amount,
                price=price,
                params=kwargs
            )
            
            # Convert to our Order model
            return Order(
                symbol=symbol,
                side=OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL,
                type=OrderType.MARKET if order_type.lower() == 'market' else OrderType.LIMIT,
                amount=Decimal(str(order['amount'])),
                price=Decimal(str(order['price'])) if order.get('price') else None,
                status=OrderStatus.OPEN if order['status'] == 'open' else OrderStatus.FILLED,
                filled=Decimal(str(order.get('filled', 0))),
                remaining=Decimal(str(order.get('remaining', amount))),
                cost=Decimal(str(order.get('cost', 0))),
                fee=Decimal(str(order.get('fee', {}).get('cost', 0))),
                exchange_order_id=str(order['id']),
                exchange=self.exchange_name,
                metadata=order.get('info', {})
            )
            
        except Exception as e:
            print(f"Error placing order: {e}")
            # Return failed order
            return Order(
                symbol=symbol,
                side=OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL,
                type=OrderType.MARKET if order_type.lower() == 'market' else OrderType.LIMIT,
                amount=Decimal(str(amount)),
                price=Decimal(str(price)) if price else None,
                status=OrderStatus.REJECTED,
                filled=Decimal("0"),
                remaining=Decimal(str(amount)),
                exchange=self.exchange_name,
                metadata={'error': str(e)}
            )
    
    async def cancel_order(self, order_id: str, symbol: str = None) -> bool:
        """Cancel an order on the exchange."""
        try:
            # Get symbol from order_id (would need to track this)
            # For now, we'll need the symbol parameter
            if not symbol:
                raise ValueError("Symbol required to cancel order")
            
            await self.exchange.cancel_order(order_id, symbol)
            return True
            
        except Exception as e:
            print(f"Error cancelling order: {e}")
            return False
    
    async def get_order_status(self, order_id: str, symbol: str = None) -> Optional[Order]:
        """Get order status from the exchange."""
        try:
            # Get symbol from order_id (would need to track this)
            if not symbol:
                raise ValueError("Symbol required to get order status")
            
            order = await self.exchange.fetch_order(order_id, symbol)
            
            return Order(
                symbol=symbol,
                side=OrderSide.BUY if order['side'] == 'buy' else OrderSide.SELL,
                type=OrderType.MARKET if order['type'] == 'market' else OrderType.LIMIT,
                amount=Decimal(str(order['amount'])),
                price=Decimal(str(order['price'])) if order.get('price') else None,
                status=OrderStatus.OPEN if order['status'] == 'open' else OrderStatus.FILLED,
                filled=Decimal(str(order.get('filled', 0))),
                remaining=Decimal(str(order.get('remaining', order['amount']))),
                cost=Decimal(str(order.get('cost', 0))),
                fee=Decimal(str(order.get('fee', {}).get('cost', 0))),
                exchange_order_id=str(order['id']),
                exchange=self.exchange_name,
                metadata=order.get('info', {})
            )
            
        except Exception as e:
            print(f"Error getting order status: {e}")
            return None
    
    async def get_trading_fees(self, symbol: str) -> Dict[str, float]:
        """Get trading fees for a symbol."""
        try:
            market = self.exchange.market(symbol)
            return {
                'maker': market.get('maker', 0.0),
                'taker': market.get('taker', 0.0)
            }
        except Exception as e:
            print(f"Error getting trading fees: {e}")
            return {"maker": 0.0, "taker": 0.0}
    
    async def get_symbol_info(self, symbol: str) -> Dict:
        """Get symbol information from the exchange."""
        try:
            market = self.exchange.market(symbol)
            return {
                'symbol': market['symbol'],
                'base': market['base'],
                'quote': market['quote'],
                'min_amount': market.get('limits', {}).get('amount', {}).get('min', 0),
                'max_amount': market.get('limits', {}).get('amount', {}).get('max', 0),
                'step': market.get('limits', {}).get('amount', {}).get('step', 0),
                'precision': market.get('precision', {}).get('amount', 0),
                'tick_size': market.get('tickSize', 0),
                'active': market.get('active', True),
                'type': market.get('type', 'spot'),
                'spot': market.get('spot', True),
                'margin': market.get('margin', False),
                'future': market.get('future', False),
                'option': market.get('option', False),
                'contract': market.get('contract', False),
                'settle': market.get('settle'),
                'settle_id': market.get('settleId'),
                'contract_size': market.get('contractSize'),
                'expiry': market.get('expiry'),
                'expiry_datetime': market.get('expiryDatetime'),
                'strike': market.get('strike'),
                'option_type': market.get('optionType'),
                'maker': market.get('maker', 0),
                'taker': market.get('taker', 0),
                'percentage': market.get('percentage', True),
                'tierBased': market.get('tierBased', False),
                'feeSide': market.get('feeSide', 'get'),
                'info': market.get('info', {})
            }
        except Exception as e:
            print(f"Error getting symbol info: {e}")
            return {}
    
    async def get_ticker(self, symbol: str) -> Dict:
        """Get current ticker for a symbol."""
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return {
                'symbol': ticker['symbol'],
                'bid': ticker.get('bid'),
                'ask': ticker.get('ask'),
                'last': ticker.get('last'),
                'high': ticker.get('high'),
                'low': ticker.get('low'),
                'volume': ticker.get('baseVolume'),
                'quote_volume': ticker.get('quoteVolume'),
                'change': ticker.get('change'),
                'percentage': ticker.get('percentage'),
                'timestamp': ticker.get('timestamp')
            }
        except Exception as e:
            print(f"Error getting ticker: {e}")
            return {}
    
    def normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol for the exchange."""
        return symbol
    
    def normalize_timeframe(self, timeframe: TimeFrame) -> str:
        """Normalize timeframe for the exchange."""
        return self.timeframe_map.get(timeframe, timeframe.value)
    
    def get_rate_limit_info(self) -> Dict[str, int]:
        """Get rate limit information for the exchange."""
        return {
            "requests_per_minute": self.exchange.rateLimit // 1000 * 60 if self.exchange.rateLimit else 60,
            "requests_per_second": self.exchange.rateLimit // 1000 if self.exchange.rateLimit else 1
        }
