"""
MetaTrader 5 adapter implementation.
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional

import MetaTrader5 as mt5
import pandas as pd

from ..core.data_models import Balance, OHLCV, Order, OrderSide, OrderStatus, OrderType, Position, PositionSide, TimeFrame
from ..core.exchange_interface import ExchangeInterface


class MT5Adapter(ExchangeInterface):
    """MetaTrader 5 adapter implementation."""
    
    def __init__(self, credentials: Dict[str, str]):
        """
        Initialize MT5 adapter.
        
        Args:
            credentials: Dictionary containing 'account', 'password', 'server'
        """
        super().__init__("MT5", credentials)
        self.account = credentials.get('account')
        self.password = credentials.get('password')
        self.server = credentials.get('server')
        self.symbol_info_cache = {}
        
        # MT5 timeframe mapping
        self.timeframe_map = {
            TimeFrame.M1: mt5.TIMEFRAME_M1,
            TimeFrame.M5: mt5.TIMEFRAME_M5,
            TimeFrame.M15: mt5.TIMEFRAME_M15,
            TimeFrame.M30: mt5.TIMEFRAME_M30,
            TimeFrame.H1: mt5.TIMEFRAME_H1,
            TimeFrame.H4: mt5.TIMEFRAME_H4,
            TimeFrame.D1: mt5.TIMEFRAME_D1,
            TimeFrame.W1: mt5.TIMEFRAME_W1,
            TimeFrame.MN1: mt5.TIMEFRAME_MN1,
        }
    
    async def connect(self) -> bool:
        """Connect to MT5."""
        try:
            # Initialize MT5
            if not mt5.initialize():
                print(f"MT5 initialization failed: {mt5.last_error()}")
                return False
            
            # Login to account
            if not mt5.login(
                login=int(self.account),
                password=self.password,
                server=self.server
            ):
                print(f"MT5 login failed: {mt5.last_error()}")
                return False
            
            # Verify connection
            account_info = mt5.account_info()
            if account_info is None:
                print("Failed to get account info")
                return False
            
            self.is_connected = True
            print(f"Connected to MT5 account {self.account}")
            return True
            
        except Exception as e:
            print(f"MT5 connection error: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from MT5."""
        try:
            mt5.shutdown()
            self.is_connected = False
            return True
        except Exception as e:
            print(f"MT5 disconnection error: {e}")
            return False
    
    async def get_ohlcv(
        self, 
        symbol: str, 
        timeframe: TimeFrame, 
        limit: int = 1000,
        since: Optional[int] = None
    ) -> List[OHLCV]:
        """Get OHLCV data from MT5."""
        try:
            # Convert timeframe
            mt5_timeframe = self.timeframe_map.get(timeframe)
            if mt5_timeframe is None:
                raise ValueError(f"Unsupported timeframe: {timeframe}")
            
            # Convert since timestamp to datetime if provided
            since_dt = None
            if since:
                since_dt = datetime.fromtimestamp(since / 1000, tz=timezone.utc)
            
            # Get rates
            rates = mt5.copy_rates_from_pos(
                symbol, 
                mt5_timeframe, 
                0, 
                limit
            )
            
            if rates is None or len(rates) == 0:
                return []
            
            # Convert to OHLCV objects
            ohlcv_data = []
            for rate in rates:
                ohlcv_data.append(OHLCV(
                    timestamp=datetime.fromtimestamp(rate['time'], tz=timezone.utc),
                    open=Decimal(str(rate['open'])),
                    high=Decimal(str(rate['high'])),
                    low=Decimal(str(rate['low'])),
                    close=Decimal(str(rate['close'])),
                    volume=Decimal(str(rate['tick_volume'])),
                    symbol=symbol,
                    timeframe=timeframe
                ))
            
            return ohlcv_data
            
        except Exception as e:
            print(f"Error getting OHLCV data: {e}")
            return []
    
    async def get_balance(self) -> Dict[str, Balance]:
        """Get account balance from MT5."""
        try:
            account_info = mt5.account_info()
            if account_info is None:
                return {}
            
            # Get account currency
            currency = account_info.currency
            
            balance = Balance(
                currency=currency,
                free=Decimal(str(account_info.balance)),
                used=Decimal(str(account_info.margin)),
                total=Decimal(str(account_info.equity))
            )
            
            return {currency: balance}
            
        except Exception as e:
            print(f"Error getting balance: {e}")
            return {}
    
    async def get_open_positions(self) -> List[Position]:
        """Get open positions from MT5."""
        try:
            positions = mt5.positions_get()
            if positions is None:
                return []
            
            position_list = []
            for pos in positions:
                # Determine position side
                side = PositionSide.LONG if pos.type == 0 else PositionSide.SHORT
                
                # Get current price
                tick = mt5.symbol_info_tick(pos.symbol)
                current_price = Decimal(str(tick.ask if side == PositionSide.LONG else tick.bid))
                
                position = Position(
                    symbol=pos.symbol,
                    side=side,
                    amount=Decimal(str(pos.volume)),
                    entry_price=Decimal(str(pos.price_open)),
                    current_price=current_price,
                    unrealized_pnl=Decimal(str(pos.profit)),
                    exchange=self.exchange_name,
                    metadata={
                        'ticket': pos.ticket,
                        'magic': pos.magic,
                        'comment': pos.comment,
                        'swap': pos.swap,
                        'commission': pos.commission
                    }
                )
                
                position_list.append(position)
            
            return position_list
            
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
        """Place an order in MT5."""
        try:
            # Prepare order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": amount,
                "type": mt5.ORDER_TYPE_BUY if side.lower() == 'buy' else mt5.ORDER_TYPE_SELL,
                "magic": kwargs.get('magic', 0),
                "comment": kwargs.get('comment', ''),
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Add price for limit orders
            if order_type.lower() == 'limit' and price:
                request["price"] = price
                request["type"] = mt5.ORDER_TYPE_BUY_LIMIT if side.lower() == 'buy' else mt5.ORDER_TYPE_SELL_LIMIT
            
            # Add stop price for stop orders
            if order_type.lower() == 'stop' and stop_price:
                request["sl"] = stop_price
            
            # Send order
            result = mt5.order_send(request)
            
            if result is None:
                raise Exception("Failed to send order")
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                raise Exception(f"Order failed: {result.retcode} - {result.comment}")
            
            # Create order object
            order = Order(
                symbol=symbol,
                side=OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL,
                type=OrderType.MARKET if order_type.lower() == 'market' else OrderType.LIMIT,
                amount=Decimal(str(amount)),
                price=Decimal(str(price)) if price else None,
                stop_price=Decimal(str(stop_price)) if stop_price else None,
                status=OrderStatus.FILLED if result.retcode == mt5.TRADE_RETCODE_DONE else OrderStatus.REJECTED,
                filled=Decimal(str(amount)),
                remaining=Decimal("0"),
                cost=Decimal(str(result.deal)),
                fee=Decimal("0"),  # MT5 doesn't provide fee info in order_send result
                exchange_order_id=str(result.order),
                exchange=self.exchange_name,
                metadata={
                    'retcode': result.retcode,
                    'comment': result.comment,
                    'request_id': result.request_id
                }
            )
            
            return order
            
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
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order in MT5."""
        try:
            # MT5 doesn't have a direct cancel order function for market orders
            # This would typically be used for pending orders
            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": int(order_id),
            }
            
            result = mt5.order_send(request)
            return result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
            
        except Exception as e:
            print(f"Error cancelling order: {e}")
            return False
    
    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get order status from MT5."""
        try:
            # Get order history
            orders = mt5.history_orders_get(
                datetime.now() - pd.Timedelta(days=1),
                datetime.now()
            )
            
            if orders is None:
                return None
            
            for order in orders:
                if str(order.ticket) == order_id:
                    return Order(
                        symbol=order.symbol,
                        side=OrderSide.BUY if order.type in [0, 2, 4] else OrderSide.SELL,
                        type=OrderType.MARKET if order.type in [0, 1] else OrderType.LIMIT,
                        amount=Decimal(str(order.volume_initial)),
                        price=Decimal(str(order.price_open)),
                        status=OrderStatus.FILLED if order.state == 2 else OrderStatus.PENDING,
                        filled=Decimal(str(order.volume_filled)),
                        remaining=Decimal(str(order.volume_initial - order.volume_filled)),
                        exchange_order_id=str(order.ticket),
                        exchange=self.exchange_name,
                        metadata={
                            'state': order.state,
                            'time_setup': order.time_setup,
                            'time_done': order.time_done
                        }
                    )
            
            return None
            
        except Exception as e:
            print(f"Error getting order status: {e}")
            return None
    
    async def get_trading_fees(self, symbol: str) -> Dict[str, float]:
        """Get trading fees for a symbol."""
        try:
            symbol_info = await self.get_symbol_info(symbol)
            if not symbol_info:
                return {"maker": 0.0, "taker": 0.0}
            
            # MT5 doesn't provide fee information directly
            # This would need to be configured or retrieved from broker
            return {
                "maker": 0.0,
                "taker": 0.001  # Default 0.1%
            }
            
        except Exception as e:
            print(f"Error getting trading fees: {e}")
            return {"maker": 0.0, "taker": 0.0}
    
    async def get_symbol_info(self, symbol: str) -> Dict:
        """Get symbol information from MT5."""
        try:
            if symbol in self.symbol_info_cache:
                return self.symbol_info_cache[symbol]
            
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return {}
            
            info = {
                'symbol': symbol_info.name,
                'base': symbol_info.currency_base,
                'quote': symbol_info.currency_profit,
                'min_amount': symbol_info.volume_min,
                'max_amount': symbol_info.volume_max,
                'step': symbol_info.volume_step,
                'precision': symbol_info.digits,
                'tick_size': symbol_info.trade_tick_size,
                'tick_value': symbol_info.trade_tick_value,
                'contract_size': symbol_info.trade_contract_size,
                'margin_initial': symbol_info.margin_initial,
                'margin_maintenance': symbol_info.margin_maintenance,
                'swap_long': symbol_info.swap_long,
                'swap_short': symbol_info.swap_short,
                'trading_mode': symbol_info.trade_mode,
                'trading_stops_level': symbol_info.trade_stops_level,
                'trading_freeze_level': symbol_info.trade_freeze_level
            }
            
            self.symbol_info_cache[symbol] = info
            return info
            
        except Exception as e:
            print(f"Error getting symbol info: {e}")
            return {}
    
    async def get_ticker(self, symbol: str) -> Dict:
        """Get current ticker for a symbol."""
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return {}
            
            return {
                'symbol': symbol,
                'bid': float(tick.bid),
                'ask': float(tick.ask),
                'last': float(tick.last),
                'volume': float(tick.volume),
                'time': tick.time
            }
            
        except Exception as e:
            print(f"Error getting ticker: {e}")
            return {}
    
    def normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol for MT5."""
        # MT5 symbols are usually in format like "EURUSD", "BTCUSD"
        return symbol.replace('/', '').upper()
    
    def normalize_timeframe(self, timeframe: TimeFrame) -> str:
        """Normalize timeframe for MT5."""
        return timeframe.value
