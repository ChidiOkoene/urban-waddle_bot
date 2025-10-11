"""
Abstract base class for exchange adapters.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from ..core.data_models import Balance, OHLCV, Order, Position, TimeFrame


class ExchangeInterface(ABC):
    """Abstract base class for exchange adapters."""
    
    def __init__(self, exchange_name: str, credentials: Dict[str, str]):
        """
        Initialize exchange adapter.
        
        Args:
            exchange_name: Name of the exchange
            credentials: Exchange credentials dictionary
        """
        self.exchange_name = exchange_name
        self.credentials = credentials
        self.is_connected = False
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to the exchange.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """
        Disconnect from the exchange.
        
        Returns:
            True if disconnection successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_ohlcv(
        self, 
        symbol: str, 
        timeframe: TimeFrame, 
        limit: int = 1000,
        since: Optional[int] = None
    ) -> List[OHLCV]:
        """
        Get OHLCV data for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe for the data
            limit: Maximum number of candles to return
            since: Start timestamp in milliseconds
            
        Returns:
            List of OHLCV data
        """
        pass
    
    @abstractmethod
    async def get_balance(self) -> Dict[str, Balance]:
        """
        Get account balance.
        
        Returns:
            Dictionary of currency balances
        """
        pass
    
    @abstractmethod
    async def get_open_positions(self) -> List[Position]:
        """
        Get open positions.
        
        Returns:
            List of open positions
        """
        pass
    
    @abstractmethod
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
        """
        Place an order.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            order_type: 'market', 'limit', 'stop', 'stop_limit'
            amount: Order amount
            price: Order price (for limit orders)
            stop_price: Stop price (for stop orders)
            **kwargs: Additional order parameters
            
        Returns:
            Order object
        """
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancellation successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """
        Get order status.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order object or None if not found
        """
        pass
    
    @abstractmethod
    async def get_trading_fees(self, symbol: str) -> Dict[str, float]:
        """
        Get trading fees for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with 'maker' and 'taker' fee rates
        """
        pass
    
    @abstractmethod
    async def get_symbol_info(self, symbol: str) -> Dict:
        """
        Get symbol information.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Symbol information dictionary
        """
        pass
    
    @abstractmethod
    async def get_ticker(self, symbol: str) -> Dict:
        """
        Get current ticker for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Ticker information dictionary
        """
        pass
    
    def normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol format for the exchange.
        
        Args:
            symbol: Input symbol
            
        Returns:
            Normalized symbol
        """
        return symbol
    
    def normalize_timeframe(self, timeframe: TimeFrame) -> str:
        """
        Normalize timeframe format for the exchange.
        
        Args:
            timeframe: Input timeframe
            
        Returns:
            Normalized timeframe string
        """
        return timeframe.value
    
    async def health_check(self) -> bool:
        """
        Check if the exchange connection is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Try to get account balance as a health check
            await self.get_balance()
            return True
        except Exception:
            return False
    
    def get_rate_limit_info(self) -> Dict[str, int]:
        """
        Get rate limit information for the exchange.
        
        Returns:
            Dictionary with rate limit information
        """
        return {
            "requests_per_minute": 60,
            "requests_per_second": 1
        }
