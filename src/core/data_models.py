"""
Data models for the trading bot using Pydantic for validation.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


class OrderSide(str, Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    """Order status enumeration."""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class PositionSide(str, Enum):
    """Position side enumeration."""
    LONG = "long"
    SHORT = "short"


class TimeFrame(str, Enum):
    """Timeframe enumeration."""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"
    MN1 = "1M"


class StrategySignal(str, Enum):
    """Strategy signal enumeration."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"


class OHLCV(BaseModel):
    """OHLCV data model."""
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    symbol: str
    timeframe: TimeFrame

    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }


class Balance(BaseModel):
    """Account balance model."""
    currency: str
    free: Decimal
    used: Decimal
    total: Decimal
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }


class Order(BaseModel):
    """Order model."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    symbol: str
    side: OrderSide
    type: OrderType
    amount: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    status: OrderStatus = OrderStatus.PENDING
    filled: Decimal = Decimal("0")
    remaining: Decimal
    cost: Decimal = Decimal("0")
    fee: Decimal = Decimal("0")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    exchange_order_id: Optional[str] = None
    exchange: str
    strategy: Optional[str] = None
    metadata: Dict[str, Union[str, int, float]] = Field(default_factory=dict)

    @validator('remaining', pre=True, always=True)
    def set_remaining(cls, v, values):
        if 'amount' in values:
            return values['amount'] - values.get('filled', Decimal("0"))
        return v

    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }


class Position(BaseModel):
    """Position model."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    symbol: str
    side: PositionSide
    amount: Decimal
    entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal = Decimal("0")
    fees: Decimal = Decimal("0")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    exchange: str
    strategy: Optional[str] = None
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    metadata: Dict[str, Union[str, int, float]] = Field(default_factory=dict)

    @validator('unrealized_pnl', pre=True, always=True)
    def calculate_unrealized_pnl(cls, v, values):
        if all(key in values for key in ['amount', 'entry_price', 'current_price', 'side']):
            amount = values['amount']
            entry_price = values['entry_price']
            current_price = values['current_price']
            side = values['side']
            
            if side == PositionSide.LONG:
                return amount * (current_price - entry_price)
            else:  # SHORT
                return amount * (entry_price - current_price)
        return v

    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }


class Trade(BaseModel):
    """Trade execution model."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    order_id: str
    symbol: str
    side: OrderSide
    amount: Decimal
    price: Decimal
    cost: Decimal
    fee: Decimal
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    exchange: str
    strategy: Optional[str] = None
    metadata: Dict[str, Union[str, int, float]] = Field(default_factory=dict)

    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }


class StrategySignal(BaseModel):
    """Strategy signal model."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    symbol: str
    signal: StrategySignal
    strength: float = Field(ge=0.0, le=1.0)  # Signal strength 0-1
    price: Decimal
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    strategy: str
    timeframe: TimeFrame
    indicators: Dict[str, Union[float, Decimal]] = Field(default_factory=dict)
    metadata: Dict[str, Union[str, int, float]] = Field(default_factory=dict)

    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }


class PerformanceMetrics(BaseModel):
    """Performance metrics model."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    profit_factor: float = 0.0
    average_win: Decimal = Decimal("0")
    average_loss: Decimal = Decimal("0")
    largest_win: Decimal = Decimal("0")
    largest_loss: Decimal = Decimal("0")
    start_date: datetime
    end_date: datetime
    strategy: str
    symbol: str

    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }


class GridLevel(BaseModel):
    """Grid level model for grid trading strategy."""
    level: int
    price: Decimal
    side: OrderSide
    amount: Decimal
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    profit_target: Decimal
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }


class Alert(BaseModel):
    """Alert model for notifications."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: str  # trade_executed, stop_loss, take_profit, error, etc.
    message: str
    symbol: Optional[str] = None
    price: Optional[Decimal] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    sent: bool = False
    channels: List[str] = Field(default_factory=list)  # telegram, discord, email

    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }
