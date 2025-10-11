"""
Database manager for SQLite with async support.
"""

import aiosqlite
import json
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..core.data_models import (
    Alert, Balance, GridLevel, OHLCV, Order, PerformanceMetrics, 
    Position, StrategySignal, Trade
)


class DatabaseManager:
    """Async SQLite database manager."""
    
    def __init__(self, db_path: str = "data/trading_bot.db"):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = None
    
    async def connect(self) -> bool:
        """Connect to the database."""
        try:
            self.connection = await aiosqlite.connect(self.db_path)
            await self._create_tables()
            return True
        except Exception as e:
            print(f"Database connection error: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from the database."""
        try:
            if self.connection:
                await self.connection.close()
            return True
        except Exception as e:
            print(f"Database disconnection error: {e}")
            return False
    
    async def _create_tables(self):
        """Create database tables."""
        tables = [
            # OHLCV data
            """
            CREATE TABLE IF NOT EXISTS ohlcv (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                open DECIMAL(20, 8) NOT NULL,
                high DECIMAL(20, 8) NOT NULL,
                low DECIMAL(20, 8) NOT NULL,
                close DECIMAL(20, 8) NOT NULL,
                volume DECIMAL(20, 8) NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(timestamp, symbol, timeframe)
            )
            """,
            
            # Orders
            """
            CREATE TABLE IF NOT EXISTS orders (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                type TEXT NOT NULL,
                amount DECIMAL(20, 8) NOT NULL,
                price DECIMAL(20, 8),
                stop_price DECIMAL(20, 8),
                status TEXT NOT NULL,
                filled DECIMAL(20, 8) DEFAULT 0,
                remaining DECIMAL(20, 8) NOT NULL,
                cost DECIMAL(20, 8) DEFAULT 0,
                fee DECIMAL(20, 8) DEFAULT 0,
                timestamp DATETIME NOT NULL,
                exchange_order_id TEXT,
                exchange TEXT NOT NULL,
                strategy TEXT,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            # Positions
            """
            CREATE TABLE IF NOT EXISTS positions (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                amount DECIMAL(20, 8) NOT NULL,
                entry_price DECIMAL(20, 8) NOT NULL,
                current_price DECIMAL(20, 8) NOT NULL,
                unrealized_pnl DECIMAL(20, 8) NOT NULL,
                realized_pnl DECIMAL(20, 8) DEFAULT 0,
                fees DECIMAL(20, 8) DEFAULT 0,
                timestamp DATETIME NOT NULL,
                exchange TEXT NOT NULL,
                strategy TEXT,
                stop_loss DECIMAL(20, 8),
                take_profit DECIMAL(20, 8),
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            # Trades
            """
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                order_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                amount DECIMAL(20, 8) NOT NULL,
                price DECIMAL(20, 8) NOT NULL,
                cost DECIMAL(20, 8) NOT NULL,
                fee DECIMAL(20, 8) NOT NULL,
                timestamp DATETIME NOT NULL,
                exchange TEXT NOT NULL,
                strategy TEXT,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            # Strategy signals
            """
            CREATE TABLE IF NOT EXISTS strategy_signals (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                signal TEXT NOT NULL,
                strength REAL NOT NULL,
                price DECIMAL(20, 8) NOT NULL,
                timestamp DATETIME NOT NULL,
                strategy TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                indicators TEXT,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            # Balances
            """
            CREATE TABLE IF NOT EXISTS balances (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                currency TEXT NOT NULL,
                free DECIMAL(20, 8) NOT NULL,
                used DECIMAL(20, 8) NOT NULL,
                total DECIMAL(20, 8) NOT NULL,
                timestamp DATETIME NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            # Performance metrics
            """
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy TEXT NOT NULL,
                symbol TEXT NOT NULL,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                losing_trades INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0.0,
                total_pnl DECIMAL(20, 8) DEFAULT 0,
                realized_pnl DECIMAL(20, 8) DEFAULT 0,
                unrealized_pnl DECIMAL(20, 8) DEFAULT 0,
                max_drawdown DECIMAL(20, 8) DEFAULT 0,
                sharpe_ratio REAL DEFAULT 0.0,
                sortino_ratio REAL DEFAULT 0.0,
                profit_factor REAL DEFAULT 0.0,
                average_win DECIMAL(20, 8) DEFAULT 0,
                average_loss DECIMAL(20, 8) DEFAULT 0,
                largest_win DECIMAL(20, 8) DEFAULT 0,
                largest_loss DECIMAL(20, 8) DEFAULT 0,
                start_date DATETIME NOT NULL,
                end_date DATETIME NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            # Grid levels
            """
            CREATE TABLE IF NOT EXISTS grid_levels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                level INTEGER NOT NULL,
                price DECIMAL(20, 8) NOT NULL,
                side TEXT NOT NULL,
                amount DECIMAL(20, 8) NOT NULL,
                order_id TEXT,
                status TEXT NOT NULL,
                profit_target DECIMAL(20, 8) NOT NULL,
                timestamp DATETIME NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            # Alerts
            """
            CREATE TABLE IF NOT EXISTS alerts (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                message TEXT NOT NULL,
                symbol TEXT,
                price DECIMAL(20, 8),
                timestamp DATETIME NOT NULL,
                sent BOOLEAN DEFAULT FALSE,
                channels TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        ]
        
        for table_sql in tables:
            await self.connection.execute(table_sql)
        
        await self.connection.commit()
    
    async def save_ohlcv(self, ohlcv_data: List[OHLCV]) -> bool:
        """Save OHLCV data to database."""
        try:
            for ohlcv in ohlcv_data:
                await self.connection.execute(
                    """
                    INSERT OR REPLACE INTO ohlcv 
                    (timestamp, symbol, timeframe, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ohlcv.timestamp,
                        ohlcv.symbol,
                        ohlcv.timeframe.value,
                        str(ohlcv.open),
                        str(ohlcv.high),
                        str(ohlcv.low),
                        str(ohlcv.close),
                        str(ohlcv.volume)
                    )
                )
            await self.connection.commit()
            return True
        except Exception as e:
            print(f"Error saving OHLCV data: {e}")
            return False
    
    async def save_order(self, order: Order) -> bool:
        """Save order to database."""
        try:
            await self.connection.execute(
                """
                INSERT OR REPLACE INTO orders 
                (id, symbol, side, type, amount, price, stop_price, status, filled, 
                 remaining, cost, fee, timestamp, exchange_order_id, exchange, strategy, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    order.id,
                    order.symbol,
                    order.side.value,
                    order.type.value,
                    str(order.amount),
                    str(order.price) if order.price else None,
                    str(order.stop_price) if order.stop_price else None,
                    order.status.value,
                    str(order.filled),
                    str(order.remaining),
                    str(order.cost),
                    str(order.fee),
                    order.timestamp,
                    order.exchange_order_id,
                    order.exchange,
                    order.strategy,
                    json.dumps(order.metadata)
                )
            )
            await self.connection.commit()
            return True
        except Exception as e:
            print(f"Error saving order: {e}")
            return False
    
    async def save_position(self, position: Position) -> bool:
        """Save position to database."""
        try:
            await self.connection.execute(
                """
                INSERT OR REPLACE INTO positions 
                (id, symbol, side, amount, entry_price, current_price, unrealized_pnl,
                 realized_pnl, fees, timestamp, exchange, strategy, stop_loss, take_profit, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    position.id,
                    position.symbol,
                    position.side.value,
                    str(position.amount),
                    str(position.entry_price),
                    str(position.current_price),
                    str(position.unrealized_pnl),
                    str(position.realized_pnl),
                    str(position.fees),
                    position.timestamp,
                    position.exchange,
                    position.strategy,
                    str(position.stop_loss) if position.stop_loss else None,
                    str(position.take_profit) if position.take_profit else None,
                    json.dumps(position.metadata)
                )
            )
            await self.connection.commit()
            return True
        except Exception as e:
            print(f"Error saving position: {e}")
            return False
    
    async def save_trade(self, trade: Trade) -> bool:
        """Save trade to database."""
        try:
            await self.connection.execute(
                """
                INSERT OR REPLACE INTO trades 
                (id, order_id, symbol, side, amount, price, cost, fee, timestamp, exchange, strategy, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade.id,
                    trade.order_id,
                    trade.symbol,
                    trade.side.value,
                    str(trade.amount),
                    str(trade.price),
                    str(trade.cost),
                    str(trade.fee),
                    trade.timestamp,
                    trade.exchange,
                    trade.strategy,
                    json.dumps(trade.metadata)
                )
            )
            await self.connection.commit()
            return True
        except Exception as e:
            print(f"Error saving trade: {e}")
            return False
    
    async def save_strategy_signal(self, signal: StrategySignal) -> bool:
        """Save strategy signal to database."""
        try:
            await self.connection.execute(
                """
                INSERT OR REPLACE INTO strategy_signals 
                (id, symbol, signal, strength, price, timestamp, strategy, timeframe, indicators, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    signal.id,
                    signal.symbol,
                    signal.signal.value,
                    signal.strength,
                    str(signal.price),
                    signal.timestamp,
                    signal.strategy,
                    signal.timeframe.value,
                    json.dumps(signal.indicators),
                    json.dumps(signal.metadata)
                )
            )
            await self.connection.commit()
            return True
        except Exception as e:
            print(f"Error saving strategy signal: {e}")
            return False
    
    async def save_balance(self, balance: Balance) -> bool:
        """Save balance to database."""
        try:
            await self.connection.execute(
                """
                INSERT INTO balances (currency, free, used, total, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    balance.currency,
                    str(balance.free),
                    str(balance.used),
                    str(balance.total),
                    balance.timestamp
                )
            )
            await self.connection.commit()
            return True
        except Exception as e:
            print(f"Error saving balance: {e}")
            return False
    
    async def save_performance_metrics(self, metrics: PerformanceMetrics) -> bool:
        """Save performance metrics to database."""
        try:
            await self.connection.execute(
                """
                INSERT OR REPLACE INTO performance_metrics 
                (strategy, symbol, total_trades, winning_trades, losing_trades, win_rate,
                 total_pnl, realized_pnl, unrealized_pnl, max_drawdown, sharpe_ratio,
                 sortino_ratio, profit_factor, average_win, average_loss, largest_win,
                 largest_loss, start_date, end_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metrics.strategy,
                    metrics.symbol,
                    metrics.total_trades,
                    metrics.winning_trades,
                    metrics.losing_trades,
                    metrics.win_rate,
                    str(metrics.total_pnl),
                    str(metrics.realized_pnl),
                    str(metrics.unrealized_pnl),
                    str(metrics.max_drawdown),
                    metrics.sharpe_ratio,
                    metrics.sortino_ratio,
                    metrics.profit_factor,
                    str(metrics.average_win),
                    str(metrics.average_loss),
                    str(metrics.largest_win),
                    str(metrics.largest_loss),
                    metrics.start_date,
                    metrics.end_date
                )
            )
            await self.connection.commit()
            return True
        except Exception as e:
            print(f"Error saving performance metrics: {e}")
            return False
    
    async def get_ohlcv(
        self, 
        symbol: str, 
        timeframe: str, 
        limit: int = 1000,
        since: Optional[datetime] = None
    ) -> List[Dict]:
        """Get OHLCV data from database."""
        try:
            query = """
                SELECT timestamp, symbol, timeframe, open, high, low, close, volume
                FROM ohlcv 
                WHERE symbol = ? AND timeframe = ?
            """
            params = [symbol, timeframe]
            
            if since:
                query += " AND timestamp >= ?"
                params.append(since)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor = await self.connection.execute(query, params)
            rows = await cursor.fetchall()
            
            return [
                {
                    'timestamp': row[0],
                    'symbol': row[1],
                    'timeframe': row[2],
                    'open': Decimal(row[3]),
                    'high': Decimal(row[4]),
                    'low': Decimal(row[5]),
                    'close': Decimal(row[6]),
                    'volume': Decimal(row[7])
                }
                for row in rows
            ]
        except Exception as e:
            print(f"Error getting OHLCV data: {e}")
            return []
    
    async def get_orders(
        self, 
        symbol: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 1000
    ) -> List[Dict]:
        """Get orders from database."""
        try:
            query = "SELECT * FROM orders WHERE 1=1"
            params = []
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            if status:
                query += " AND status = ?"
                params.append(status)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor = await self.connection.execute(query, params)
            rows = await cursor.fetchall()
            
            return [dict(row) for row in rows]
        except Exception as e:
            print(f"Error getting orders: {e}")
            return []
    
    async def get_positions(
        self, 
        symbol: Optional[str] = None,
        exchange: Optional[str] = None
    ) -> List[Dict]:
        """Get positions from database."""
        try:
            query = "SELECT * FROM positions WHERE 1=1"
            params = []
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            if exchange:
                query += " AND exchange = ?"
                params.append(exchange)
            
            query += " ORDER BY timestamp DESC"
            
            cursor = await self.connection.execute(query, params)
            rows = await cursor.fetchall()
            
            return [dict(row) for row in rows]
        except Exception as e:
            print(f"Error getting positions: {e}")
            return []
    
    async def get_performance_metrics(
        self, 
        strategy: Optional[str] = None,
        symbol: Optional[str] = None
    ) -> List[Dict]:
        """Get performance metrics from database."""
        try:
            query = "SELECT * FROM performance_metrics WHERE 1=1"
            params = []
            
            if strategy:
                query += " AND strategy = ?"
                params.append(strategy)
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            query += " ORDER BY created_at DESC"
            
            cursor = await self.connection.execute(query, params)
            rows = await cursor.fetchall()
            
            return [dict(row) for row in rows]
        except Exception as e:
            print(f"Error getting performance metrics: {e}")
            return []
    
    async def cleanup_old_data(self, days: int = 30):
        """Clean up old data from database."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Clean up old OHLCV data (keep more recent)
            await self.connection.execute(
                "DELETE FROM ohlcv WHERE timestamp < ?",
                (cutoff_date,)
            )
            
            # Clean up old balances (keep more recent)
            await self.connection.execute(
                "DELETE FROM balances WHERE timestamp < ?",
                (cutoff_date,)
            )
            
            await self.connection.commit()
            print(f"Cleaned up data older than {days} days")
        except Exception as e:
            print(f"Error cleaning up old data: {e}")
