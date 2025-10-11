"""
Event Logger for Trading Bot

This module logs all trading events, including signals, orders, trades,
and system events for monitoring and debugging purposes.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
import json
import uuid

from ..core.data_models import (
    StrategySignal, Order, Trade, Position, OrderSide, OrderType, 
    OrderStatus, PositionStatus
)
from ..database.db_manager import DatabaseManager


class EventType(Enum):
    """Types of events that can be logged."""
    SIGNAL_GENERATED = "signal_generated"
    ORDER_PLACED = "order_placed"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_FAILED = "order_failed"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_MODIFIED = "position_modified"
    STOP_LOSS_TRIGGERED = "stop_loss_triggered"
    TAKE_PROFIT_TRIGGERED = "take_profit_triggered"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    STRATEGY_CHANGE = "strategy_change"
    RISK_LIMIT_HIT = "risk_limit_hit"
    BALANCE_UPDATE = "balance_update"


class EventLogger:
    """Logs trading events and system events."""
    
    def __init__(self, database_manager: DatabaseManager):
        """
        Initialize the event logger.
        
        Args:
            database_manager: Database manager for persistence
        """
        self.database_manager = database_manager
        self.logger = logging.getLogger("EventLogger")
        
        # Event storage
        self.events = []  # In-memory event storage
        self.max_events = 10000  # Maximum events to keep in memory
        
        # Event filtering
        self.enabled_event_types = set(EventType)
        self.event_filters = {}
        
        # Performance tracking
        self.event_counts = {event_type: 0 for event_type in EventType}
        self.last_cleanup = datetime.now()
    
    def enable_event_type(self, event_type: EventType):
        """Enable logging for a specific event type."""
        self.enabled_event_types.add(event_type)
    
    def disable_event_type(self, event_type: EventType):
        """Disable logging for a specific event type."""
        self.enabled_event_types.discard(event_type)
    
    def set_event_filter(self, event_type: EventType, filter_func: callable):
        """Set a filter function for a specific event type."""
        self.event_filters[event_type] = filter_func
    
    async def log_event(self, 
                       event_type: EventType,
                       message: str,
                       data: Dict[str, Any] = None,
                       level: str = "INFO") -> str:
        """
        Log a general event.
        
        Args:
            event_type: Type of event
            message: Event message
            data: Additional event data
            level: Log level (DEBUG, INFO, WARNING, ERROR)
            
        Returns:
            Event ID
        """
        if event_type not in self.enabled_event_types:
            return None
        
        # Apply event filter if exists
        if event_type in self.event_filters:
            if not self.event_filters[event_type](data or {}):
                return None
        
        event_id = str(uuid.uuid4())
        event = {
            'id': event_id,
            'type': event_type.value,
            'message': message,
            'data': data or {},
            'level': level,
            'timestamp': datetime.now(),
            'created_at': datetime.now().isoformat()
        }
        
        # Add to in-memory storage
        self.events.append(event)
        
        # Update event count
        self.event_counts[event_type] += 1
        
        # Save to database
        await self._save_event_to_database(event)
        
        # Log to standard logger
        log_message = f"[{event_type.value}] {message}"
        if data:
            log_message += f" | Data: {json.dumps(data, default=str)}"
        
        if level == "ERROR":
            self.logger.error(log_message)
        elif level == "WARNING":
            self.logger.warning(log_message)
        elif level == "DEBUG":
            self.logger.debug(log_message)
        else:
            self.logger.info(log_message)
        
        # Cleanup old events if needed
        await self._cleanup_old_events()
        
        return event_id
    
    async def log_signal(self, signal: StrategySignal) -> str:
        """Log a trading signal."""
        data = {
            'symbol': signal.symbol,
            'side': signal.side.value,
            'price': signal.price,
            'strength': signal.strength,
            'timeframe': signal.timeframe,
            'strategy_name': signal.strategy_name,
            'metadata': signal.metadata
        }
        
        message = f"Signal generated: {signal.symbol} {signal.side.value} at {signal.price}"
        
        return await self.log_event(
            EventType.SIGNAL_GENERATED,
            message,
            data
        )
    
    async def log_order_execution(self, order: Order, signal: StrategySignal = None) -> str:
        """Log order execution."""
        data = {
            'order_id': order.id,
            'symbol': order.symbol,
            'side': order.side.value,
            'type': order.type.value,
            'quantity': order.quantity,
            'price': order.price,
            'status': order.status.value,
            'strategy_name': order.strategy_name,
            'metadata': order.metadata
        }
        
        if signal:
            data['signal_id'] = signal.metadata.get('signal_id')
            data['signal_strength'] = signal.strength
        
        message = f"Order executed: {order.symbol} {order.side.value} {order.quantity} at {order.price}"
        
        return await self.log_event(
            EventType.ORDER_PLACED,
            message,
            data
        )
    
    async def log_order_filled(self, order: Order) -> str:
        """Log order fill."""
        data = {
            'order_id': order.id,
            'symbol': order.symbol,
            'side': order.side.value,
            'quantity': order.quantity,
            'price': order.price,
            'filled_at': order.filled_at.isoformat() if order.filled_at else None,
            'strategy_name': order.strategy_name
        }
        
        message = f"Order filled: {order.symbol} {order.side.value} {order.quantity}"
        
        return await self.log_event(
            EventType.ORDER_FILLED,
            message,
            data
        )
    
    async def log_order_cancelled(self, order: Order) -> str:
        """Log order cancellation."""
        data = {
            'order_id': order.id,
            'symbol': order.symbol,
            'side': order.side.value,
            'quantity': order.quantity,
            'price': order.price,
            'cancelled_at': order.cancelled_at.isoformat() if order.cancelled_at else None,
            'strategy_name': order.strategy_name
        }
        
        message = f"Order cancelled: {order.symbol} {order.side.value} {order.quantity}"
        
        return await self.log_event(
            EventType.ORDER_CANCELLED,
            message,
            data
        )
    
    async def log_order_failed(self, order: Order, error: str) -> str:
        """Log order failure."""
        data = {
            'order_id': order.id,
            'symbol': order.symbol,
            'side': order.side.value,
            'quantity': order.quantity,
            'price': order.price,
            'error': error,
            'failed_at': order.failed_at.isoformat() if order.failed_at else None,
            'strategy_name': order.strategy_name
        }
        
        message = f"Order failed: {order.symbol} {order.side.value} {order.quantity} - {error}"
        
        return await self.log_event(
            EventType.ORDER_FAILED,
            message,
            data,
            level="ERROR"
        )
    
    async def log_position_opened(self, position: Position) -> str:
        """Log position opening."""
        data = {
            'position_id': position.id,
            'symbol': position.symbol,
            'side': position.side.value,
            'quantity': position.quantity,
            'entry_price': position.entry_price,
            'strategy_name': position.strategy_name,
            'metadata': position.metadata
        }
        
        message = f"Position opened: {position.symbol} {position.side.value} {position.quantity} at {position.entry_price}"
        
        return await self.log_event(
            EventType.POSITION_OPENED,
            message,
            data
        )
    
    async def log_position_closed(self, position: Position, reason: str) -> str:
        """Log position closing."""
        data = {
            'position_id': position.id,
            'symbol': position.symbol,
            'side': position.side.value,
            'quantity': position.quantity,
            'entry_price': position.entry_price,
            'exit_price': position.current_price,
            'close_reason': reason,
            'final_pnl': position.metadata.get('final_pnl', 0),
            'strategy_name': position.strategy_name,
            'metadata': position.metadata
        }
        
        pnl = position.metadata.get('final_pnl', 0)
        message = f"Position closed: {position.symbol} {position.side.value} {position.quantity} (PnL: ${pnl:.2f}) - {reason}"
        
        return await self.log_event(
            EventType.POSITION_CLOSED,
            message,
            data
        )
    
    async def log_position_modified(self, position: Position, changes: Dict[str, Any]) -> str:
        """Log position modification."""
        data = {
            'position_id': position.id,
            'symbol': position.symbol,
            'changes': changes,
            'strategy_name': position.strategy_name
        }
        
        message = f"Position modified: {position.symbol} - {', '.join(f'{k}: {v}' for k, v in changes.items())}"
        
        return await self.log_event(
            EventType.POSITION_MODIFIED,
            message,
            data
        )
    
    async def log_stop_loss_triggered(self, position: Position, stop_price: float) -> str:
        """Log stop-loss trigger."""
        data = {
            'position_id': position.id,
            'symbol': position.symbol,
            'side': position.side.value,
            'stop_price': stop_price,
            'current_price': position.current_price,
            'strategy_name': position.strategy_name
        }
        
        message = f"Stop-loss triggered: {position.symbol} at {stop_price}"
        
        return await self.log_event(
            EventType.STOP_LOSS_TRIGGERED,
            message,
            data,
            level="WARNING"
        )
    
    async def log_take_profit_triggered(self, position: Position, take_profit_price: float) -> str:
        """Log take-profit trigger."""
        data = {
            'position_id': position.id,
            'symbol': position.symbol,
            'side': position.side.value,
            'take_profit_price': take_profit_price,
            'current_price': position.current_price,
            'strategy_name': position.strategy_name
        }
        
        message = f"Take-profit triggered: {position.symbol} at {take_profit_price}"
        
        return await self.log_event(
            EventType.TAKE_PROFIT_TRIGGERED,
            message,
            data
        )
    
    async def log_error(self, error: str, context: Dict[str, Any] = None) -> str:
        """Log an error."""
        data = {
            'error': error,
            'context': context or {}
        }
        
        message = f"Error: {error}"
        
        return await self.log_event(
            EventType.ERROR,
            message,
            data,
            level="ERROR"
        )
    
    async def log_warning(self, warning: str, context: Dict[str, Any] = None) -> str:
        """Log a warning."""
        data = {
            'warning': warning,
            'context': context or {}
        }
        
        message = f"Warning: {warning}"
        
        return await self.log_event(
            EventType.WARNING,
            message,
            data,
            level="WARNING"
        )
    
    async def log_info(self, info: str, context: Dict[str, Any] = None) -> str:
        """Log an info message."""
        data = {
            'info': info,
            'context': context or {}
        }
        
        message = f"Info: {info}"
        
        return await self.log_event(
            EventType.INFO,
            message,
            data
        )
    
    async def log_system_start(self, config: Dict[str, Any]) -> str:
        """Log system start."""
        data = {
            'config': config,
            'start_time': datetime.now().isoformat()
        }
        
        message = "Trading system started"
        
        return await self.log_event(
            EventType.SYSTEM_START,
            message,
            data
        )
    
    async def log_system_stop(self, reason: str = "manual") -> str:
        """Log system stop."""
        data = {
            'reason': reason,
            'stop_time': datetime.now().isoformat()
        }
        
        message = f"Trading system stopped - {reason}"
        
        return await self.log_event(
            EventType.SYSTEM_STOP,
            message,
            data
        )
    
    async def log_strategy_change(self, old_strategy: str, new_strategy: str) -> str:
        """Log strategy change."""
        data = {
            'old_strategy': old_strategy,
            'new_strategy': new_strategy,
            'change_time': datetime.now().isoformat()
        }
        
        message = f"Strategy changed from {old_strategy} to {new_strategy}"
        
        return await self.log_event(
            EventType.STRATEGY_CHANGE,
            message,
            data
        )
    
    async def log_risk_limit_hit(self, limit_type: str, current_value: float, limit_value: float) -> str:
        """Log risk limit hit."""
        data = {
            'limit_type': limit_type,
            'current_value': current_value,
            'limit_value': limit_value,
            'hit_time': datetime.now().isoformat()
        }
        
        message = f"Risk limit hit: {limit_type} ({current_value} > {limit_value})"
        
        return await self.log_event(
            EventType.RISK_LIMIT_HIT,
            message,
            data,
            level="WARNING"
        )
    
    async def log_balance_update(self, balance: Dict[str, float]) -> str:
        """Log balance update."""
        data = {
            'balance': balance,
            'update_time': datetime.now().isoformat()
        }
        
        total_balance = sum(balance.values())
        message = f"Balance updated: ${total_balance:.2f}"
        
        return await self.log_event(
            EventType.BALANCE_UPDATE,
            message,
            data
        )
    
    async def _save_event_to_database(self, event: Dict[str, Any]):
        """Save event to database."""
        try:
            # This would normally save to database
            # For now, just log that we would save it
            pass
        except Exception as e:
            self.logger.error(f"Error saving event to database: {e}")
    
    async def _cleanup_old_events(self):
        """Clean up old events from memory."""
        # Only cleanup every hour
        if (datetime.now() - self.last_cleanup).total_seconds() < 3600:
            return
        
        # Keep only recent events
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        old_events = [
            event for event in self.events
            if event['timestamp'] < cutoff_time
        ]
        
        for event in old_events:
            self.events.remove(event)
        
        self.last_cleanup = datetime.now()
        
        if old_events:
            self.logger.info(f"Cleaned up {len(old_events)} old events")
    
    def get_events(self, 
                   event_type: EventType = None,
                   limit: int = 100,
                   since: datetime = None) -> List[Dict[str, Any]]:
        """
        Get events from memory.
        
        Args:
            event_type: Filter by event type
            limit: Maximum number of events to return
            since: Only return events since this time
            
        Returns:
            List of events
        """
        filtered_events = self.events
        
        # Filter by event type
        if event_type:
            filtered_events = [
                event for event in filtered_events
                if event['type'] == event_type.value
            ]
        
        # Filter by time
        if since:
            filtered_events = [
                event for event in filtered_events
                if event['timestamp'] >= since
            ]
        
        # Sort by timestamp (newest first)
        filtered_events.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Limit results
        return filtered_events[:limit]
    
    def get_event_statistics(self) -> Dict[str, Any]:
        """Get event logging statistics."""
        return {
            'total_events': len(self.events),
            'event_counts': {event_type.value: count for event_type, count in self.event_counts.items()},
            'enabled_event_types': [event_type.value for event_type in self.enabled_event_types],
            'last_cleanup': self.last_cleanup.isoformat(),
            'max_events': self.max_events
        }
    
    def get_recent_events(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent events from the last N hours."""
        since = datetime.now() - timedelta(hours=hours)
        return self.get_events(since=since)
    
    def get_events_by_symbol(self, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get events for a specific symbol."""
        symbol_events = [
            event for event in self.events
            if event.get('data', {}).get('symbol') == symbol
        ]
        
        symbol_events.sort(key=lambda x: x['timestamp'], reverse=True)
        return symbol_events[:limit]
    
    def get_error_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent error events."""
        error_events = [
            event for event in self.events
            if event['level'] == 'ERROR'
        ]
        
        error_events.sort(key=lambda x: x['timestamp'], reverse=True)
        return error_events[:limit]
    
    def get_warning_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent warning events."""
        warning_events = [
            event for event in self.events
            if event['level'] == 'WARNING'
        ]
        
        warning_events.sort(key=lambda x: x['timestamp'], reverse=True)
        return warning_events[:limit]
    
    def clear_events(self):
        """Clear all events from memory."""
        self.events.clear()
        self.event_counts = {event_type: 0 for event_type in EventType}
        self.logger.info("All events cleared from memory")
    
    def export_events(self, file_path: str, event_type: EventType = None):
        """Export events to a JSON file."""
        try:
            events = self.get_events(event_type=event_type)
            
            # Convert datetime objects to strings
            export_data = []
            for event in events:
                export_event = event.copy()
                export_event['timestamp'] = event['timestamp'].isoformat()
                export_data.append(export_event)
            
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Exported {len(export_data)} events to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting events: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from events."""
        # Count different event types
        signal_count = self.event_counts[EventType.SIGNAL_GENERATED]
        order_count = self.event_counts[EventType.ORDER_PLACED]
        filled_count = self.event_counts[EventType.ORDER_FILLED]
        failed_count = self.event_counts[EventType.ORDER_FAILED]
        
        # Calculate success rate
        success_rate = (filled_count / max(1, order_count)) * 100 if order_count > 0 else 0
        
        return {
            'signals_generated': signal_count,
            'orders_placed': order_count,
            'orders_filled': filled_count,
            'orders_failed': failed_count,
            'success_rate': success_rate,
            'total_events': sum(self.event_counts.values())
        }
