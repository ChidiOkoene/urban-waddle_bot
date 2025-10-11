"""
Position Monitor for Trading Bot

This module monitors active positions, manages stop-loss and take-profit orders,
and handles position lifecycle events.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import uuid

from ..core.exchange_interface import ExchangeInterface
from ..core.data_models import (
    Position, Order, Trade, OrderSide, OrderType, PositionStatus
)
from ..database.db_manager import DatabaseManager


class PositionMonitor:
    """Monitors and manages trading positions."""
    
    def __init__(self, 
                 exchange_adapter: ExchangeInterface,
                 database_manager: DatabaseManager):
        """
        Initialize the position monitor.
        
        Args:
            exchange_adapter: Exchange adapter for market data
            database_manager: Database manager for persistence
        """
        self.exchange_adapter = exchange_adapter
        self.database_manager = database_manager
        self.logger = logging.getLogger("PositionMonitor")
        
        # Position tracking
        self.active_positions = {}  # position_id -> Position
        self.closed_positions = {}  # position_id -> Position
        
        # Order tracking for positions
        self.position_orders = {}  # position_id -> List[Order]
        
        # Monitoring settings
        self.update_interval = 30  # seconds
        self.max_position_age = 30 * 24 * 60 * 60  # 30 days in seconds
    
    async def initialize(self):
        """Initialize the position monitor."""
        # Load existing positions from database
        await self._load_existing_positions()
        
        self.logger.info("Position monitor initialized")
    
    async def _load_existing_positions(self):
        """Load existing positions from database."""
        try:
            positions = await self.database_manager.get_active_positions()
            for position in positions:
                self.active_positions[position.id] = position
                
                # Load associated orders
                orders = await self.database_manager.get_position_orders(position.id)
                self.position_orders[position.id] = orders
            
            self.logger.info(f"Loaded {len(positions)} existing positions")
            
        except Exception as e:
            self.logger.error(f"Error loading existing positions: {e}")
    
    async def create_position(self, 
                            order: Order,
                            entry_price: float,
                            quantity: float) -> Position:
        """
        Create a new position from an executed order.
        
        Args:
            order: Executed order
            entry_price: Entry price of the position
            quantity: Quantity of the position
            
        Returns:
            Created position
        """
        try:
            position = Position(
                id=str(uuid.uuid4()),
                symbol=order.symbol,
                side=order.side,
                quantity=quantity,
                entry_price=entry_price,
                current_price=entry_price,
                status=PositionStatus.OPEN,
                timestamp=datetime.now(),
                strategy_name=order.strategy_name,
                metadata={
                    'order_id': order.id,
                    'created_from_order': True,
                    'entry_timestamp': datetime.now().isoformat()
                }
            )
            
            # Add to active positions
            self.active_positions[position.id] = position
            self.position_orders[position.id] = [order]
            
            # Save to database
            await self.database_manager.save_position(position)
            
            self.logger.info(f"Created position: {position.symbol} {position.side.value} {position.quantity}")
            
            return position
            
        except Exception as e:
            self.logger.error(f"Error creating position: {e}")
            raise
    
    async def update_position_pnl(self, position: Position) -> float:
        """
        Update position PnL based on current market price.
        
        Args:
            position: Position to update
            
        Returns:
            Updated unrealized PnL
        """
        try:
            # Get current market price
            ticker = await self.exchange_adapter.get_ticker(position.symbol)
            current_price = ticker.get('last', position.current_price)
            
            # Update current price
            position.current_price = current_price
            
            # Calculate unrealized PnL
            if position.side == OrderSide.BUY:
                unrealized_pnl = (current_price - position.entry_price) * position.quantity
            else:
                unrealized_pnl = (position.entry_price - current_price) * position.quantity
            
            # Update position metadata
            position.metadata['unrealized_pnl'] = unrealized_pnl
            position.metadata['pnl_percentage'] = (unrealized_pnl / (position.entry_price * position.quantity)) * 100
            position.metadata['last_price_update'] = datetime.now().isoformat()
            
            # Update in database
            await self.database_manager.update_position(position)
            
            return unrealized_pnl
            
        except Exception as e:
            self.logger.error(f"Error updating position PnL: {e}")
            return 0.0
    
    async def check_exit_conditions(self, position: Position) -> bool:
        """
        Check if position should be closed based on exit conditions.
        
        Args:
            position: Position to check
            
        Returns:
            True if position should be closed
        """
        try:
            # Update PnL first
            unrealized_pnl = await self.update_position_pnl(position)
            
            # Check stop-loss
            stop_loss_price = position.metadata.get('stop_loss_price')
            if stop_loss_price:
                if self._should_trigger_stop_loss(position, stop_loss_price):
                    self.logger.info(f"Stop-loss triggered for {position.symbol} at {stop_loss_price}")
                    await self.close_position(position, reason="stop_loss")
                    return True
            
            # Check take-profit
            take_profit_price = position.metadata.get('take_profit_price')
            if take_profit_price:
                if self._should_trigger_take_profit(position, take_profit_price):
                    self.logger.info(f"Take-profit triggered for {position.symbol} at {take_profit_price}")
                    await self.close_position(position, reason="take_profit")
                    return True
            
            # Check time-based exit
            max_hold_time = position.metadata.get('max_hold_hours')
            if max_hold_time:
                hold_time = (datetime.now() - position.timestamp).total_seconds() / 3600
                if hold_time >= max_hold_time:
                    self.logger.info(f"Max hold time reached for {position.symbol}")
                    await self.close_position(position, reason="max_hold_time")
                    return True
            
            # Check trailing stop
            trailing_stop = position.metadata.get('trailing_stop')
            if trailing_stop:
                await self._update_trailing_stop(position, trailing_stop)
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking exit conditions: {e}")
            return False
    
    def _should_trigger_stop_loss(self, position: Position, stop_loss_price: float) -> bool:
        """Check if stop-loss should be triggered."""
        if position.side == OrderSide.BUY:
            return position.current_price <= stop_loss_price
        else:
            return position.current_price >= stop_loss_price
    
    def _should_trigger_take_profit(self, position: Position, take_profit_price: float) -> bool:
        """Check if take-profit should be triggered."""
        if position.side == OrderSide.BUY:
            return position.current_price >= take_profit_price
        else:
            return position.current_price <= take_profit_price
    
    async def _update_trailing_stop(self, position: Position, trailing_stop_pct: float):
        """Update trailing stop price."""
        try:
            current_price = position.current_price
            entry_price = position.entry_price
            
            if position.side == OrderSide.BUY:
                # For long positions, trail stop below current price
                new_stop_price = current_price * (1 - trailing_stop_pct / 100)
                current_stop = position.metadata.get('stop_loss_price', 0)
                
                if new_stop_price > current_stop:
                    position.metadata['stop_loss_price'] = new_stop_price
                    await self.database_manager.update_position(position)
                    
            else:
                # For short positions, trail stop above current price
                new_stop_price = current_price * (1 + trailing_stop_pct / 100)
                current_stop = position.metadata.get('stop_loss_price', float('inf'))
                
                if new_stop_price < current_stop:
                    position.metadata['stop_loss_price'] = new_stop_price
                    await self.database_manager.update_position(position)
                    
        except Exception as e:
            self.logger.error(f"Error updating trailing stop: {e}")
    
    async def close_position(self, 
                           position: Position, 
                           reason: str = "manual") -> Optional[Order]:
        """
        Close a position.
        
        Args:
            position: Position to close
            reason: Reason for closing
            
        Returns:
            Closing order if successful
        """
        try:
            # Create closing order
            from .order_executor import OrderExecutor
            order_executor = OrderExecutor(self.exchange_adapter, self.database_manager)
            
            closing_order = await order_executor.close_position(position)
            
            if closing_order:
                # Update position status
                position.status = PositionStatus.CLOSED
                position.closed_at = datetime.now()
                position.metadata['close_reason'] = reason
                position.metadata['closing_order_id'] = closing_order.id
                
                # Calculate final PnL
                final_pnl = await self.update_position_pnl(position)
                position.metadata['final_pnl'] = final_pnl
                
                # Move to closed positions
                self.closed_positions[position.id] = position
                del self.active_positions[position.id]
                
                # Update in database
                await self.database_manager.update_position(position)
                
                self.logger.info(f"Position closed: {position.symbol} (reason: {reason})")
                
                return closing_order
            else:
                self.logger.error(f"Failed to close position: {position.symbol}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            return None
    
    async def modify_position(self, position: Position) -> bool:
        """
        Modify position parameters (stop-loss, take-profit, etc.).
        
        Args:
            position: Position to modify
            
        Returns:
            True if modified successfully
        """
        try:
            # This would normally implement position modification logic
            # For now, just update the position in the database
            await self.database_manager.update_position(position)
            
            self.logger.info(f"Position modified: {position.symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error modifying position: {e}")
            return False
    
    async def should_modify_position(self, position: Position) -> bool:
        """
        Check if position should be modified.
        
        Args:
            position: Position to check
            
        Returns:
            True if position should be modified
        """
        try:
            # Check if position has been open for a certain time
            time_open = (datetime.now() - position.timestamp).total_seconds()
            
            # Modify if position has been open for more than 1 hour
            if time_open > 3600:
                return True
            
            # Check if PnL has changed significantly
            unrealized_pnl = position.metadata.get('unrealized_pnl', 0)
            entry_value = position.entry_price * position.quantity
            
            if abs(unrealized_pnl) > entry_value * 0.05:  # 5% change
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking if position should be modified: {e}")
            return False
    
    async def get_active_positions(self) -> List[Position]:
        """Get all active positions."""
        return list(self.active_positions.values())
    
    async def get_closed_positions(self) -> List[Position]:
        """Get all closed positions."""
        return list(self.closed_positions.values())
    
    async def get_position_by_id(self, position_id: str) -> Optional[Position]:
        """Get position by ID."""
        if position_id in self.active_positions:
            return self.active_positions[position_id]
        elif position_id in self.closed_positions:
            return self.closed_positions[position_id]
        return None
    
    async def get_positions_by_symbol(self, symbol: str) -> List[Position]:
        """Get all positions for a specific symbol."""
        positions = []
        
        for position in self.active_positions.values():
            if position.symbol == symbol:
                positions.append(position)
        
        for position in self.closed_positions.values():
            if position.symbol == symbol:
                positions.append(position)
        
        return positions
    
    async def set_stop_loss(self, 
                           position: Position, 
                           stop_price: float) -> bool:
        """
        Set stop-loss for a position.
        
        Args:
            position: Position to set stop-loss for
            stop_price: Stop-loss price
            
        Returns:
            True if set successfully
        """
        try:
            position.metadata['stop_loss_price'] = stop_price
            await self.database_manager.update_position(position)
            
            self.logger.info(f"Stop-loss set for {position.symbol} at {stop_price}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting stop-loss: {e}")
            return False
    
    async def set_take_profit(self, 
                            position: Position, 
                            take_profit_price: float) -> bool:
        """
        Set take-profit for a position.
        
        Args:
            position: Position to set take-profit for
            take_profit_price: Take-profit price
            
        Returns:
            True if set successfully
        """
        try:
            position.metadata['take_profit_price'] = take_profit_price
            await self.database_manager.update_position(position)
            
            self.logger.info(f"Take-profit set for {position.symbol} at {take_profit_price}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting take-profit: {e}")
            return False
    
    async def set_trailing_stop(self, 
                               position: Position, 
                               trailing_stop_pct: float) -> bool:
        """
        Set trailing stop for a position.
        
        Args:
            position: Position to set trailing stop for
            trailing_stop_pct: Trailing stop percentage
            
        Returns:
            True if set successfully
        """
        try:
            position.metadata['trailing_stop'] = trailing_stop_pct
            await self.database_manager.update_position(position)
            
            self.logger.info(f"Trailing stop set for {position.symbol} at {trailing_stop_pct}%")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting trailing stop: {e}")
            return False
    
    async def monitor_all_positions(self):
        """Monitor all active positions."""
        try:
            for position in list(self.active_positions.values()):
                # Check exit conditions
                should_close = await self.check_exit_conditions(position)
                
                if should_close:
                    continue  # Position was closed
                
                # Check if position should be modified
                if await self.should_modify_position(position):
                    await self.modify_position(position)
            
        except Exception as e:
            self.logger.error(f"Error monitoring positions: {e}")
    
    def get_position_statistics(self) -> Dict[str, Any]:
        """Get position statistics."""
        active_positions = list(self.active_positions.values())
        closed_positions = list(self.closed_positions.values())
        
        # Calculate statistics
        total_positions = len(active_positions) + len(closed_positions)
        open_positions = len(active_positions)
        closed_positions_count = len(closed_positions)
        
        # Calculate PnL statistics
        total_pnl = 0.0
        winning_positions = 0
        losing_positions = 0
        
        for position in closed_positions:
            pnl = position.metadata.get('final_pnl', 0)
            total_pnl += pnl
            
            if pnl > 0:
                winning_positions += 1
            elif pnl < 0:
                losing_positions += 1
        
        win_rate = (winning_positions / max(1, closed_positions_count)) * 100
        
        return {
            'total_positions': total_positions,
            'open_positions': open_positions,
            'closed_positions': closed_positions_count,
            'total_pnl': total_pnl,
            'winning_positions': winning_positions,
            'losing_positions': losing_positions,
            'win_rate': win_rate,
            'avg_position_value': sum(p.entry_price * p.quantity for p in active_positions) / max(1, open_positions)
        }
    
    async def cleanup_old_positions(self, max_age_days: int = 30):
        """Clean up old closed positions from memory."""
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        
        old_positions = {
            pos_id: position for pos_id, position in self.closed_positions.items()
            if position.closed_at and position.closed_at < cutoff_time
        }
        
        for pos_id in old_positions:
            del self.closed_positions[pos_id]
        
        self.logger.info(f"Cleaned up {len(old_positions)} old positions")
    
    async def get_position_performance(self, symbol: str = None) -> Dict[str, Any]:
        """Get position performance metrics."""
        positions = []
        
        if symbol:
            positions = await self.get_positions_by_symbol(symbol)
        else:
            positions = list(self.closed_positions.values())
        
        if not positions:
            return {}
        
        # Calculate performance metrics
        total_pnl = sum(p.metadata.get('final_pnl', 0) for p in positions)
        avg_pnl = total_pnl / len(positions)
        
        winning_positions = [p for p in positions if p.metadata.get('final_pnl', 0) > 0]
        losing_positions = [p for p in positions if p.metadata.get('final_pnl', 0) < 0]
        
        avg_win = sum(p.metadata.get('final_pnl', 0) for p in winning_positions) / max(1, len(winning_positions))
        avg_loss = sum(p.metadata.get('final_pnl', 0) for p in losing_positions) / max(1, len(losing_positions))
        
        return {
            'total_positions': len(positions),
            'winning_positions': len(winning_positions),
            'losing_positions': len(losing_positions),
            'win_rate': (len(winning_positions) / len(positions)) * 100,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0
        }
