"""
Order Executor for Trading Bot

This module handles order execution, including order placement, modification,
and cancellation on various exchanges.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import uuid

from ..core.exchange_interface import ExchangeInterface
from ..core.data_models import (
    Order, Trade, Position, StrategySignal, OrderSide, OrderType, 
    OrderStatus, PositionStatus
)
from ..database.db_manager import DatabaseManager


class OrderExecutor:
    """Handles order execution and management."""
    
    def __init__(self, 
                 exchange_adapter: ExchangeInterface,
                 database_manager: DatabaseManager):
        """
        Initialize the order executor.
        
        Args:
            exchange_adapter: Exchange adapter for order execution
            database_manager: Database manager for persistence
        """
        self.exchange_adapter = exchange_adapter
        self.database_manager = database_manager
        self.logger = logging.getLogger("OrderExecutor")
        
        # Order tracking
        self.pending_orders = {}  # order_id -> Order
        self.executed_orders = {}  # order_id -> Order
        self.failed_orders = {}    # order_id -> Order
        
        # Execution settings
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds
        self.order_timeout = 300  # 5 minutes
    
    async def initialize(self):
        """Initialize the order executor."""
        self.logger.info("Order executor initialized")
    
    async def execute_order(self, 
                          signal: StrategySignal,
                          position_size: float,
                          order_type: OrderType = OrderType.MARKET) -> Optional[Order]:
        """
        Execute an order based on a trading signal.
        
        Args:
            signal: Trading signal
            position_size: Size of the position
            order_type: Type of order to place
            
        Returns:
            Created order or None if failed
        """
        try:
            # Create order object
            order = Order(
                id=str(uuid.uuid4()),
                symbol=signal.symbol,
                side=signal.side,
                type=order_type,
                quantity=position_size,
                price=signal.price if order_type == OrderType.LIMIT else None,
                status=OrderStatus.PENDING,
                timestamp=datetime.now(),
                strategy_name=signal.strategy_name,
                metadata={
                    'signal_id': signal.metadata.get('signal_id'),
                    'signal_strength': signal.strength,
                    'signal_timestamp': signal.timestamp.isoformat()
                }
            )
            
            # Add to pending orders
            self.pending_orders[order.id] = order
            
            # Save to database
            await self.database_manager.save_order(order)
            
            # Execute the order
            success = await self._execute_order_on_exchange(order)
            
            if success:
                # Move to executed orders
                self.executed_orders[order.id] = order
                del self.pending_orders[order.id]
                
                # Create trade record
                trade = await self._create_trade_from_order(order)
                if trade:
                    await self.database_manager.save_trade(trade)
                
                self.logger.info(f"Order executed successfully: {order.symbol} {order.side.value} {order.quantity}")
                return order
            else:
                # Move to failed orders
                self.failed_orders[order.id] = order
                del self.pending_orders[order.id]
                
                self.logger.error(f"Order execution failed: {order.symbol} {order.side.value} {order.quantity}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error executing order: {e}")
            return None
    
    async def _execute_order_on_exchange(self, order: Order) -> bool:
        """Execute order on the exchange."""
        retry_count = 0
        
        while retry_count < self.max_retries:
            try:
                # Place order on exchange
                exchange_order_id = await self.exchange_adapter.place_order(
                    symbol=order.symbol,
                    side=order.side,
                    order_type=order.type,
                    quantity=order.quantity,
                    price=order.price
                )
                
                if exchange_order_id:
                    # Update order with exchange ID
                    order.exchange_order_id = exchange_order_id
                    order.status = OrderStatus.FILLED
                    order.filled_at = datetime.now()
                    
                    # Update in database
                    await self.database_manager.update_order(order)
                    
                    return True
                else:
                    retry_count += 1
                    if retry_count < self.max_retries:
                        await asyncio.sleep(self.retry_delay * retry_count)
                    
            except Exception as e:
                self.logger.error(f"Error placing order on exchange (attempt {retry_count + 1}): {e}")
                retry_count += 1
                if retry_count < self.max_retries:
                    await asyncio.sleep(self.retry_delay * retry_count)
        
        # Mark order as failed
        order.status = OrderStatus.FAILED
        order.failed_at = datetime.now()
        await self.database_manager.update_order(order)
        
        return False
    
    async def _create_trade_from_order(self, order: Order) -> Optional[Trade]:
        """Create trade record from executed order."""
        try:
            # Get current market price for PnL calculation
            ticker = await self.exchange_adapter.get_ticker(order.symbol)
            current_price = ticker.get('last', order.price or 0)
            
            trade = Trade(
                id=str(uuid.uuid4()),
                order_id=order.id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=current_price,
                timestamp=datetime.now(),
                strategy_name=order.strategy_name,
                metadata=order.metadata.copy()
            )
            
            return trade
            
        except Exception as e:
            self.logger.error(f"Error creating trade from order: {e}")
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancelled successfully
        """
        try:
            if order_id not in self.pending_orders:
                self.logger.warning(f"Order {order_id} not found in pending orders")
                return False
            
            order = self.pending_orders[order_id]
            
            # Cancel on exchange
            success = await self.exchange_adapter.cancel_order(
                order_id=order.exchange_order_id or order_id,
                symbol=order.symbol
            )
            
            if success:
                # Update order status
                order.status = OrderStatus.CANCELLED
                order.cancelled_at = datetime.now()
                
                # Move to executed orders (for tracking)
                self.executed_orders[order_id] = order
                del self.pending_orders[order_id]
                
                # Update in database
                await self.database_manager.update_order(order)
                
                self.logger.info(f"Order cancelled: {order_id}")
                return True
            else:
                self.logger.error(f"Failed to cancel order: {order_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    async def modify_order(self, 
                          order_id: str,
                          new_quantity: Optional[float] = None,
                          new_price: Optional[float] = None) -> bool:
        """
        Modify a pending order.
        
        Args:
            order_id: Order ID to modify
            new_quantity: New quantity (if None, keep current)
            new_price: New price (if None, keep current)
            
        Returns:
            True if modified successfully
        """
        try:
            if order_id not in self.pending_orders:
                self.logger.warning(f"Order {order_id} not found in pending orders")
                return False
            
            order = self.pending_orders[order_id]
            
            # Update order parameters
            if new_quantity is not None:
                order.quantity = new_quantity
            if new_price is not None:
                order.price = new_price
            
            order.modified_at = datetime.now()
            
            # Modify on exchange (if supported)
            success = await self.exchange_adapter.modify_order(
                order_id=order.exchange_order_id or order_id,
                symbol=order.symbol,
                quantity=new_quantity,
                price=new_price
            )
            
            if success:
                # Update in database
                await self.database_manager.update_order(order)
                
                self.logger.info(f"Order modified: {order_id}")
                return True
            else:
                self.logger.error(f"Failed to modify order: {order_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error modifying order {order_id}: {e}")
            return False
    
    async def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """
        Get order status from exchange.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order status or None if not found
        """
        try:
            # Check pending orders first
            if order_id in self.pending_orders:
                order = self.pending_orders[order_id]
                if order.exchange_order_id:
                    status = await self.exchange_adapter.get_order_status(
                        order_id=order.exchange_order_id,
                        symbol=order.symbol
                    )
                    if status:
                        order.status = status.status
                        await self.database_manager.update_order(order)
                        return status.status
            
            # Check executed orders
            elif order_id in self.executed_orders:
                return self.executed_orders[order_id].status
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting order status {order_id}: {e}")
            return None
    
    async def get_recent_trades(self, limit: int = 50) -> List[Trade]:
        """
        Get recent trades.
        
        Args:
            limit: Maximum number of trades to return
            
        Returns:
            List of recent trades
        """
        try:
            return await self.database_manager.get_recent_trades(limit)
        except Exception as e:
            self.logger.error(f"Error getting recent trades: {e}")
            return []
    
    async def get_pending_orders(self) -> List[Order]:
        """Get all pending orders."""
        return list(self.pending_orders.values())
    
    async def get_executed_orders(self) -> List[Order]:
        """Get all executed orders."""
        return list(self.executed_orders.values())
    
    async def get_failed_orders(self) -> List[Order]:
        """Get all failed orders."""
        return list(self.failed_orders.values())
    
    async def place_stop_loss_order(self, 
                                  position: Position,
                                  stop_price: float) -> Optional[Order]:
        """
        Place a stop-loss order for a position.
        
        Args:
            position: Position to protect
            stop_price: Stop-loss price
            
        Returns:
            Created stop-loss order
        """
        try:
            # Determine order side (opposite of position)
            stop_side = OrderSide.SELL if position.side == OrderSide.BUY else OrderSide.BUY
            
            order = Order(
                id=str(uuid.uuid4()),
                symbol=position.symbol,
                side=stop_side,
                type=OrderType.STOP_LOSS,
                quantity=position.quantity,
                price=stop_price,
                status=OrderStatus.PENDING,
                timestamp=datetime.now(),
                strategy_name=position.strategy_name,
                metadata={
                    'position_id': position.id,
                    'stop_loss': True,
                    'original_position_side': position.side.value
                }
            )
            
            # Add to pending orders
            self.pending_orders[order.id] = order
            
            # Save to database
            await self.database_manager.save_order(order)
            
            # Execute the order
            success = await self._execute_order_on_exchange(order)
            
            if success:
                self.executed_orders[order.id] = order
                del self.pending_orders[order.id]
                
                self.logger.info(f"Stop-loss order placed: {position.symbol} at {stop_price}")
                return order
            else:
                self.failed_orders[order.id] = order
                del self.pending_orders[order.id]
                
                self.logger.error(f"Stop-loss order failed: {position.symbol} at {stop_price}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error placing stop-loss order: {e}")
            return None
    
    async def place_take_profit_order(self, 
                                    position: Position,
                                    take_profit_price: float) -> Optional[Order]:
        """
        Place a take-profit order for a position.
        
        Args:
            position: Position to close
            take_profit_price: Take-profit price
            
        Returns:
            Created take-profit order
        """
        try:
            # Determine order side (opposite of position)
            tp_side = OrderSide.SELL if position.side == OrderSide.BUY else OrderSide.BUY
            
            order = Order(
                id=str(uuid.uuid4()),
                symbol=position.symbol,
                side=tp_side,
                type=OrderType.TAKE_PROFIT,
                quantity=position.quantity,
                price=take_profit_price,
                status=OrderStatus.PENDING,
                timestamp=datetime.now(),
                strategy_name=position.strategy_name,
                metadata={
                    'position_id': position.id,
                    'take_profit': True,
                    'original_position_side': position.side.value
                }
            )
            
            # Add to pending orders
            self.pending_orders[order.id] = order
            
            # Save to database
            await self.database_manager.save_order(order)
            
            # Execute the order
            success = await self._execute_order_on_exchange(order)
            
            if success:
                self.executed_orders[order.id] = order
                del self.pending_orders[order.id]
                
                self.logger.info(f"Take-profit order placed: {position.symbol} at {take_profit_price}")
                return order
            else:
                self.failed_orders[order.id] = order
                del self.pending_orders[order.id]
                
                self.logger.error(f"Take-profit order failed: {position.symbol} at {take_profit_price}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error placing take-profit order: {e}")
            return None
    
    async def close_position(self, position: Position) -> Optional[Order]:
        """
        Close a position with a market order.
        
        Args:
            position: Position to close
            
        Returns:
            Created closing order
        """
        try:
            # Determine order side (opposite of position)
            close_side = OrderSide.SELL if position.side == OrderSide.BUY else OrderSide.BUY
            
            order = Order(
                id=str(uuid.uuid4()),
                symbol=position.symbol,
                side=close_side,
                type=OrderType.MARKET,
                quantity=position.quantity,
                price=None,  # Market order
                status=OrderStatus.PENDING,
                timestamp=datetime.now(),
                strategy_name=position.strategy_name,
                metadata={
                    'position_id': position.id,
                    'close_position': True,
                    'original_position_side': position.side.value
                }
            )
            
            # Add to pending orders
            self.pending_orders[order.id] = order
            
            # Save to database
            await self.database_manager.save_order(order)
            
            # Execute the order
            success = await self._execute_order_on_exchange(order)
            
            if success:
                self.executed_orders[order.id] = order
                del self.pending_orders[order.id]
                
                self.logger.info(f"Position closed: {position.symbol}")
                return order
            else:
                self.failed_orders[order.id] = order
                del self.pending_orders[order.id]
                
                self.logger.error(f"Failed to close position: {position.symbol}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            return None
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get order execution statistics."""
        return {
            'pending_orders': len(self.pending_orders),
            'executed_orders': len(self.executed_orders),
            'failed_orders': len(self.failed_orders),
            'success_rate': len(self.executed_orders) / max(1, len(self.executed_orders) + len(self.failed_orders)),
            'total_orders': len(self.pending_orders) + len(self.executed_orders) + len(self.failed_orders)
        }
    
    async def cleanup_old_orders(self, max_age_hours: int = 24):
        """Clean up old orders from memory."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        # Clean up executed orders
        old_executed = {
            order_id: order for order_id, order in self.executed_orders.items()
            if order.timestamp < cutoff_time
        }
        
        for order_id in old_executed:
            del self.executed_orders[order_id]
        
        # Clean up failed orders
        old_failed = {
            order_id: order for order_id, order in self.failed_orders.items()
            if order.timestamp < cutoff_time
        }
        
        for order_id in old_failed:
            del self.failed_orders[order_id]
        
        self.logger.info(f"Cleaned up {len(old_executed)} executed and {len(old_failed)} failed orders")
