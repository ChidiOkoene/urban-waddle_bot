"""
Discord Notifier for Trading Bot

This module provides Discord notifications for trading events,
including signals, orders, positions, and system alerts.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import aiohttp
import json
from dataclasses import dataclass

from ..core.data_models import (
    StrategySignal, Order, Trade, Position, OrderSide, OrderType, 
    OrderStatus, PositionStatus
)


@dataclass
class DiscordConfig:
    """Configuration for Discord notifications."""
    webhook_url: str
    enabled: bool = True
    send_signals: bool = True
    send_orders: bool = True
    send_positions: bool = True
    send_alerts: bool = True
    send_performance: bool = True
    username: str = "Trading Bot"
    avatar_url: str = ""
    max_message_length: int = 2000
    rate_limit_per_minute: int = 30


class DiscordNotifier:
    """Sends notifications via Discord webhook."""
    
    def __init__(self, config: DiscordConfig):
        """
        Initialize the Discord notifier.
        
        Args:
            config: Discord configuration
        """
        self.config = config
        self.logger = logging.getLogger("DiscordNotifier")
        
        # Rate limiting
        self.message_times = []
        self.last_cleanup = datetime.now()
    
    async def send_message(self, 
                          content: str,
                          embeds: List[Dict[str, Any]] = None,
                          username: str = None) -> bool:
        """
        Send a message to Discord.
        
        Args:
            content: Message content
            embeds: Discord embeds for rich formatting
            username: Override username
            
        Returns:
            True if sent successfully
        """
        if not self.config.enabled:
            return False
        
        # Check rate limit
        if not self._check_rate_limit():
            self.logger.warning("Rate limit exceeded, skipping message")
            return False
        
        # Truncate content if too long
        if len(content) > self.config.max_message_length:
            content = content[:self.config.max_message_length - 3] + "..."
        
        try:
            payload = {
                'content': content,
                'username': username or self.config.username
            }
            
            if self.config.avatar_url:
                payload['avatar_url'] = self.config.avatar_url
            
            if embeds:
                payload['embeds'] = embeds
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.config.webhook_url, json=payload) as response:
                    if response.status == 204:  # Discord returns 204 for successful webhook
                        self.message_times.append(datetime.now())
                        self.logger.debug("Message sent to Discord successfully")
                        return True
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Failed to send Discord message: {response.status} - {error_text}")
                        return False
        
        except Exception as e:
            self.logger.error(f"Error sending Discord message: {e}")
            return False
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        now = datetime.now()
        
        # Clean up old message times
        if (now - self.last_cleanup).total_seconds() > 60:
            self.message_times = [
                msg_time for msg_time in self.message_times
                if (now - msg_time).total_seconds() < 60
            ]
            self.last_cleanup = now
        
        # Check if we're within rate limit
        return len(self.message_times) < self.config.rate_limit_per_minute
    
    def _create_embed(self, 
                     title: str,
                     description: str,
                     color: int,
                     fields: List[Dict[str, str]] = None,
                     footer: str = None) -> Dict[str, Any]:
        """Create a Discord embed."""
        embed = {
            'title': title,
            'description': description,
            'color': color,
            'timestamp': datetime.now().isoformat()
        }
        
        if fields:
            embed['fields'] = fields
        
        if footer:
            embed['footer'] = {'text': footer}
        
        return embed
    
    async def send_signal(self, signal: StrategySignal) -> bool:
        """Send trading signal notification."""
        if not self.config.send_signals:
            return False
        
        # Determine color based on signal side
        color = 0x00ff00 if signal.side == OrderSide.BUY else 0xff0000
        
        fields = [
            {'name': 'Symbol', 'value': signal.symbol, 'inline': True},
            {'name': 'Side', 'value': signal.side.value.upper(), 'inline': True},
            {'name': 'Price', 'value': f"${signal.price:.2f}", 'inline': True},
            {'name': 'Strength', 'value': f"{signal.strength:.1%}", 'inline': True},
            {'name': 'Strategy', 'value': signal.strategy_name, 'inline': True},
            {'name': 'Timeframe', 'value': signal.timeframe, 'inline': True}
        ]
        
        embed = self._create_embed(
            title="🚨 Trading Signal",
            description=f"New trading signal generated",
            color=color,
            fields=fields,
            footer=f"Generated at {signal.timestamp.strftime('%H:%M:%S')}"
        )
        
        return await self.send_message("", embeds=[embed])
    
    async def send_order_placed(self, order: Order) -> bool:
        """Send order placed notification."""
        if not self.config.send_orders:
            return False
        
        color = 0x00ff00 if order.side == OrderSide.BUY else 0xff0000
        
        fields = [
            {'name': 'Symbol', 'value': order.symbol, 'inline': True},
            {'name': 'Side', 'value': order.side.value.upper(), 'inline': True},
            {'name': 'Quantity', 'value': f"{order.quantity:.6f}", 'inline': True},
            {'name': 'Price', 'value': f"${order.price:.2f}" if order.price else "Market", 'inline': True},
            {'name': 'Type', 'value': order.type.value.upper(), 'inline': True},
            {'name': 'Strategy', 'value': order.strategy_name, 'inline': True}
        ]
        
        embed = self._create_embed(
            title="📋 Order Placed",
            description=f"New order has been placed",
            color=color,
            fields=fields,
            footer=f"Placed at {order.timestamp.strftime('%H:%M:%S')}"
        )
        
        return await self.send_message("", embeds=[embed])
    
    async def send_order_filled(self, order: Order) -> bool:
        """Send order filled notification."""
        if not self.config.send_orders:
            return False
        
        color = 0x00ff00
        
        fields = [
            {'name': 'Symbol', 'value': order.symbol, 'inline': True},
            {'name': 'Side', 'value': order.side.value.upper(), 'inline': True},
            {'name': 'Quantity', 'value': f"{order.quantity:.6f}", 'inline': True},
            {'name': 'Price', 'value': f"${order.price:.2f}" if order.price else "Market", 'inline': True},
            {'name': 'Strategy', 'value': order.strategy_name, 'inline': True}
        ]
        
        embed = self._create_embed(
            title="✅ Order Filled",
            description=f"Order has been successfully filled",
            color=color,
            fields=fields,
            footer=f"Filled at {order.filled_at.strftime('%H:%M:%S') if order.filled_at else order.timestamp.strftime('%H:%M:%S')}"
        )
        
        return await self.send_message("", embeds=[embed])
    
    async def send_order_failed(self, order: Order, error: str) -> bool:
        """Send order failed notification."""
        if not self.config.send_orders:
            return False
        
        color = 0xff0000
        
        fields = [
            {'name': 'Symbol', 'value': order.symbol, 'inline': True},
            {'name': 'Side', 'value': order.side.value.upper(), 'inline': True},
            {'name': 'Quantity', 'value': f"{order.quantity:.6f}", 'inline': True},
            {'name': 'Price', 'value': f"${order.price:.2f}" if order.price else "Market", 'inline': True},
            {'name': 'Error', 'value': error, 'inline': False},
            {'name': 'Strategy', 'value': order.strategy_name, 'inline': True}
        ]
        
        embed = self._create_embed(
            title="❌ Order Failed",
            description=f"Order execution failed",
            color=color,
            fields=fields,
            footer=f"Failed at {order.timestamp.strftime('%H:%M:%S')}"
        )
        
        return await self.send_message("", embeds=[embed])
    
    async def send_position_opened(self, position: Position) -> bool:
        """Send position opened notification."""
        if not self.config.send_positions:
            return False
        
        color = 0x00ff00 if position.side == OrderSide.BUY else 0xff0000
        
        fields = [
            {'name': 'Symbol', 'value': position.symbol, 'inline': True},
            {'name': 'Side', 'value': position.side.value.upper(), 'inline': True},
            {'name': 'Quantity', 'value': f"{position.quantity:.6f}", 'inline': True},
            {'name': 'Entry Price', 'value': f"${position.entry_price:.2f}", 'inline': True},
            {'name': 'Strategy', 'value': position.strategy_name, 'inline': True}
        ]
        
        embed = self._create_embed(
            title="🟢 Position Opened",
            description=f"New position has been opened",
            color=color,
            fields=fields,
            footer=f"Opened at {position.timestamp.strftime('%H:%M:%S')}"
        )
        
        return await self.send_message("", embeds=[embed])
    
    async def send_position_closed(self, position: Position, reason: str) -> bool:
        """Send position closed notification."""
        if not self.config.send_positions:
            return False
        
        pnl = position.metadata.get('final_pnl', 0)
        pnl_pct = (pnl / (position.entry_price * position.quantity)) * 100 if position.entry_price * position.quantity > 0 else 0
        
        # Determine color based on PnL
        color = 0x00ff00 if pnl > 0 else 0xff0000 if pnl < 0 else 0x808080
        
        fields = [
            {'name': 'Symbol', 'value': position.symbol, 'inline': True},
            {'name': 'Side', 'value': position.side.value.upper(), 'inline': True},
            {'name': 'Quantity', 'value': f"{position.quantity:.6f}", 'inline': True},
            {'name': 'Entry Price', 'value': f"${position.entry_price:.2f}", 'inline': True},
            {'name': 'Exit Price', 'value': f"${position.current_price:.2f}", 'inline': True},
            {'name': 'PnL', 'value': f"${pnl:.2f} ({pnl_pct:.1%})", 'inline': True},
            {'name': 'Reason', 'value': reason, 'inline': True},
            {'name': 'Strategy', 'value': position.strategy_name, 'inline': True}
        ]
        
        embed = self._create_embed(
            title="🔴 Position Closed",
            description=f"Position has been closed",
            color=color,
            fields=fields,
            footer=f"Closed at {position.timestamp.strftime('%H:%M:%S')}"
        )
        
        return await self.send_message("", embeds=[embed])
    
    async def send_stop_loss(self, position: Position, stop_price: float) -> bool:
        """Send stop-loss triggered notification."""
        if not self.config.send_alerts:
            return False
        
        unrealized_pnl = position.metadata.get('unrealized_pnl', 0)
        
        fields = [
            {'name': 'Symbol', 'value': position.symbol, 'inline': True},
            {'name': 'Stop Price', 'value': f"${stop_price:.2f}", 'inline': True},
            {'name': 'Current Price', 'value': f"${position.current_price:.2f}", 'inline': True},
            {'name': 'PnL', 'value': f"${unrealized_pnl:.2f}", 'inline': True},
            {'name': 'Strategy', 'value': position.strategy_name, 'inline': True}
        ]
        
        embed = self._create_embed(
            title="🛑 Stop Loss Triggered",
            description=f"Stop loss has been triggered",
            color=0xff0000,
            fields=fields,
            footer=f"Triggered at {datetime.now().strftime('%H:%M:%S')}"
        )
        
        return await self.send_message("", embeds=[embed])
    
    async def send_take_profit(self, position: Position, take_profit_price: float) -> bool:
        """Send take-profit triggered notification."""
        if not self.config.send_alerts:
            return False
        
        unrealized_pnl = position.metadata.get('unrealized_pnl', 0)
        
        fields = [
            {'name': 'Symbol', 'value': position.symbol, 'inline': True},
            {'name': 'Take Profit Price', 'value': f"${take_profit_price:.2f}", 'inline': True},
            {'name': 'Current Price', 'value': f"${position.current_price:.2f}", 'inline': True},
            {'name': 'PnL', 'value': f"${unrealized_pnl:.2f}", 'inline': True},
            {'name': 'Strategy', 'value': position.strategy_name, 'inline': True}
        ]
        
        embed = self._create_embed(
            title="🎯 Take Profit Triggered",
            description=f"Take profit has been triggered",
            color=0x00ff00,
            fields=fields,
            footer=f"Triggered at {datetime.now().strftime('%H:%M:%S')}"
        )
        
        return await self.send_message("", embeds=[embed])
    
    async def send_error(self, error: str, context: Dict[str, Any] = None) -> bool:
        """Send error notification."""
        if not self.config.send_alerts:
            return False
        
        context_str = json.dumps(context or {}, indent=2) if context else "No context provided"
        
        fields = [
            {'name': 'Error', 'value': error, 'inline': False},
            {'name': 'Context', 'value': f"```json\n{context_str}\n```", 'inline': False}
        ]
        
        embed = self._create_embed(
            title="🚨 Error Alert",
            description=f"An error has occurred",
            color=0xff0000,
            fields=fields,
            footer=f"Error at {datetime.now().strftime('%H:%M:%S')}"
        )
        
        return await self.send_message("", embeds=[embed])
    
    async def send_warning(self, warning: str, context: Dict[str, Any] = None) -> bool:
        """Send warning notification."""
        if not self.config.send_alerts:
            return False
        
        context_str = json.dumps(context or {}, indent=2) if context else "No context provided"
        
        fields = [
            {'name': 'Warning', 'value': warning, 'inline': False},
            {'name': 'Context', 'value': f"```json\n{context_str}\n```", 'inline': False}
        ]
        
        embed = self._create_embed(
            title="⚠️ Warning Alert",
            description=f"A warning has been triggered",
            color=0xffaa00,
            fields=fields,
            footer=f"Warning at {datetime.now().strftime('%H:%M:%S')}"
        )
        
        return await self.send_message("", embeds=[embed])
    
    async def send_performance_update(self, 
                                    total_pnl: float,
                                    win_rate: float,
                                    open_positions: int,
                                    total_trades: int) -> bool:
        """Send performance update notification."""
        if not self.config.send_performance:
            return False
        
        # Determine color based on PnL
        color = 0x00ff00 if total_pnl > 0 else 0xff0000 if total_pnl < 0 else 0x808080
        
        fields = [
            {'name': 'Total PnL', 'value': f"${total_pnl:.2f}", 'inline': True},
            {'name': 'Win Rate', 'value': f"{win_rate:.1%}", 'inline': True},
            {'name': 'Open Positions', 'value': str(open_positions), 'inline': True},
            {'name': 'Total Trades', 'value': str(total_trades), 'inline': True}
        ]
        
        embed = self._create_embed(
            title="📊 Performance Update",
            description=f"Trading performance update",
            color=color,
            fields=fields,
            footer=f"Update at {datetime.now().strftime('%H:%M:%S')}"
        )
        
        return await self.send_message("", embeds=[embed])
    
    async def send_system_start(self, strategy: str, exchange: str) -> bool:
        """Send system start notification."""
        if not self.config.send_alerts:
            return False
        
        fields = [
            {'name': 'Strategy', 'value': strategy, 'inline': True},
            {'name': 'Exchange', 'value': exchange, 'inline': True}
        ]
        
        embed = self._create_embed(
            title="🚀 Trading System Started",
            description=f"Trading system has been started",
            color=0x00ff00,
            fields=fields,
            footer=f"Started at {datetime.now().strftime('%H:%M:%S')}"
        )
        
        return await self.send_message("", embeds=[embed])
    
    async def send_system_stop(self, reason: str) -> bool:
        """Send system stop notification."""
        if not self.config.send_alerts:
            return False
        
        fields = [
            {'name': 'Reason', 'value': reason, 'inline': False}
        ]
        
        embed = self._create_embed(
            title="🛑 Trading System Stopped",
            description=f"Trading system has been stopped",
            color=0xff0000,
            fields=fields,
            footer=f"Stopped at {datetime.now().strftime('%H:%M:%S')}"
        )
        
        return await self.send_message("", embeds=[embed])
    
    async def send_daily_summary(self, 
                               total_pnl: float,
                               trades_today: int,
                               win_rate: float,
                               open_positions: int,
                               top_performers: List[Dict[str, Any]]) -> bool:
        """Send daily summary notification."""
        if not self.config.send_performance:
            return False
        
        # Determine color based on PnL
        color = 0x00ff00 if total_pnl > 0 else 0xff0000 if total_pnl < 0 else 0x808080
        
        fields = [
            {'name': 'Total PnL', 'value': f"${total_pnl:.2f}", 'inline': True},
            {'name': 'Win Rate', 'value': f"{win_rate:.1%}", 'inline': True},
            {'name': 'Open Positions', 'value': str(open_positions), 'inline': True},
            {'name': 'Trades Today', 'value': str(trades_today), 'inline': True}
        ]
        
        # Add top performers
        if top_performers:
            performers_text = "\n".join([
                f"{i}. {performer['symbol']}: ${performer['pnl']:.2f}"
                for i, performer in enumerate(top_performers[:3], 1)
            ])
            fields.append({'name': 'Top Performers', 'value': performers_text, 'inline': False})
        
        embed = self._create_embed(
            title="📊 Daily Trading Summary",
            description=f"Trading summary for {datetime.now().strftime('%Y-%m-%d')}",
            color=color,
            fields=fields,
            footer=f"Generated at {datetime.now().strftime('%H:%M:%S')}"
        )
        
        return await self.send_message("", embeds=[embed])
    
    async def send_risk_alert(self, 
                            alert_type: str,
                            current_value: float,
                            limit_value: float,
                            symbol: str = None) -> bool:
        """Send risk alert notification."""
        if not self.config.send_alerts:
            return False
        
        fields = [
            {'name': 'Type', 'value': alert_type, 'inline': True},
            {'name': 'Symbol', 'value': symbol or 'Portfolio', 'inline': True},
            {'name': 'Current Value', 'value': f"{current_value:.2f}", 'inline': True},
            {'name': 'Limit Value', 'value': f"{limit_value:.2f}", 'inline': True}
        ]
        
        embed = self._create_embed(
            title="🚨 Risk Alert",
            description=f"Risk limit has been exceeded",
            color=0xff0000,
            fields=fields,
            footer=f"Alert at {datetime.now().strftime('%H:%M:%S')}"
        )
        
        return await self.send_message("", embeds=[embed])
    
    async def send_custom_message(self, content: str, embeds: List[Dict[str, Any]] = None) -> bool:
        """Send a custom message."""
        return await self.send_message(content, embeds)
    
    def update_config(self, config: DiscordConfig):
        """Update Discord configuration."""
        self.config = config
        self.logger.info("Discord configuration updated")
    
    def get_status(self) -> Dict[str, Any]:
        """Get notifier status."""
        return {
            'enabled': self.config.enabled,
            'send_signals': self.config.send_signals,
            'send_orders': self.config.send_orders,
            'send_positions': self.config.send_positions,
            'send_alerts': self.config.send_alerts,
            'send_performance': self.config.send_performance,
            'messages_sent_last_hour': len([
                t for t in self.message_times
                if (datetime.now() - t).total_seconds() < 3600
            ]),
            'rate_limit_per_minute': self.config.rate_limit_per_minute
        }
    
    async def test_connection(self) -> bool:
        """Test Discord connection."""
        try:
            test_message = f"🤖 Trading Bot Test Message\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            return await self.send_message(test_message)
        except Exception as e:
            self.logger.error(f"Discord connection test failed: {e}")
            return False
