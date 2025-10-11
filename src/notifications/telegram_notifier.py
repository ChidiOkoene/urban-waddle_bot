"""
Telegram Notifier for Trading Bot

This module provides Telegram notifications for trading events,
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
class TelegramConfig:
    """Configuration for Telegram notifications."""
    bot_token: str
    chat_id: str
    enabled: bool = True
    send_signals: bool = True
    send_orders: bool = True
    send_positions: bool = True
    send_alerts: bool = True
    send_performance: bool = True
    max_message_length: int = 4096
    rate_limit_per_minute: int = 30


class TelegramNotifier:
    """Sends notifications via Telegram."""
    
    def __init__(self, config: TelegramConfig):
        """
        Initialize the Telegram notifier.
        
        Args:
            config: Telegram configuration
        """
        self.config = config
        self.logger = logging.getLogger("TelegramNotifier")
        
        # Rate limiting
        self.message_times = []
        self.last_cleanup = datetime.now()
        
        # Message templates
        self.templates = self._setup_templates()
    
    def _setup_templates(self) -> Dict[str, str]:
        """Setup message templates."""
        return {
            'signal': """
🚨 **Trading Signal**
📊 Symbol: {symbol}
📈 Side: {side}
💰 Price: ${price:.2f}
💪 Strength: {strength:.1%}
⏰ Time: {time}
🤖 Strategy: {strategy}
            """,
            'order_placed': """
📋 **Order Placed**
📊 Symbol: {symbol}
📈 Side: {side}
📦 Quantity: {quantity:.6f}
💰 Price: ${price:.2f}
📝 Type: {order_type}
⏰ Time: {time}
🤖 Strategy: {strategy}
            """,
            'order_filled': """
✅ **Order Filled**
📊 Symbol: {symbol}
📈 Side: {side}
📦 Quantity: {quantity:.6f}
💰 Price: ${price:.2f}
⏰ Time: {time}
🤖 Strategy: {strategy}
            """,
            'order_failed': """
❌ **Order Failed**
📊 Symbol: {symbol}
📈 Side: {side}
📦 Quantity: {quantity:.6f}
💰 Price: ${price:.2f}
⚠️ Error: {error}
⏰ Time: {time}
🤖 Strategy: {strategy}
            """,
            'position_opened': """
🟢 **Position Opened**
📊 Symbol: {symbol}
📈 Side: {side}
📦 Quantity: {quantity:.6f}
💰 Entry Price: ${entry_price:.2f}
⏰ Time: {time}
🤖 Strategy: {strategy}
            """,
            'position_closed': """
🔴 **Position Closed**
📊 Symbol: {symbol}
📈 Side: {side}
📦 Quantity: {quantity:.6f}
💰 Entry Price: ${entry_price:.2f}
💰 Exit Price: ${exit_price:.2f}
💵 PnL: ${pnl:.2f} ({pnl_pct:.1%})
📝 Reason: {reason}
⏰ Time: {time}
🤖 Strategy: {strategy}
            """,
            'stop_loss': """
🛑 **Stop Loss Triggered**
📊 Symbol: {symbol}
💰 Stop Price: ${stop_price:.2f}
💰 Current Price: ${current_price:.2f}
💵 PnL: ${pnl:.2f}
⏰ Time: {time}
🤖 Strategy: {strategy}
            """,
            'take_profit': """
🎯 **Take Profit Triggered**
📊 Symbol: {symbol}
💰 Take Profit Price: ${take_profit_price:.2f}
💰 Current Price: ${current_price:.2f}
💵 PnL: ${pnl:.2f}
⏰ Time: {time}
🤖 Strategy: {strategy}
            """,
            'error': """
🚨 **Error Alert**
⚠️ Error: {error}
📝 Context: {context}
⏰ Time: {time}
            """,
            'warning': """
⚠️ **Warning Alert**
📝 Warning: {warning}
📝 Context: {context}
⏰ Time: {time}
            """,
            'performance': """
📊 **Performance Update**
💵 Total PnL: ${total_pnl:.2f}
📈 Win Rate: {win_rate:.1%}
📦 Open Positions: {open_positions}
📋 Total Trades: {total_trades}
⏰ Time: {time}
            """,
            'system_start': """
🚀 **Trading System Started**
🤖 Strategy: {strategy}
📊 Exchange: {exchange}
⏰ Time: {time}
            """,
            'system_stop': """
🛑 **Trading System Stopped**
📝 Reason: {reason}
⏰ Time: {time}
            """
        }
    
    async def send_message(self, message: str, parse_mode: str = "Markdown") -> bool:
        """
        Send a message to Telegram.
        
        Args:
            message: Message to send
            parse_mode: Parse mode for formatting
            
        Returns:
            True if sent successfully
        """
        if not self.config.enabled:
            return False
        
        # Check rate limit
        if not self._check_rate_limit():
            self.logger.warning("Rate limit exceeded, skipping message")
            return False
        
        # Truncate message if too long
        if len(message) > self.config.max_message_length:
            message = message[:self.config.max_message_length - 3] + "..."
        
        try:
            url = f"https://api.telegram.org/bot{self.config.bot_token}/sendMessage"
            
            data = {
                'chat_id': self.config.chat_id,
                'text': message,
                'parse_mode': parse_mode
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        self.message_times.append(datetime.now())
                        self.logger.debug("Message sent to Telegram successfully")
                        return True
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Failed to send Telegram message: {response.status} - {error_text}")
                        return False
        
        except Exception as e:
            self.logger.error(f"Error sending Telegram message: {e}")
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
    
    async def send_signal(self, signal: StrategySignal) -> bool:
        """Send trading signal notification."""
        if not self.config.send_signals:
            return False
        
        message = self.templates['signal'].format(
            symbol=signal.symbol,
            side=signal.side.value.upper(),
            price=signal.price,
            strength=signal.strength,
            time=signal.timestamp.strftime('%H:%M:%S'),
            strategy=signal.strategy_name
        )
        
        return await self.send_message(message)
    
    async def send_order_placed(self, order: Order) -> bool:
        """Send order placed notification."""
        if not self.config.send_orders:
            return False
        
        message = self.templates['order_placed'].format(
            symbol=order.symbol,
            side=order.side.value.upper(),
            quantity=order.quantity,
            price=order.price or 0,
            order_type=order.type.value.upper(),
            time=order.timestamp.strftime('%H:%M:%S'),
            strategy=order.strategy_name
        )
        
        return await self.send_message(message)
    
    async def send_order_filled(self, order: Order) -> bool:
        """Send order filled notification."""
        if not self.config.send_orders:
            return False
        
        message = self.templates['order_filled'].format(
            symbol=order.symbol,
            side=order.side.value.upper(),
            quantity=order.quantity,
            price=order.price or 0,
            time=order.filled_at.strftime('%H:%M:%S') if order.filled_at else order.timestamp.strftime('%H:%M:%S'),
            strategy=order.strategy_name
        )
        
        return await self.send_message(message)
    
    async def send_order_failed(self, order: Order, error: str) -> bool:
        """Send order failed notification."""
        if not self.config.send_orders:
            return False
        
        message = self.templates['order_failed'].format(
            symbol=order.symbol,
            side=order.side.value.upper(),
            quantity=order.quantity,
            price=order.price or 0,
            error=error,
            time=order.timestamp.strftime('%H:%M:%S'),
            strategy=order.strategy_name
        )
        
        return await self.send_message(message)
    
    async def send_position_opened(self, position: Position) -> bool:
        """Send position opened notification."""
        if not self.config.send_positions:
            return False
        
        message = self.templates['position_opened'].format(
            symbol=position.symbol,
            side=position.side.value.upper(),
            quantity=position.quantity,
            entry_price=position.entry_price,
            time=position.timestamp.strftime('%H:%M:%S'),
            strategy=position.strategy_name
        )
        
        return await self.send_message(message)
    
    async def send_position_closed(self, position: Position, reason: str) -> bool:
        """Send position closed notification."""
        if not self.config.send_positions:
            return False
        
        pnl = position.metadata.get('final_pnl', 0)
        pnl_pct = (pnl / (position.entry_price * position.quantity)) * 100 if position.entry_price * position.quantity > 0 else 0
        
        message = self.templates['position_closed'].format(
            symbol=position.symbol,
            side=position.side.value.upper(),
            quantity=position.quantity,
            entry_price=position.entry_price,
            exit_price=position.current_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            reason=reason,
            time=position.timestamp.strftime('%H:%M:%S'),
            strategy=position.strategy_name
        )
        
        return await self.send_message(message)
    
    async def send_stop_loss(self, position: Position, stop_price: float) -> bool:
        """Send stop-loss triggered notification."""
        if not self.config.send_alerts:
            return False
        
        unrealized_pnl = position.metadata.get('unrealized_pnl', 0)
        
        message = self.templates['stop_loss'].format(
            symbol=position.symbol,
            stop_price=stop_price,
            current_price=position.current_price,
            pnl=unrealized_pnl,
            time=datetime.now().strftime('%H:%M:%S'),
            strategy=position.strategy_name
        )
        
        return await self.send_message(message)
    
    async def send_take_profit(self, position: Position, take_profit_price: float) -> bool:
        """Send take-profit triggered notification."""
        if not self.config.send_alerts:
            return False
        
        unrealized_pnl = position.metadata.get('unrealized_pnl', 0)
        
        message = self.templates['take_profit'].format(
            symbol=position.symbol,
            take_profit_price=take_profit_price,
            current_price=position.current_price,
            pnl=unrealized_pnl,
            time=datetime.now().strftime('%H:%M:%S'),
            strategy=position.strategy_name
        )
        
        return await self.send_message(message)
    
    async def send_error(self, error: str, context: Dict[str, Any] = None) -> bool:
        """Send error notification."""
        if not self.config.send_alerts:
            return False
        
        context_str = json.dumps(context or {}, indent=2)
        
        message = self.templates['error'].format(
            error=error,
            context=context_str,
            time=datetime.now().strftime('%H:%M:%S')
        )
        
        return await self.send_message(message)
    
    async def send_warning(self, warning: str, context: Dict[str, Any] = None) -> bool:
        """Send warning notification."""
        if not self.config.send_alerts:
            return False
        
        context_str = json.dumps(context or {}, indent=2)
        
        message = self.templates['warning'].format(
            warning=warning,
            context=context_str,
            time=datetime.now().strftime('%H:%M:%S')
        )
        
        return await self.send_message(message)
    
    async def send_performance_update(self, 
                                    total_pnl: float,
                                    win_rate: float,
                                    open_positions: int,
                                    total_trades: int) -> bool:
        """Send performance update notification."""
        if not self.config.send_performance:
            return False
        
        message = self.templates['performance'].format(
            total_pnl=total_pnl,
            win_rate=win_rate,
            open_positions=open_positions,
            total_trades=total_trades,
            time=datetime.now().strftime('%H:%M:%S')
        )
        
        return await self.send_message(message)
    
    async def send_system_start(self, strategy: str, exchange: str) -> bool:
        """Send system start notification."""
        if not self.config.send_alerts:
            return False
        
        message = self.templates['system_start'].format(
            strategy=strategy,
            exchange=exchange,
            time=datetime.now().strftime('%H:%M:%S')
        )
        
        return await self.send_message(message)
    
    async def send_system_stop(self, reason: str) -> bool:
        """Send system stop notification."""
        if not self.config.send_alerts:
            return False
        
        message = self.templates['system_stop'].format(
            reason=reason,
            time=datetime.now().strftime('%H:%M:%S')
        )
        
        return await self.send_message(message)
    
    async def send_custom_message(self, message: str, parse_mode: str = "Markdown") -> bool:
        """Send a custom message."""
        return await self.send_message(message, parse_mode)
    
    async def send_daily_summary(self, 
                               total_pnl: float,
                               trades_today: int,
                               win_rate: float,
                               open_positions: int,
                               top_performers: List[Dict[str, Any]]) -> bool:
        """Send daily summary notification."""
        if not self.config.send_performance:
            return False
        
        # Create daily summary message
        message = f"""
📊 **Daily Trading Summary**
📅 Date: {datetime.now().strftime('%Y-%m-%d')}

💰 **Performance**
💵 Total PnL: ${total_pnl:.2f}
📈 Win Rate: {win_rate:.1%}
📦 Open Positions: {open_positions}
📋 Trades Today: {trades_today}

🏆 **Top Performers**
"""
        
        for i, performer in enumerate(top_performers[:3], 1):
            message += f"{i}. {performer['symbol']}: ${performer['pnl']:.2f}\n"
        
        message += f"\n⏰ Generated at: {datetime.now().strftime('%H:%M:%S')}"
        
        return await self.send_message(message)
    
    async def send_risk_alert(self, 
                            alert_type: str,
                            current_value: float,
                            limit_value: float,
                            symbol: str = None) -> bool:
        """Send risk alert notification."""
        if not self.config.send_alerts:
            return False
        
        message = f"""
🚨 **Risk Alert**
⚠️ Type: {alert_type}
📊 Symbol: {symbol or 'Portfolio'}
📈 Current: {current_value:.2f}
🛑 Limit: {limit_value:.2f}
⏰ Time: {datetime.now().strftime('%H:%M:%S')}
        """
        
        return await self.send_message(message)
    
    def update_config(self, config: TelegramConfig):
        """Update Telegram configuration."""
        self.config = config
        self.logger.info("Telegram configuration updated")
    
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
        """Test Telegram connection."""
        try:
            test_message = f"🤖 Trading Bot Test Message\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            return await self.send_message(test_message)
        except Exception as e:
            self.logger.error(f"Telegram connection test failed: {e}")
            return False
