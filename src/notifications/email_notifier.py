"""
Email Notifier for Trading Bot

This module provides email notifications for trading events,
including signals, orders, positions, and system alerts.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from dataclasses import dataclass
import json

from ..core.data_models import (
    StrategySignal, Order, Trade, Position, OrderSide, OrderType, 
    OrderStatus, PositionStatus
)


@dataclass
class EmailConfig:
    """Configuration for email notifications."""
    smtp_server: str
    smtp_port: int = 587
    username: str
    password: str
    from_email: str
    recipients: List[str]
    enabled: bool = True
    send_signals: bool = True
    send_orders: bool = True
    send_positions: bool = True
    send_alerts: bool = True
    send_performance: bool = True
    use_tls: bool = True
    use_ssl: bool = False
    rate_limit_per_hour: int = 50


class EmailNotifier:
    """Sends notifications via email."""
    
    def __init__(self, config: EmailConfig):
        """
        Initialize the email notifier.
        
        Args:
            config: Email configuration
        """
        self.config = config
        self.logger = logging.getLogger("EmailNotifier")
        
        # Rate limiting
        self.message_times = []
        self.last_cleanup = datetime.now()
        
        # Email templates
        self.templates = self._setup_templates()
    
    def _setup_templates(self) -> Dict[str, str]:
        """Setup email templates."""
        return {
            'signal': """
Subject: Trading Signal - {symbol} {side}

A new trading signal has been generated:

Symbol: {symbol}
Side: {side}
Price: ${price:.2f}
Strength: {strength:.1%}
Time: {time}
Strategy: {strategy}

This is an automated message from your trading bot.
            """,
            'order_placed': """
Subject: Order Placed - {symbol} {side}

A new order has been placed:

Symbol: {symbol}
Side: {side}
Quantity: {quantity:.6f}
Price: ${price:.2f}
Type: {order_type}
Time: {time}
Strategy: {strategy}

This is an automated message from your trading bot.
            """,
            'order_filled': """
Subject: Order Filled - {symbol} {side}

An order has been successfully filled:

Symbol: {symbol}
Side: {side}
Quantity: {quantity:.6f}
Price: ${price:.2f}
Time: {time}
Strategy: {strategy}

This is an automated message from your trading bot.
            """,
            'order_failed': """
Subject: Order Failed - {symbol} {side}

An order has failed to execute:

Symbol: {symbol}
Side: {side}
Quantity: {quantity:.6f}
Price: ${price:.2f}
Error: {error}
Time: {time}
Strategy: {strategy}

This is an automated message from your trading bot.
            """,
            'position_opened': """
Subject: Position Opened - {symbol} {side}

A new position has been opened:

Symbol: {symbol}
Side: {side}
Quantity: {quantity:.6f}
Entry Price: ${entry_price:.2f}
Time: {time}
Strategy: {strategy}

This is an automated message from your trading bot.
            """,
            'position_closed': """
Subject: Position Closed - {symbol} {side}

A position has been closed:

Symbol: {symbol}
Side: {side}
Quantity: {quantity:.6f}
Entry Price: ${entry_price:.2f}
Exit Price: ${exit_price:.2f}
PnL: ${pnl:.2f} ({pnl_pct:.1%})
Reason: {reason}
Time: {time}
Strategy: {strategy}

This is an automated message from your trading bot.
            """,
            'stop_loss': """
Subject: Stop Loss Triggered - {symbol}

A stop loss has been triggered:

Symbol: {symbol}
Stop Price: ${stop_price:.2f}
Current Price: ${current_price:.2f}
PnL: ${pnl:.2f}
Time: {time}
Strategy: {strategy}

This is an automated message from your trading bot.
            """,
            'take_profit': """
Subject: Take Profit Triggered - {symbol}

A take profit has been triggered:

Symbol: {symbol}
Take Profit Price: ${take_profit_price:.2f}
Current Price: ${current_price:.2f}
PnL: ${pnl:.2f}
Time: {time}
Strategy: {strategy}

This is an automated message from your trading bot.
            """,
            'error': """
Subject: Trading Bot Error Alert

An error has occurred in your trading bot:

Error: {error}
Context: {context}
Time: {time}

Please check your trading bot immediately.

This is an automated message from your trading bot.
            """,
            'warning': """
Subject: Trading Bot Warning Alert

A warning has been triggered in your trading bot:

Warning: {warning}
Context: {context}
Time: {time}

Please review your trading bot configuration.

This is an automated message from your trading bot.
            """,
            'performance': """
Subject: Trading Performance Update

Trading performance update:

Total PnL: ${total_pnl:.2f}
Win Rate: {win_rate:.1%}
Open Positions: {open_positions}
Total Trades: {total_trades}
Time: {time}

This is an automated message from your trading bot.
            """,
            'system_start': """
Subject: Trading System Started

Your trading system has been started:

Strategy: {strategy}
Exchange: {exchange}
Time: {time}

This is an automated message from your trading bot.
            """,
            'system_stop': """
Subject: Trading System Stopped

Your trading system has been stopped:

Reason: {reason}
Time: {time}

This is an automated message from your trading bot.
            """
        }
    
    async def send_email(self, 
                        subject: str,
                        body: str,
                        recipients: List[str] = None,
                        attachments: List[str] = None) -> bool:
        """
        Send an email.
        
        Args:
            subject: Email subject
            body: Email body
            recipients: List of recipients (if None, uses config recipients)
            attachments: List of file paths to attach
            
        Returns:
            True if sent successfully
        """
        if not self.config.enabled:
            return False
        
        # Check rate limit
        if not self._check_rate_limit():
            self.logger.warning("Rate limit exceeded, skipping email")
            return False
        
        recipients = recipients or self.config.recipients
        
        if not recipients:
            self.logger.error("No recipients specified")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.config.from_email
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = subject
            
            # Add body
            msg.attach(MIMEText(body, 'plain'))
            
            # Add attachments if any
            if attachments:
                for file_path in attachments:
                    try:
                        with open(file_path, 'rb') as attachment:
                            part = MIMEBase('application', 'octet-stream')
                            part.set_payload(attachment.read())
                        
                        encoders.encode_base64(part)
                        part.add_header(
                            'Content-Disposition',
                            f'attachment; filename= {file_path}'
                        )
                        msg.attach(part)
                    except Exception as e:
                        self.logger.error(f"Error attaching file {file_path}: {e}")
            
            # Send email
            await self._send_smtp_email(msg, recipients)
            
            self.message_times.append(datetime.now())
            self.logger.debug(f"Email sent successfully to {len(recipients)} recipients")
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending email: {e}")
            return False
    
    async def _send_smtp_email(self, msg: MIMEMultipart, recipients: List[str]):
        """Send email via SMTP."""
        try:
            # Create SMTP session
            if self.config.use_ssl:
                server = smtplib.SMTP_SSL(self.config.smtp_server, self.config.smtp_port)
            else:
                server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
                
                if self.config.use_tls:
                    server.starttls()
            
            # Login
            server.login(self.config.username, self.config.password)
            
            # Send email
            text = msg.as_string()
            server.sendmail(self.config.from_email, recipients, text)
            server.quit()
            
        except Exception as e:
            self.logger.error(f"SMTP error: {e}")
            raise
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        now = datetime.now()
        
        # Clean up old message times
        if (now - self.last_cleanup).total_seconds() > 3600:
            self.message_times = [
                msg_time for msg_time in self.message_times
                if (now - msg_time).total_seconds() < 3600
            ]
            self.last_cleanup = now
        
        # Check if we're within rate limit
        return len(self.message_times) < self.config.rate_limit_per_hour
    
    async def send_signal(self, signal: StrategySignal) -> bool:
        """Send trading signal notification."""
        if not self.config.send_signals:
            return False
        
        subject = f"Trading Signal - {signal.symbol} {signal.side.value.upper()}"
        body = self.templates['signal'].format(
            symbol=signal.symbol,
            side=signal.side.value.upper(),
            price=signal.price,
            strength=signal.strength,
            time=signal.timestamp.strftime('%H:%M:%S'),
            strategy=signal.strategy_name
        )
        
        return await self.send_email(subject, body)
    
    async def send_order_placed(self, order: Order) -> bool:
        """Send order placed notification."""
        if not self.config.send_orders:
            return False
        
        subject = f"Order Placed - {order.symbol} {order.side.value.upper()}"
        body = self.templates['order_placed'].format(
            symbol=order.symbol,
            side=order.side.value.upper(),
            quantity=order.quantity,
            price=order.price or 0,
            order_type=order.type.value.upper(),
            time=order.timestamp.strftime('%H:%M:%S'),
            strategy=order.strategy_name
        )
        
        return await self.send_email(subject, body)
    
    async def send_order_filled(self, order: Order) -> bool:
        """Send order filled notification."""
        if not self.config.send_orders:
            return False
        
        subject = f"Order Filled - {order.symbol} {order.side.value.upper()}"
        body = self.templates['order_filled'].format(
            symbol=order.symbol,
            side=order.side.value.upper(),
            quantity=order.quantity,
            price=order.price or 0,
            time=order.filled_at.strftime('%H:%M:%S') if order.filled_at else order.timestamp.strftime('%H:%M:%S'),
            strategy=order.strategy_name
        )
        
        return await self.send_email(subject, body)
    
    async def send_order_failed(self, order: Order, error: str) -> bool:
        """Send order failed notification."""
        if not self.config.send_orders:
            return False
        
        subject = f"Order Failed - {order.symbol} {order.side.value.upper()}"
        body = self.templates['order_failed'].format(
            symbol=order.symbol,
            side=order.side.value.upper(),
            quantity=order.quantity,
            price=order.price or 0,
            error=error,
            time=order.timestamp.strftime('%H:%M:%S'),
            strategy=order.strategy_name
        )
        
        return await self.send_email(subject, body)
    
    async def send_position_opened(self, position: Position) -> bool:
        """Send position opened notification."""
        if not self.config.send_positions:
            return False
        
        subject = f"Position Opened - {position.symbol} {position.side.value.upper()}"
        body = self.templates['position_opened'].format(
            symbol=position.symbol,
            side=position.side.value.upper(),
            quantity=position.quantity,
            entry_price=position.entry_price,
            time=position.timestamp.strftime('%H:%M:%S'),
            strategy=position.strategy_name
        )
        
        return await self.send_email(subject, body)
    
    async def send_position_closed(self, position: Position, reason: str) -> bool:
        """Send position closed notification."""
        if not self.config.send_positions:
            return False
        
        pnl = position.metadata.get('final_pnl', 0)
        pnl_pct = (pnl / (position.entry_price * position.quantity)) * 100 if position.entry_price * position.quantity > 0 else 0
        
        subject = f"Position Closed - {position.symbol} {position.side.value.upper()}"
        body = self.templates['position_closed'].format(
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
        
        return await self.send_email(subject, body)
    
    async def send_stop_loss(self, position: Position, stop_price: float) -> bool:
        """Send stop-loss triggered notification."""
        if not self.config.send_alerts:
            return False
        
        unrealized_pnl = position.metadata.get('unrealized_pnl', 0)
        
        subject = f"Stop Loss Triggered - {position.symbol}"
        body = self.templates['stop_loss'].format(
            symbol=position.symbol,
            stop_price=stop_price,
            current_price=position.current_price,
            pnl=unrealized_pnl,
            time=datetime.now().strftime('%H:%M:%S'),
            strategy=position.strategy_name
        )
        
        return await self.send_email(subject, body)
    
    async def send_take_profit(self, position: Position, take_profit_price: float) -> bool:
        """Send take-profit triggered notification."""
        if not self.config.send_alerts:
            return False
        
        unrealized_pnl = position.metadata.get('unrealized_pnl', 0)
        
        subject = f"Take Profit Triggered - {position.symbol}"
        body = self.templates['take_profit'].format(
            symbol=position.symbol,
            take_profit_price=take_profit_price,
            current_price=position.current_price,
            pnl=unrealized_pnl,
            time=datetime.now().strftime('%H:%M:%S'),
            strategy=position.strategy_name
        )
        
        return await self.send_email(subject, body)
    
    async def send_error(self, error: str, context: Dict[str, Any] = None) -> bool:
        """Send error notification."""
        if not self.config.send_alerts:
            return False
        
        context_str = json.dumps(context or {}, indent=2)
        
        subject = "Trading Bot Error Alert"
        body = self.templates['error'].format(
            error=error,
            context=context_str,
            time=datetime.now().strftime('%H:%M:%S')
        )
        
        return await self.send_email(subject, body)
    
    async def send_warning(self, warning: str, context: Dict[str, Any] = None) -> bool:
        """Send warning notification."""
        if not self.config.send_alerts:
            return False
        
        context_str = json.dumps(context or {}, indent=2)
        
        subject = "Trading Bot Warning Alert"
        body = self.templates['warning'].format(
            warning=warning,
            context=context_str,
            time=datetime.now().strftime('%H:%M:%S')
        )
        
        return await self.send_email(subject, body)
    
    async def send_performance_update(self, 
                                    total_pnl: float,
                                    win_rate: float,
                                    open_positions: int,
                                    total_trades: int) -> bool:
        """Send performance update notification."""
        if not self.config.send_performance:
            return False
        
        subject = "Trading Performance Update"
        body = self.templates['performance'].format(
            total_pnl=total_pnl,
            win_rate=win_rate,
            open_positions=open_positions,
            total_trades=total_trades,
            time=datetime.now().strftime('%H:%M:%S')
        )
        
        return await self.send_email(subject, body)
    
    async def send_system_start(self, strategy: str, exchange: str) -> bool:
        """Send system start notification."""
        if not self.config.send_alerts:
            return False
        
        subject = "Trading System Started"
        body = self.templates['system_start'].format(
            strategy=strategy,
            exchange=exchange,
            time=datetime.now().strftime('%H:%M:%S')
        )
        
        return await self.send_email(subject, body)
    
    async def send_system_stop(self, reason: str) -> bool:
        """Send system stop notification."""
        if not self.config.send_alerts:
            return False
        
        subject = "Trading System Stopped"
        body = self.templates['system_stop'].format(
            reason=reason,
            time=datetime.now().strftime('%H:%M:%S')
        )
        
        return await self.send_email(subject, body)
    
    async def send_daily_summary(self, 
                               total_pnl: float,
                               trades_today: int,
                               win_rate: float,
                               open_positions: int,
                               top_performers: List[Dict[str, Any]]) -> bool:
        """Send daily summary notification."""
        if not self.config.send_performance:
            return False
        
        subject = f"Daily Trading Summary - {datetime.now().strftime('%Y-%m-%d')}"
        
        body = f"""
Daily Trading Summary
Date: {datetime.now().strftime('%Y-%m-%d')}

Performance:
- Total PnL: ${total_pnl:.2f}
- Win Rate: {win_rate:.1%}
- Open Positions: {open_positions}
- Trades Today: {trades_today}

Top Performers:
"""
        
        for i, performer in enumerate(top_performers[:3], 1):
            body += f"{i}. {performer['symbol']}: ${performer['pnl']:.2f}\n"
        
        body += f"\nGenerated at: {datetime.now().strftime('%H:%M:%S')}\n"
        body += "\nThis is an automated message from your trading bot."
        
        return await self.send_email(subject, body)
    
    async def send_risk_alert(self, 
                            alert_type: str,
                            current_value: float,
                            limit_value: float,
                            symbol: str = None) -> bool:
        """Send risk alert notification."""
        if not self.config.send_alerts:
            return False
        
        subject = f"Risk Alert - {alert_type}"
        
        body = f"""
Risk Alert

Type: {alert_type}
Symbol: {symbol or 'Portfolio'}
Current Value: {current_value:.2f}
Limit Value: {limit_value:.2f}
Time: {datetime.now().strftime('%H:%M:%S')}

Please review your trading bot configuration immediately.

This is an automated message from your trading bot.
        """
        
        return await self.send_email(subject, body)
    
    async def send_custom_email(self, subject: str, body: str, recipients: List[str] = None) -> bool:
        """Send a custom email."""
        return await self.send_email(subject, body, recipients)
    
    def update_config(self, config: EmailConfig):
        """Update email configuration."""
        self.config = config
        self.logger.info("Email configuration updated")
    
    def get_status(self) -> Dict[str, Any]:
        """Get notifier status."""
        return {
            'enabled': self.config.enabled,
            'send_signals': self.config.send_signals,
            'send_orders': self.config.send_orders,
            'send_positions': self.config.send_positions,
            'send_alerts': self.config.send_alerts,
            'send_performance': self.config.send_performance,
            'recipients': len(self.config.recipients),
            'emails_sent_last_hour': len([
                t for t in self.message_times
                if (datetime.now() - t).total_seconds() < 3600
            ]),
            'rate_limit_per_hour': self.config.rate_limit_per_hour
        }
    
    async def test_connection(self) -> bool:
        """Test email connection."""
        try:
            subject = "Trading Bot Test Email"
            body = f"🤖 Trading Bot Test Email\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            return await self.send_email(subject, body)
        except Exception as e:
            self.logger.error(f"Email connection test failed: {e}")
            return False
