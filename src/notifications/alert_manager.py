"""
Alert Manager for Trading Bot

This module coordinates all notification systems and manages alert routing,
filtering, and delivery across multiple channels.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from ..core.data_models import (
    StrategySignal, Order, Trade, Position, OrderSide, OrderType, 
    OrderStatus, PositionStatus
)
from .telegram_notifier import TelegramNotifier, TelegramConfig
from .discord_notifier import DiscordNotifier, DiscordConfig
from .email_notifier import EmailNotifier, EmailConfig


class AlertLevel(Enum):
    """Alert priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AlertRule:
    """Alert routing rule."""
    name: str
    enabled: bool = True
    channels: List[str] = None  # ['telegram', 'discord', 'email']
    alert_levels: List[AlertLevel] = None
    event_types: List[str] = None  # ['signal', 'order', 'position', 'error']
    symbols: List[str] = None  # Filter by symbols
    strategies: List[str] = None  # Filter by strategies
    conditions: Dict[str, Any] = None  # Custom conditions


class AlertManager:
    """Manages and coordinates all notification systems."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the alert manager.
        
        Args:
            config: Alert manager configuration
        """
        self.config = config
        self.logger = logging.getLogger("AlertManager")
        
        # Initialize notifiers
        self.notifiers = {}
        self._initialize_notifiers()
        
        # Alert rules
        self.alert_rules = []
        self._setup_default_rules()
        
        # Alert history
        self.alert_history = []
        self.max_history = 1000
        
        # Rate limiting
        self.rate_limits = {
            'telegram': {'count': 0, 'reset_time': datetime.now()},
            'discord': {'count': 0, 'reset_time': datetime.now()},
            'email': {'count': 0, 'reset_time': datetime.now()}
        }
    
    def _initialize_notifiers(self):
        """Initialize notification systems."""
        try:
            # Initialize Telegram
            if 'telegram' in self.config and self.config['telegram'].get('enabled', False):
                telegram_config = TelegramConfig(**self.config['telegram'])
                self.notifiers['telegram'] = TelegramNotifier(telegram_config)
                self.logger.info("Telegram notifier initialized")
            
            # Initialize Discord
            if 'discord' in self.config and self.config['discord'].get('enabled', False):
                discord_config = DiscordConfig(**self.config['discord'])
                self.notifiers['discord'] = DiscordNotifier(discord_config)
                self.logger.info("Discord notifier initialized")
            
            # Initialize Email
            if 'email' in self.config and self.config['email'].get('enabled', False):
                email_config = EmailConfig(**self.config['email'])
                self.notifiers['email'] = EmailNotifier(email_config)
                self.logger.info("Email notifier initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing notifiers: {e}")
    
    def _setup_default_rules(self):
        """Setup default alert rules."""
        default_rules = [
            AlertRule(
                name="critical_alerts",
                channels=['telegram', 'discord', 'email'],
                alert_levels=[AlertLevel.CRITICAL],
                event_types=['error', 'system_stop']
            ),
            AlertRule(
                name="trading_signals",
                channels=['telegram', 'discord'],
                alert_levels=[AlertLevel.MEDIUM, AlertLevel.HIGH],
                event_types=['signal']
            ),
            AlertRule(
                name="order_notifications",
                channels=['telegram'],
                alert_levels=[AlertLevel.MEDIUM],
                event_types=['order_placed', 'order_filled', 'order_failed']
            ),
            AlertRule(
                name="position_updates",
                channels=['telegram', 'discord'],
                alert_levels=[AlertLevel.MEDIUM, AlertLevel.HIGH],
                event_types=['position_opened', 'position_closed']
            ),
            AlertRule(
                name="performance_updates",
                channels=['email'],
                alert_levels=[AlertLevel.LOW],
                event_types=['performance']
            )
        ]
        
        self.alert_rules = default_rules
    
    async def send_alert(self, 
                        event_type: str,
                        data: Dict[str, Any],
                        alert_level: AlertLevel = AlertLevel.MEDIUM,
                        channels: List[str] = None) -> Dict[str, bool]:
        """
        Send alert to appropriate channels.
        
        Args:
            event_type: Type of event
            data: Event data
            alert_level: Alert priority level
            channels: Specific channels to use (if None, uses rules)
            
        Returns:
            Dictionary of channel -> success status
        """
        # Determine channels based on rules
        if channels is None:
            channels = self._get_channels_for_event(event_type, alert_level, data)
        
        # Send to each channel
        results = {}
        for channel in channels:
            if channel in self.notifiers:
                try:
                    success = await self._send_to_channel(channel, event_type, data, alert_level)
                    results[channel] = success
                except Exception as e:
                    self.logger.error(f"Error sending alert to {channel}: {e}")
                    results[channel] = False
            else:
                self.logger.warning(f"Channel {channel} not available")
                results[channel] = False
        
        # Log alert
        self._log_alert(event_type, data, alert_level, channels, results)
        
        return results
    
    def _get_channels_for_event(self, 
                               event_type: str, 
                               alert_level: AlertLevel, 
                               data: Dict[str, Any]) -> List[str]:
        """Get channels for event based on rules."""
        channels = set()
        
        for rule in self.alert_rules:
            if not rule.enabled:
                continue
            
            # Check event type
            if rule.event_types and event_type not in rule.event_types:
                continue
            
            # Check alert level
            if rule.alert_levels and alert_level not in rule.alert_levels:
                continue
            
            # Check symbol filter
            if rule.symbols and data.get('symbol') not in rule.symbols:
                continue
            
            # Check strategy filter
            if rule.strategies and data.get('strategy_name') not in rule.strategies:
                continue
            
            # Check custom conditions
            if rule.conditions and not self._check_conditions(rule.conditions, data):
                continue
            
            # Add channels
            if rule.channels:
                channels.update(rule.channels)
        
        return list(channels)
    
    def _check_conditions(self, conditions: Dict[str, Any], data: Dict[str, Any]) -> bool:
        """Check custom conditions."""
        try:
            for key, expected_value in conditions.items():
                actual_value = data.get(key)
                
                if isinstance(expected_value, dict):
                    # Range condition
                    if 'min' in expected_value and actual_value < expected_value['min']:
                        return False
                    if 'max' in expected_value and actual_value > expected_value['max']:
                        return False
                else:
                    # Exact match
                    if actual_value != expected_value:
                        return False
            
            return True
        except Exception as e:
            self.logger.error(f"Error checking conditions: {e}")
            return False
    
    async def _send_to_channel(self, 
                              channel: str, 
                              event_type: str, 
                              data: Dict[str, Any], 
                              alert_level: AlertLevel) -> bool:
        """Send alert to specific channel."""
        notifier = self.notifiers[channel]
        
        # Check rate limit
        if not self._check_rate_limit(channel):
            self.logger.warning(f"Rate limit exceeded for {channel}")
            return False
        
        # Route to appropriate notifier method
        if event_type == 'signal':
            signal = data.get('signal')
            if signal:
                return await notifier.send_signal(signal)
        
        elif event_type == 'order_placed':
            order = data.get('order')
            if order:
                return await notifier.send_order_placed(order)
        
        elif event_type == 'order_filled':
            order = data.get('order')
            if order:
                return await notifier.send_order_filled(order)
        
        elif event_type == 'order_failed':
            order = data.get('order')
            error = data.get('error')
            if order and error:
                return await notifier.send_order_failed(order, error)
        
        elif event_type == 'position_opened':
            position = data.get('position')
            if position:
                return await notifier.send_position_opened(position)
        
        elif event_type == 'position_closed':
            position = data.get('position')
            reason = data.get('reason')
            if position and reason:
                return await notifier.send_position_closed(position, reason)
        
        elif event_type == 'stop_loss':
            position = data.get('position')
            stop_price = data.get('stop_price')
            if position and stop_price:
                return await notifier.send_stop_loss(position, stop_price)
        
        elif event_type == 'take_profit':
            position = data.get('position')
            take_profit_price = data.get('take_profit_price')
            if position and take_profit_price:
                return await notifier.send_take_profit(position, take_profit_price)
        
        elif event_type == 'error':
            error = data.get('error')
            context = data.get('context')
            if error:
                return await notifier.send_error(error, context)
        
        elif event_type == 'warning':
            warning = data.get('warning')
            context = data.get('context')
            if warning:
                return await notifier.send_warning(warning, context)
        
        elif event_type == 'performance':
            return await notifier.send_performance_update(
                data.get('total_pnl', 0),
                data.get('win_rate', 0),
                data.get('open_positions', 0),
                data.get('total_trades', 0)
            )
        
        elif event_type == 'system_start':
            return await notifier.send_system_start(
                data.get('strategy', ''),
                data.get('exchange', '')
            )
        
        elif event_type == 'system_stop':
            return await notifier.send_system_stop(data.get('reason', ''))
        
        else:
            self.logger.warning(f"Unknown event type: {event_type}")
            return False
    
    def _check_rate_limit(self, channel: str) -> bool:
        """Check rate limit for channel."""
        if channel not in self.rate_limits:
            return True
        
        now = datetime.now()
        rate_info = self.rate_limits[channel]
        
        # Reset counter if needed
        if (now - rate_info['reset_time']).total_seconds() > 60:
            rate_info['count'] = 0
            rate_info['reset_time'] = now
        
        # Check limit (assuming 30 messages per minute)
        if rate_info['count'] >= 30:
            return False
        
        rate_info['count'] += 1
        return True
    
    def _log_alert(self, 
                   event_type: str, 
                   data: Dict[str, Any], 
                   alert_level: AlertLevel, 
                   channels: List[str], 
                   results: Dict[str, bool]):
        """Log alert to history."""
        alert_record = {
            'timestamp': datetime.now(),
            'event_type': event_type,
            'alert_level': alert_level.value,
            'channels': channels,
            'results': results,
            'data_summary': {
                'symbol': data.get('symbol'),
                'strategy': data.get('strategy_name'),
                'error': data.get('error'),
                'warning': data.get('warning')
            }
        }
        
        self.alert_history.append(alert_record)
        
        # Keep only recent alerts
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history:]
    
    # Convenience methods for common alerts
    
    async def send_signal_alert(self, signal: StrategySignal, alert_level: AlertLevel = AlertLevel.MEDIUM):
        """Send signal alert."""
        return await self.send_alert(
            'signal',
            {'signal': signal, 'symbol': signal.symbol, 'strategy_name': signal.strategy_name},
            alert_level
        )
    
    async def send_order_alert(self, order: Order, event_type: str, error: str = None):
        """Send order alert."""
        data = {'order': order, 'symbol': order.symbol, 'strategy_name': order.strategy_name}
        if error:
            data['error'] = error
        
        alert_level = AlertLevel.HIGH if event_type == 'order_failed' else AlertLevel.MEDIUM
        return await self.send_alert(event_type, data, alert_level)
    
    async def send_position_alert(self, position: Position, event_type: str, reason: str = None, stop_price: float = None, take_profit_price: float = None):
        """Send position alert."""
        data = {'position': position, 'symbol': position.symbol, 'strategy_name': position.strategy_name}
        
        if reason:
            data['reason'] = reason
        if stop_price:
            data['stop_price'] = stop_price
        if take_profit_price:
            data['take_profit_price'] = take_profit_price
        
        alert_level = AlertLevel.HIGH if event_type in ['stop_loss', 'take_profit'] else AlertLevel.MEDIUM
        return await self.send_alert(event_type, data, alert_level)
    
    async def send_error_alert(self, error: str, context: Dict[str, Any] = None):
        """Send error alert."""
        return await self.send_alert(
            'error',
            {'error': error, 'context': context},
            AlertLevel.CRITICAL
        )
    
    async def send_warning_alert(self, warning: str, context: Dict[str, Any] = None):
        """Send warning alert."""
        return await self.send_alert(
            'warning',
            {'warning': warning, 'context': context},
            AlertLevel.MEDIUM
        )
    
    async def send_performance_alert(self, 
                                   total_pnl: float,
                                   win_rate: float,
                                   open_positions: int,
                                   total_trades: int):
        """Send performance alert."""
        return await self.send_alert(
            'performance',
            {
                'total_pnl': total_pnl,
                'win_rate': win_rate,
                'open_positions': open_positions,
                'total_trades': total_trades
            },
            AlertLevel.LOW
        )
    
    async def send_system_alert(self, event_type: str, strategy: str = None, exchange: str = None, reason: str = None):
        """Send system alert."""
        data = {}
        if strategy:
            data['strategy'] = strategy
        if exchange:
            data['exchange'] = exchange
        if reason:
            data['reason'] = reason
        
        alert_level = AlertLevel.CRITICAL if event_type == 'system_stop' else AlertLevel.MEDIUM
        return await self.send_alert(event_type, data, alert_level)
    
    # Alert rule management
    
    def add_alert_rule(self, rule: AlertRule):
        """Add a new alert rule."""
        self.alert_rules.append(rule)
        self.logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_name: str):
        """Remove an alert rule."""
        self.alert_rules = [rule for rule in self.alert_rules if rule.name != rule_name]
        self.logger.info(f"Removed alert rule: {rule_name}")
    
    def update_alert_rule(self, rule_name: str, updates: Dict[str, Any]):
        """Update an alert rule."""
        for rule in self.alert_rules:
            if rule.name == rule_name:
                for key, value in updates.items():
                    if hasattr(rule, key):
                        setattr(rule, key, value)
                self.logger.info(f"Updated alert rule: {rule_name}")
                break
    
    def get_alert_rules(self) -> List[AlertRule]:
        """Get all alert rules."""
        return self.alert_rules.copy()
    
    # Status and monitoring
    
    def get_status(self) -> Dict[str, Any]:
        """Get alert manager status."""
        notifier_status = {}
        for channel, notifier in self.notifiers.items():
            notifier_status[channel] = notifier.get_status()
        
        return {
            'enabled_channels': list(self.notifiers.keys()),
            'notifier_status': notifier_status,
            'alert_rules_count': len(self.alert_rules),
            'alert_history_count': len(self.alert_history),
            'rate_limits': self.rate_limits
        }
    
    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent alert history."""
        return self.alert_history[-limit:] if self.alert_history else []
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        if not self.alert_history:
            return {}
        
        # Count by event type
        event_counts = {}
        channel_counts = {}
        level_counts = {}
        
        for alert in self.alert_history:
            event_type = alert['event_type']
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
            for channel in alert['channels']:
                channel_counts[channel] = channel_counts.get(channel, 0) + 1
            
            alert_level = alert['alert_level']
            level_counts[alert_level] = level_counts.get(alert_level, 0) + 1
        
        return {
            'total_alerts': len(self.alert_history),
            'event_counts': event_counts,
            'channel_counts': channel_counts,
            'level_counts': level_counts,
            'success_rate': self._calculate_success_rate()
        }
    
    def _calculate_success_rate(self) -> float:
        """Calculate alert success rate."""
        if not self.alert_history:
            return 0.0
        
        total_sends = 0
        successful_sends = 0
        
        for alert in self.alert_history:
            for channel, success in alert['results'].items():
                total_sends += 1
                if success:
                    successful_sends += 1
        
        return (successful_sends / total_sends * 100) if total_sends > 0 else 0.0
    
    async def test_all_channels(self) -> Dict[str, bool]:
        """Test all notification channels."""
        results = {}
        
        for channel, notifier in self.notifiers.items():
            try:
                if hasattr(notifier, 'test_connection'):
                    success = await notifier.test_connection()
                    results[channel] = success
                else:
                    results[channel] = False
            except Exception as e:
                self.logger.error(f"Error testing {channel}: {e}")
                results[channel] = False
        
        return results
    
    def clear_alert_history(self):
        """Clear alert history."""
        self.alert_history.clear()
        self.logger.info("Alert history cleared")
    
    def update_config(self, config: Dict[str, Any]):
        """Update alert manager configuration."""
        self.config = config
        self._initialize_notifiers()
        self.logger.info("Alert manager configuration updated")
