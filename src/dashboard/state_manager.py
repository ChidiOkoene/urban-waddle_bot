"""
State Manager for Trading Dashboard

This module manages the application state, including bot status, configuration,
and data persistence across sessions.
"""

import streamlit as st
import json
import yaml
import pickle
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
import threading
import time

from ..core.data_models import Trade, Position, OHLCV, StrategySignal


class StateManager:
    """Manages application state and persistence."""
    
    def __init__(self, state_file: str = "dashboard_state.json"):
        self.state_file = Path(state_file)
        self.lock = threading.Lock()
        self._initialize_state()
    
    def _initialize_state(self):
        """Initialize the application state."""
        if 'dashboard_state' not in st.session_state:
            st.session_state.dashboard_state = {
                'bot_status': {
                    'running': False,
                    'start_time': None,
                    'last_update': None,
                    'active_strategy': None,
                    'exchange': None
                },
                'configuration': {
                    'exchanges': {
                        'selected': 'binance',
                        'binance': {
                            'api_key': '',
                            'secret': '',
                            'sandbox': True
                        },
                        'bitget': {
                            'api_key': '',
                            'secret': '',
                            'sandbox': True
                        },
                        'mt5': {
                            'account': '',
                            'password': '',
                            'server': ''
                        }
                    },
                    'strategies': {
                        'active': 'rsi_macd',
                        'enabled': ['rsi_macd', 'bollinger', 'ema_crossover'],
                        'parameters': {}
                    },
                    'risk_management': {
                        'max_position_size': 0.1,
                        'max_drawdown': 0.2,
                        'stop_loss': 0.02,
                        'take_profit': 0.04,
                        'max_positions': 10
                    },
                    'dashboard': {
                        'refresh_interval': 5,
                        'max_candles': 1000,
                        'theme': 'dark',
                        'auto_refresh': True
                    }
                },
                'data': {
                    'market_data': {},
                    'positions': [],
                    'trades': [],
                    'equity_curve': [],
                    'signals': []
                },
                'performance': {
                    'metrics': {},
                    'backtest_results': {},
                    'optimization_results': {}
                },
                'alerts': {
                    'enabled': True,
                    'telegram': {
                        'enabled': False,
                        'bot_token': '',
                        'chat_id': ''
                    },
                    'discord': {
                        'enabled': False,
                        'webhook_url': ''
                    },
                    'email': {
                        'enabled': False,
                        'smtp_server': '',
                        'port': 587,
                        'username': '',
                        'password': '',
                        'recipients': []
                    }
                },
                'logs': {
                    'level': 'INFO',
                    'max_entries': 1000,
                    'entries': []
                },
                'session': {
                    'start_time': datetime.now().isoformat(),
                    'last_activity': datetime.now().isoformat(),
                    'page_views': {},
                    'user_actions': []
                }
            }
    
    def get_state(self, key: str = None) -> Union[Dict[str, Any], Any]:
        """
        Get state value by key.
        
        Args:
            key: State key (dot notation supported, e.g., 'bot_status.running')
        
        Returns:
            State value or entire state if key is None
        """
        with self.lock:
            state = st.session_state.dashboard_state
            
            if key is None:
                return state
            
            # Support dot notation
            keys = key.split('.')
            value = state
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return None
            
            return value
    
    def set_state(self, key: str, value: Any):
        """
        Set state value by key.
        
        Args:
            key: State key (dot notation supported)
            value: Value to set
        """
        with self.lock:
            state = st.session_state.dashboard_state
            
            # Support dot notation
            keys = key.split('.')
            current = state
            
            # Navigate to the parent of the target key
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            # Set the value
            current[keys[-1]] = value
            
            # Update last activity
            state['session']['last_activity'] = datetime.now().isoformat()
    
    def update_state(self, updates: Dict[str, Any]):
        """
        Update multiple state values.
        
        Args:
            updates: Dictionary of key-value pairs to update
        """
        with self.lock:
            for key, value in updates.items():
                self.set_state(key, value)
    
    def get_bot_status(self) -> Dict[str, Any]:
        """Get bot status."""
        return self.get_state('bot_status')
    
    def set_bot_status(self, status: Dict[str, Any]):
        """Set bot status."""
        self.set_state('bot_status', status)
    
    def start_bot(self, strategy: str, exchange: str):
        """Start the bot."""
        self.set_state('bot_status', {
            'running': True,
            'start_time': datetime.now().isoformat(),
            'last_update': datetime.now().isoformat(),
            'active_strategy': strategy,
            'exchange': exchange
        })
        
        self.log_action('bot_started', {
            'strategy': strategy,
            'exchange': exchange,
            'timestamp': datetime.now().isoformat()
        })
    
    def stop_bot(self):
        """Stop the bot."""
        self.set_state('bot_status', {
            'running': False,
            'start_time': None,
            'last_update': datetime.now().isoformat(),
            'active_strategy': None,
            'exchange': None
        })
        
        self.log_action('bot_stopped', {
            'timestamp': datetime.now().isoformat()
        })
    
    def update_bot_status(self, **kwargs):
        """Update bot status with new values."""
        current_status = self.get_bot_status()
        current_status.update(kwargs)
        current_status['last_update'] = datetime.now().isoformat()
        self.set_bot_status(current_status)
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get configuration."""
        return self.get_state('configuration')
    
    def set_configuration(self, config: Dict[str, Any]):
        """Set configuration."""
        self.set_state('configuration', config)
    
    def update_configuration(self, updates: Dict[str, Any]):
        """Update configuration."""
        current_config = self.get_configuration()
        current_config.update(updates)
        self.set_configuration(current_config)
    
    def get_strategy_parameters(self, strategy: str) -> Dict[str, Any]:
        """Get strategy parameters."""
        return self.get_state(f'configuration.strategies.parameters.{strategy}')
    
    def set_strategy_parameters(self, strategy: str, parameters: Dict[str, Any]):
        """Set strategy parameters."""
        self.set_state(f'configuration.strategies.parameters.{strategy}', parameters)
    
    def get_market_data(self, symbol: str) -> List[OHLCV]:
        """Get market data for symbol."""
        data = self.get_state(f'data.market_data.{symbol}')
        if data:
            return [OHLCV(**item) for item in data]
        return []
    
    def set_market_data(self, symbol: str, data: List[OHLCV]):
        """Set market data for symbol."""
        data_dict = [item.__dict__ for item in data]
        self.set_state(f'data.market_data.{symbol}', data_dict)
    
    def get_positions(self) -> List[Position]:
        """Get current positions."""
        positions = self.get_state('data.positions')
        if positions:
            return [Position(**pos) for pos in positions]
        return []
    
    def set_positions(self, positions: List[Position]):
        """Set current positions."""
        positions_dict = [pos.__dict__ for pos in positions]
        self.set_state('data.positions', positions_dict)
    
    def get_trades(self) -> List[Trade]:
        """Get recent trades."""
        trades = self.get_state('data.trades')
        if trades:
            return [Trade(**trade) for trade in trades]
        return []
    
    def set_trades(self, trades: List[Trade]):
        """Set recent trades."""
        trades_dict = [trade.__dict__ for trade in trades]
        self.set_state('data.trades', trades_dict)
    
    def get_equity_curve(self) -> List[tuple]:
        """Get equity curve data."""
        return self.get_state('data.equity_curve')
    
    def set_equity_curve(self, equity_curve: List[tuple]):
        """Set equity curve data."""
        self.set_state('data.equity_curve', equity_curve)
    
    def get_signals(self) -> List[StrategySignal]:
        """Get recent signals."""
        signals = self.get_state('data.signals')
        if signals:
            return [StrategySignal(**signal) for signal in signals]
        return []
    
    def set_signals(self, signals: List[StrategySignal]):
        """Set recent signals."""
        signals_dict = [signal.__dict__ for signal in signals]
        self.set_state('data.signals', signals_dict)
    
    def add_signal(self, signal: StrategySignal):
        """Add a new signal."""
        signals = self.get_signals()
        signals.append(signal)
        
        # Keep only recent signals (last 100)
        if len(signals) > 100:
            signals = signals[-100:]
        
        self.set_signals(signals)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return self.get_state('performance.metrics')
    
    def set_performance_metrics(self, metrics: Dict[str, Any]):
        """Set performance metrics."""
        self.set_state('performance.metrics', metrics)
    
    def get_backtest_results(self, strategy: str = None) -> Union[Dict[str, Any], Any]:
        """Get backtest results."""
        if strategy:
            return self.get_state(f'performance.backtest_results.{strategy}')
        return self.get_state('performance.backtest_results')
    
    def set_backtest_results(self, strategy: str, results: Dict[str, Any]):
        """Set backtest results."""
        self.set_state(f'performance.backtest_results.{strategy}', results)
    
    def get_optimization_results(self, strategy: str = None) -> Union[Dict[str, Any], Any]:
        """Get optimization results."""
        if strategy:
            return self.get_state(f'performance.optimization_results.{strategy}')
        return self.get_state('performance.optimization_results')
    
    def set_optimization_results(self, strategy: str, results: Dict[str, Any]):
        """Set optimization results."""
        self.set_state(f'performance.optimization_results.{strategy}', results)
    
    def get_alerts_config(self) -> Dict[str, Any]:
        """Get alerts configuration."""
        return self.get_state('alerts')
    
    def set_alerts_config(self, config: Dict[str, Any]):
        """Set alerts configuration."""
        self.set_state('alerts', config)
    
    def log_action(self, action: str, data: Dict[str, Any] = None):
        """Log user action."""
        log_entry = {
            'action': action,
            'timestamp': datetime.now().isoformat(),
            'data': data or {}
        }
        
        # Get current logs
        logs = self.get_state('logs.entries')
        logs.append(log_entry)
        
        # Keep only recent logs
        max_entries = self.get_state('logs.max_entries')
        if len(logs) > max_entries:
            logs = logs[-max_entries:]
        
        self.set_state('logs.entries', logs)
    
    def get_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent logs."""
        logs = self.get_state('logs.entries')
        return logs[-limit:] if logs else []
    
    def clear_logs(self):
        """Clear all logs."""
        self.set_state('logs.entries', [])
    
    def save_state_to_file(self):
        """Save state to file."""
        try:
            with self.lock:
                state = st.session_state.dashboard_state
                
                # Convert datetime objects to strings
                state = self._serialize_state(state)
                
                with open(self.state_file, 'w') as f:
                    json.dump(state, f, indent=2)
                
                self.log_action('state_saved', {
                    'file': str(self.state_file),
                    'timestamp': datetime.now().isoformat()
                })
                
        except Exception as e:
            st.error(f"Error saving state: {e}")
    
    def load_state_from_file(self):
        """Load state from file."""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                
                # Convert string dates back to datetime objects
                state = self._deserialize_state(state)
                
                st.session_state.dashboard_state = state
                
                self.log_action('state_loaded', {
                    'file': str(self.state_file),
                    'timestamp': datetime.now().isoformat()
                })
                
        except Exception as e:
            st.error(f"Error loading state: {e}")
    
    def export_state(self, format: str = 'json') -> str:
        """Export state in specified format."""
        state = self.get_state()
        
        if format == 'json':
            return json.dumps(state, indent=2, default=str)
        elif format == 'yaml':
            return yaml.dump(state, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def import_state(self, data: str, format: str = 'json'):
        """Import state from string."""
        try:
            if format == 'json':
                state = json.loads(data)
            elif format == 'yaml':
                state = yaml.safe_load(data)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Convert string dates back to datetime objects
            state = self._deserialize_state(state)
            
            st.session_state.dashboard_state = state
            
            self.log_action('state_imported', {
                'format': format,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            st.error(f"Error importing state: {e}")
    
    def reset_state(self):
        """Reset state to defaults."""
        st.session_state.dashboard_state = {}
        self._initialize_state()
        
        self.log_action('state_reset', {
            'timestamp': datetime.now().isoformat()
        })
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        session = self.get_state('session')
        logs = self.get_logs()
        
        return {
            'start_time': session.get('start_time'),
            'last_activity': session.get('last_activity'),
            'duration': self._calculate_duration(session.get('start_time')),
            'total_actions': len(logs),
            'bot_running': self.get_state('bot_status.running'),
            'active_strategy': self.get_state('bot_status.active_strategy'),
            'positions_count': len(self.get_positions()),
            'trades_count': len(self.get_trades())
        }
    
    def _serialize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize state for JSON storage."""
        serialized = {}
        
        for key, value in state.items():
            if isinstance(value, dict):
                serialized[key] = self._serialize_state(value)
            elif isinstance(value, datetime):
                serialized[key] = value.isoformat()
            elif isinstance(value, list):
                serialized[key] = [
                    self._serialize_state(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                serialized[key] = value
        
        return serialized
    
    def _deserialize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize state from JSON storage."""
        deserialized = {}
        
        for key, value in state.items():
            if isinstance(value, dict):
                deserialized[key] = self._deserialize_state(value)
            elif isinstance(value, str) and self._is_datetime_string(value):
                try:
                    deserialized[key] = datetime.fromisoformat(value)
                except ValueError:
                    deserialized[key] = value
            elif isinstance(value, list):
                deserialized[key] = [
                    self._deserialize_state(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                deserialized[key] = value
        
        return deserialized
    
    def _is_datetime_string(self, value: str) -> bool:
        """Check if string is a datetime."""
        try:
            datetime.fromisoformat(value)
            return True
        except ValueError:
            return False
    
    def _calculate_duration(self, start_time: str) -> str:
        """Calculate session duration."""
        if not start_time:
            return "Unknown"
        
        try:
            start = datetime.fromisoformat(start_time)
            duration = datetime.now() - start
            
            hours, remainder = divmod(duration.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            
            return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        except ValueError:
            return "Unknown"
    
    def cleanup_old_data(self, max_age_hours: int = 24):
        """Clean up old data."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        # Clean up old logs
        logs = self.get_logs()
        filtered_logs = [
            log for log in logs
            if datetime.fromisoformat(log['timestamp']) > cutoff_time
        ]
        self.set_state('logs.entries', filtered_logs)
        
        # Clean up old signals
        signals = self.get_signals()
        filtered_signals = [
            signal for signal in signals
            if signal.timestamp > cutoff_time
        ]
        self.set_signals(filtered_signals)
        
        self.log_action('data_cleanup', {
            'max_age_hours': max_age_hours,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get dashboard metrics for display."""
        bot_status = self.get_bot_status()
        positions = self.get_positions()
        trades = self.get_trades()
        session_stats = self.get_session_stats()
        
        return {
            'bot_running': bot_status['running'],
            'active_strategy': bot_status['active_strategy'],
            'exchange': bot_status['exchange'],
            'positions_count': len(positions),
            'trades_count': len(trades),
            'session_duration': session_stats['duration'],
            'last_activity': session_stats['last_activity'],
            'total_actions': session_stats['total_actions']
        }
