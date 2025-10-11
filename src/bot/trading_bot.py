"""
Trading Bot Execution Engine

This module contains the main trading bot that orchestrates all components
to execute automated trading strategies.
"""

import asyncio
import threading
import time
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import yaml
from pathlib import Path

from ..core.exchange_interface import ExchangeInterface
from ..core.data_models import OHLCV, Trade, Position, StrategySignal, OrderSide, OrderType
from ..strategies.base_strategy import BaseStrategy
from ..risk.risk_manager import RiskManager
from ..risk.position_sizer import PositionSizer
from ..ml.predictor import MLPredictor
from ..database.db_manager import DatabaseManager
from .signal_generator import SignalGenerator
from .order_executor import OrderExecutor
from .position_monitor import PositionMonitor
from .event_logger import EventLogger


@dataclass
class BotConfig:
    """Configuration for the trading bot."""
    strategy_name: str
    exchange_name: str
    symbols: List[str]
    timeframes: List[str]
    risk_params: Dict[str, Any]
    ml_enabled: bool = False
    paper_trading: bool = True
    max_positions: int = 10
    refresh_interval: int = 5
    log_level: str = "INFO"


class TradingBot:
    """Main trading bot that orchestrates all components."""
    
    def __init__(self, 
                 exchange_adapter: ExchangeInterface,
                 strategy_name: str,
                 config: Dict[str, Any],
                 database_manager: Optional[DatabaseManager] = None):
        """
        Initialize the trading bot.
        
        Args:
            exchange_adapter: Exchange adapter instance
            strategy_name: Name of the strategy to use
            config: Bot configuration
            database_manager: Database manager instance
        """
        self.exchange_adapter = exchange_adapter
        self.strategy_name = strategy_name
        self.config = config
        self.database_manager = database_manager or DatabaseManager()
        
        # Bot state
        self.running = False
        self.start_time = None
        self.last_update = None
        self.active_positions = []
        self.recent_trades = []
        self.equity_curve = []
        self.signals_history = []
        
        # Components
        self.signal_generator = SignalGenerator()
        self.order_executor = OrderExecutor(exchange_adapter, self.database_manager)
        self.position_monitor = PositionMonitor(exchange_adapter, self.database_manager)
        self.event_logger = EventLogger(self.database_manager)
        self.risk_manager = RiskManager(config.get('risk_management', {}))
        self.position_sizer = PositionSizer(config.get('risk_management', {}))
        
        # ML predictor (optional)
        self.ml_predictor = None
        if config.get('ml_enabled', False):
            self.ml_predictor = MLPredictor()
        
        # Strategy instance
        self.strategy = self._create_strategy_instance()
        
        # Setup logging
        self._setup_logging()
        
        # Performance tracking
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0,
            'peak_equity': 0.0,
            'current_equity': 0.0
        }
    
    def _create_strategy_instance(self) -> BaseStrategy:
        """Create strategy instance based on strategy name."""
        strategy_map = {
            'rsi_macd': 'RSIMACDStrategy',
            'bollinger': 'BollingerMeanReversionStrategy',
            'ema_crossover': 'EMACrossoverStrategy',
            'grid_bot': 'GridBotStrategy',
            'dca': 'DCAStrategy',
            'breakout': 'BreakoutStrategy',
            'momentum': 'MomentumStrategy',
            'ichimoku': 'IchimokuStrategy',
            'arbitrage': 'ArbitrageStrategy'
        }
        
        strategy_class_name = strategy_map.get(self.strategy_name)
        if not strategy_class_name:
            raise ValueError(f"Unknown strategy: {self.strategy_name}")
        
        # Import strategy class dynamically
        try:
            if self.strategy_name == 'rsi_macd':
                from ..strategies.rsi_macd_strategy import RSIMACDStrategy
                return RSIMACDStrategy()
            elif self.strategy_name == 'bollinger':
                from ..strategies.bollinger_mean_reversion import BollingerMeanReversionStrategy
                return BollingerMeanReversionStrategy()
            elif self.strategy_name == 'ema_crossover':
                from ..strategies.ema_crossover_strategy import EMACrossoverStrategy
                return EMACrossoverStrategy()
            elif self.strategy_name == 'grid_bot':
                from ..strategies.grid_bot_strategy import GridBotStrategy
                return GridBotStrategy()
            elif self.strategy_name == 'dca':
                from ..strategies.dca_strategy import DCAStrategy
                return DCAStrategy()
            elif self.strategy_name == 'breakout':
                from ..strategies.breakout_strategy import BreakoutStrategy
                return BreakoutStrategy()
            elif self.strategy_name == 'momentum':
                from ..strategies.momentum_strategy import MomentumStrategy
                return MomentumStrategy()
            elif self.strategy_name == 'ichimoku':
                from ..strategies.ichimoku_strategy import IchimokuStrategy
                return IchimokuStrategy()
            elif self.strategy_name == 'arbitrage':
                from ..strategies.arbitrage_strategy import ArbitrageStrategy
                return ArbitrageStrategy()
            else:
                # Fallback to base strategy
                return BaseStrategy()
        except ImportError as e:
            self.logger.error(f"Error importing strategy {self.strategy_name}: {e}")
            return BaseStrategy()
    
    def _setup_logging(self):
        """Setup logging for the bot."""
        self.logger = logging.getLogger(f"TradingBot_{self.strategy_name}")
        self.logger.setLevel(getattr(logging, self.config.get('log_level', 'INFO')))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Create file handler
        log_file = Path("logs") / f"trading_bot_{self.strategy_name}.log"
        log_file.parent.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    async def start(self):
        """Start the trading bot."""
        if self.running:
            self.logger.warning("Bot is already running")
            return
        
        self.running = True
        self.start_time = datetime.now()
        self.last_update = datetime.now()
        
        self.logger.info(f"Starting trading bot with strategy: {self.strategy_name}")
        
        # Initialize exchange connection
        try:
            await self.exchange_adapter.connect()
            self.logger.info("Exchange connection established")
        except Exception as e:
            self.logger.error(f"Failed to connect to exchange: {e}")
            self.running = False
            return
        
        # Initialize components
        await self._initialize_components()
        
        # Start main trading loop
        await self._run_trading_loop()
    
    async def stop(self):
        """Stop the trading bot."""
        if not self.running:
            self.logger.warning("Bot is not running")
            return
        
        self.logger.info("Stopping trading bot")
        self.running = False
        
        # Close all positions if configured to do so
        if self.config.get('close_positions_on_stop', False):
            await self._close_all_positions()
        
        # Disconnect from exchange
        try:
            await self.exchange_adapter.disconnect()
            self.logger.info("Exchange connection closed")
        except Exception as e:
            self.logger.error(f"Error disconnecting from exchange: {e}")
        
        # Log final performance
        self._log_final_performance()
    
    async def _initialize_components(self):
        """Initialize bot components."""
        try:
            # Initialize signal generator
            await self.signal_generator.initialize(self.strategy, self.exchange_adapter)
            
            # Initialize order executor
            await self.order_executor.initialize()
            
            # Initialize position monitor
            await self.position_monitor.initialize()
            
            # Initialize ML predictor if enabled
            if self.ml_predictor:
                await self.ml_predictor.initialize()
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            raise
    
    async def _run_trading_loop(self):
        """Main trading loop."""
        self.logger.info("Starting main trading loop")
        
        while self.running:
            try:
                # Update bot state
                await self._update_bot_state()
                
                # Generate signals
                signals = await self._generate_signals()
                
                # Process signals
                if signals:
                    await self._process_signals(signals)
                
                # Monitor positions
                await self._monitor_positions()
                
                # Update performance metrics
                self._update_performance_metrics()
                
                # Log periodic status
                if datetime.now().minute % 5 == 0:  # Every 5 minutes
                    self._log_status()
                
                # Wait for next iteration
                await asyncio.sleep(self.config.get('refresh_interval', 5))
                
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(10)  # Wait longer on error
    
    async def _update_bot_state(self):
        """Update bot state information."""
        self.last_update = datetime.now()
        
        # Update active positions
        self.active_positions = await self.position_monitor.get_active_positions()
        
        # Update recent trades
        self.recent_trades = await self.order_executor.get_recent_trades(limit=50)
        
        # Update equity curve
        current_equity = await self._calculate_current_equity()
        self.equity_curve.append((datetime.now(), current_equity))
        
        # Keep only recent equity data (last 1000 points)
        if len(self.equity_curve) > 1000:
            self.equity_curve = self.equity_curve[-1000:]
    
    async def _generate_signals(self) -> List[StrategySignal]:
        """Generate trading signals."""
        try:
            # Get market data for all configured symbols
            market_data = {}
            for symbol in self.config.get('symbols', ['BTC/USDT']):
                for timeframe in self.config.get('timeframes', ['1h']):
                    data = await self.exchange_adapter.get_ohlcv(
                        symbol=symbol,
                        timeframe=timeframe,
                        limit=100
                    )
                    market_data[f"{symbol}_{timeframe}"] = data
            
            # Generate signals using the strategy
            signals = await self.signal_generator.generate_signals(
                strategy=self.strategy,
                market_data=market_data
            )
            
            # Apply ML filter if enabled
            if self.ml_predictor and signals:
                signals = await self.ml_predictor.filter_signals(signals, market_data)
            
            # Log signals
            for signal in signals:
                self.logger.info(f"Generated signal: {signal.symbol} {signal.side.value} at {signal.price}")
                self.event_logger.log_signal(signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return []
    
    async def _process_signals(self, signals: List[StrategySignal]):
        """Process generated signals."""
        for signal in signals:
            try:
                # Check risk limits
                if not await self.risk_manager.check_signal_risk(signal, self.active_positions):
                    self.logger.warning(f"Signal rejected by risk manager: {signal.symbol}")
                    continue
                
                # Calculate position size
                position_size = await self.position_sizer.calculate_position_size(
                    signal=signal,
                    current_positions=self.active_positions,
                    account_balance=await self.exchange_adapter.get_balance()
                )
                
                if position_size <= 0:
                    self.logger.warning(f"Position size too small: {signal.symbol}")
                    continue
                
                # Execute order
                order_result = await self.order_executor.execute_order(
                    signal=signal,
                    position_size=position_size
                )
                
                if order_result:
                    self.logger.info(f"Order executed: {signal.symbol} {signal.side.value} {position_size}")
                    self.event_logger.log_order_execution(order_result, signal)
                else:
                    self.logger.error(f"Failed to execute order: {signal.symbol}")
                
            except Exception as e:
                self.logger.error(f"Error processing signal {signal.symbol}: {e}")
    
    async def _monitor_positions(self):
        """Monitor active positions."""
        try:
            for position in self.active_positions:
                # Check stop loss and take profit
                await self.position_monitor.check_exit_conditions(position)
                
                # Update position PnL
                await self.position_monitor.update_position_pnl(position)
                
                # Check for position modifications
                if await self.position_monitor.should_modify_position(position):
                    await self.position_monitor.modify_position(position)
            
        except Exception as e:
            self.logger.error(f"Error monitoring positions: {e}")
    
    async def _calculate_current_equity(self) -> float:
        """Calculate current account equity."""
        try:
            balance = await self.exchange_adapter.get_balance()
            total_equity = balance.get('total', 0.0)
            
            # Add unrealized PnL from positions
            for position in self.active_positions:
                if position.current_price:
                    unrealized_pnl = (position.current_price - position.entry_price) * position.quantity
                    total_equity += unrealized_pnl
            
            return total_equity
            
        except Exception as e:
            self.logger.error(f"Error calculating equity: {e}")
            return 0.0
    
    def _update_performance_metrics(self):
        """Update performance metrics."""
        try:
            # Update trade counts
            self.performance_metrics['total_trades'] = len(self.recent_trades)
            
            winning_trades = len([t for t in self.recent_trades if t.pnl and t.pnl > 0])
            losing_trades = len([t for t in self.recent_trades if t.pnl and t.pnl < 0])
            
            self.performance_metrics['winning_trades'] = winning_trades
            self.performance_metrics['losing_trades'] = losing_trades
            
            # Update PnL
            total_pnl = sum([t.pnl for t in self.recent_trades if t.pnl])
            self.performance_metrics['total_pnl'] = total_pnl
            
            # Update equity and drawdown
            if self.equity_curve:
                current_equity = self.equity_curve[-1][1]
                self.performance_metrics['current_equity'] = current_equity
                
                # Update peak equity
                if current_equity > self.performance_metrics['peak_equity']:
                    self.performance_metrics['peak_equity'] = current_equity
                
                # Calculate current drawdown
                peak = self.performance_metrics['peak_equity']
                if peak > 0:
                    current_drawdown = (peak - current_equity) / peak
                    self.performance_metrics['current_drawdown'] = current_drawdown
                    
                    # Update max drawdown
                    if current_drawdown > self.performance_metrics['max_drawdown']:
                        self.performance_metrics['max_drawdown'] = current_drawdown
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    def _log_status(self):
        """Log periodic status."""
        self.logger.info(f"Bot Status - Positions: {len(self.active_positions)}, "
                        f"Trades: {self.performance_metrics['total_trades']}, "
                        f"PnL: ${self.performance_metrics['total_pnl']:.2f}, "
                        f"Drawdown: {self.performance_metrics['current_drawdown']:.2%}")
    
    def _log_final_performance(self):
        """Log final performance metrics."""
        duration = datetime.now() - self.start_time if self.start_time else timedelta(0)
        
        self.logger.info("=== FINAL PERFORMANCE ===")
        self.logger.info(f"Duration: {duration}")
        self.logger.info(f"Total Trades: {self.performance_metrics['total_trades']}")
        self.logger.info(f"Winning Trades: {self.performance_metrics['winning_trades']}")
        self.logger.info(f"Losing Trades: {self.performance_metrics['losing_trades']}")
        self.logger.info(f"Total PnL: ${self.performance_metrics['total_pnl']:.2f}")
        self.logger.info(f"Max Drawdown: {self.performance_metrics['max_drawdown']:.2%}")
        self.logger.info(f"Final Equity: ${self.performance_metrics['current_equity']:.2f}")
    
    async def _close_all_positions(self):
        """Close all active positions."""
        self.logger.info("Closing all active positions")
        
        for position in self.active_positions:
            try:
                await self.position_monitor.close_position(position)
                self.logger.info(f"Closed position: {position.symbol}")
            except Exception as e:
                self.logger.error(f"Error closing position {position.symbol}: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current bot status."""
        return {
            'running': self.running,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'strategy': self.strategy_name,
            'active_positions': len(self.active_positions),
            'recent_trades': len(self.recent_trades),
            'performance': self.performance_metrics.copy()
        }
    
    def get_open_positions(self) -> List[Position]:
        """Get current open positions."""
        return self.active_positions.copy()
    
    def get_recent_trades(self) -> List[Trade]:
        """Get recent trades."""
        return self.recent_trades.copy()
    
    def get_equity_curve(self) -> List[tuple]:
        """Get equity curve data."""
        return self.equity_curve.copy()
    
    def get_signals_history(self) -> List[StrategySignal]:
        """Get signals history."""
        return self.signals_history.copy()
    
    def update(self):
        """Update method for synchronous calls (used by dashboard)."""
        # This method is called by the dashboard to update bot state
        # In a real implementation, this would trigger the async update
        pass


class BotManager:
    """Manages multiple trading bot instances."""
    
    def __init__(self):
        self.bots = {}
        self.running = False
    
    async def create_bot(self, 
                        bot_id: str,
                        exchange_adapter: ExchangeInterface,
                        strategy_name: str,
                        config: Dict[str, Any]) -> TradingBot:
        """Create a new trading bot instance."""
        bot = TradingBot(exchange_adapter, strategy_name, config)
        self.bots[bot_id] = bot
        return bot
    
    async def start_bot(self, bot_id: str):
        """Start a specific bot."""
        if bot_id in self.bots:
            await self.bots[bot_id].start()
    
    async def stop_bot(self, bot_id: str):
        """Stop a specific bot."""
        if bot_id in self.bots:
            await self.bots[bot_id].stop()
    
    async def stop_all_bots(self):
        """Stop all running bots."""
        for bot in self.bots.values():
            if bot.running:
                await bot.stop()
    
    def get_bot_status(self, bot_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific bot."""
        if bot_id in self.bots:
            return self.bots[bot_id].get_status()
        return None
    
    def get_all_bots_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all bots."""
        return {bot_id: bot.get_status() for bot_id, bot in self.bots.items()}
    
    def remove_bot(self, bot_id: str):
        """Remove a bot instance."""
        if bot_id in self.bots:
            del self.bots[bot_id]
