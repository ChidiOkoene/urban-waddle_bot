#!/usr/bin/env python3
"""
Urban Waddle Bot - Main Entry Point

This is the main entry point for the Urban Waddle Trading Bot.
It initializes all components and starts the trading bot with production features.
"""

import asyncio
import argparse
import logging
import sys
import os
import signal
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.exchange_interface import ExchangeInterface
from src.adapters.mt5_adapter import MT5Adapter
from src.adapters.ccxt_adapter import CCXTAdapter
from src.database.db_manager import DatabaseManager
from src.bot.trading_bot import TradingBot
from src.monitoring.production_manager import ProductionManager
from src.notifications.alert_manager import AlertManager
import yaml


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/trading_bot.log'),
            logging.StreamHandler()
        ]
    )
    
    # Set specific logger levels
    logging.getLogger('src.exchange').setLevel(logging.INFO)
    logging.getLogger('src.strategies').setLevel(logging.INFO)
    logging.getLogger('src.risk').setLevel(logging.INFO)
    logging.getLogger('src.bot').setLevel(logging.INFO)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)


def load_environment_variables():
    """Load environment variables."""
    env_vars = {}
    
    # Exchange configuration
    env_vars['exchange_type'] = os.getenv('EXCHANGE_TYPE', 'ccxt')
    env_vars['exchange_name'] = os.getenv('EXCHANGE_NAME', 'binance')
    env_vars['api_key'] = os.getenv('API_KEY')
    env_vars['api_secret'] = os.getenv('API_SECRET')
    env_vars['api_passphrase'] = os.getenv('API_PASSPHRASE')
    
    # Trading configuration
    env_vars['bot_mode'] = os.getenv('BOT_MODE', 'paper')
    env_vars['default_symbol'] = os.getenv('DEFAULT_SYMBOL', 'BTC/USDT')
    env_vars['default_timeframe'] = os.getenv('DEFAULT_TIMEFRAME', '1h')
    env_vars['initial_capital'] = float(os.getenv('INITIAL_CAPITAL', '10000.0'))
    
    # Risk management
    env_vars['max_risk_per_trade'] = float(os.getenv('MAX_RISK_PER_TRADE', '0.02'))
    env_vars['max_positions'] = int(os.getenv('MAX_POSITIONS', '5'))
    env_vars['max_drawdown'] = float(os.getenv('MAX_DRAWDOWN', '0.05'))
    
    # Notifications
    env_vars['telegram_bot_token'] = os.getenv('TELEGRAM_BOT_TOKEN')
    env_vars['telegram_chat_id'] = os.getenv('TELEGRAM_CHAT_ID')
    env_vars['discord_webhook_url'] = os.getenv('DISCORD_WEBHOOK_URL')
    
    # Database
    env_vars['database_url'] = os.getenv('DATABASE_URL', 'sqlite:///data/trading_bot.db')
    
    # Logging
    env_vars['log_level'] = os.getenv('LOG_LEVEL', 'INFO')
    
    # Dashboard
    env_vars['dashboard_port'] = int(os.getenv('DASHBOARD_PORT', '8501'))
    
    return env_vars


class BotManager:
    """Main bot manager with production features."""
    
    def __init__(self, config: dict, env_vars: dict):
        self.config = config
        self.env_vars = env_vars
        self.logger = logging.getLogger(__name__)
        self.bot = None
        self.production_manager = None
        self.alert_manager = None
        self.running = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False
    
    async def initialize(self):
        """Initialize all components."""
        self.logger.info("Initializing Urban Waddle Trading Bot...")
        
        # Initialize database
        self.logger.info("Initializing database...")
        db_manager = DatabaseManager(self.env_vars['database_url'])
        await db_manager.initialize()
        
        # Initialize production manager
        self.production_manager = ProductionManager(db_manager)
        await self.production_manager.start_monitoring()
        
        # Initialize exchange adapter
        self.logger.info("Initializing exchange adapter...")
        exchange = await self._initialize_exchange()
        
        # Initialize alert manager
        self.logger.info("Initializing alert manager...")
        self.alert_manager = AlertManager()
        await self.alert_manager.configure_notifiers({
            'telegram': self.env_vars.get('telegram_bot_token'),
            'discord': self.env_vars.get('discord_webhook_url')
        })
        
        # Initialize trading bot
        self.logger.info("Initializing trading bot...")
        self.bot = TradingBot()
        await self.bot.configure(
            exchange=exchange,
            strategies=[],  # Will be loaded from config
            risk_manager=None,  # Will be initialized
            db_manager=db_manager,
            alert_manager=self.alert_manager
        )
        
        self.logger.info("Bot initialization completed successfully")
    
    async def _initialize_exchange(self):
        """Initialize exchange adapter."""
        exchange_type = self.env_vars['exchange_type']
        
        if exchange_type == 'mt5':
            exchange = MT5Adapter()
        elif exchange_type == 'ccxt':
            exchange_name = self.env_vars['exchange_name']
            exchange = CCXTAdapter(exchange_name=exchange_name)
        else:
            raise ValueError(f"Unsupported exchange type: {exchange_type}")
        
        # Connect to exchange
        await exchange.connect()
        return exchange
    
    async def start(self):
        """Start the bot."""
        if not self.bot:
            raise RuntimeError("Bot not initialized. Call initialize() first.")
        
        self.logger.info("Starting trading bot...")
        self.running = True
        
        try:
            await self.bot.start()
            
            # Send startup notification
            if self.alert_manager:
                await self.alert_manager.send_alert(
                    message="🚀 Urban Waddle Bot started successfully",
                    level="info"
                )
            
            # Main loop
            while self.running:
                try:
                    # Update production manager heartbeat
                    self.production_manager.bot_monitor.update_heartbeat()
                    
                    # Check health status
                    health_status = await self.production_manager.get_health_status()
                    if health_status['overall_status'] == 'critical':
                        self.logger.critical("Critical health issues detected")
                        if self.alert_manager:
                            await self.alert_manager.send_alert(
                                message="🚨 Critical health issues detected in trading bot",
                                level="critical"
                            )
                    
                    await asyncio.sleep(60)  # Check every minute
                    
                except Exception as e:
                    self.logger.error(f"Error in main loop: {e}")
                    self.production_manager.increment_error_count()
                    await asyncio.sleep(60)
        
        except Exception as e:
            self.logger.error(f"Fatal error in bot: {e}")
            if self.alert_manager:
                await self.alert_manager.send_alert(
                    message=f"💥 Fatal error in trading bot: {str(e)}",
                    level="critical"
                )
            raise
    
    async def stop(self):
        """Stop the bot gracefully."""
        self.logger.info("Stopping trading bot...")
        self.running = False
        
        if self.bot:
            await self.bot.stop()
        
        # Send shutdown notification
        if self.alert_manager:
            await self.alert_manager.send_alert(
                message="🛑 Urban Waddle Bot stopped",
                level="info"
            )
        
        self.logger.info("Bot stopped successfully")


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Urban Waddle Trading Bot')
    parser.add_argument('--mode', choices=['paper', 'live', 'backtest'], default='paper',
                       help='Trading mode')
    parser.add_argument('--strategy', default='rsi_macd',
                       help='Trading strategy to use')
    parser.add_argument('--config', default='config/config.yaml',
                       help='Configuration file path')
    parser.add_argument('--log-level', default='INFO',
                       help='Logging level')
    parser.add_argument('--dashboard-port', type=int, default=8501,
                       help='Dashboard port')
    
    args = parser.parse_args()
    
    # Load environment variables
    env_vars = load_environment_variables()
    
    # Override with command line arguments
    env_vars['bot_mode'] = args.mode
    env_vars['log_level'] = args.log_level
    env_vars['dashboard_port'] = args.dashboard_port
    
    # Setup logging
    setup_logging(env_vars['log_level'])
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 50)
    logger.info("Urban Waddle Trading Bot")
    logger.info("=" * 50)
    logger.info(f"Mode: {env_vars['bot_mode']}")
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Exchange: {env_vars['exchange_type']} ({env_vars['exchange_name']})")
    logger.info(f"Symbol: {env_vars['default_symbol']}")
    logger.info(f"Timeframe: {env_vars['default_timeframe']}")
    logger.info(f"Initial Capital: ${env_vars['initial_capital']:,.2f}")
    logger.info(f"Dashboard Port: {env_vars['dashboard_port']}")
    logger.info("=" * 50)
    
    # Load configuration
    config = load_config(args.config)
    
    # Create bot manager
    bot_manager = BotManager(config, env_vars)
    
    try:
        # Initialize bot
        await bot_manager.initialize()
        
        # Start bot
        await bot_manager.start()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        # Stop bot gracefully
        await bot_manager.stop()


if __name__ == "__main__":
    asyncio.run(main())