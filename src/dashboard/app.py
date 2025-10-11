"""
Streamlit Dashboard for Trading Bot

This is the main dashboard application that provides real-time visualization
and control of the trading bot.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import asyncio
import threading
import time
from typing import Dict, List, Any, Optional
import json
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.data_models import OHLCV, Trade, Position, StrategySignal, OrderSide
from src.database.db_manager import DatabaseManager
from src.bot.trading_bot import TradingBot
from src.backtesting.backtest_engine import BacktestEngine
from src.backtesting.performance_metrics import PerformanceMetrics
from src.backtesting.report_generator import ReportGenerator
from src.strategies.base_strategy import BaseStrategy
from src.adapters.ccxt_adapter import CCXTAdapter
from src.adapters.mt5_adapter import MT5Adapter

# Import dashboard components
from .components.charts import ChartComponents
from .components.positions_table import PositionsTable
from .components.performance_metrics import PerformanceMetricsComponent
from .components.strategy_controls import StrategyControls
from .components.backtest_viewer import BacktestViewer
from .state_manager import StateManager


class TradingDashboard:
    """Main trading dashboard application."""
    
    def __init__(self):
        self.state_manager = StateManager()
        self.charts = ChartComponents()
        self.positions_table = PositionsTable()
        self.performance_metrics = PerformanceMetricsComponent()
        self.strategy_controls = StrategyControls()
        self.backtest_viewer = BacktestViewer()
        
        # Initialize database
        self.db_manager = DatabaseManager()
        
        # Bot instance (will be initialized when needed)
        self.trading_bot = None
        self.bot_thread = None
        self.bot_running = False
        
        # Data cache
        self.market_data_cache = {}
        self.positions_cache = []
        self.trades_cache = []
        self.equity_curve_cache = []
        
        # Configuration
        self.config = self._load_config()
        
        # Setup page config
        st.set_page_config(
            page_title="Trading Bot Dashboard",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            config_path = Path("config/config.yaml")
            if config_path.exists():
                import yaml
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                return self._get_default_config()
        except Exception as e:
            st.error(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'exchanges': {
                'binance': {
                    'api_key': '',
                    'secret': '',
                    'sandbox': True
                }
            },
            'strategies': {
                'default': 'rsi_macd',
                'enabled': ['rsi_macd', 'bollinger', 'ema_crossover']
            },
            'risk_management': {
                'max_position_size': 0.1,
                'max_drawdown': 0.2,
                'stop_loss': 0.02
            },
            'dashboard': {
                'refresh_interval': 5,
                'max_candles': 1000
            }
        }
    
    def run(self):
        """Run the dashboard application."""
        # Sidebar
        self._render_sidebar()
        
        # Main content
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Live Trading", 
            "📈 Backtesting", 
            "⚙️ Strategies", 
            "📋 Positions", 
            "📊 Performance"
        ])
        
        with tab1:
            self._render_live_trading_tab()
        
        with tab2:
            self._render_backtesting_tab()
        
        with tab3:
            self._render_strategies_tab()
        
        with tab4:
            self._render_positions_tab()
        
        with tab5:
            self._render_performance_tab()
    
    def _render_sidebar(self):
        """Render the sidebar with controls."""
        st.sidebar.title("🤖 Trading Bot Control")
        
        # Bot status
        status_color = "🟢" if self.bot_running else "🔴"
        st.sidebar.markdown(f"**Status:** {status_color} {'Running' if self.bot_running else 'Stopped'}")
        
        # Bot controls
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("▶️ Start Bot", disabled=self.bot_running):
                self._start_bot()
        
        with col2:
            if st.button("⏹️ Stop Bot", disabled=not self.bot_running):
                self._stop_bot()
        
        # Exchange selection
        st.sidebar.subheader("Exchange Settings")
        exchange = st.sidebar.selectbox(
            "Select Exchange",
            ["binance", "bitget", "mt5"],
            index=0
        )
        
        # Strategy selection
        st.sidebar.subheader("Strategy Settings")
        strategy = st.sidebar.selectbox(
            "Active Strategy",
            ["rsi_macd", "bollinger", "ema_crossover", "grid_bot", "dca"],
            index=0
        )
        
        # Risk settings
        st.sidebar.subheader("Risk Management")
        max_position_size = st.sidebar.slider(
            "Max Position Size (%)",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1
        )
        
        stop_loss = st.sidebar.slider(
            "Stop Loss (%)",
            min_value=0.1,
            max_value=10.0,
            value=2.0,
            step=0.1
        )
        
        # Update config
        self.config['exchanges']['selected'] = exchange
        self.config['strategies']['active'] = strategy
        self.config['risk_management']['max_position_size'] = max_position_size / 100
        self.config['risk_management']['stop_loss'] = stop_loss / 100
        
        # Refresh controls
        st.sidebar.subheader("Data Refresh")
        if st.sidebar.button("🔄 Refresh Data"):
            self._refresh_data()
        
        auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
        if auto_refresh:
            refresh_interval = st.sidebar.slider(
                "Refresh Interval (seconds)",
                min_value=1,
                max_value=60,
                value=5
            )
            self.config['dashboard']['refresh_interval'] = refresh_interval
    
    def _render_live_trading_tab(self):
        """Render the live trading tab."""
        st.header("📊 Live Trading Dashboard")
        
        # Market data section
        st.subheader("Market Data")
        
        # Symbol selection
        col1, col2, col3 = st.columns(3)
        with col1:
            symbol = st.selectbox("Symbol", ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT"])
        with col2:
            timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"])
        with col3:
            if st.button("📊 Load Chart"):
                self._load_market_data(symbol, timeframe)
        
        # Display chart
        if symbol in self.market_data_cache:
            self.charts.render_candlestick_chart(
                self.market_data_cache[symbol],
                symbol,
                timeframe
            )
        
        # Current positions
        st.subheader("Current Positions")
        if self.positions_cache:
            self.positions_table.render_positions_table(self.positions_cache)
        else:
            st.info("No open positions")
        
        # Recent trades
        st.subheader("Recent Trades")
        if self.trades_cache:
            self._render_recent_trades()
        else:
            st.info("No recent trades")
        
        # Equity curve
        st.subheader("Equity Curve")
        if self.equity_curve_cache:
            self.charts.render_equity_curve(self.equity_curve_cache)
        else:
            st.info("No equity data available")
    
    def _render_backtesting_tab(self):
        """Render the backtesting tab."""
        st.header("📈 Strategy Backtesting")
        
        # Backtest controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            strategy = st.selectbox(
                "Strategy",
                ["rsi_macd", "bollinger", "ema_crossover", "grid_bot", "dca"],
                key="backtest_strategy"
            )
        
        with col2:
            symbol = st.selectbox(
                "Symbol",
                ["BTC/USDT", "ETH/USDT", "BNB/USDT"],
                key="backtest_symbol"
            )
        
        with col3:
            timeframe = st.selectbox(
                "Timeframe",
                ["1h", "4h", "1d"],
                key="backtest_timeframe"
            )
        
        # Date range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
        with col2:
            end_date = st.date_input("End Date", value=datetime.now())
        
        # Run backtest
        if st.button("🚀 Run Backtest"):
            with st.spinner("Running backtest..."):
                self._run_backtest(strategy, symbol, timeframe, start_date, end_date)
        
        # Display backtest results
        if 'backtest_results' in st.session_state:
            self.backtest_viewer.render_backtest_results(st.session_state['backtest_results'])
    
    def _render_strategies_tab(self):
        """Render the strategies tab."""
        st.header("⚙️ Strategy Configuration")
        
        # Strategy selection
        strategy = st.selectbox(
            "Select Strategy",
            ["rsi_macd", "bollinger", "ema_crossover", "grid_bot", "dca"],
            key="strategy_config"
        )
        
        # Strategy parameters
        self.strategy_controls.render_strategy_parameters(strategy)
        
        # Strategy performance
        st.subheader("Strategy Performance")
        self._render_strategy_performance(strategy)
    
    def _render_positions_tab(self):
        """Render the positions tab."""
        st.header("📋 Position Management")
        
        # Open positions
        st.subheader("Open Positions")
        if self.positions_cache:
            self.positions_table.render_positions_table(self.positions_cache)
        else:
            st.info("No open positions")
        
        # Position history
        st.subheader("Position History")
        self._render_position_history()
    
    def _render_performance_tab(self):
        """Render the performance tab."""
        st.header("📊 Performance Analytics")
        
        # Performance metrics
        self.performance_metrics.render_performance_metrics()
        
        # Performance charts
        st.subheader("Performance Charts")
        if self.equity_curve_cache:
            self.charts.render_equity_curve(self.equity_curve_cache)
            self.charts.render_drawdown_chart(self.equity_curve_cache)
        
        # Risk metrics
        st.subheader("Risk Metrics")
        self._render_risk_metrics()
    
    def _start_bot(self):
        """Start the trading bot."""
        try:
            # Initialize exchange adapter
            exchange = self.config['exchanges']['selected']
            if exchange == 'mt5':
                adapter = MT5Adapter()
            else:
                adapter = CCXTAdapter(exchange)
            
            # Initialize trading bot
            self.trading_bot = TradingBot(
                exchange_adapter=adapter,
                strategy_name=self.config['strategies']['active'],
                config=self.config
            )
            
            # Start bot in separate thread
            self.bot_thread = threading.Thread(target=self._run_bot_loop)
            self.bot_thread.daemon = True
            self.bot_thread.start()
            
            self.bot_running = True
            st.success("Bot started successfully!")
            
        except Exception as e:
            st.error(f"Error starting bot: {e}")
    
    def _stop_bot(self):
        """Stop the trading bot."""
        self.bot_running = False
        if self.trading_bot:
            self.trading_bot.stop()
        st.success("Bot stopped!")
    
    def _run_bot_loop(self):
        """Run the bot main loop."""
        while self.bot_running and self.trading_bot:
            try:
                # Update bot state
                self.trading_bot.update()
                
                # Update caches
                self._update_caches()
                
                # Sleep for refresh interval
                time.sleep(self.config['dashboard']['refresh_interval'])
                
            except Exception as e:
                st.error(f"Bot error: {e}")
                break
    
    def _update_caches(self):
        """Update data caches."""
        try:
            if self.trading_bot:
                # Update positions
                self.positions_cache = self.trading_bot.get_open_positions()
                
                # Update trades
                self.trades_cache = self.trading_bot.get_recent_trades()
                
                # Update equity curve
                self.equity_curve_cache = self.trading_bot.get_equity_curve()
                
        except Exception as e:
            print(f"Error updating caches: {e}")
    
    def _load_market_data(self, symbol: str, timeframe: str):
        """Load market data for the given symbol and timeframe."""
        try:
            # This would normally fetch from exchange
            # For now, generate sample data
            data = self._generate_sample_data(symbol, timeframe)
            self.market_data_cache[symbol] = data
            
        except Exception as e:
            st.error(f"Error loading market data: {e}")
    
    def _generate_sample_data(self, symbol: str, timeframe: str) -> List[OHLCV]:
        """Generate sample market data."""
        # This is placeholder data - in real implementation, fetch from exchange
        data = []
        base_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 300
        
        for i in range(100):
            timestamp = datetime.now() - timedelta(minutes=5*i)
            price = base_price + np.random.normal(0, base_price * 0.01)
            
            data.append(OHLCV(
                symbol=symbol,
                timestamp=timestamp,
                open=price,
                high=price * (1 + abs(np.random.normal(0, 0.005))),
                low=price * (1 - abs(np.random.normal(0, 0.005))),
                close=price + np.random.normal(0, price * 0.002),
                volume=np.random.uniform(1000, 10000)
            ))
        
        return data
    
    def _run_backtest(self, strategy: str, symbol: str, timeframe: str, start_date, end_date):
        """Run backtest for the given parameters."""
        try:
            # Load historical data
            data = self._load_historical_data(symbol, timeframe, start_date, end_date)
            
            # Create strategy instance
            strategy_class = self._get_strategy_class(strategy)
            strategy_instance = strategy_class()
            
            # Run backtest
            engine = BacktestEngine()
            results = engine.run_backtest(strategy_instance, data)
            
            # Store results in session state
            st.session_state['backtest_results'] = results
            
            st.success("Backtest completed successfully!")
            
        except Exception as e:
            st.error(f"Error running backtest: {e}")
    
    def _load_historical_data(self, symbol: str, timeframe: str, start_date, end_date) -> List[OHLCV]:
        """Load historical data for backtesting."""
        # This would normally fetch from exchange
        # For now, generate sample data
        data = []
        base_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 300
        
        current_date = start_date
        while current_date <= end_date:
            price = base_price + np.random.normal(0, base_price * 0.01)
            
            data.append(OHLCV(
                symbol=symbol,
                timestamp=datetime.combine(current_date, datetime.min.time()),
                open=price,
                high=price * (1 + abs(np.random.normal(0, 0.005))),
                low=price * (1 - abs(np.random.normal(0, 0.005))),
                close=price + np.random.normal(0, price * 0.002),
                volume=np.random.uniform(1000, 10000)
            ))
            
            current_date += timedelta(days=1)
        
        return data
    
    def _get_strategy_class(self, strategy_name: str):
        """Get strategy class by name."""
        strategy_map = {
            'rsi_macd': 'RSIMACDStrategy',
            'bollinger': 'BollingerMeanReversionStrategy',
            'ema_crossover': 'EMACrossoverStrategy',
            'grid_bot': 'GridBotStrategy',
            'dca': 'DCAStrategy'
        }
        
        # This would normally import the actual strategy classes
        # For now, return a placeholder
        return BaseStrategy
    
    def _render_recent_trades(self):
        """Render recent trades table."""
        if not self.trades_cache:
            return
        
        df = pd.DataFrame([{
            'Time': trade.timestamp.strftime('%H:%M:%S'),
            'Symbol': trade.symbol,
            'Side': trade.side.value,
            'Quantity': trade.quantity,
            'Price': f"${trade.price:.2f}",
            'PnL': f"${trade.pnl:.2f}" if trade.pnl else "N/A"
        } for trade in self.trades_cache[-10:]])  # Last 10 trades
        
        st.dataframe(df, use_container_width=True)
    
    def _render_position_history(self):
        """Render position history."""
        # This would normally fetch from database
        st.info("Position history will be loaded from database")
    
    def _render_strategy_performance(self, strategy: str):
        """Render strategy performance metrics."""
        # This would normally calculate from historical data
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Return", "12.5%", "2.1%")
        with col2:
            st.metric("Sharpe Ratio", "1.85", "0.15")
        with col3:
            st.metric("Max Drawdown", "-8.2%", "1.3%")
        with col4:
            st.metric("Win Rate", "68%", "3%")
    
    def _render_risk_metrics(self):
        """Render risk metrics."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("VaR (95%)", "-2.1%")
        with col2:
            st.metric("CVaR (95%)", "-3.2%")
        with col3:
            st.metric("Volatility", "15.8%")
        with col4:
            st.metric("Beta", "0.95")
    
    def _refresh_data(self):
        """Refresh all data."""
        st.rerun()


def main():
    """Main function to run the dashboard."""
    dashboard = TradingDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
