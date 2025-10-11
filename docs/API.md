# Urban Waddle Bot - API Documentation

## Overview

This document provides comprehensive API documentation for the Urban Waddle Bot, including all classes, methods, and their usage examples.

## Core Components

### Exchange Interface

#### `ExchangeInterface`
**File**: `src/core/exchange_interface.py`

Base abstract class for all exchange adapters.

```python
from src.core.exchange_interface import ExchangeInterface

class CustomExchange(ExchangeInterface):
    async def connect(self) -> bool:
        """Connect to the exchange."""
        pass
    
    async def disconnect(self) -> bool:
        """Disconnect from the exchange."""
        pass
    
    async def get_balance(self, asset: str) -> Balance:
        """Get balance for a specific asset."""
        pass
```

**Abstract Methods**:
- `connect()`: Establish connection to exchange
- `disconnect()`: Close connection to exchange
- `get_balance(asset)`: Get balance for specific asset
- `get_ohlcv(symbol, timeframe, limit)`: Get OHLCV data
- `place_order(symbol, side, order_type, quantity, price)`: Place trading order
- `cancel_order(order_id, symbol)`: Cancel existing order
- `get_order_status(order_id, symbol)`: Get order status
- `get_open_positions()`: Get open positions
- `get_trade_history(symbol, limit)`: Get trade history

### Data Models

#### `OHLCV`
**File**: `src/core/data_models.py`

Represents OHLCV (Open, High, Low, Close, Volume) data.

```python
from src.core.data_models import OHLCV

ohlcv = OHLCV(
    symbol='BTC/USDT',
    timestamp=datetime.now(),
    open=100.0,
    high=105.0,
    low=95.0,
    close=102.0,
    volume=1000.0
)
```

**Attributes**:
- `symbol` (str): Trading pair symbol
- `timestamp` (datetime): Data timestamp
- `open` (float): Opening price
- `high` (float): Highest price
- `low` (float): Lowest price
- `close` (float): Closing price
- `volume` (float): Trading volume

#### `Order`
**File**: `src/core/data_models.py`

Represents a trading order.

```python
from src.core.data_models import Order, OrderSide, OrderType

order = Order(
    symbol='BTC/USDT',
    side=OrderSide.BUY,
    type=OrderType.MARKET,
    quantity=0.1,
    price=100.0,
    timestamp=datetime.now()
)
```

**Attributes**:
- `symbol` (str): Trading pair symbol
- `side` (OrderSide): BUY or SELL
- `type` (OrderType): MARKET or LIMIT
- `quantity` (float): Order quantity
- `price` (float): Order price
- `timestamp` (datetime): Order timestamp

#### `Position`
**File**: `src/core/data_models.py`

Represents an open trading position.

```python
from src.core.data_models import Position, OrderSide

position = Position(
    symbol='BTC/USDT',
    side=OrderSide.BUY,
    size=0.1,
    entry_price=100.0,
    current_price=105.0,
    unrealized_pnl=0.5
)
```

**Attributes**:
- `symbol` (str): Trading pair symbol
- `side` (OrderSide): BUY or SELL
- `size` (float): Position size
- `entry_price` (float): Entry price
- `current_price` (float): Current market price
- `unrealized_pnl` (float): Unrealized profit/loss

#### `Trade`
**File**: `src/core/data_models.py`

Represents an executed trade.

```python
from src.core.data_models import Trade, OrderSide

trade = Trade(
    symbol='BTC/USDT',
    side=OrderSide.BUY,
    quantity=0.1,
    price=100.0,
    timestamp=datetime.now(),
    commission=0.1
)
```

**Attributes**:
- `symbol` (str): Trading pair symbol
- `side` (OrderSide): BUY or SELL
- `quantity` (float): Trade quantity
- `price` (float): Trade price
- `timestamp` (datetime): Trade timestamp
- `commission` (float): Commission paid

## Exchange Adapters

### MetaTrader 5 Adapter

#### `MT5Adapter`
**File**: `src/adapters/mt5_adapter.py`

Adapter for MetaTrader 5 platform.

```python
from src.adapters.mt5_adapter import MT5Adapter

# Initialize adapter
mt5_adapter = MT5Adapter()

# Connect to MT5
await mt5_adapter.connect()

# Get balance
balance = await mt5_adapter.get_balance('USD')

# Place order
order_id = await mt5_adapter.place_order(
    symbol='EURUSD',
    side=OrderSide.BUY,
    order_type=OrderType.MARKET,
    quantity=0.1
)
```

**Methods**:
- `connect()`: Connect to MT5 terminal
- `disconnect()`: Disconnect from MT5
- `get_balance(asset)`: Get account balance
- `get_ohlcv(symbol, timeframe, limit)`: Get OHLCV data
- `place_order(symbol, side, order_type, quantity, price)`: Place order
- `cancel_order(order_id, symbol)`: Cancel order
- `get_order_status(order_id, symbol)`: Get order status
- `get_open_positions()`: Get open positions
- `get_trade_history(symbol, limit)`: Get trade history

### CCXT Adapter

#### `CCXTAdapter`
**File**: `src/adapters/ccxt_adapter.py`

Adapter for CCXT-supported exchanges.

```python
from src.adapters.ccxt_adapter import CCXTAdapter

# Initialize adapter
ccxt_adapter = CCXTAdapter(exchange_name='binance')

# Connect to exchange
await ccxt_adapter.connect()

# Get balance
balance = await ccxt_adapter.get_balance('USDT')

# Place order
order_id = await ccxt_adapter.place_order(
    symbol='BTC/USDT',
    side=OrderSide.BUY,
    order_type=OrderType.MARKET,
    quantity=0.1
)
```

**Methods**:
- `connect()`: Connect to exchange
- `disconnect()`: Disconnect from exchange
- `get_balance(asset)`: Get account balance
- `get_ohlcv(symbol, timeframe, limit)`: Get OHLCV data
- `place_order(symbol, side, order_type, quantity, price)`: Place order
- `cancel_order(order_id, symbol)`: Cancel order
- `get_order_status(order_id, symbol)`: Get order status
- `get_open_positions()`: Get open positions
- `get_trade_history(symbol, limit)`: Get trade history

## Technical Indicators

### `TechnicalIndicators`
**File**: `src/indicators/technical_indicators.py`

Collection of technical indicators.

```python
from src.indicators.technical_indicators import TechnicalIndicators

# Initialize indicators
indicators = TechnicalIndicators()

# Calculate RSI
rsi = indicators.calculate_rsi(prices, period=14)

# Calculate MACD
macd_line, signal_line, histogram = indicators.calculate_macd(prices, fast=12, slow=26, signal=9)

# Calculate Bollinger Bands
upper, middle, lower = indicators.calculate_bollinger_bands(prices, period=20, std_dev=2.0)
```

**Available Indicators**:
- `calculate_rsi(prices, period)`: Relative Strength Index
- `calculate_macd(prices, fast, slow, signal)`: MACD
- `calculate_bollinger_bands(prices, period, std_dev)`: Bollinger Bands
- `calculate_sma(prices, period)`: Simple Moving Average
- `calculate_ema(prices, period)`: Exponential Moving Average
- `calculate_atr(high, low, close, period)`: Average True Range
- `calculate_adx(high, low, close, period)`: Average Directional Index
- `calculate_parabolic_sar(high, low, close)`: Parabolic SAR
- `calculate_ichimoku(high, low, close)`: Ichimoku Cloud
- `calculate_stochastic(high, low, close, period)`: Stochastic Oscillator
- `calculate_cci(high, low, close, period)`: Commodity Channel Index
- `calculate_williams_r(high, low, close, period)`: Williams %R
- `calculate_roc(prices, period)`: Rate of Change
- `calculate_obv(close, volume)`: On-Balance Volume
- `calculate_volume_profile(high, low, close, volume)`: Volume Profile
- `calculate_vwap(high, low, close, volume)`: Volume Weighted Average Price
- `calculate_mfi(high, low, close, volume, period)`: Money Flow Index

## Trading Strategies

### Base Strategy

#### `BaseStrategy`
**File**: `src/strategies/base_strategy.py`

Abstract base class for all trading strategies.

```python
from src.strategies.base_strategy import BaseStrategy

class CustomStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.name = "CustomStrategy"
        self.parameters = {
            'param1': 10,
            'param2': 20
        }
    
    async def generate_signals(self, symbol, timeframe, data, indicators):
        """Generate trading signals."""
        signals = []
        # Strategy logic here
        return signals
```

**Methods**:
- `set_parameters(params)`: Set strategy parameters
- `get_parameter(name, default)`: Get parameter value
- `enable()`: Enable strategy
- `disable()`: Disable strategy
- `generate_signals(symbol, timeframe, data, indicators)`: Generate signals

### RSI + MACD Strategy

#### `RSIMACDStrategy`
**File**: `src/strategies/rsi_macd_strategy.py`

Combines RSI and MACD indicators.

```python
from src.strategies.rsi_macd_strategy import RSIMACDStrategy

# Initialize strategy
strategy = RSIMACDStrategy()

# Set parameters
strategy.set_parameters({
    'rsi_period': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9
})

# Generate signals
signals = await strategy.generate_signals(
    symbol='BTC/USDT',
    timeframe='1h',
    data=ohlcv_data,
    indicators=indicators
)
```

**Parameters**:
- `rsi_period` (int): RSI calculation period
- `rsi_overbought` (float): RSI overbought threshold
- `rsi_oversold` (float): RSI oversold threshold
- `macd_fast` (int): MACD fast EMA period
- `macd_slow` (int): MACD slow EMA period
- `macd_signal` (int): MACD signal line period

### Bollinger Bands Mean Reversion

#### `BollingerMeanReversionStrategy`
**File**: `src/strategies/bollinger_mean_reversion.py`

Mean reversion strategy using Bollinger Bands.

```python
from src.strategies.bollinger_mean_reversion import BollingerMeanReversionStrategy

# Initialize strategy
strategy = BollingerMeanReversionStrategy()

# Set parameters
strategy.set_parameters({
    'period': 20,
    'std_dev': 2.0,
    'entry_threshold': 1.0
})

# Generate signals
signals = await strategy.generate_signals(
    symbol='BTC/USDT',
    timeframe='1h',
    data=ohlcv_data,
    indicators=indicators
)
```

**Parameters**:
- `period` (int): Moving average period
- `std_dev` (float): Standard deviation multiplier
- `entry_threshold` (float): Entry threshold from bands

### Grid Bot Strategy

#### `GridBotStrategy`
**File**: `src/strategies/grid_bot_strategy.py`

Grid trading strategy.

```python
from src.strategies.grid_bot_strategy import GridBotStrategy

# Initialize strategy
strategy = GridBotStrategy()

# Set parameters
strategy.set_parameters({
    'grid_levels': 10,
    'grid_spacing': 0.01,
    'max_position_size': 0.1
})

# Generate signals
signals = await strategy.generate_signals(
    symbol='BTC/USDT',
    timeframe='1h',
    data=ohlcv_data,
    indicators=indicators
)
```

**Parameters**:
- `grid_levels` (int): Number of grid levels
- `grid_spacing` (float): Price spacing between levels
- `max_position_size` (float): Maximum position size per level

## Risk Management

### Position Sizer

#### `PositionSizer`
**File**: `src/risk/position_sizer.py`

Calculates optimal position sizes.

```python
from src.risk.position_sizer import PositionSizer

# Initialize position sizer
sizer = PositionSizer()

# Calculate position size
position_size = sizer.calculate_position_size(
    balance=balance,
    risk_percentage=0.01,
    entry_price=100.0,
    stop_loss_price=95.0
)
```

**Methods**:
- `calculate_position_size(balance, risk_percentage, entry_price, stop_loss_price)`: Calculate position size
- `calculate_kelly_position_size(balance, win_rate, avg_win_loss_ratio)`: Kelly criterion sizing
- `calculate_volatility_position_size(balance, atr, risk_percentage)`: Volatility-based sizing

### Risk Manager

#### `RiskManager`
**File**: `src/risk/risk_manager.py`

Manages overall portfolio risk.

```python
from src.risk.risk_manager import RiskManager

# Initialize risk manager
risk_manager = RiskManager()

# Set risk limits
risk_manager.set_risk_limits({
    'max_drawdown': 0.05,
    'max_positions': 5,
    'max_correlation': 0.7,
    'max_risk_per_trade': 0.02,
    'max_portfolio_risk': 0.10
})

# Check risk limits
can_trade = risk_manager.check_risk_per_trade(0.01)
```

**Methods**:
- `set_risk_limits(limits)`: Set risk limits
- `check_max_drawdown(current_drawdown)`: Check maximum drawdown
- `check_max_positions(current_positions)`: Check maximum positions
- `check_correlation(correlation)`: Check correlation limit
- `check_risk_per_trade(risk_per_trade)`: Check risk per trade
- `check_portfolio_risk(portfolio_risk)`: Check portfolio risk
- `activate_emergency_stop(reason)`: Activate emergency stop
- `deactivate_emergency_stop()`: Deactivate emergency stop

### Stop Loss Manager

#### `StopLossManager`
**File**: `src/risk/stop_loss_manager.py`

Manages stop-loss orders.

```python
from src.risk.stop_loss_manager import StopLossManager

# Initialize stop-loss manager
stop_loss_manager = StopLossManager()

# Calculate fixed stop-loss
stop_price = stop_loss_manager.calculate_fixed_stop_loss(
    position=position,
    stop_loss_percentage=0.05
)

# Calculate trailing stop-loss
stop_price = stop_loss_manager.calculate_trailing_stop_loss(
    position=position,
    trailing_percentage=0.03,
    highest_price=110.0
)
```

**Methods**:
- `calculate_fixed_stop_loss(position, stop_loss_percentage)`: Fixed stop-loss
- `calculate_trailing_stop_loss(position, trailing_percentage, highest_price)`: Trailing stop-loss
- `calculate_atr_stop_loss(position, atr, atr_multiplier)`: ATR-based stop-loss
- `check_time_based_stop_loss(position, max_hold_hours)`: Time-based stop-loss
- `set_stop_loss(symbol, stop_price)`: Set stop-loss
- `update_stop_loss(symbol, stop_price)`: Update stop-loss
- `remove_stop_loss(symbol)`: Remove stop-loss
- `get_stop_loss(symbol)`: Get stop-loss

## Backtesting Framework

### Backtest Engine

#### `BacktestEngine`
**File**: `src/backtesting/backtest_engine.py`

Core backtesting engine.

```python
from src.backtesting.backtest_engine import BacktestEngine

# Initialize backtest engine
backtest_engine = BacktestEngine()

# Set parameters
backtest_engine.set_parameters({
    'initial_capital': 10000.0,
    'commission_rate': 0.001,
    'slippage_rate': 0.0005
})

# Run backtest
results = await backtest_engine.run_backtest(
    strategy=strategy,
    data=ohlcv_data,
    symbol='BTC/USDT',
    timeframe='1h'
)
```

**Methods**:
- `set_parameters(params)`: Set backtest parameters
- `run_backtest(strategy, data, symbol, timeframe)`: Run backtest
- `execute_trade(order, current_price)`: Execute trade
- `update_position(position, current_price)`: Update position
- `calculate_commission(amount, rate)`: Calculate commission
- `calculate_slippage(price, rate)`: Calculate slippage
- `calculate_equity_curve(trades)`: Calculate equity curve

### Performance Metrics

#### `PerformanceMetrics`
**File**: `src/backtesting/performance_metrics.py`

Calculates performance metrics.

```python
from src.backtesting.performance_metrics import PerformanceMetrics

# Initialize performance metrics
metrics = PerformanceMetrics()

# Calculate comprehensive metrics
comprehensive_metrics = metrics.calculate_comprehensive_metrics(
    equity_curve=equity_curve,
    trades=trades
)
```

**Methods**:
- `calculate_total_return(equity_curve)`: Total return
- `calculate_annualized_return(equity_curve)`: Annualized return
- `calculate_volatility(equity_curve)`: Volatility
- `calculate_sharpe_ratio(equity_curve)`: Sharpe ratio
- `calculate_sortino_ratio(equity_curve)`: Sortino ratio
- `calculate_max_drawdown(equity_curve)`: Maximum drawdown
- `calculate_calmar_ratio(equity_curve)`: Calmar ratio
- `calculate_win_rate(trades)`: Win rate
- `calculate_profit_factor(trades)`: Profit factor
- `calculate_average_trade_duration(trades)`: Average trade duration
- `calculate_comprehensive_metrics(equity_curve, trades)`: All metrics

### Strategy Comparison

#### `StrategyComparison`
**File**: `src/backtesting/strategy_comparison.py`

Compares multiple strategies.

```python
from src.backtesting.strategy_comparison import StrategyComparison

# Initialize strategy comparison
comparison = StrategyComparison()

# Compare strategies
comparison_results = comparison.compare_strategies(strategy_results)

# Rank strategies
ranking = comparison.rank_strategies(strategy_results)
```

**Methods**:
- `compare_strategies(results)`: Compare strategies
- `rank_strategies(results)`: Rank strategies
- `calculate_strategy_correlation(equity_curves)`: Calculate correlation
- `calculate_risk_adjusted_returns(results)`: Risk-adjusted returns

### Optimization Engine

#### `OptimizationEngine`
**File**: `src/backtesting/optimization_engine.py`

Optimizes strategy parameters.

```python
from src.backtesting.optimization_engine import OptimizationEngine

# Initialize optimization engine
optimizer = OptimizationEngine()

# Define parameter space
parameter_space = {
    'rsi_period': [10, 14, 20],
    'rsi_overbought': [65, 70, 75],
    'rsi_oversold': [25, 30, 35]
}

# Run grid search optimization
best_params = optimizer.grid_search_optimization(
    strategy=strategy,
    parameter_space=parameter_space,
    backtest_function=backtest_function,
    max_iterations=100
)
```

**Methods**:
- `define_parameter_space(strategy)`: Define parameter space
- `grid_search_optimization(strategy, parameter_space, backtest_function, max_iterations)`: Grid search
- `genetic_algorithm_optimization(strategy, parameter_space, backtest_function, population_size, generations)`: Genetic algorithm
- `bayesian_optimization(strategy, parameter_space, backtest_function, n_iterations)`: Bayesian optimization
- `validate_parameters(strategy, parameters)`: Validate parameters

## ML Pipeline

### Feature Engineering

#### `FeatureEngineering`
**File**: `src/ml/feature_engineering.py`

Creates features for ML models.

```python
from src.ml.feature_engineering import FeatureEngineering

# Initialize feature engineering
feature_engineering = FeatureEngineering()

# Create features
features = feature_engineering.create_features(
    ohlcv_data=ohlcv_data,
    indicators=indicators,
    lookback_period=20
)
```

**Methods**:
- `create_features(ohlcv_data, indicators, lookback_period)`: Create features
- `create_price_features(ohlcv_data)`: Price-based features
- `create_technical_features(indicators)`: Technical indicator features
- `create_volume_features(ohlcv_data)`: Volume-based features
- `create_time_features(timestamps)`: Time-based features
- `create_lag_features(data, lags)`: Lag features
- `create_rolling_features(data, windows)`: Rolling window features

### Model Trainer

#### `ModelTrainer`
**File**: `src/ml/model_trainer.py`

Trains ML models.

```python
from src.ml.model_trainer import ModelTrainer

# Initialize model trainer
trainer = ModelTrainer()

# Train model
model = trainer.train_model(
    X_train=X_train,
    y_train=y_train,
    model_type='random_forest',
    parameters={'n_estimators': 100, 'max_depth': 10}
)
```

**Methods**:
- `train_model(X_train, y_train, model_type, parameters)`: Train model
- `train_random_forest(X_train, y_train, parameters)`: Random Forest
- `train_xgboost(X_train, y_train, parameters)`: XGBoost
- `train_lightgbm(X_train, y_train, parameters)`: LightGBM
- `train_neural_network(X_train, y_train, parameters)`: Neural Network
- `cross_validate_model(model, X, y, cv_folds)`: Cross-validation
- `save_model(model, filepath)`: Save model
- `load_model(filepath)`: Load model

### Predictor

#### `Predictor`
**File**: `src/ml/predictor.py`

Makes predictions using trained models.

```python
from src.ml.predictor import Predictor

# Initialize predictor
predictor = Predictor()

# Load model
model = predictor.load_model('models/trading_model.pkl')

# Make prediction
prediction = predictor.predict(
    model=model,
    features=features
)
```

**Methods**:
- `load_model(filepath)`: Load model
- `predict(model, features)`: Make prediction
- `predict_proba(model, features)`: Prediction probabilities
- `predict_batch(model, features_batch)`: Batch predictions
- `update_model(model, new_data)`: Update model
- `evaluate_model(model, X_test, y_test)`: Evaluate model

## Dashboard

### Streamlit App

#### `DashboardApp`
**File**: `src/dashboard/app.py`

Main Streamlit dashboard application.

```python
import streamlit as st
from src.dashboard.app import DashboardApp

# Initialize dashboard
app = DashboardApp()

# Run dashboard
app.run()
```

**Components**:
- `Charts`: Candlestick charts, performance charts
- `PositionsTable`: Open positions display
- `PerformanceMetrics`: Performance metrics display
- `StrategyControls`: Strategy configuration
- `BacktestViewer`: Backtest results viewer

### State Manager

#### `StateManager`
**File**: `src/dashboard/state_manager.py`

Manages dashboard state.

```python
from src.dashboard.state_manager import StateManager

# Initialize state manager
state_manager = StateManager()

# Get current state
state = state_manager.get_state()

# Update state
state_manager.update_state(new_state)
```

**Methods**:
- `get_state()`: Get current state
- `update_state(new_state)`: Update state
- `reset_state()`: Reset state
- `save_state(filepath)`: Save state
- `load_state(filepath)`: Load state

## Bot Execution Engine

### Trading Bot

#### `TradingBot`
**File**: `src/bot/trading_bot.py`

Main trading bot execution engine.

```python
from src.bot.trading_bot import TradingBot

# Initialize trading bot
bot = TradingBot()

# Configure bot
bot.configure(
    exchange=exchange,
    strategies=[strategy1, strategy2],
    risk_manager=risk_manager
)

# Start bot
await bot.start()

# Stop bot
await bot.stop()
```

**Methods**:
- `configure(exchange, strategies, risk_manager)`: Configure bot
- `start()`: Start bot
- `stop()`: Stop bot
- `pause()`: Pause bot
- `resume()`: Resume bot
- `get_status()`: Get bot status
- `get_performance()`: Get performance metrics

### Signal Generator

#### `SignalGenerator`
**File**: `src/bot/signal_generator.py`

Generates trading signals.

```python
from src.bot.signal_generator import SignalGenerator

# Initialize signal generator
signal_generator = SignalGenerator()

# Generate signals
signals = await signal_generator.generate_signals(
    symbol='BTC/USDT',
    timeframe='1h',
    strategies=[strategy1, strategy2]
)
```

**Methods**:
- `generate_signals(symbol, timeframe, strategies)`: Generate signals
- `combine_signals(signals)`: Combine multiple signals
- `filter_signals(signals, filters)`: Filter signals
- `validate_signals(signals)`: Validate signals

### Order Executor

#### `OrderExecutor`
**File**: `src/bot/order_executor.py`

Executes trading orders.

```python
from src.bot.order_executor import OrderExecutor

# Initialize order executor
executor = OrderExecutor(exchange=exchange)

# Execute order
order_id = await executor.execute_order(
    symbol='BTC/USDT',
    side=OrderSide.BUY,
    order_type=OrderType.MARKET,
    quantity=0.1
)
```

**Methods**:
- `execute_order(symbol, side, order_type, quantity, price)`: Execute order
- `cancel_order(order_id, symbol)`: Cancel order
- `get_order_status(order_id, symbol)`: Get order status
- `execute_batch_orders(orders)`: Execute multiple orders

### Position Monitor

#### `PositionMonitor`
**File**: `src/bot/position_monitor.py`

Monitors open positions.

```python
from src.bot.position_monitor import PositionMonitor

# Initialize position monitor
monitor = PositionMonitor(exchange=exchange)

# Monitor positions
positions = await monitor.monitor_positions()

# Update position
updated_position = monitor.update_position(position, current_price)
```

**Methods**:
- `monitor_positions()`: Monitor all positions
- `update_position(position, current_price)`: Update position
- `check_stop_loss(position)`: Check stop-loss
- `check_take_profit(position)`: Check take-profit
- `close_position(position)`: Close position

## Notifications

### Alert Manager

#### `AlertManager`
**File**: `src/notifications/alert_manager.py`

Manages and dispatches alerts.

```python
from src.notifications.alert_manager import AlertManager

# Initialize alert manager
alert_manager = AlertManager()

# Configure notifiers
alert_manager.configure_notifiers({
    'telegram': telegram_notifier,
    'discord': discord_notifier,
    'email': email_notifier
})

# Send alert
await alert_manager.send_alert(
    message='Trade executed',
    level='info',
    notifiers=['telegram', 'discord']
)
```

**Methods**:
- `configure_notifiers(notifiers)`: Configure notifiers
- `send_alert(message, level, notifiers)`: Send alert
- `send_trade_alert(trade)`: Send trade alert
- `send_performance_alert(performance)`: Send performance alert
- `send_error_alert(error)`: Send error alert

### Telegram Notifier

#### `TelegramNotifier`
**File**: `src/notifications/telegram_notifier.py`

Sends notifications via Telegram.

```python
from src.notifications.telegram_notifier import TelegramNotifier

# Initialize Telegram notifier
telegram_notifier = TelegramNotifier(
    bot_token='your_bot_token',
    chat_id='your_chat_id'
)

# Send message
await telegram_notifier.send_message('Hello from trading bot!')
```

**Methods**:
- `send_message(message)`: Send text message
- `send_trade_notification(trade)`: Send trade notification
- `send_performance_notification(performance)`: Send performance notification
- `send_error_notification(error)`: Send error notification

### Discord Notifier

#### `DiscordNotifier`
**File**: `src/notifications/discord_notifier.py`

Sends notifications via Discord.

```python
from src.notifications.discord_notifier import DiscordNotifier

# Initialize Discord notifier
discord_notifier = DiscordNotifier(
    webhook_url='your_webhook_url'
)

# Send message
await discord_notifier.send_message('Hello from trading bot!')
```

**Methods**:
- `send_message(message)`: Send text message
- `send_trade_notification(trade)`: Send trade notification
- `send_performance_notification(performance)`: Send performance notification
- `send_error_notification(error)`: Send error notification

### Email Notifier

#### `EmailNotifier`
**File**: `src/notifications/email_notifier.py`

Sends notifications via email.

```python
from src.notifications.email_notifier import EmailNotifier

# Initialize email notifier
email_notifier = EmailNotifier(
    smtp_server='smtp.gmail.com',
    smtp_port=587,
    username='your_email@gmail.com',
    password='your_password'
)

# Send email
await email_notifier.send_email(
    to='recipient@example.com',
    subject='Trading Bot Alert',
    body='Hello from trading bot!'
)
```

**Methods**:
- `send_email(to, subject, body)`: Send email
- `send_trade_notification(trade, recipient)`: Send trade notification
- `send_performance_notification(performance, recipient)`: Send performance notification
- `send_error_notification(error, recipient)`: Send error notification

## Database

### Database Manager

#### `DatabaseManager`
**File**: `src/database/db_manager.py`

Manages database operations.

```python
from src.database.db_manager import DatabaseManager

# Initialize database manager
db_manager = DatabaseManager('sqlite:///trading_bot.db')

# Initialize database
await db_manager.initialize()

# Save trade
await db_manager.save_trade(trade)

# Get trades
trades = await db_manager.get_trades(symbol='BTC/USDT', limit=100)
```

**Methods**:
- `initialize()`: Initialize database
- `save_trade(trade)`: Save trade
- `get_trades(symbol, limit)`: Get trades
- `save_position(position)`: Save position
- `get_positions(symbol)`: Get positions
- `save_performance_metrics(metrics)`: Save performance metrics
- `get_performance_metrics(strategy, start_date, end_date)`: Get performance metrics
- `save_alert(alert)`: Save alert
- `get_alerts(limit)`: Get alerts

## Usage Examples

### Basic Trading Bot Setup

```python
import asyncio
from src.adapters.ccxt_adapter import CCXTAdapter
from src.strategies.rsi_macd_strategy import RSIMACDStrategy
from src.risk.risk_manager import RiskManager
from src.bot.trading_bot import TradingBot

async def main():
    # Initialize components
    exchange = CCXTAdapter(exchange_name='binance')
    strategy = RSIMACDStrategy()
    risk_manager = RiskManager()
    bot = TradingBot()
    
    # Configure bot
    bot.configure(
        exchange=exchange,
        strategies=[strategy],
        risk_manager=risk_manager
    )
    
    # Start bot
    await bot.start()
    
    # Keep running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await bot.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

### Backtesting Example

```python
import asyncio
from src.backtesting.backtest_engine import BacktestEngine
from src.strategies.bollinger_mean_reversion import BollingerMeanReversionStrategy
import pandas as pd

async def run_backtest():
    # Initialize components
    backtest_engine = BacktestEngine()
    strategy = BollingerMeanReversionStrategy()
    
    # Load historical data
    data = pd.read_csv('data/btc_usdt_1h.csv')
    
    # Run backtest
    results = await backtest_engine.run_backtest(
        strategy=strategy,
        data=data,
        symbol='BTC/USDT',
        timeframe='1h'
    )
    
    # Print results
    print(f"Total Return: {results['performance']['total_return']:.2%}")
    print(f"Sharpe Ratio: {results['performance']['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['performance']['max_drawdown']:.2%}")

if __name__ == "__main__":
    asyncio.run(run_backtest())
```

### Strategy Optimization Example

```python
import asyncio
from src.backtesting.optimization_engine import OptimizationEngine
from src.strategies.rsi_macd_strategy import RSIMACDStrategy

async def optimize_strategy():
    # Initialize components
    optimizer = OptimizationEngine()
    strategy = RSIMACDStrategy()
    
    # Define parameter space
    parameter_space = {
        'rsi_period': [10, 14, 20],
        'rsi_overbought': [65, 70, 75],
        'rsi_oversold': [25, 30, 35],
        'macd_fast': [10, 12, 15],
        'macd_slow': [20, 26, 30],
        'macd_signal': [7, 9, 12]
    }
    
    # Mock backtest function
    def backtest_function(params):
        strategy.set_parameters(params)
        # Run backtest and return performance metric
        return {'sharpe_ratio': 1.2}  # Mock result
    
    # Run optimization
    best_params = optimizer.grid_search_optimization(
        strategy=strategy,
        parameter_space=parameter_space,
        backtest_function=backtest_function,
        max_iterations=100
    )
    
    print(f"Best parameters: {best_params}")

if __name__ == "__main__":
    asyncio.run(optimize_strategy())
```

## Error Handling

### Common Exceptions

#### `ExchangeConnectionError`
Raised when unable to connect to exchange.

```python
try:
    await exchange.connect()
except ExchangeConnectionError as e:
    print(f"Failed to connect to exchange: {e}")
```

#### `InsufficientFundsError`
Raised when insufficient funds for trade.

```python
try:
    await exchange.place_order(symbol, side, order_type, quantity)
except InsufficientFundsError as e:
    print(f"Insufficient funds: {e}")
```

#### `InvalidOrderError`
Raised when order parameters are invalid.

```python
try:
    await exchange.place_order(symbol, side, order_type, quantity, price)
except InvalidOrderError as e:
    print(f"Invalid order: {e}")
```

#### `RiskLimitExceededError`
Raised when risk limits are exceeded.

```python
try:
    risk_manager.check_risk_per_trade(0.05)
except RiskLimitExceededError as e:
    print(f"Risk limit exceeded: {e}")
```

## Best Practices

### Code Organization

1. **Modular Design**: Keep components separate and focused
2. **Error Handling**: Always handle exceptions gracefully
3. **Logging**: Use comprehensive logging for debugging
4. **Testing**: Write tests for all components
5. **Documentation**: Document all public methods and classes

### Performance Optimization

1. **Async Operations**: Use async/await for I/O operations
2. **Caching**: Cache frequently accessed data
3. **Batch Operations**: Group operations when possible
4. **Memory Management**: Clean up resources properly
5. **Database Optimization**: Use appropriate indexes and queries

### Security Considerations

1. **API Keys**: Never hardcode API keys
2. **Input Validation**: Validate all inputs
3. **Rate Limiting**: Respect exchange rate limits
4. **Error Messages**: Don't expose sensitive information
5. **Access Control**: Implement proper access controls

## Conclusion

This API documentation provides comprehensive coverage of all components in the Urban Waddle Bot. Each component is designed to be modular, testable, and extensible. Follow the examples and best practices to build robust trading applications.

For additional support and examples, refer to the test files in the `tests/` directory and the configuration files in the `config/` directory.
