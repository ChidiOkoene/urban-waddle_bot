<!-- df02c9a7-c802-46c9-be8a-fe69e61a779d 474090e9-3f2a-4a70-8a21-a2bb30863f57 -->
# Cross-Platform Automated Trading Bot with Advanced Strategies

## Architecture Overview

Modular architecture with complete separation of concerns:

- **Exchange Adapters**: Abstract interface with MT5 and CCXT implementations
- **Strategy Engine**: Multiple proven strategies with backtesting framework
- **Risk Management**: Position sizing, stop-loss, take-profit, trailing stops
- **ML Module**: Feature engineering and prediction pipeline
- **Dashboard**: Real-time Streamlit interface with live updates
- **Data Layer**: SQLite for persistence with async support

## Implementation Phases

### Phase 1: Core Infrastructure & Exchange Adapters

**Files to create:**

- `config/config.yaml` - Configuration template with all parameters
- `src/core/exchange_interface.py` - Abstract base class for all exchanges
- `src/adapters/mt5_adapter.py` - MetaTrader 5 implementation
- `src/adapters/ccxt_adapter.py` - Universal crypto exchange adapter (Bitget, Binance, etc.)
- `src/core/data_models.py` - Pydantic models for OHLCV, Order, Position, Balance
- `src/database/db_manager.py` - SQLite async database manager
- `requirements.txt` - All dependencies with pinned versions

**Key Features:**

- Exchange interface with methods: `get_ohlcv()`, `place_order()`, `get_balance()`, `get_open_positions()`, `cancel_order()`
- Normalized data models across all exchanges
- Connection pooling and error handling with retries
- Rate limiting per exchange
- Database schema for trades, positions, strategy performance

### Phase 2: Technical Indicators & Analysis

**Files to create:**

- `src/indicators/technical_indicators.py` - All indicator calculations
- `src/indicators/pattern_recognition.py` - Chart patterns and candlestick patterns
- `tests/test_indicators.py` - Comprehensive indicator tests

**Indicators to implement:**

- Trend: SMA, EMA, MACD, ADX, Parabolic SAR, Ichimoku Cloud
- Momentum: RSI, Stochastic, CCI, Williams %R, ROC
- Volatility: Bollinger Bands, ATR, Keltner Channels, Standard Deviation
- Volume: OBV, Volume Profile, VWAP, MFI
- Pattern Recognition: Support/Resistance, Trend Lines, Head & Shoulders, Double Top/Bottom

### Phase 3: Strategy Implementation

**Files to create:**

- `src/strategies/base_strategy.py` - Abstract strategy class with backtesting interface
- `src/strategies/rsi_macd_strategy.py` - RSI + MACD combination
- `src/strategies/bollinger_mean_reversion.py` - Bollinger Bands mean reversion
- `src/strategies/ema_crossover_strategy.py` - Multiple EMA crossover with trend filter
- `src/strategies/grid_bot_strategy.py` - Dynamic grid with trailing grid support
- `src/strategies/dca_strategy.py` - Dollar Cost Averaging with smart entry
- `src/strategies/breakout_strategy.py` - Support/resistance breakout with volume confirmation
- `src/strategies/momentum_strategy.py` - Multi-timeframe momentum
- `src/strategies/ichimoku_strategy.py` - Ichimoku Cloud strategy
- `src/strategies/arbitrage_strategy.py` - Cross-exchange arbitrage detection
- `src/strategies/strategy_optimizer.py` - Parameter optimization with genetic algorithms

**Strategy Details:**

1. **RSI + MACD Strategy**: Buy when RSI < 30 and MACD crosses above signal, sell when RSI > 70 and MACD crosses below
2. **Bollinger Mean Reversion**: Buy at lower band, sell at upper band with RSI confirmation
3. **EMA Crossover**: Fast/medium/slow EMA with ADX trend strength filter
4. **Grid Bot**: Dynamic grid levels with profit taking at each level, adjustable spacing
5. **DCA Strategy**: Accumulate on dips with increasing position size
6. **Breakout Strategy**: Volume-confirmed breakouts of support/resistance with ATR-based stops
7. **Momentum Strategy**: Combine 1h, 4h, daily timeframes for aligned momentum signals
8. **Ichimoku**: Full Ichimoku system with cloud, TK cross, and lagging span
9. **Arbitrage**: Real-time price comparison across exchanges with fee calculation

### Phase 4: Risk Management System

**Files to create:**

- `src/risk/position_sizer.py` - Dynamic position sizing based on account risk
- `src/risk/risk_manager.py` - Overall risk management coordinator
- `src/risk/stop_loss_manager.py` - Multiple stop-loss types (fixed, trailing, ATR-based)
- `src/risk/portfolio_manager.py` - Multi-position and correlation management

**Features:**

- Risk per trade as % of account (default 1-2%)
- Position sizing using Kelly Criterion or fixed fractional
- Trailing stop-loss with multiple algorithms (percentage, ATR, parabolic SAR)
- Take-profit levels (fixed, scaled, trailing)
- Maximum drawdown protection
- Correlation-based position limits
- Maximum simultaneous positions per symbol and total

### Phase 5: Machine Learning Integration

**Files to create:**

- `src/ml/feature_engineering.py` - Create ML features from price/indicators
- `src/ml/model_trainer.py` - Train classification models for trade signals
- `src/ml/predictor.py` - Real-time prediction interface
- `src/ml/model_evaluator.py` - Backtesting and performance metrics
- `models/` - Directory for saved models

**ML Features:**

- Feature engineering: normalized indicators, price patterns, volatility metrics, volume analysis
- Models: Random Forest, XGBoost, LightGBM for signal classification
- Target: Predict if trade will be profitable in next N candles
- Training pipeline with walk-forward validation
- Feature importance analysis
- Model retraining scheduler

### Phase 6: Backtesting Framework

**Files to create:**

- `src/backtesting/backtest_engine.py` - Historical simulation engine
- `src/backtesting/performance_metrics.py` - Sharpe, Sortino, max drawdown, win rate, etc.
- `src/backtesting/strategy_comparison.py` - Compare multiple strategies
- `src/backtesting/optimization_engine.py` - Grid search and genetic algorithm optimization
- `src/backtesting/report_generator.py` - HTML/PDF reports with charts

**Capabilities:**

- Realistic order execution with slippage and fees
- Multi-timeframe backtesting
- Walk-forward analysis
- Monte Carlo simulation
- Strategy parameter optimization
- Performance metrics: Sharpe ratio, Sortino ratio, Calmar ratio, max drawdown, win rate, profit factor
- Equity curve visualization
- Trade-by-trade analysis

### Phase 7: Real-Time Dashboard

**Files to create:**

- `src/dashboard/app.py` - Main Streamlit application
- `src/dashboard/components/charts.py` - Plotly candlestick and indicator charts
- `src/dashboard/components/positions_table.py` - Live positions display
- `src/dashboard/components/performance_metrics.py` - PnL, win rate, Sharpe ratio
- `src/dashboard/components/strategy_controls.py` - Strategy selection and parameter adjustment
- `src/dashboard/components/backtest_viewer.py` - Backtesting results visualization
- `src/dashboard/state_manager.py` - Shared state between bot and dashboard

**Dashboard Features:**

- Multi-page layout: Live Trading, Backtesting, Strategy Comparison, Settings
- Real-time candlestick charts with indicators overlay
- Live positions table with entry price, current price, PnL, unrealized profit
- Active orders table (pending, grid orders)
- Performance metrics dashboard (daily/weekly/monthly PnL)
- Strategy selector with parameter controls
- Risk management controls
- Multi-symbol monitoring
- Alert configuration
- Trade history with filtering
- Auto-refresh with configurable interval

### Phase 8: Bot Execution Engine

**Files to create:**

- `src/bot/trading_bot.py` - Main bot orchestrator
- `src/bot/signal_generator.py` - Combine strategy signals with ML filter
- `src/bot/order_executor.py` - Execute orders with retry logic
- `src/bot/position_monitor.py` - Monitor and manage open positions
- `src/bot/event_logger.py` - Comprehensive logging system
- `main.py` - Entry point for bot execution

**Execution Loop:**

1. Fetch latest OHLCV data from exchange
2. Calculate all indicators
3. Generate strategy signals
4. Apply ML filter (optional)
5. Check risk management constraints
6. Calculate position size
7. Execute orders
8. Monitor open positions for stop-loss/take-profit
9. Update dashboard state
10. Log all events to database
11. Sleep until next interval

**Features:**

- Multi-threaded execution (data fetching, signal generation, order execution)
- Graceful shutdown with position preservation
- Paper trading mode (simulated execution)
- Dry-run mode (signal generation only)
- Emergency stop mechanism
- Automatic reconnection on connection loss

### Phase 9: Notifications & Alerts

**Files to create:**

- `src/notifications/telegram_notifier.py` - Telegram bot integration
- `src/notifications/discord_notifier.py` - Discord webhook integration
- `src/notifications/email_notifier.py` - Email alerts via SMTP
- `src/notifications/alert_manager.py` - Centralized alert management

**Alert Types:**

- Trade executed (entry/exit)
- Stop-loss triggered
- Take-profit hit
- Daily PnL summary
- Strategy performance alerts
- Error notifications
- Connection issues

### Phase 10: Testing & Documentation

**Files to create:**

- `tests/test_adapters.py` - Exchange adapter tests
- `tests/test_strategies.py` - Strategy logic tests
- `tests/test_risk_management.py` - Risk management tests
- `tests/test_backtesting.py` - Backtesting engine tests
- `tests/mocks/mock_exchange.py` - Mock exchange for testing
- `docs/SETUP.md` - Setup instructions
- `docs/STRATEGIES.md` - Strategy documentation
- `docs/API.md` - API reference
- `docs/CONFIGURATION.md` - Configuration guide
- `README.md` - Project overview and quick start

### Phase 11: Deployment & Production Features

**Files to create:**

- `docker/Dockerfile` - Container for bot
- `docker/docker-compose.yml` - Multi-container setup
- `.env.example` - Environment variables template
- `scripts/setup.sh` - Automated setup script
- `scripts/start_bot.sh` - Bot startup script
- `scripts/backup_db.sh` - Database backup script

**Production Features:**

- Docker containerization
- Environment-based configuration
- Automated database backups
- Log rotation
- Health check endpoints
- Monitoring integration (Prometheus/Grafana ready)
- Graceful restart without losing positions

## Technology Stack

- **Python 3.11+**: Core language
- **MetaTrader5**: MT5 Python package
- **CCXT**: Universal crypto exchange library
- **Pandas/NumPy**: Data manipulation
- **TA-Lib**: Technical indicators (with pure Python fallback)
- **Scikit-learn/XGBoost**: Machine learning
- **Streamlit**: Dashboard framework
- **Plotly**: Interactive charts
- **SQLite**: Database with aiosqlite for async
- **Pydantic**: Data validation
- **PyYAML**: Configuration management
- **Python-telegram-bot**: Telegram integration
- **Pytest**: Testing framework
- **Docker**: Containerization

## Configuration Structure

```yaml
exchange:
  type: "ccxt"  # or "mt5"
  name: "bitget"  # exchange name
  credentials:
    api_key: "${API_KEY}"
    secret: "${API_SECRET}"
    password: "${API_PASSWORD}"
  
trading:
  symbols: ["BTC/USDT", "ETH/USDT"]
  timeframe: "1h"
  mode: "paper"  # paper, live, dry-run
  
strategy:
  name: "rsi_macd"  # or grid, ema_crossover, etc.
  parameters:
    rsi_period: 14
    rsi_oversold: 30
    rsi_overbought: 70
    macd_fast: 12
    macd_slow: 26
    macd_signal: 9
  use_ml_filter: true
  ml_threshold: 0.6

risk:
  max_risk_per_trade: 0.02  # 2% of account
  max_positions: 3
  position_sizing: "kelly"  # or "fixed_fractional"
  stop_loss_type: "atr"  # or "percentage", "trailing"
  stop_loss_atr_multiplier: 2.0
  take_profit_ratio: 2.0  # risk:reward
  trailing_stop_activation: 0.015  # 1.5%
  trailing_stop_distance: 0.01  # 1%

backtesting:
  start_date: "2023-01-01"
  end_date: "2024-12-31"
  initial_capital: 10000
  commission: 0.001  # 0.1%
  slippage: 0.0005  # 0.05%

dashboard:
  port: 8501
  update_interval: 5  # seconds
  enable_notifications: true

notifications:
  telegram:
    enabled: true
    bot_token: "${TELEGRAM_BOT_TOKEN}"
    chat_id: "${TELEGRAM_CHAT_ID}"
  discord:
    enabled: false
    webhook_url: "${DISCORD_WEBHOOK}"
```

## Project Structure

```
urban-waddle_bot/
├── config/
│   ├── config.yaml
│   └── config_schema.json
├── src/
│   ├── core/
│   │   ├── exchange_interface.py
│   │   └── data_models.py
│   ├── adapters/
│   │   ├── mt5_adapter.py
│   │   └── ccxt_adapter.py
│   ├── indicators/
│   │   ├── technical_indicators.py
│   │   └── pattern_recognition.py
│   ├── strategies/
│   │   ├── base_strategy.py
│   │   ├── rsi_macd_strategy.py
│   │   ├── bollinger_mean_reversion.py
│   │   ├── ema_crossover_strategy.py
│   │   ├── grid_bot_strategy.py
│   │   ├── dca_strategy.py
│   │   ├── breakout_strategy.py
│   │   ├── momentum_strategy.py
│   │   ├── ichimoku_strategy.py
│   │   ├── arbitrage_strategy.py
│   │   └── strategy_optimizer.py
│   ├── risk/
│   │   ├── position_sizer.py
│   │   ├── risk_manager.py
│   │   ├── stop_loss_manager.py
│   │   └── portfolio_manager.py
│   ├── ml/
│   │   ├── feature_engineering.py
│   │   ├── model_trainer.py
│   │   ├── predictor.py
│   │   └── model_evaluator.py
│   ├── backtesting/
│   │   ├── backtest_engine.py
│   │   ├── performance_metrics.py
│   │   ├── strategy_comparison.py
│   │   ├── optimization_engine.py
│   │   └── report_generator.py
│   ├── dashboard/
│   │   ├── app.py
│   │   ├── components/
│   │   │   ├── charts.py
│   │   │   ├── positions_table.py
│   │   │   ├── performance_metrics.py
│   │   │   ├── strategy_controls.py
│   │   │   └── backtest_viewer.py
│   │   └── state_manager.py
│   ├── bot/
│   │   ├── trading_bot.py
│   │   ├── signal_generator.py
│   │   ├── order_executor.py
│   │   ├── position_monitor.py
│   │   └── event_logger.py
│   ├── notifications/
│   │   ├── telegram_notifier.py
│   │   ├── discord_notifier.py
│   │   ├── email_notifier.py
│   │   └── alert_manager.py
│   └── database/
│       └── db_manager.py
├── tests/
│   ├── test_adapters.py
│   ├── test_strategies.py
│   ├── test_indicators.py
│   ├── test_risk_management.py
│   ├── test_backtesting.py
│   └── mocks/
│       └── mock_exchange.py
├── docs/
│   ├── SETUP.md
│   ├── STRATEGIES.md
│   ├── API.md
│   └── CONFIGURATION.md
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── scripts/
│   ├── setup.sh
│   ├── start_bot.sh
│   └── backup_db.sh
├── models/
│   └── .gitkeep
├── logs/
│   └── .gitkeep
├── data/
│   └── .gitkeep
├── main.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── LICENSE
```

## Execution Order

Each phase will be completed fully before moving to the next, ensuring each module is production-ready with tests and documentation.

### To-dos

- [ ] Build core infrastructure: config system, exchange interface, MT5/CCXT adapters, data models, database manager
- [ ] Implement all technical indicators: trend, momentum, volatility, volume indicators, and pattern recognition
- [ ] Create 9 trading strategies: RSI+MACD, Bollinger, EMA crossover, Grid, DCA, Breakout, Momentum, Ichimoku, Arbitrage, plus optimizer
- [ ] Build risk management system: position sizer, risk manager, stop-loss manager, portfolio manager
- [ ] Implement ML pipeline: feature engineering, model training, predictor, evaluator
- [ ] Create backtesting framework: engine, metrics, comparison, optimization, report generator
- [ ] Build Streamlit dashboard: charts, positions table, metrics, controls, backtest viewer, state manager
- [ ] Implement bot execution engine: trading bot, signal generator, order executor, position monitor, event logger
- [ ] Add notification system: Telegram, Discord, email notifiers, alert manager
- [ ] Complete testing suite: adapter tests, strategy tests, risk tests, backtesting tests, mock exchange, documentation
- [ ] Setup deployment: Docker containers, scripts, environment config, monitoring, production features