# Urban Waddle Bot - Configuration Guide

## Overview

This guide provides comprehensive information about configuring the Urban Waddle Bot, including all configuration options, environment variables, and best practices.

## Configuration Files

### Main Configuration File

**File**: `config/config.yaml`

The main configuration file contains all bot settings organized by category.

```yaml
# Exchange Configuration
exchange:
  type: ccxt  # or mt5
  name: binance  # or bitget, bybit, okx, etc.
  api_key: ${API_KEY}
  api_secret: ${API_SECRET}
  api_passphrase: ${API_PASSPHRASE}  # for some exchanges
  sandbox: false  # use sandbox/testnet
  rate_limit: 100  # requests per minute

# Trading Configuration
trading:
  default_symbol: BTC/USDT
  default_timeframe: 1h
  initial_capital: 10000.0
  commission_rate: 0.001
  slippage_rate: 0.0005
  
  # Order settings
  order_timeout: 30  # seconds
  max_retries: 3
  retry_delay: 1  # seconds

# Risk Management
risk_management:
  max_risk_per_trade: 0.02  # 2%
  max_positions: 5
  max_drawdown: 0.05  # 5%
  max_correlation: 0.7
  max_portfolio_risk: 0.10  # 10%
  
  # Position sizing
  position_sizing:
    method: fixed_percentage  # or kelly, volatility
    risk_percentage: 0.01  # 1%
    max_position_size: 0.1  # 10%
    min_position_size: 0.001  # 0.1%
  
  # Stop loss
  stop_loss:
    method: fixed_percentage  # or trailing, atr, time
    stop_loss_percentage: 0.05  # 5%
    trailing_percentage: 0.03  # 3%
    atr_multiplier: 2.0
    max_hold_hours: 24

# Strategies Configuration
strategies:
  rsi_macd:
    enabled: true
    weight: 0.4
    parameters:
      rsi_period: 14
      rsi_overbought: 70
      rsi_oversold: 30
      macd_fast: 12
      macd_slow: 26
      macd_signal: 9
  
  bollinger_mean_reversion:
    enabled: true
    weight: 0.3
    parameters:
      period: 20
      std_dev: 2.0
      entry_threshold: 1.0
  
  ema_crossover:
    enabled: false
    weight: 0.2
    parameters:
      fast_period: 12
      slow_period: 26
      signal_threshold: 0.005
  
  grid_bot:
    enabled: false
    weight: 0.1
    parameters:
      grid_levels: 10
      grid_spacing: 0.01
      max_position_size: 0.1
  
  dca:
    enabled: false
    weight: 0.0
    parameters:
      interval_hours: 24
      position_size: 0.05
      max_positions: 10
  
  breakout:
    enabled: false
    weight: 0.0
    parameters:
      consolidation_periods: 20
      breakout_threshold: 0.02
      volume_confirmation: true
  
  momentum:
    enabled: false
    weight: 0.0
    parameters:
      momentum_period: 14
      acceleration_period: 5
      momentum_threshold: 0.01
  
  ichimoku:
    enabled: false
    weight: 0.0
    parameters:
      tenkan_period: 9
      kijun_period: 26
      senkou_span_b_period: 52
      displacement: 26
  
  arbitrage:
    enabled: false
    weight: 0.0
    parameters:
      min_spread: 0.005
      max_position_size: 0.1
      execution_delay: 1.0

# Technical Indicators
indicators:
  rsi:
    period: 14
    overbought: 70
    oversold: 30
  
  macd:
    fast: 12
    slow: 26
    signal: 9
  
  bollinger_bands:
    period: 20
    std_dev: 2.0
  
  sma:
    periods: [10, 20, 50, 200]
  
  ema:
    periods: [12, 26, 50]
  
  atr:
    period: 14
  
  adx:
    period: 14
  
  parabolic_sar:
    acceleration: 0.02
    maximum: 0.2
  
  ichimoku:
    tenkan_period: 9
    kijun_period: 26
    senkou_span_b_period: 52
    displacement: 26
  
  stochastic:
    k_period: 14
    d_period: 3
    smooth_k: 3
  
  cci:
    period: 20
  
  williams_r:
    period: 14
  
  roc:
    period: 10
  
  obv:
    period: 10
  
  volume_profile:
    bins: 20
  
  vwap:
    period: 14
  
  mfi:
    period: 14

# ML Configuration
ml:
  enabled: true
  model_type: random_forest  # or xgboost, lightgbm, neural_network
  retrain_interval: 24  # hours
  lookback_period: 100  # periods
  
  # Feature engineering
  features:
    price_features: true
    technical_features: true
    volume_features: true
    time_features: true
    lag_features: true
    rolling_features: true
  
  # Model parameters
  random_forest:
    n_estimators: 100
    max_depth: 10
    min_samples_split: 2
    min_samples_leaf: 1
  
  xgboost:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
    subsample: 0.8
  
  lightgbm:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
    subsample: 0.8
  
  neural_network:
    hidden_layers: [64, 32]
    activation: relu
    optimizer: adam
    learning_rate: 0.001
    epochs: 100
    batch_size: 32

# Backtesting Configuration
backtesting:
  initial_capital: 10000.0
  commission_rate: 0.001
  slippage_rate: 0.0005
  start_date: 2023-01-01
  end_date: 2023-12-31
  
  # Optimization
  optimization:
    method: grid_search  # or genetic_algorithm, bayesian_optimization
    max_iterations: 100
    population_size: 50
    generations: 100
  
  # Walk-forward analysis
  walk_forward:
    train_period: 252  # days
    test_period: 63   # days
    step_size: 21     # days

# Notifications
notifications:
  enabled: true
  
  telegram:
    enabled: true
    bot_token: ${TELEGRAM_BOT_TOKEN}
    chat_id: ${TELEGRAM_CHAT_ID}
    send_trades: true
    send_performance: true
    send_errors: true
  
  discord:
    enabled: true
    webhook_url: ${DISCORD_WEBHOOK_URL}
    send_trades: true
    send_performance: true
    send_errors: true
  
  email:
    enabled: false
    smtp_server: smtp.gmail.com
    smtp_port: 587
    username: ${EMAIL_USERNAME}
    password: ${EMAIL_PASSWORD}
    to_email: ${EMAIL_TO}
    send_trades: true
    send_performance: true
    send_errors: true

# Database Configuration
database:
  url: sqlite:///trading_bot.db
  echo: false  # log SQL queries
  
  # Connection pool
  pool_size: 10
  max_overflow: 20
  pool_timeout: 30
  pool_recycle: 3600
  
  # Backup
  backup:
    enabled: true
    interval: 24  # hours
    retention: 7  # days
    path: backups/

# Logging Configuration
logging:
  level: INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  
  # File logging
  file:
    enabled: true
    path: logs/
    max_size: 10MB
    backup_count: 5
  
  # Console logging
  console:
    enabled: true
  
  # Loggers
  loggers:
    trading_bot: INFO
    exchange: INFO
    strategies: INFO
    risk_management: INFO
    backtesting: INFO
    ml: INFO
    notifications: INFO

# Dashboard Configuration
dashboard:
  enabled: true
  host: localhost
  port: 8501
  theme: dark  # or light
  
  # Update intervals
  update_intervals:
    charts: 5  # seconds
    positions: 10  # seconds
    performance: 30  # seconds
  
  # Chart settings
  charts:
    candlestick:
      enabled: true
      periods: [1h, 4h, 1d]
    
    performance:
      enabled: true
      periods: [1d, 1w, 1m, 3m, 1y]
    
    indicators:
      enabled: true
      indicators: [rsi, macd, bollinger_bands]

# Bot Execution
bot:
  mode: paper  # paper, live, backtest
  update_interval: 60  # seconds
  
  # Paper trading
  paper_trading:
    initial_balance: 10000.0
    commission_rate: 0.001
    slippage_rate: 0.0005
  
  # Live trading
  live_trading:
    emergency_stop: false
    max_daily_trades: 100
    max_daily_loss: 0.05  # 5%
  
  # Backtesting
  backtesting:
    start_date: 2023-01-01
    end_date: 2023-12-31
    initial_capital: 10000.0

# Security
security:
  # API key encryption
  encrypt_api_keys: true
  encryption_key: ${ENCRYPTION_KEY}
  
  # Rate limiting
  rate_limits:
    api_calls: 100  # per minute
    trades: 10      # per minute
  
  # Access control
  access_control:
    allowed_ips: []  # empty for all IPs
    require_authentication: false
    session_timeout: 3600  # seconds
```

### Configuration Schema

**File**: `config/config_schema.json`

JSON schema for validating configuration files.

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "exchange": {
      "type": "object",
      "properties": {
        "type": {"type": "string", "enum": ["ccxt", "mt5"]},
        "name": {"type": "string"},
        "api_key": {"type": "string"},
        "api_secret": {"type": "string"},
        "api_passphrase": {"type": "string"},
        "sandbox": {"type": "boolean"},
        "rate_limit": {"type": "number", "minimum": 1}
      },
      "required": ["type", "name"]
    },
    "trading": {
      "type": "object",
      "properties": {
        "default_symbol": {"type": "string"},
        "default_timeframe": {"type": "string"},
        "initial_capital": {"type": "number", "minimum": 0},
        "commission_rate": {"type": "number", "minimum": 0, "maximum": 1},
        "slippage_rate": {"type": "number", "minimum": 0, "maximum": 1}
      }
    },
    "risk_management": {
      "type": "object",
      "properties": {
        "max_risk_per_trade": {"type": "number", "minimum": 0, "maximum": 1},
        "max_positions": {"type": "integer", "minimum": 1},
        "max_drawdown": {"type": "number", "minimum": 0, "maximum": 1},
        "max_correlation": {"type": "number", "minimum": 0, "maximum": 1},
        "max_portfolio_risk": {"type": "number", "minimum": 0, "maximum": 1}
      }
    }
  },
  "required": ["exchange", "trading", "risk_management"]
}
```

## Environment Variables

### Required Variables

**File**: `.env`

```env
# Exchange API Credentials
API_KEY=your_api_key_here
API_SECRET=your_api_secret_here
API_PASSPHRASE=your_passphrase_here  # for some exchanges

# Telegram Notifications
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id

# Discord Notifications
DISCORD_WEBHOOK_URL=your_discord_webhook_url

# Email Notifications
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
EMAIL_TO=recipient@example.com

# Database
DATABASE_URL=sqlite:///trading_bot.db

# Security
ENCRYPTION_KEY=your_encryption_key_here

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/trading_bot.log
```

### Optional Variables

```env
# Exchange Settings
EXCHANGE_TYPE=ccxt
EXCHANGE_NAME=binance
SANDBOX_MODE=false

# Trading Settings
DEFAULT_SYMBOL=BTC/USDT
DEFAULT_TIMEFRAME=1h
INITIAL_CAPITAL=10000.0

# Risk Management
MAX_RISK_PER_TRADE=0.02
MAX_POSITIONS=5
MAX_DRAWDOWN=0.05

# Bot Settings
BOT_MODE=paper
UPDATE_INTERVAL=60

# Dashboard Settings
DASHBOARD_HOST=localhost
DASHBOARD_PORT=8501

# ML Settings
ML_ENABLED=true
MODEL_TYPE=random_forest
RETRAIN_INTERVAL=24

# Backtesting Settings
BACKTEST_START_DATE=2023-01-01
BACKTEST_END_DATE=2023-12-31
BACKTEST_INITIAL_CAPITAL=10000.0
```

## Configuration Loading

### Python Configuration Loader

**File**: `src/config/config_loader.py`

```python
import os
import yaml
from typing import Dict, Any
from pathlib import Path

class ConfigLoader:
    """Loads and validates configuration from files and environment variables."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.config = {}
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file and environment variables."""
        # Load YAML configuration
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Substitute environment variables
        self.config = self._substitute_env_vars(self.config)
        
        # Validate configuration
        self._validate_config()
        
        return self.config
    
    def _substitute_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Substitute environment variables in configuration."""
        if isinstance(config, dict):
            return {key: self._substitute_env_vars(value) for key, value in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith('${') and config.endswith('}'):
            env_var = config[2:-1]
            return os.getenv(env_var, config)
        else:
            return config
    
    def _validate_config(self):
        """Validate configuration against schema."""
        # Load schema
        schema_path = self.config_path.parent / "config_schema.json"
        with open(schema_path, 'r') as file:
            schema = json.load(file)
        
        # Validate using jsonschema
        import jsonschema
        jsonschema.validate(self.config, schema)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value by key."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save_config(self, filepath: str = None):
        """Save configuration to file."""
        if filepath is None:
            filepath = self.config_path
        
        with open(filepath, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False)
```

### Usage Example

```python
from src.config.config_loader import ConfigLoader

# Load configuration
config_loader = ConfigLoader()
config = config_loader.load_config()

# Access configuration values
exchange_type = config_loader.get('exchange.type')
api_key = config_loader.get('exchange.api_key')
max_risk = config_loader.get('risk_management.max_risk_per_trade')

# Update configuration
config_loader.set('trading.default_symbol', 'ETH/USDT')
config_loader.save_config()
```

## Configuration Categories

### Exchange Configuration

#### CCXT Exchanges

```yaml
exchange:
  type: ccxt
  name: binance  # or bitget, bybit, okx, etc.
  api_key: ${API_KEY}
  api_secret: ${API_SECRET}
  api_passphrase: ${API_PASSPHRASE}  # for some exchanges
  sandbox: false
  rate_limit: 100
  timeout: 30
  retries: 3
```

#### MetaTrader 5

```yaml
exchange:
  type: mt5
  name: mt5
  account: 123456
  password: ${MT5_PASSWORD}
  server: MetaQuotes-Demo
  timeout: 30
  retries: 3
```

### Trading Configuration

```yaml
trading:
  default_symbol: BTC/USDT
  default_timeframe: 1h
  initial_capital: 10000.0
  commission_rate: 0.001
  slippage_rate: 0.0005
  
  # Order settings
  order_timeout: 30
  max_retries: 3
  retry_delay: 1
  
  # Position settings
  max_position_size: 0.1
  min_position_size: 0.001
  
  # Execution settings
  execution_mode: immediate  # or batch
  batch_size: 10
  batch_delay: 1
```

### Risk Management Configuration

```yaml
risk_management:
  # Risk limits
  max_risk_per_trade: 0.02
  max_positions: 5
  max_drawdown: 0.05
  max_correlation: 0.7
  max_portfolio_risk: 0.10
  
  # Position sizing
  position_sizing:
    method: fixed_percentage  # or kelly, volatility
    risk_percentage: 0.01
    max_position_size: 0.1
    min_position_size: 0.001
    
    # Kelly criterion
    kelly:
      win_rate: 0.6
      avg_win_loss_ratio: 1.5
      max_kelly_fraction: 0.25
    
    # Volatility-based
    volatility:
      atr_period: 14
      volatility_multiplier: 2.0
  
  # Stop loss
  stop_loss:
    method: fixed_percentage  # or trailing, atr, time
    stop_loss_percentage: 0.05
    trailing_percentage: 0.03
    atr_multiplier: 2.0
    max_hold_hours: 24
    
    # Dynamic stop loss
    dynamic:
      enabled: true
      volatility_adjustment: true
      trend_adjustment: true
  
  # Take profit
  take_profit:
    enabled: true
    method: fixed_percentage  # or atr, trailing
    take_profit_percentage: 0.10
    atr_multiplier: 3.0
    trailing_percentage: 0.05
  
  # Portfolio management
  portfolio:
    rebalancing:
      enabled: true
      interval: 24  # hours
      threshold: 0.05  # 5% deviation
    
    correlation:
      enabled: true
      max_correlation: 0.7
      correlation_period: 30  # days
    
    diversification:
      enabled: true
      max_single_asset: 0.3
      min_assets: 3
```

### Strategy Configuration

#### RSI + MACD Strategy

```yaml
strategies:
  rsi_macd:
    enabled: true
    weight: 0.4
    parameters:
      rsi_period: 14
      rsi_overbought: 70
      rsi_oversold: 30
      macd_fast: 12
      macd_slow: 26
      macd_signal: 9
      
      # Signal filters
      volume_confirmation: true
      trend_filter: true
      volatility_filter: true
      
      # Risk management
      stop_loss_percentage: 0.05
      take_profit_percentage: 0.10
      max_hold_hours: 24
```

#### Bollinger Bands Mean Reversion

```yaml
strategies:
  bollinger_mean_reversion:
    enabled: true
    weight: 0.3
    parameters:
      period: 20
      std_dev: 2.0
      entry_threshold: 1.0
      
      # Signal filters
      volume_confirmation: true
      momentum_filter: true
      
      # Risk management
      stop_loss_percentage: 0.03
      take_profit_percentage: 0.06
      max_hold_hours: 12
```

#### Grid Bot Strategy

```yaml
strategies:
  grid_bot:
    enabled: false
    weight: 0.1
    parameters:
      grid_levels: 10
      grid_spacing: 0.01
      max_position_size: 0.1
      
      # Grid settings
      grid_type: fixed  # or dynamic
      grid_range: 0.10  # 10% range
      
      # Risk management
      stop_loss_percentage: 0.20
      take_profit_percentage: 0.05
      max_hold_hours: 168  # 1 week
```

### ML Configuration

```yaml
ml:
  enabled: true
  model_type: random_forest  # or xgboost, lightgbm, neural_network
  retrain_interval: 24  # hours
  lookback_period: 100  # periods
  
  # Feature engineering
  features:
    price_features: true
    technical_features: true
    volume_features: true
    time_features: true
    lag_features: true
    rolling_features: true
    
    # Feature selection
    feature_selection: true
    max_features: 50
    correlation_threshold: 0.8
  
  # Model parameters
  random_forest:
    n_estimators: 100
    max_depth: 10
    min_samples_split: 2
    min_samples_leaf: 1
    random_state: 42
  
  xgboost:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
    subsample: 0.8
    random_state: 42
  
  lightgbm:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
    subsample: 0.8
    random_state: 42
  
  neural_network:
    hidden_layers: [64, 32]
    activation: relu
    optimizer: adam
    learning_rate: 0.001
    epochs: 100
    batch_size: 32
    validation_split: 0.2
  
  # Model evaluation
  evaluation:
    cross_validation: true
    cv_folds: 5
    scoring: accuracy
    test_size: 0.2
  
  # Model persistence
  persistence:
    save_models: true
    model_path: models/
    backup_models: true
    max_backups: 5
```

### Backtesting Configuration

```yaml
backtesting:
  initial_capital: 10000.0
  commission_rate: 0.001
  slippage_rate: 0.0005
  start_date: 2023-01-01
  end_date: 2023-12-31
  
  # Data settings
  data:
    source: exchange  # or file
    file_path: data/historical_data.csv
    symbols: [BTC/USDT, ETH/USDT]
    timeframes: [1h, 4h, 1d]
  
  # Optimization
  optimization:
    method: grid_search  # or genetic_algorithm, bayesian_optimization
    max_iterations: 100
    population_size: 50
    generations: 100
    
    # Grid search
    grid_search:
      n_jobs: -1
      cv_folds: 5
    
    # Genetic algorithm
    genetic_algorithm:
      mutation_rate: 0.1
      crossover_rate: 0.8
      selection_method: tournament
    
    # Bayesian optimization
    bayesian_optimization:
      acquisition_function: expected_improvement
      n_initial_points: 10
  
  # Walk-forward analysis
  walk_forward:
    train_period: 252  # days
    test_period: 63   # days
    step_size: 21     # days
    min_train_period: 126  # days
  
  # Performance metrics
  metrics:
    calculate_all: true
    custom_metrics: []
    benchmark: SPY  # or custom benchmark
```

### Notifications Configuration

```yaml
notifications:
  enabled: true
  
  # Global settings
  global:
    send_trades: true
    send_performance: true
    send_errors: true
    send_alerts: true
  
  # Telegram
  telegram:
    enabled: true
    bot_token: ${TELEGRAM_BOT_TOKEN}
    chat_id: ${TELEGRAM_CHAT_ID}
    
    # Message settings
    message_format: markdown
    include_charts: true
    chart_period: 1d
    
    # Rate limiting
    rate_limit: 30  # messages per minute
    burst_limit: 5
  
  # Discord
  discord:
    enabled: true
    webhook_url: ${DISCORD_WEBHOOK_URL}
    
    # Message settings
    message_format: markdown
    include_charts: true
    chart_period: 1d
    
    # Rate limiting
    rate_limit: 30  # messages per minute
    burst_limit: 5
  
  # Email
  email:
    enabled: false
    smtp_server: smtp.gmail.com
    smtp_port: 587
    username: ${EMAIL_USERNAME}
    password: ${EMAIL_PASSWORD}
    to_email: ${EMAIL_TO}
    
    # Email settings
    subject_prefix: [Trading Bot]
    include_charts: true
    chart_period: 1d
    
    # Rate limiting
    rate_limit: 10  # emails per hour
    burst_limit: 2
```

### Database Configuration

```yaml
database:
  url: sqlite:///trading_bot.db
  echo: false  # log SQL queries
  
  # Connection pool
  pool_size: 10
  max_overflow: 20
  pool_timeout: 30
  pool_recycle: 3600
  
  # Backup
  backup:
    enabled: true
    interval: 24  # hours
    retention: 7  # days
    path: backups/
    compression: true
  
  # Performance
  performance:
    query_timeout: 30
    connection_timeout: 10
    max_retries: 3
  
  # Security
  security:
    encrypt_sensitive_data: true
    encryption_key: ${ENCRYPTION_KEY}
```

### Logging Configuration

```yaml
logging:
  level: INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  
  # File logging
  file:
    enabled: true
    path: logs/
    max_size: 10MB
    backup_count: 5
    compression: true
  
  # Console logging
  console:
    enabled: true
    colorize: true
  
  # Loggers
  loggers:
    trading_bot: INFO
    exchange: INFO
    strategies: INFO
    risk_management: INFO
    backtesting: INFO
    ml: INFO
    notifications: INFO
    database: WARNING
  
  # Filters
  filters:
    sensitive_data: true
    api_keys: true
    passwords: true
```

### Dashboard Configuration

```yaml
dashboard:
  enabled: true
  host: localhost
  port: 8501
  theme: dark  # or light
  
  # Update intervals
  update_intervals:
    charts: 5  # seconds
    positions: 10  # seconds
    performance: 30  # seconds
    alerts: 60  # seconds
  
  # Chart settings
  charts:
    candlestick:
      enabled: true
      periods: [1h, 4h, 1d]
      indicators: [rsi, macd, bollinger_bands]
    
    performance:
      enabled: true
      periods: [1d, 1w, 1m, 3m, 1y]
      metrics: [total_return, sharpe_ratio, max_drawdown]
    
    indicators:
      enabled: true
      indicators: [rsi, macd, bollinger_bands, atr, adx]
  
  # Layout
  layout:
    sidebar_width: 300
    main_width: 800
    chart_height: 400
  
  # Security
  security:
    require_authentication: false
    session_timeout: 3600
    allowed_ips: []
```

## Configuration Validation

### Schema Validation

```python
import jsonschema
import yaml
import json

def validate_config(config_path: str, schema_path: str):
    """Validate configuration against schema."""
    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Load schema
    with open(schema_path, 'r') as file:
        schema = json.load(file)
    
    # Validate
    try:
        jsonschema.validate(config, schema)
        print("Configuration is valid!")
    except jsonschema.ValidationError as e:
        print(f"Configuration validation error: {e}")
    except Exception as e:
        print(f"Error validating configuration: {e}")
```

### Custom Validation

```python
def validate_trading_config(config: dict):
    """Validate trading-specific configuration."""
    errors = []
    
    # Check risk limits
    risk_mgmt = config.get('risk_management', {})
    
    if risk_mgmt.get('max_risk_per_trade', 0) > 0.1:
        errors.append("max_risk_per_trade should not exceed 10%")
    
    if risk_mgmt.get('max_positions', 0) > 20:
        errors.append("max_positions should not exceed 20")
    
    if risk_mgmt.get('max_drawdown', 0) > 0.2:
        errors.append("max_drawdown should not exceed 20%")
    
    # Check strategy weights
    strategies = config.get('strategies', {})
    total_weight = sum(strategy.get('weight', 0) for strategy in strategies.values())
    
    if abs(total_weight - 1.0) > 0.01:
        errors.append(f"Strategy weights should sum to 1.0, got {total_weight}")
    
    return errors
```

## Configuration Management

### Dynamic Configuration Updates

```python
class ConfigManager:
    """Manages dynamic configuration updates."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config()
        self.callbacks = {}
    
    def register_callback(self, key: str, callback: callable):
        """Register callback for configuration changes."""
        if key not in self.callbacks:
            self.callbacks[key] = []
        self.callbacks[key].append(callback)
    
    def update_config(self, key: str, value: Any):
        """Update configuration and trigger callbacks."""
        old_value = self.get(key)
        self.set(key, value)
        
        # Trigger callbacks
        if key in self.callbacks:
            for callback in self.callbacks[key]:
                callback(key, old_value, value)
    
    def save_config(self):
        """Save configuration to file."""
        with open(self.config_path, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False)
```

### Configuration Hot Reloading

```python
import watchdog
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ConfigReloadHandler(FileSystemEventHandler):
    """Handles configuration file changes."""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
    
    def on_modified(self, event):
        if event.src_path.endswith('config.yaml'):
            print("Configuration file changed, reloading...")
            self.config_manager.reload_config()

# Usage
config_manager = ConfigManager('config/config.yaml')
observer = Observer()
observer.schedule(ConfigReloadHandler(config_manager), 'config/', recursive=False)
observer.start()
```

## Best Practices

### Configuration Organization

1. **Hierarchical Structure**: Organize configuration in logical hierarchies
2. **Environment Separation**: Use different configs for different environments
3. **Sensitive Data**: Never commit sensitive data to version control
4. **Validation**: Always validate configuration before use
5. **Documentation**: Document all configuration options

### Security Considerations

1. **API Keys**: Use environment variables for API keys
2. **Encryption**: Encrypt sensitive configuration data
3. **Access Control**: Restrict access to configuration files
4. **Audit Logging**: Log configuration changes
5. **Backup**: Regularly backup configuration files

### Performance Optimization

1. **Caching**: Cache frequently accessed configuration values
2. **Lazy Loading**: Load configuration only when needed
3. **Validation**: Validate configuration once at startup
4. **Hot Reloading**: Use hot reloading for development
5. **Monitoring**: Monitor configuration changes

## Troubleshooting

### Common Configuration Issues

#### Invalid YAML Syntax
```yaml
# Error: Missing quotes around string with special characters
symbol: BTC/USDT  # Should be "BTC/USDT"

# Error: Incorrect indentation
strategies:
rsi_macd:  # Should be indented
  enabled: true
```

#### Missing Environment Variables
```bash
# Error: Environment variable not set
API_KEY=  # Empty value

# Solution: Set environment variable
export API_KEY=your_actual_api_key
```

#### Invalid Parameter Values
```yaml
# Error: Risk percentage > 1
risk_management:
  max_risk_per_trade: 1.5  # Should be <= 1.0

# Error: Negative values
trading:
  initial_capital: -1000  # Should be >= 0
```

### Debugging Configuration

#### Enable Debug Logging
```yaml
logging:
  level: DEBUG
  loggers:
    config_loader: DEBUG
```

#### Validate Configuration
```python
from src.config.config_loader import ConfigLoader

config_loader = ConfigLoader()
try:
    config = config_loader.load_config()
    print("Configuration loaded successfully!")
except Exception as e:
    print(f"Configuration error: {e}")
```

#### Test Configuration
```python
def test_config():
    """Test configuration loading and validation."""
    config_loader = ConfigLoader()
    config = config_loader.load_config()
    
    # Test specific values
    assert config['exchange']['type'] in ['ccxt', 'mt5']
    assert 0 <= config['risk_management']['max_risk_per_trade'] <= 1
    assert config['trading']['initial_capital'] > 0
    
    print("Configuration test passed!")
```

## Conclusion

This configuration guide provides comprehensive coverage of all configuration options in the Urban Waddle Bot. Proper configuration is essential for successful trading bot operation. Always validate your configuration and test thoroughly before deploying to production.

For additional help with configuration, refer to the example configuration files and the test suite for configuration validation examples.
