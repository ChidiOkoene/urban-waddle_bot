# Urban Waddle Bot - Setup Guide

## Overview

Urban Waddle Bot is a comprehensive, cross-platform automated trading bot that supports multiple exchanges and trading strategies. This guide will help you set up and configure the bot for your trading needs.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)
- MetaTrader 5 (for MT5 trading)
- Exchange API keys (for crypto trading)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd urban-waddle_bot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Copy the example environment file and configure it:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```env
# Exchange Configuration
EXCHANGE_TYPE=ccxt  # or mt5
EXCHANGE_NAME=binance  # or bitget, etc.

# API Credentials
API_KEY=your_api_key
API_SECRET=your_api_secret
API_PASSPHRASE=your_passphrase  # for some exchanges

# Trading Configuration
DEFAULT_SYMBOL=BTC/USDT
DEFAULT_TIMEFRAME=1h
INITIAL_CAPITAL=10000.0

# Risk Management
MAX_RISK_PER_TRADE=0.02
MAX_POSITIONS=5
MAX_DRAWDOWN=0.05

# Notifications
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
DISCORD_WEBHOOK_URL=your_discord_webhook_url

# Database
DATABASE_URL=sqlite:///trading_bot.db

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/trading_bot.log
```

## Configuration

### 1. Exchange Configuration

The bot supports two main exchange types:

#### CCXT Exchanges (Crypto)
- Binance
- Bitget
- Bybit
- OKX
- And many more

#### MetaTrader 5
- Forex
- CFDs
- Commodities
- Indices

### 2. Strategy Configuration

Edit `config/config.yaml` to configure your trading strategies:

```yaml
strategies:
  rsi_macd:
    enabled: true
    parameters:
      rsi_period: 14
      rsi_overbought: 70
      rsi_oversold: 30
      macd_fast: 12
      macd_slow: 26
      macd_signal: 9
  
  bollinger_mean_reversion:
    enabled: true
    parameters:
      period: 20
      std_dev: 2.0
      entry_threshold: 1.0
  
  grid_bot:
    enabled: false
    parameters:
      grid_levels: 10
      grid_spacing: 0.01
      max_position_size: 0.1
```

### 3. Risk Management Configuration

Configure risk management parameters:

```yaml
risk_management:
  max_risk_per_trade: 0.02
  max_positions: 5
  max_drawdown: 0.05
  max_correlation: 0.7
  max_portfolio_risk: 0.10
  
  position_sizing:
    method: fixed_percentage
    risk_percentage: 0.01
    
  stop_loss:
    method: fixed_percentage
    stop_loss_percentage: 0.05
```

## Usage

### 1. Paper Trading Mode

Start with paper trading to test your strategies:

```bash
python main.py --mode paper --strategy rsi_macd
```

### 2. Live Trading Mode

Once you're confident with paper trading, switch to live trading:

```bash
python main.py --mode live --strategy rsi_macd
```

### 3. Dashboard

Launch the Streamlit dashboard:

```bash
streamlit run src/dashboard/app.py
```

The dashboard will be available at `http://localhost:8501`

### 4. Backtesting

Run backtests on historical data:

```bash
python main.py --mode backtest --strategy rsi_macd --start-date 2023-01-01 --end-date 2023-12-31
```

## Strategies

### Available Strategies

1. **RSI + MACD Strategy**
   - Combines RSI momentum with MACD trend
   - Good for trending markets

2. **Bollinger Bands Mean Reversion**
   - Trades price reversals at band extremes
   - Good for ranging markets

3. **EMA Crossover Strategy**
   - Uses fast and slow EMA crossovers
   - Simple trend-following strategy

4. **Grid Bot Strategy**
   - Places buy/sell orders at regular intervals
   - Good for sideways markets

5. **DCA Strategy**
   - Dollar Cost Averaging approach
   - Reduces average entry price

6. **Breakout Strategy**
   - Trades price breakouts from consolidation
   - Good for volatile markets

7. **Momentum Strategy**
   - Trades based on price momentum
   - Good for trending markets

8. **Ichimoku Strategy**
   - Uses Ichimoku Cloud indicators
   - Comprehensive trend analysis

9. **Arbitrage Strategy**
   - Trades price differences between exchanges
   - Requires multiple exchange connections

### Strategy Optimization

Use the built-in optimization engine to find optimal parameters:

```bash
python main.py --mode optimize --strategy rsi_macd --optimization-method genetic_algorithm
```

## Risk Management

### Position Sizing Methods

1. **Fixed Percentage**: Risk a fixed percentage of capital per trade
2. **Kelly Criterion**: Optimal position sizing based on win rate and payoff ratio
3. **Volatility-Based**: Adjust position size based on market volatility

### Stop Loss Methods

1. **Fixed Percentage**: Stop loss at fixed percentage from entry
2. **Trailing Stop**: Stop loss follows price in favorable direction
3. **ATR-Based**: Stop loss based on Average True Range
4. **Time-Based**: Stop loss after maximum holding period

### Risk Limits

- Maximum risk per trade
- Maximum number of positions
- Maximum drawdown
- Maximum correlation between positions
- Maximum portfolio risk

## Notifications

### Telegram Notifications

1. Create a Telegram bot via @BotFather
2. Get your bot token
3. Get your chat ID
4. Configure in `.env` file

### Discord Notifications

1. Create a Discord webhook
2. Configure webhook URL in `.env` file

### Email Notifications

1. Configure SMTP settings in `.env` file
2. Set up email alerts for important events

## Monitoring and Logging

### Log Files

Logs are stored in the `logs/` directory:
- `trading_bot.log`: Main bot activity
- `errors.log`: Error messages
- `trades.log`: Trade execution logs

### Database

The bot uses SQLite for data storage:
- `trades`: Trade history
- `positions`: Open positions
- `performance`: Strategy performance metrics
- `alerts`: Alert history

## Troubleshooting

### Common Issues

1. **API Connection Errors**
   - Check API keys and permissions
   - Verify exchange connectivity
   - Check rate limits

2. **Strategy Not Generating Signals**
   - Verify strategy parameters
   - Check market data availability
   - Review strategy logic

3. **Risk Management Errors**
   - Check risk limits configuration
   - Verify position sizing parameters
   - Review stop-loss settings

### Debug Mode

Enable debug logging:

```bash
python main.py --log-level DEBUG
```

### Testing

Run the test suite:

```bash
pytest tests/
```

## Performance Optimization

### Backtesting

- Use appropriate timeframes for your strategy
- Ensure sufficient historical data
- Validate strategy parameters

### Live Trading

- Monitor performance metrics
- Adjust risk parameters as needed
- Regular strategy optimization

## Security Best Practices

1. **API Keys**
   - Use environment variables
   - Never commit keys to version control
   - Use read-only keys when possible

2. **Risk Management**
   - Set appropriate position limits
   - Use stop-losses
   - Monitor drawdown

3. **Monitoring**
   - Regular performance reviews
   - Alert on unusual activity
   - Backup trading data

## Support

For issues and questions:
1. Check the documentation
2. Review log files
3. Test with paper trading
4. Contact support if needed

## License

This project is licensed under the MIT License - see the LICENSE file for details.
