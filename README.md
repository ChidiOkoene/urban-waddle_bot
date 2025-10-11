# Urban Waddle Bot

A comprehensive, cross-platform automated trading bot with advanced strategies, real-time dashboard, and production-ready features.

## 🚀 Features

### Core Trading Features
- **Multi-Exchange Support**: MetaTrader 5 and CCXT-compatible exchanges (Binance, Bitget, Bybit, OKX, etc.)
- **9 Trading Strategies**: RSI+MACD, Bollinger Bands, EMA Crossover, Grid Bot, DCA, Breakout, Momentum, Ichimoku, Arbitrage
- **Advanced Risk Management**: Position sizing, stop-loss, take-profit, trailing stops, portfolio management
- **Machine Learning Integration**: Feature engineering, model training, real-time prediction
- **Comprehensive Backtesting**: Historical simulation, performance metrics, strategy optimization

### Production Features
- **Real-time Dashboard**: Streamlit-based interface with live charts and metrics
- **Health Monitoring**: System resource monitoring, health checks, alerting
- **Notifications**: Telegram, Discord, and email notifications
- **Docker Support**: Containerized deployment with Docker Compose
- **Database**: SQLite with async support for persistent storage
- **Logging**: Comprehensive logging with rotation and levels

### Technical Features
- **Cross-Platform**: Windows, macOS, Linux support
- **Async Architecture**: High-performance async/await implementation
- **Modular Design**: Clean separation of concerns with dependency injection
- **Configuration Management**: YAML configuration with environment variable support
- **Testing Suite**: Comprehensive test coverage with mock exchanges

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)
- MetaTrader 5 (for MT5 trading)
- Exchange API keys (for crypto trading)

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd urban-waddle_bot
```

### 2. Automated Setup

```bash
# Make scripts executable
chmod +x scripts/*.sh

# Run automated setup
./scripts/setup.sh
```

### 3. Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p data logs models backups config monitoring/grafana/dashboards monitoring/grafana/datasources nginx/ssl

# Copy environment template
cp env.example .env
```

### 4. Configuration

Edit `.env` file with your configuration:

```env
# Exchange Configuration
EXCHANGE_TYPE=ccxt
EXCHANGE_NAME=binance
API_KEY=your_api_key_here
API_SECRET=your_api_secret_here

# Trading Configuration
BOT_MODE=paper
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
```

## 🚀 Usage

### Paper Trading

```bash
# Start with paper trading
./scripts/start_bot.sh --mode paper --strategy rsi_macd
```

### Live Trading

```bash
# Switch to live trading (make sure to configure API keys)
./scripts/start_bot.sh --mode live --strategy rsi_macd
```

### Backtesting

```bash
# Run backtest
./scripts/start_bot.sh --mode backtest --strategy rsi_macd
```

### Dashboard

```bash
# Launch dashboard
streamlit run src/dashboard/app.py
```

The dashboard will be available at `http://localhost:8501`

## 🐳 Docker Deployment

### Quick Start

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f trading-bot

# Stop services
docker-compose down
```

### Services

- **trading-bot**: Main bot container
- **redis**: Caching and session storage
- **prometheus**: Metrics collection
- **grafana**: Monitoring dashboard
- **nginx**: Reverse proxy and load balancer

### Access Points

- **Dashboard**: http://localhost:8501
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

## 📊 Trading Strategies

### 1. RSI + MACD Strategy
Combines RSI momentum with MACD trend indicators.

**Parameters**:
- RSI Period: 14
- RSI Overbought: 70
- RSI Oversold: 30
- MACD Fast: 12
- MACD Slow: 26
- MACD Signal: 9

### 2. Bollinger Bands Mean Reversion
Trades price reversals at Bollinger Band extremes.

**Parameters**:
- Period: 20
- Standard Deviation: 2.0
- Entry Threshold: 1.0

### 3. EMA Crossover Strategy
Uses exponential moving average crossovers.

**Parameters**:
- Fast Period: 12
- Slow Period: 26
- Signal Threshold: 0.005

### 4. Grid Bot Strategy
Places buy/sell orders at regular intervals.

**Parameters**:
- Grid Levels: 10
- Grid Spacing: 0.01
- Max Position Size: 0.1

### 5. DCA Strategy
Dollar Cost Averaging approach.

**Parameters**:
- Interval Hours: 24
- Position Size: 0.05
- Max Positions: 10

### 6. Breakout Strategy
Trades price breakouts from consolidation.

**Parameters**:
- Consolidation Periods: 20
- Breakout Threshold: 0.02
- Volume Confirmation: true

### 7. Momentum Strategy
Trades based on price momentum.

**Parameters**:
- Momentum Period: 14
- Acceleration Period: 5
- Momentum Threshold: 0.01

### 8. Ichimoku Strategy
Comprehensive trend analysis using Ichimoku Cloud.

**Parameters**:
- Tenkan Period: 9
- Kijun Period: 26
- Senkou Span B Period: 52
- Displacement: 26

### 9. Arbitrage Strategy
Trades price differences between exchanges.

**Parameters**:
- Min Spread: 0.005
- Max Position Size: 0.1
- Execution Delay: 1.0

## ⚙️ Configuration

### Strategy Configuration

Edit `config/config.yaml`:

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
  
  bollinger_mean_reversion:
    enabled: true
    weight: 0.3
    parameters:
      period: 20
      std_dev: 2.0
      entry_threshold: 1.0
```

### Risk Management

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

## 🔧 Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_strategies.py

# Run with coverage
pytest --cov=src tests/
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Adding New Strategies

1. Create strategy file in `src/strategies/`
2. Inherit from `BaseStrategy`
3. Implement `generate_signals()` method
4. Add tests in `tests/test_strategies.py`
5. Update configuration schema

## 📈 Monitoring

### Health Checks

The bot includes comprehensive health monitoring:

- **Database Health**: Connection status and response time
- **System Resources**: CPU, memory, and disk usage
- **Bot Health**: Heartbeat monitoring and error tracking
- **Trading Metrics**: Performance and position monitoring

### Metrics

Prometheus metrics are available at `/metrics`:

- `system_cpu_usage`: CPU usage percentage
- `system_memory_usage`: Memory usage percentage
- `system_disk_usage`: Disk usage percentage
- `trading_total_trades`: Total number of trades
- `trading_win_rate`: Win rate percentage
- `trading_total_pnl`: Total profit/loss
- `health_status`: Overall health status

### Alerts

Configure alerts for:

- Trade execution
- Stop-loss triggers
- Performance metrics
- System health issues
- Connection problems

## 🔒 Security

### API Key Management

- Never commit API keys to version control
- Use environment variables for sensitive data
- Enable API key encryption in production
- Use read-only keys when possible

### Risk Management

- Set appropriate position limits
- Use stop-losses on all trades
- Monitor drawdown limits
- Regular performance reviews

### Access Control

- Restrict dashboard access
- Use strong authentication
- Monitor access logs
- Regular security updates

## 📚 Documentation

- [Setup Guide](docs/SETUP.md) - Detailed setup instructions
- [Strategies Guide](docs/STRATEGIES.md) - Trading strategies documentation
- [API Reference](docs/API.md) - Complete API documentation
- [Configuration Guide](docs/CONFIGURATION.md) - Configuration options

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading cryptocurrencies and other financial instruments involves substantial risk of loss. Past performance is not indicative of future results. Always do your own research and consider consulting with a financial advisor before making investment decisions.

## 🆘 Support

For issues and questions:

1. Check the documentation
2. Review log files
3. Test with paper trading
4. Open an issue on GitHub

## 🎯 Roadmap

- [ ] Additional exchange integrations
- [ ] More trading strategies
- [ ] Advanced ML models
- [ ] Mobile app
- [ ] Cloud deployment options
- [ ] Social trading features

---

**Happy Trading! 🚀**