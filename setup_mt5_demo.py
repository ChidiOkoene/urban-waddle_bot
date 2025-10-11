#!/usr/bin/env python3
"""
MT5 Demo Account Quick Setup

This script helps you quickly set up the Urban Waddle Bot
to work with your MT5 demo account.
"""

import os
import sys
from pathlib import Path

def create_env_file():
    """Create .env file with MT5 configuration template."""
    
    env_content = """# Urban Waddle Bot - MT5 Demo Account Configuration
# Copy this file to .env and fill in your actual MT5 demo account details

# Exchange Configuration
EXCHANGE_TYPE=mt5
EXCHANGE_NAME=mt5

# MT5 Demo Account Credentials
# Get these from your MT5 platform
MT5_ACCOUNT=12345678
MT5_PASSWORD=your_demo_password
MT5_SERVER=MetaQuotes-Demo

# Trading Configuration
BOT_MODE=paper
DEFAULT_SYMBOL=EURUSD
DEFAULT_TIMEFRAME=1h
INITIAL_CAPITAL=10000.0

# Risk Management
MAX_RISK_PER_TRADE=0.02
MAX_POSITIONS=5
MAX_DRAWDOWN=0.05

# Notifications (Optional)
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
DISCORD_WEBHOOK_URL=

# Database
DATABASE_URL=sqlite:///data/trading_bot.db

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/trading_bot.log

# Dashboard
DASHBOARD_PORT=8501
"""
    
    env_file = Path('.env')
    if env_file.exists():
        print("WARNING: .env file already exists. Backing up to .env.backup")
        env_file.rename('.env.backup')
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("SUCCESS: Created .env file with MT5 configuration template")
    print("Please edit .env file with your actual MT5 demo account details")

def create_config_file():
    """Create config.yaml file for MT5."""
    
    config_content = """# Urban Waddle Bot Configuration for MT5 Demo Account

# Exchange Configuration
exchange:
  type: mt5
  name: mt5
  credentials:
    account: ${MT5_ACCOUNT}
    password: ${MT5_PASSWORD}
    server: ${MT5_SERVER}

# Trading Configuration
trading:
  default_symbol: EURUSD
  default_timeframe: 1h
  initial_capital: 10000.0
  commission_rate: 0.0
  slippage_rate: 0.0001

# Risk Management
risk_management:
  max_risk_per_trade: 0.02
  max_positions: 5
  max_drawdown: 0.05
  max_correlation: 0.7
  max_portfolio_risk: 0.10
  
  position_sizing:
    method: fixed_percentage
    risk_percentage: 0.01
    max_position_size: 0.1
    min_position_size: 0.01
  
  stop_loss:
    method: fixed_percentage
    stop_loss_percentage: 0.05
    trailing_percentage: 0.03
    max_hold_hours: 24

# Strategies Configuration
strategies:
  rsi_macd:
    enabled: true
    weight: 1.0
    parameters:
      rsi_period: 14
      rsi_overbought: 70
      rsi_oversold: 30
      macd_fast: 12
      macd_slow: 26
      macd_signal: 9
  
  bollinger_mean_reversion:
    enabled: false
    weight: 0.0
    parameters:
      period: 20
      std_dev: 2.0
      entry_threshold: 1.0

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
  
  sma:
    periods: [10, 20, 50]
  
  ema:
    periods: [12, 26]

# Notifications
notifications:
  enabled: false
  
  telegram:
    enabled: false
    bot_token: ${TELEGRAM_BOT_TOKEN}
    chat_id: ${TELEGRAM_CHAT_ID}
  
  discord:
    enabled: false
    webhook_url: ${DISCORD_WEBHOOK_URL}

# Database Configuration
database:
  url: ${DATABASE_URL}
  echo: false

# Logging Configuration
logging:
  level: ${LOG_LEVEL}
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  
  file:
    enabled: true
    path: logs/
    max_size: 10MB
    backup_count: 5
  
  console:
    enabled: true

# Dashboard Configuration
dashboard:
  enabled: true
  host: localhost
  port: ${DASHBOARD_PORT}
  theme: dark
  
  update_intervals:
    charts: 5
    positions: 10
    performance: 30

# Bot Execution
bot:
  mode: ${BOT_MODE}
  update_interval: 60
"""
    
    config_dir = Path('config')
    config_dir.mkdir(exist_ok=True)
    
    config_file = config_dir / 'config.yaml'
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print("SUCCESS: Created config/config.yaml file for MT5")

def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")
    
    try:
        import subprocess
        result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("SUCCESS: Dependencies installed successfully")
        else:
            print(f"ERROR: Failed to install dependencies: {result.stderr}")
            return False
    except Exception as e:
        print(f"ERROR: Error installing dependencies: {e}")
        return False
    
    return True

def create_directories():
    """Create necessary directories."""
    directories = ['data', 'logs', 'models', 'backups', 'config']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("SUCCESS: Created necessary directories")

def main():
    """Main setup function."""
    print("Urban Waddle Bot - MT5 Demo Account Setup")
    print("=" * 50)
    
    # Create directories
    create_directories()
    
    # Create configuration files
    create_env_file()
    create_config_file()
    
    # Install dependencies
    if not install_dependencies():
        print("ERROR: Setup failed during dependency installation")
        return
    
    print("\n" + "=" * 50)
    print("SUCCESS: MT5 Demo Account Setup Complete!")
    print("=" * 50)
    
    print("\nNext Steps:")
    print("1. Edit .env file with your MT5 demo account details:")
    print("   - MT5_ACCOUNT: Your demo account number")
    print("   - MT5_PASSWORD: Your demo password")
    print("   - MT5_SERVER: Your broker's server name")
    
    print("\n2. Test your MT5 connection:")
    print("   python test_mt5_connection.py")
    
    print("\n3. Start the bot in paper trading mode:")
    print("   python main.py --mode paper --strategy rsi_macd")
    
    print("\n4. Launch the dashboard:")
    print("   streamlit run src/dashboard/app.py")
    
    print("\nFor detailed setup instructions, see:")
    print("   docs/MT5_SETUP.md")
    
    print("\nImportant Notes:")
    print("- Always test with demo accounts first")
    print("- Start with small position sizes")
    print("- Use proper risk management")
    print("- Monitor your trades closely")

if __name__ == "__main__":
    main()