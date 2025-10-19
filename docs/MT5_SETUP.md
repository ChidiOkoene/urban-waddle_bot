# MT5 Demo Account Setup Guide

## Prerequisites

1. **MetaTrader 5 Platform**: Download and install MT5 from your broker
2. **Demo Account**: Create a demo account with your broker
3. **Python Environment**: Ensure Python 3.8+ is installed

## Step 1: Install MetaTrader5 Python Package

```bash
pip install MetaTrader5
```

## Step 2: Get Your Demo Account Details

From your MT5 platform, note down:
- **Account Number**: Your demo account number
- **Password**: Your demo account password  
- **Server**: Your broker's server name (e.g., "MetaQuotes-Demo", "ICMarkets-Demo")

## Step 3: Configure Environment Variables

Create or edit your `.env` file:

```env
# Exchange Configuration
EXCHANGE_TYPE=mt5
EXCHANGE_NAME=mt5

# MT5 Demo Account Credentials
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
```

## Step 4: Update Configuration File

Edit `config/config.yaml`:

```yaml
exchange:
  type: mt5
  name: mt5
  credentials:
    account: ${MT5_ACCOUNT}
    password: ${MT5_PASSWORD}
    server: ${MT5_SERVER}

trading:
  default_symbol: EURUSD
  default_timeframe: 1h
  initial_capital: 10000.0

risk_management:
  max_risk_per_trade: 0.02
  max_positions: 5
  max_drawdown: 0.05
```

## Step 5: Test MT5 Connection

Create a test script to verify connection:

```python
# test_mt5_connection.py
import MetaTrader5 as mt5
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MT5 credentials
account = int(os.getenv('MT5_ACCOUNT'))
password = os.getenv('MT5_PASSWORD')
server = os.getenv('MT5_SERVER')

print(f"Connecting to MT5...")
print(f"Account: {account}")
print(f"Server: {server}")

# Initialize MT5
if not mt5.initialize():
    print("MT5 initialization failed")
    exit()

# Login to account
if not mt5.login(account, password=password, server=server):
    print(f"Login failed: {mt5.last_error()}")
    mt5.shutdown()
    exit()

print("✅ Successfully connected to MT5 demo account!")

# Get account info
account_info = mt5.account_info()
if account_info:
    print(f"Account Balance: {account_info.balance}")
    print(f"Account Equity: {account_info.equity}")
    print(f"Account Currency: {account_info.currency}")
    print(f"Account Server: {account_info.server}")

# Get available symbols
symbols = mt5.symbols_get()
if symbols:
    print(f"\nAvailable symbols: {len(symbols)}")
    for i, symbol in enumerate(symbols[:10]):  # Show first 10
        print(f"  {i+1}. {symbol.name}")

# Disconnect
mt5.shutdown()
print("\n✅ MT5 connection test completed successfully!")
```

## Step 6: Run the Test

```bash
python test_mt5_connection.py
```

## Step 7: Start the Bot with MT5

```bash
# Start with paper trading mode first
python main.py --mode paper --strategy rsi_macd

# Or use the start script
./scripts/start_bot.sh --mode paper --strategy rsi_macd
```

## Common MT5 Demo Account Servers

Here are some popular broker demo servers:

- **MetaQuotes**: `MetaQuotes-Demo`
- **IC Markets**: `ICMarkets-Demo`
- **XM**: `XM.COM-Demo`
- **FXCM**: `FXCM-Demo`
- **OANDA**: `OANDA-Demo`
- **Pepperstone**: `Pepperstone-Demo`

## Troubleshooting

### Connection Issues

1. **MT5 Not Running**: Ensure MT5 platform is running
2. **Wrong Server**: Double-check your broker's server name
3. **Account Details**: Verify account number and password
4. **Firewall**: Check if firewall is blocking MT5

### Common Errors

```python
# Error: MT5 initialization failed
# Solution: Make sure MT5 is installed and running

# Error: Login failed
# Solution: Check account credentials and server name

# Error: Symbol not found
# Solution: Use correct symbol names (e.g., EURUSD, GBPUSD)
```

## Testing with Real Demo Trades

Once connected, you can test with small demo trades:

```python
# Example: Place a small demo buy order
import MetaTrader5 as mt5

# Place order
request = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": "EURUSD",
    "volume": 0.01,  # Small lot size
    "type": mt5.ORDER_TYPE_BUY,
    "price": mt5.symbol_info_tick("EURUSD").ask,
    "deviation": 20,
    "magic": 234000,
    "comment": "Python script open",
    "type_time": mt5.ORDER_TIME_GTC,
    "type_filling": mt5.ORDER_FILLING_IOC,
}

result = mt5.order_send(request)
print(f"Order result: {result}")
```

## Next Steps

1. **Test Connection**: Run the connection test script
2. **Paper Trading**: Start with paper trading mode
3. **Small Demo Trades**: Test with minimal lot sizes
4. **Monitor Performance**: Use the dashboard to monitor trades
5. **Scale Up**: Gradually increase position sizes

## Important Notes

- **Demo Account**: Always test with demo accounts first
- **Lot Sizes**: Start with minimum lot sizes (0.01)
- **Risk Management**: Use proper stop-loss and take-profit
- **Market Hours**: Be aware of forex market hours
- **Spread**: Consider spreads in your strategy

## Support

If you encounter issues:
1. Check MT5 terminal logs
2. Verify broker-specific requirements
3. Test with simple buy/sell orders first
4. Contact your broker for demo account support
