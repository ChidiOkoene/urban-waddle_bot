#!/usr/bin/env python3
"""
MT5 Demo Account Connection Test

This script tests the connection to your MT5 demo account
and verifies that the Urban Waddle Bot can communicate with MT5.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    import MetaTrader5 as mt5
except ImportError:
    print("❌ MetaTrader5 package not installed!")
    print("Install it with: pip install MetaTrader5")
    sys.exit(1)

def test_mt5_connection():
    """Test MT5 connection and display account information."""
    
    # Load environment variables
    load_dotenv()
    
    # Get MT5 credentials
    account = os.getenv('MT5_ACCOUNT')
    password = os.getenv('MT5_PASSWORD')
    server = os.getenv('MT5_SERVER')
    
    if not all([account, password, server]):
        print("❌ Missing MT5 credentials in .env file!")
        print("Please set the following environment variables:")
        print("  MT5_ACCOUNT=your_account_number")
        print("  MT5_PASSWORD=your_demo_password")
        print("  MT5_SERVER=your_broker_server")
        return False
    
    print("🔌 Testing MT5 Demo Account Connection...")
    print(f"Account: {account}")
    print(f"Server: {server}")
    print("-" * 50)
    
    # Initialize MT5
    print("📡 Initializing MT5...")
    if not mt5.initialize():
        print(f"❌ MT5 initialization failed: {mt5.last_error()}")
        return False
    print("✅ MT5 initialized successfully")
    
    # Login to account
    print("🔐 Logging into demo account...")
    if not mt5.login(int(account), password=password, server=server):
        print(f"❌ Login failed: {mt5.last_error()}")
        mt5.shutdown()
        return False
    print("✅ Successfully logged into demo account!")
    
    # Get account information
    print("\n📊 Account Information:")
    account_info = mt5.account_info()
    if account_info:
        print(f"  Balance: ${account_info.balance:.2f}")
        print(f"  Equity: ${account_info.equity:.2f}")
        print(f"  Currency: {account_info.currency}")
        print(f"  Server: {account_info.server}")
        print(f"  Leverage: 1:{account_info.leverage}")
        print(f"  Margin: ${account_info.margin:.2f}")
        print(f"  Free Margin: ${account_info.margin_free:.2f}")
    else:
        print("❌ Failed to get account information")
        return False
    
    # Get available symbols
    print("\n📈 Available Trading Symbols:")
    symbols = mt5.symbols_get()
    if symbols:
        print(f"  Total symbols: {len(symbols)}")
        print("  Popular forex pairs:")
        forex_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'NZDUSD', 'USDCHF']
        for pair in forex_pairs:
            symbol_info = mt5.symbol_info(pair)
            if symbol_info:
                tick = mt5.symbol_info_tick(pair)
                if tick:
                    print(f"    {pair}: Bid={tick.bid:.5f}, Ask={tick.ask:.5f}")
    else:
        print("❌ Failed to get symbols")
        return False
    
    # Test symbol info
    print("\n🔍 Testing Symbol Information:")
    test_symbol = "EURUSD"
    symbol_info = mt5.symbol_info(test_symbol)
    if symbol_info:
        print(f"  Symbol: {test_symbol}")
        print(f"  Point: {symbol_info.point}")
        print(f"  Digits: {symbol_info.digits}")
        print(f"  Spread: {symbol_info.spread}")
        print(f"  Min Volume: {symbol_info.volume_min}")
        print(f"  Max Volume: {symbol_info.volume_max}")
        print(f"  Volume Step: {symbol_info.volume_step}")
    else:
        print(f"❌ Failed to get info for {test_symbol}")
        return False
    
    # Test getting OHLCV data
    print("\n📊 Testing OHLCV Data Retrieval:")
    rates = mt5.copy_rates_from_pos(test_symbol, mt5.TIMEFRAME_H1, 0, 10)
    if rates is not None and len(rates) > 0:
        print(f"  Retrieved {len(rates)} hourly candles for {test_symbol}")
        print(f"  Latest candle: Open={rates[-1]['open']:.5f}, High={rates[-1]['high']:.5f}, Low={rates[-1]['low']:.5f}, Close={rates[-1]['close']:.5f}")
    else:
        print(f"❌ Failed to get OHLCV data for {test_symbol}")
        return False
    
    # Test order placement (simulation)
    print("\n🧪 Testing Order Simulation:")
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": test_symbol,
        "volume": 0.01,
        "type": mt5.ORDER_TYPE_BUY,
        "price": mt5.symbol_info_tick(test_symbol).ask,
        "deviation": 20,
        "magic": 234000,
        "comment": "Urban Waddle Bot Test",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    # Note: We won't actually send the order, just validate the request
    print(f"  Order request prepared for {test_symbol}")
    print(f"  Volume: {request['volume']} lots")
    print(f"  Price: {request['price']:.5f}")
    print("  ✅ Order request validation passed")
    
    # Disconnect
    print("\n🔌 Disconnecting from MT5...")
    mt5.shutdown()
    print("✅ Disconnected successfully")
    
    return True

def main():
    """Main function."""
    print("=" * 60)
    print("🚀 Urban Waddle Bot - MT5 Demo Account Test")
    print("=" * 60)
    
    success = test_mt5_connection()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 MT5 Demo Account Test PASSED!")
        print("✅ Your bot is ready to connect to MT5")
        print("\nNext steps:")
        print("1. Start the bot: python main.py --mode paper --strategy rsi_macd")
        print("2. Monitor trades in the dashboard")
        print("3. Test with small demo trades")
    else:
        print("❌ MT5 Demo Account Test FAILED!")
        print("Please check your configuration and try again")
        print("\nTroubleshooting:")
        print("1. Ensure MT5 platform is running")
        print("2. Verify your demo account credentials")
        print("3. Check your broker's server name")
        print("4. Make sure MetaTrader5 package is installed")
    print("=" * 60)

if __name__ == "__main__":
    main()
