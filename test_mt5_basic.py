#!/usr/bin/env python3
"""
Simple MT5 Connection Test

This script tests basic MT5 connection without requiring the full bot setup.
"""

import os
import sys

def test_mt5_basic():
    """Test basic MT5 functionality."""
    
    print("Testing MT5 Connection...")
    print("=" * 40)
    
    try:
        import MetaTrader5 as mt5
        print("SUCCESS: MetaTrader5 package imported successfully")
    except ImportError:
        print("ERROR: MetaTrader5 package not installed!")
        print("Install it with: pip install MetaTrader5")
        return False
    
    # Initialize MT5
    print("\nInitializing MT5...")
    if not mt5.initialize():
        print(f"ERROR: MT5 initialization failed: {mt5.last_error()}")
        print("\nTroubleshooting:")
        print("1. Make sure MT5 platform is installed and running")
        print("2. Check if MT5 is properly installed")
        return False
    
    print("SUCCESS: MT5 initialized successfully")
    
    # Get terminal info
    terminal_info = mt5.terminal_info()
    if terminal_info:
        print(f"SUCCESS: Terminal: {terminal_info.name}")
        print(f"SUCCESS: Company: {terminal_info.company}")
        print(f"SUCCESS: Path: {terminal_info.path}")
    
    # Get account info (if logged in)
    account_info = mt5.account_info()
    if account_info:
        print(f"SUCCESS: Account: {account_info.login}")
        print(f"SUCCESS: Server: {account_info.server}")
        print(f"SUCCESS: Balance: ${account_info.balance:.2f}")
        print(f"SUCCESS: Currency: {account_info.currency}")
    else:
        print("INFO: No account logged in (this is normal for testing)")
    
    # Get symbols
    symbols = mt5.symbols_get()
    if symbols:
        print(f"SUCCESS: Available symbols: {len(symbols)}")
        print("  Popular forex pairs:")
        forex_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
        for pair in forex_pairs:
            symbol_info = mt5.symbol_info(pair)
            if symbol_info:
                print(f"    {pair}: Available")
            else:
                print(f"    {pair}: Not available")
    else:
        print("ERROR: Failed to get symbols")
    
    # Test getting rates
    print("\nTesting data retrieval...")
    rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_H1, 0, 5)
    if rates is not None and len(rates) > 0:
        print(f"SUCCESS: Retrieved {len(rates)} hourly candles for EURUSD")
        latest = rates[-1]
        print(f"  Latest: Open={latest['open']:.5f}, Close={latest['close']:.5f}")
    else:
        print("ERROR: Failed to get EURUSD data")
    
    # Shutdown
    mt5.shutdown()
    print("\nSUCCESS: MT5 connection test completed")
    
    return True

def main():
    """Main function."""
    print("Urban Waddle Bot - MT5 Basic Connection Test")
    print("=" * 50)
    
    success = test_mt5_basic()
    
    print("\n" + "=" * 50)
    if success:
        print("SUCCESS: MT5 Basic Test PASSED!")
        print("\nNext steps:")
        print("1. Copy env_mt5_template to .env")
        print("2. Edit .env with your MT5 demo account details")
        print("3. Run: python test_mt5_connection.py")
        print("4. Start the bot: python main.py --mode paper")
    else:
        print("ERROR: MT5 Basic Test FAILED!")
        print("\nPlease check:")
        print("1. MT5 platform is installed and running")
        print("2. MetaTrader5 Python package is installed")
        print("3. Your system meets MT5 requirements")
    print("=" * 50)

if __name__ == "__main__":
    main()
