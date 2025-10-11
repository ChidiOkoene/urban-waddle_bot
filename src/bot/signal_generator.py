"""
Signal Generator for Trading Bot

This module generates trading signals based on strategy logic and market data.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from ..core.exchange_interface import ExchangeInterface
from ..core.data_models import OHLCV, StrategySignal, OrderSide, OrderType, TimeFrame
from ..strategies.base_strategy import BaseStrategy
from ..indicators.technical_indicators import TechnicalIndicators


class SignalGenerator:
    """Generates trading signals based on strategy logic."""
    
    def __init__(self):
        self.logger = logging.getLogger("SignalGenerator")
        self.indicators = TechnicalIndicators()
        self.last_signals = {}  # Track last signals per symbol to avoid duplicates
        self.signal_cooldown = {}  # Cooldown periods for signals
        
    async def initialize(self, 
                        strategy: BaseStrategy, 
                        exchange_adapter: ExchangeInterface):
        """
        Initialize the signal generator.
        
        Args:
            strategy: Strategy instance to use for signal generation
            exchange_adapter: Exchange adapter for market data
        """
        self.strategy = strategy
        self.exchange_adapter = exchange_adapter
        
        # Set up signal cooldowns based on strategy
        self._setup_signal_cooldowns()
        
        self.logger.info("Signal generator initialized")
    
    def _setup_signal_cooldowns(self):
        """Setup signal cooldown periods."""
        # Default cooldown periods (in minutes)
        default_cooldowns = {
            'rsi_macd': 30,
            'bollinger': 60,
            'ema_crossover': 15,
            'grid_bot': 5,
            'dca': 1440,  # 24 hours
            'breakout': 120,
            'momentum': 60,
            'ichimoku': 240,
            'arbitrage': 1
        }
        
        strategy_name = getattr(self.strategy, 'name', 'default')
        self.default_cooldown = default_cooldowns.get(strategy_name, 30)
    
    async def generate_signals(self, 
                             strategy: BaseStrategy,
                             market_data: Dict[str, List[OHLCV]]) -> List[StrategySignal]:
        """
        Generate trading signals based on market data and strategy.
        
        Args:
            strategy: Strategy instance
            market_data: Dictionary of market data by symbol_timeframe
            
        Returns:
            List of generated signals
        """
        signals = []
        
        try:
            # Process each symbol-timeframe combination
            for data_key, ohlcv_data in market_data.items():
                if not ohlcv_data:
                    continue
                
                symbol, timeframe = data_key.rsplit('_', 1)
                
                # Check cooldown for this symbol
                if self._is_in_cooldown(symbol):
                    continue
                
                # Generate signals for this symbol
                symbol_signals = await self._generate_symbol_signals(
                    strategy, symbol, timeframe, ohlcv_data
                )
                
                signals.extend(symbol_signals)
                
                # Update cooldown if signals were generated
                if symbol_signals:
                    self._update_cooldown(symbol)
            
            # Filter and validate signals
            signals = await self._filter_signals(signals)
            
            self.logger.info(f"Generated {len(signals)} signals")
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
        
        return signals
    
    async def _generate_symbol_signals(self, 
                                     strategy: BaseStrategy,
                                     symbol: str,
                                     timeframe: str,
                                     ohlcv_data: List[OHLCV]) -> List[StrategySignal]:
        """Generate signals for a specific symbol."""
        signals = []
        
        try:
            # Convert OHLCV data to DataFrame for easier processing
            df = self._ohlcv_to_dataframe(ohlcv_data)
            
            if len(df) < 50:  # Need minimum data for indicators
                return signals
            
            # Calculate technical indicators
            indicators = await self._calculate_indicators(df)
            
            # Generate signals using strategy
            strategy_signals = await strategy.generate_signals(
                symbol=symbol,
                timeframe=timeframe,
                data=df,
                indicators=indicators
            )
            
            # Convert strategy signals to StrategySignal objects
            for signal_data in strategy_signals:
                signal = StrategySignal(
                    symbol=symbol,
                    side=OrderSide.BUY if signal_data.get('side') == 'buy' else OrderSide.SELL,
                    price=signal_data.get('price', df['close'].iloc[-1]),
                    timestamp=datetime.now(),
                    strength=signal_data.get('strength', 0.5),
                    timeframe=timeframe,
                    strategy_name=strategy.name,
                    metadata=signal_data.get('metadata', {})
                )
                
                signals.append(signal)
            
        except Exception as e:
            self.logger.error(f"Error generating signals for {symbol}: {e}")
        
        return signals
    
    def _ohlcv_to_dataframe(self, ohlcv_data: List[OHLCV]) -> pd.DataFrame:
        """Convert OHLCV data to pandas DataFrame."""
        data = []
        for candle in ohlcv_data:
            data.append({
                'timestamp': candle.timestamp,
                'open': candle.open,
                'high': candle.high,
                'low': candle.low,
                'close': candle.close,
                'volume': candle.volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    async def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators for the data."""
        indicators = {}
        
        try:
            # Price-based indicators
            indicators['sma_20'] = self.indicators.sma(df['close'], 20)
            indicators['sma_50'] = self.indicators.sma(df['close'], 50)
            indicators['ema_12'] = self.indicators.ema(df['close'], 12)
            indicators['ema_26'] = self.indicators.ema(df['close'], 26)
            
            # Momentum indicators
            indicators['rsi'] = self.indicators.rsi(df['close'], 14)
            indicators['macd'] = self.indicators.macd(df['close'])
            indicators['stochastic'] = self.indicators.stochastic(df['high'], df['low'], df['close'])
            
            # Volatility indicators
            indicators['bollinger'] = self.indicators.bollinger_bands(df['close'], 20, 2)
            indicators['atr'] = self.indicators.atr(df['high'], df['low'], df['close'], 14)
            
            # Volume indicators
            indicators['obv'] = self.indicators.obv(df['close'], df['volume'])
            indicators['mfi'] = self.indicators.mfi(df['high'], df['low'], df['close'], df['volume'], 14)
            
            # Trend indicators
            indicators['adx'] = self.indicators.adx(df['high'], df['low'], df['close'], 14)
            indicators['parabolic_sar'] = self.indicators.parabolic_sar(df['high'], df['low'], df['close'])
            
            # Ichimoku Cloud
            ichimoku = self.indicators.ichimoku_cloud(df['high'], df['low'], df['close'])
            indicators.update(ichimoku)
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
        
        return indicators
    
    async def _filter_signals(self, signals: List[StrategySignal]) -> List[StrategySignal]:
        """Filter and validate signals."""
        filtered_signals = []
        
        for signal in signals:
            # Check signal strength
            if signal.strength < 0.3:  # Minimum strength threshold
                continue
            
            # Check for duplicate signals
            if self._is_duplicate_signal(signal):
                continue
            
            # Validate signal price
            if not await self._validate_signal_price(signal):
                continue
            
            # Add signal metadata
            signal.metadata.update({
                'generated_at': datetime.now().isoformat(),
                'signal_id': f"{signal.symbol}_{signal.side.value}_{int(datetime.now().timestamp())}"
            })
            
            filtered_signals.append(signal)
        
        return filtered_signals
    
    def _is_duplicate_signal(self, signal: StrategySignal) -> bool:
        """Check if signal is a duplicate of recent signals."""
        signal_key = f"{signal.symbol}_{signal.side.value}"
        
        if signal_key in self.last_signals:
            last_signal = self.last_signals[signal_key]
            time_diff = (signal.timestamp - last_signal.timestamp).total_seconds()
            
            # Consider duplicate if same side within cooldown period
            if time_diff < self.default_cooldown * 60:
                return True
        
        # Update last signal
        self.last_signals[signal_key] = signal
        return False
    
    async def _validate_signal_price(self, signal: StrategySignal) -> bool:
        """Validate signal price against current market price."""
        try:
            # Get current market price
            ticker = await self.exchange_adapter.get_ticker(signal.symbol)
            current_price = ticker.get('last', 0)
            
            if current_price == 0:
                return False
            
            # Check if signal price is within reasonable range (within 5% of current price)
            price_diff = abs(signal.price - current_price) / current_price
            if price_diff > 0.05:
                self.logger.warning(f"Signal price {signal.price} differs significantly from current price {current_price}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating signal price: {e}")
            return False
    
    def _is_in_cooldown(self, symbol: str) -> bool:
        """Check if symbol is in cooldown period."""
        if symbol in self.signal_cooldown:
            cooldown_end = self.signal_cooldown[symbol]
            if datetime.now() < cooldown_end:
                return True
        
        return False
    
    def _update_cooldown(self, symbol: str):
        """Update cooldown period for symbol."""
        cooldown_end = datetime.now() + timedelta(minutes=self.default_cooldown)
        self.signal_cooldown[symbol] = cooldown_end
    
    async def generate_manual_signal(self, 
                                   symbol: str,
                                   side: OrderSide,
                                   price: float,
                                   timeframe: str = "1h",
                                   strength: float = 1.0) -> StrategySignal:
        """
        Generate a manual trading signal.
        
        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            price: Signal price
            timeframe: Timeframe
            strength: Signal strength (0-1)
            
        Returns:
            Generated signal
        """
        signal = StrategySignal(
            symbol=symbol,
            side=side,
            price=price,
            timestamp=datetime.now(),
            strength=strength,
            timeframe=timeframe,
            strategy_name="manual",
            metadata={
                'manual': True,
                'generated_at': datetime.now().isoformat()
            }
        )
        
        self.logger.info(f"Generated manual signal: {symbol} {side.value} at {price}")
        return signal
    
    async def generate_grid_signals(self, 
                                  symbol: str,
                                  center_price: float,
                                  grid_levels: int,
                                  grid_spacing: float,
                                  side: OrderSide) -> List[StrategySignal]:
        """
        Generate grid trading signals.
        
        Args:
            symbol: Trading symbol
            center_price: Center price for grid
            grid_levels: Number of grid levels
            grid_spacing: Spacing between grid levels
            side: Order side (BUY/SELL)
            
        Returns:
            List of grid signals
        """
        signals = []
        
        for i in range(grid_levels):
            if side == OrderSide.BUY:
                price = center_price * (1 - grid_spacing * (i + 1))
            else:
                price = center_price * (1 + grid_spacing * (i + 1))
            
            signal = StrategySignal(
                symbol=symbol,
                side=side,
                price=price,
                timestamp=datetime.now(),
                strength=0.8,  # High strength for grid signals
                timeframe="1h",
                strategy_name="grid_bot",
                metadata={
                    'grid_level': i + 1,
                    'center_price': center_price,
                    'grid_spacing': grid_spacing,
                    'generated_at': datetime.now().isoformat()
                }
            )
            
            signals.append(signal)
        
        self.logger.info(f"Generated {len(signals)} grid signals for {symbol}")
        return signals
    
    async def generate_dca_signals(self, 
                                 symbol: str,
                                 interval_hours: int,
                                 position_size: float,
                                 max_positions: int) -> List[StrategySignal]:
        """
        Generate DCA (Dollar Cost Averaging) signals.
        
        Args:
            symbol: Trading symbol
            interval_hours: Interval between DCA purchases
            position_size: Size of each DCA position
            max_positions: Maximum number of positions
            
        Returns:
            List of DCA signals
        """
        signals = []
        
        # Get current price
        try:
            ticker = await self.exchange_adapter.get_ticker(symbol)
            current_price = ticker.get('last', 0)
            
            if current_price == 0:
                return signals
            
            # Generate DCA signals
            for i in range(max_positions):
                signal = StrategySignal(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    price=current_price,
                    timestamp=datetime.now() + timedelta(hours=i * interval_hours),
                    strength=0.9,  # High strength for DCA
                    timeframe="1h",
                    strategy_name="dca",
                    metadata={
                        'dca_level': i + 1,
                        'interval_hours': interval_hours,
                        'position_size': position_size,
                        'scheduled_time': (datetime.now() + timedelta(hours=i * interval_hours)).isoformat(),
                        'generated_at': datetime.now().isoformat()
                    }
                )
                
                signals.append(signal)
            
        except Exception as e:
            self.logger.error(f"Error generating DCA signals: {e}")
        
        self.logger.info(f"Generated {len(signals)} DCA signals for {symbol}")
        return signals
    
    def get_signal_statistics(self) -> Dict[str, Any]:
        """Get signal generation statistics."""
        return {
            'total_signals_generated': len(self.last_signals),
            'active_cooldowns': len(self.signal_cooldown),
            'default_cooldown_minutes': self.default_cooldown,
            'last_signals': {
                symbol: signal.timestamp.isoformat() 
                for symbol, signal in self.last_signals.items()
            }
        }
    
    def clear_cooldowns(self):
        """Clear all signal cooldowns."""
        self.signal_cooldown.clear()
        self.logger.info("All signal cooldowns cleared")
    
    def set_cooldown(self, symbol: str, minutes: int):
        """Set custom cooldown for a symbol."""
        cooldown_end = datetime.now() + timedelta(minutes=minutes)
        self.signal_cooldown[symbol] = cooldown_end
        self.logger.info(f"Set {minutes} minute cooldown for {symbol}")
