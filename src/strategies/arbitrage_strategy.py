"""
Arbitrage Strategy implementation.
"""

import pandas as pd
from typing import Dict, List, Optional, Any
from decimal import Decimal

from ..core.data_models import OHLCV, StrategySignal, OrderSide, TimeFrame
from .base_strategy import BaseStrategy


class ArbitrageStrategy(BaseStrategy):
    """Cross-exchange arbitrage detection strategy."""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize Arbitrage strategy.
        
        Args:
            parameters: Strategy parameters
        """
        default_params = {
            'arbitrage_threshold': 0.005,  # 0.5%
            'arbitrage_min_volume': 1000,  # USD
            'arbitrage_max_volume': 10000,  # USD
            'fee_rate': 0.001,  # 0.1%
            'min_profit_after_fees': 0.002,  # 0.2%
            'min_signal_strength': 0.6,
            'stop_loss_type': 'percentage',
            'stop_loss_percentage': 0.01,  # 1% stop loss for arbitrage
            'take_profit_ratio': 1.0,
            'max_arbitrage_duration': 300,  # 5 minutes
            'volume_confirmation': True,
            'price_stability_check': True
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__("Arbitrage", default_params)
        
        # Arbitrage state
        self.price_data = {}  # Store prices from different exchanges
        self.arbitrage_opportunities = []
        self.active_arbitrage = None
    
    def generate_signal(self, ohlcv_data: List[OHLCV], indicators: Dict[str, List[float]], 
                       patterns: Dict[str, Any]) -> Optional[StrategySignal]:
        """Generate arbitrage trading signal."""
        if len(ohlcv_data) < 2:
            return None
        
        current_price = ohlcv_data[-1].close
        symbol = ohlcv_data[-1].symbol
        
        # In a real implementation, you would fetch prices from multiple exchanges
        # For now, we'll simulate arbitrage opportunities
        arbitrage_opportunity = self._detect_arbitrage_opportunity(symbol, current_price)
        
        if not arbitrage_opportunity:
            return None
        
        signal_strength = 0.0
        signal_type = None
        
        # Determine signal based on arbitrage opportunity
        if arbitrage_opportunity['type'] == 'buy_low_sell_high':
            signal_type = OrderSide.BUY
            signal_strength = 0.7
            
            # Increase strength based on profit margin
            profit_margin = arbitrage_opportunity['profit_margin']
            if profit_margin > self.parameters['arbitrage_threshold'] * 2:
                signal_strength += 0.2
            
            # Increase strength if volume is sufficient
            if arbitrage_opportunity['volume'] > self.parameters['arbitrage_min_volume']:
                signal_strength += 0.1
        
        # Additional confirmation signals
        if signal_type:
            # Check for volume confirmation
            if self.parameters['volume_confirmation'] and self._check_volume_confirmation(ohlcv_data, signal_type):
                signal_strength += 0.1
            
            # Check for price stability
            if self.parameters['price_stability_check'] and self._check_price_stability(ohlcv_data):
                signal_strength += 0.1
            
            # Check for market conditions
            if self._check_market_conditions(ohlcv_data, indicators):
                signal_strength += 0.1
        
        # Only generate signal if strength meets minimum threshold
        if signal_type and signal_strength >= self.parameters['min_signal_strength']:
            return StrategySignal(
                symbol=symbol,
                signal=signal_type,
                strength=min(signal_strength, 1.0),
                price=current_price,
                strategy=self.name,
                timeframe=ohlcv_data[-1].timeframe,
                indicators={
                    'arbitrage_profit': arbitrage_opportunity['profit_margin'],
                    'arbitrage_volume': arbitrage_opportunity['volume'],
                    'price_difference': arbitrage_opportunity['price_difference'],
                    'fee_cost': arbitrage_opportunity['fee_cost']
                },
                metadata={
                    'arbitrage_threshold': self.parameters['arbitrage_threshold'],
                    'arbitrage_min_volume': self.parameters['arbitrage_min_volume'],
                    'arbitrage_type': arbitrage_opportunity['type'],
                    'exchange_low': arbitrage_opportunity.get('exchange_low', 'simulated'),
                    'exchange_high': arbitrage_opportunity.get('exchange_high', 'simulated')
                }
            )
        
        return None
    
    def _detect_arbitrage_opportunity(self, symbol: str, current_price: Decimal) -> Optional[Dict[str, Any]]:
        """Detect arbitrage opportunity."""
        # In a real implementation, this would fetch prices from multiple exchanges
        # For simulation, we'll create mock arbitrage opportunities
        
        # Simulate price differences between exchanges
        price_variation = 0.01  # 1% variation
        exchange_prices = {
            'exchange_a': float(current_price) * (1 + price_variation),
            'exchange_b': float(current_price) * (1 - price_variation),
            'exchange_c': float(current_price) * (1 + price_variation * 0.5)
        }
        
        # Find lowest and highest prices
        min_price = min(exchange_prices.values())
        max_price = max(exchange_prices.values())
        
        # Calculate price difference
        price_difference = (max_price - min_price) / min_price
        
        # Check if difference exceeds threshold
        if price_difference < self.parameters['arbitrage_threshold']:
            return None
        
        # Calculate profit after fees
        fee_cost = self.parameters['fee_rate'] * 2  # Buy and sell fees
        profit_margin = price_difference - fee_cost
        
        # Check if profit is sufficient
        if profit_margin < self.parameters['min_profit_after_fees']:
            return None
        
        # Simulate volume
        volume = self.parameters['arbitrage_min_volume'] * 2
        
        return {
            'type': 'buy_low_sell_high',
            'price_difference': price_difference,
            'profit_margin': profit_margin,
            'volume': volume,
            'fee_cost': fee_cost,
            'exchange_low': min(exchange_prices, key=exchange_prices.get),
            'exchange_high': max(exchange_prices, key=exchange_prices.get),
            'low_price': min_price,
            'high_price': max_price
        }
    
    def _check_volume_confirmation(self, ohlcv_data: List[OHLCV], signal_type: OrderSide) -> bool:
        """Check for volume confirmation."""
        if len(ohlcv_data) < 5:
            return True  # No volume filter if not enough data
        
        # Calculate average volume
        recent_volumes = [float(candle.volume) for candle in ohlcv_data[-5:]]
        avg_volume = sum(recent_volumes) / len(recent_volumes)
        current_volume = float(ohlcv_data[-1].volume)
        
        # Volume should be sufficient for arbitrage
        return current_volume > avg_volume * 0.8
    
    def _check_price_stability(self, ohlcv_data: List[OHLCV]) -> bool:
        """Check for price stability."""
        if len(ohlcv_data) < 10:
            return True  # No stability check if not enough data
        
        # Calculate price volatility
        recent_prices = [float(candle.close) for candle in ohlcv_data[-10:]]
        price_changes = [abs(recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1] 
                        for i in range(1, len(recent_prices))]
        
        avg_volatility = sum(price_changes) / len(price_changes)
        
        # Price should be relatively stable for arbitrage
        return avg_volatility < 0.02  # Less than 2% average volatility
    
    def _check_market_conditions(self, ohlcv_data: List[OHLCV], indicators: Dict[str, List[float]]) -> bool:
        """Check market conditions for arbitrage."""
        # Check for low volatility (good for arbitrage)
        atr_values = indicators.get('atr', [])
        if atr_values and len(atr_values) > 0:
            current_atr = atr_values[-1]
            current_price = float(ohlcv_data[-1].close)
            
            # ATR should be relatively low compared to price
            atr_ratio = current_atr / current_price
            if atr_ratio > 0.05:  # More than 5% ATR
                return False
        
        # Check for trend strength (arbitrage works better in sideways markets)
        ema_20 = indicators.get('sma_20', [])
        ema_50 = indicators.get('sma_50', [])
        
        if ema_20 and ema_50 and len(ema_20) > 0 and len(ema_50) > 0:
            ema_diff = abs(ema_20[-1] - ema_50[-1]) / ema_50[-1]
            if ema_diff > 0.03:  # More than 3% difference between EMAs
                return False
        
        return True
    
    def calculate_arbitrage_profit(self, amount: Decimal, buy_price: Decimal, sell_price: Decimal) -> Decimal:
        """Calculate arbitrage profit after fees."""
        fee_rate = Decimal(str(self.parameters['fee_rate']))
        
        # Buy cost
        buy_cost = amount * buy_price * (1 + fee_rate)
        
        # Sell proceeds
        sell_proceeds = amount * sell_price * (1 - fee_rate)
        
        # Profit
        profit = sell_proceeds - buy_cost
        
        return profit
    
    def get_arbitrage_status(self) -> Dict[str, Any]:
        """Get current arbitrage status."""
        return {
            'active_arbitrage': self.active_arbitrage,
            'arbitrage_opportunities': len(self.arbitrage_opportunities),
            'arbitrage_threshold': self.parameters['arbitrage_threshold'],
            'min_profit_after_fees': self.parameters['min_profit_after_fees'],
            'fee_rate': self.parameters['fee_rate']
        }
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        return self.parameters.copy()
    
    def set_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Set strategy parameters."""
        if self.validate_parameters(parameters):
            self.parameters.update(parameters)
            return True
        return False
    
    def get_required_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get required parameters for the strategy."""
        return {
            'arbitrage_threshold': {'type': float, 'min': 0.001, 'max': 0.02, 'default': 0.005},
            'arbitrage_min_volume': {'type': float, 'min': 100, 'max': 50000, 'default': 1000},
            'arbitrage_max_volume': {'type': float, 'min': 1000, 'max': 100000, 'default': 10000},
            'fee_rate': {'type': float, 'min': 0.0001, 'max': 0.01, 'default': 0.001},
            'min_profit_after_fees': {'type': float, 'min': 0.0001, 'max': 0.01, 'default': 0.002},
            'min_signal_strength': {'type': float, 'min': 0.1, 'max': 1.0, 'default': 0.6},
            'stop_loss_type': {'type': str, 'options': ['percentage', 'atr'], 'default': 'percentage'},
            'stop_loss_percentage': {'type': float, 'min': 0.001, 'max': 0.05, 'default': 0.01},
            'take_profit_ratio': {'type': float, 'min': 0.5, 'max': 3.0, 'default': 1.0},
            'max_arbitrage_duration': {'type': int, 'min': 60, 'max': 3600, 'default': 300},
            'volume_confirmation': {'type': bool, 'default': True},
            'price_stability_check': {'type': bool, 'default': True}
        }
    
    def get_description(self) -> str:
        """Get strategy description."""
        return ("Arbitrage Strategy: Detects price differences between exchanges for risk-free profit. "
                "Buy on exchange with lower price, sell on exchange with higher price. "
                "Requires sufficient volume and price stability for execution.")
