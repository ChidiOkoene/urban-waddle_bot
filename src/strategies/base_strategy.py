"""
Base strategy class for all trading strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal
from datetime import datetime

from ..core.data_models import OHLCV, StrategySignal, OrderSide, TimeFrame
from ..indicators.technical_indicators import TechnicalIndicators
from ..indicators.pattern_recognition import PatternRecognition


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""
    
    def __init__(self, name: str, parameters: Dict[str, Any]):
        """
        Initialize strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters
        """
        self.name = name
        self.parameters = parameters
        self.indicators = TechnicalIndicators()
        self.patterns = PatternRecognition()
        
        # Strategy state
        self.last_signal = None
        self.last_signal_time = None
        self.position_size = Decimal("0")
        self.entry_price = Decimal("0")
        self.stop_loss = Decimal("0")
        self.take_profit = Decimal("0")
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = Decimal("0")
    
    @abstractmethod
    def generate_signal(self, ohlcv_data: List[OHLCV], indicators: Dict[str, List[float]], 
                       patterns: Dict[str, Any]) -> Optional[StrategySignal]:
        """
        Generate trading signal based on OHLCV data and indicators.
        
        Args:
            ohlcv_data: List of OHLCV candles
            indicators: Dictionary of calculated indicators
            patterns: Dictionary of detected patterns
            
        Returns:
            StrategySignal or None if no signal
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        pass
    
    @abstractmethod
    def set_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Set strategy parameters."""
        pass
    
    def calculate_indicators(self, ohlcv_data: List[OHLCV]) -> Dict[str, List[float]]:
        """Calculate all indicators for OHLCV data."""
        return self.indicators.calculate_all_indicators(ohlcv_data)
    
    def detect_patterns(self, ohlcv_data: List[OHLCV]) -> Dict[str, Any]:
        """Detect all patterns for OHLCV data."""
        return self.patterns.detect_all_patterns(ohlcv_data)
    
    def analyze(self, ohlcv_data: List[OHLCV]) -> Optional[StrategySignal]:
        """
        Analyze OHLCV data and generate signal.
        
        Args:
            ohlcv_data: List of OHLCV candles
            
        Returns:
            StrategySignal or None if no signal
        """
        if not ohlcv_data:
            return None
        
        # Calculate indicators
        indicators = self.calculate_indicators(ohlcv_data)
        
        # Detect patterns
        patterns = self.detect_patterns(ohlcv_data)
        
        # Generate signal
        signal = self.generate_signal(ohlcv_data, indicators, patterns)
        
        if signal:
            self.last_signal = signal
            self.last_signal_time = datetime.utcnow()
        
        return signal
    
    def get_signal_strength(self, signal: StrategySignal) -> float:
        """Calculate signal strength (0-1)."""
        # Base strength from signal
        base_strength = signal.strength
        
        # Adjust based on indicators
        if 'rsi' in signal.indicators:
            rsi = signal.indicators['rsi']
            if signal.signal == OrderSide.BUY and rsi < 30:
                base_strength += 0.2
            elif signal.signal == OrderSide.SELL and rsi > 70:
                base_strength += 0.2
        
        if 'macd' in signal.indicators and 'macd_signal' in signal.indicators:
            macd = signal.indicators['macd']
            macd_signal = signal.indicators['macd_signal']
            if signal.signal == OrderSide.BUY and macd > macd_signal:
                base_strength += 0.1
            elif signal.signal == OrderSide.SELL and macd < macd_signal:
                base_strength += 0.1
        
        return min(max(base_strength, 0.0), 1.0)
    
    def calculate_position_size(self, account_balance: Decimal, risk_per_trade: float, 
                              current_price: Decimal, stop_loss_price: Decimal) -> Decimal:
        """
        Calculate position size based on risk management.
        
        Args:
            account_balance: Current account balance
            risk_per_trade: Risk percentage per trade (0-1)
            current_price: Current price
            stop_loss_price: Stop loss price
            
        Returns:
            Position size
        """
        if stop_loss_price == 0 or current_price == 0:
            return Decimal("0")
        
        # Calculate risk amount
        risk_amount = account_balance * Decimal(str(risk_per_trade))
        
        # Calculate price difference
        price_diff = abs(current_price - stop_loss_price)
        
        # Calculate position size
        position_size = risk_amount / price_diff
        
        return position_size
    
    def calculate_stop_loss(self, entry_price: Decimal, signal: StrategySignal, 
                          atr: Optional[float] = None) -> Decimal:
        """
        Calculate stop loss price.
        
        Args:
            entry_price: Entry price
            signal: Trading signal
            atr: Average True Range value
            
        Returns:
            Stop loss price
        """
        stop_loss_type = self.parameters.get('stop_loss_type', 'percentage')
        stop_loss_value = self.parameters.get('stop_loss_percentage', 0.02)
        
        if stop_loss_type == 'atr' and atr:
            atr_multiplier = self.parameters.get('stop_loss_atr_multiplier', 2.0)
            if signal.signal == OrderSide.BUY:
                return entry_price - Decimal(str(atr * atr_multiplier))
            else:
                return entry_price + Decimal(str(atr * atr_multiplier))
        else:
            # Percentage-based stop loss
            if signal.signal == OrderSide.BUY:
                return entry_price * (1 - Decimal(str(stop_loss_value)))
            else:
                return entry_price * (1 + Decimal(str(stop_loss_value)))
    
    def calculate_take_profit(self, entry_price: Decimal, signal: StrategySignal, 
                            stop_loss_price: Decimal) -> Decimal:
        """
        Calculate take profit price.
        
        Args:
            entry_price: Entry price
            signal: Trading signal
            stop_loss_price: Stop loss price
            
        Returns:
            Take profit price
        """
        risk_reward_ratio = self.parameters.get('take_profit_ratio', 2.0)
        
        risk = abs(entry_price - stop_loss_price)
        reward = risk * Decimal(str(risk_reward_ratio))
        
        if signal.signal == OrderSide.BUY:
            return entry_price + reward
        else:
            return entry_price - reward
    
    def update_performance(self, pnl: Decimal):
        """Update strategy performance metrics."""
        self.total_trades += 1
        self.total_pnl += pnl
        
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get strategy performance metrics."""
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'total_pnl': float(self.total_pnl),
            'last_signal': self.last_signal.signal.value if self.last_signal else None,
            'last_signal_time': self.last_signal_time
        }
    
    def reset_performance(self):
        """Reset performance metrics."""
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = Decimal("0")
        self.last_signal = None
        self.last_signal_time = None
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate strategy parameters."""
        required_params = self.get_required_parameters()
        
        for param in required_params:
            if param not in parameters:
                return False
            
            # Type validation
            param_type = required_params[param]['type']
            if not isinstance(parameters[param], param_type):
                return False
            
            # Range validation
            if 'min' in required_params[param]:
                if parameters[param] < required_params[param]['min']:
                    return False
            
            if 'max' in required_params[param]:
                if parameters[param] > required_params[param]['max']:
                    return False
        
        return True
    
    @abstractmethod
    def get_required_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get required parameters for the strategy."""
        pass
    
    def get_description(self) -> str:
        """Get strategy description."""
        return f"{self.name} strategy"
    
    def __str__(self) -> str:
        """String representation of strategy."""
        return f"{self.name}({self.parameters})"
    
    def __repr__(self) -> str:
        """Detailed string representation of strategy."""
        return f"{self.__class__.__name__}(name='{self.name}', parameters={self.parameters})"
