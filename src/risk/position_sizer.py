"""
Position sizing implementation with multiple algorithms.
"""

from typing import Dict, List, Optional, Any
from decimal import Decimal
from enum import Enum
import math

from ..core.data_models import Balance, Position, OrderSide


class PositionSizingMethod(str, Enum):
    """Position sizing methods."""
    FIXED_FRACTIONAL = "fixed_fractional"
    KELLY_CRITERION = "kelly"
    VOLATILITY_BASED = "volatility_based"
    RISK_PARITY = "risk_parity"
    EQUAL_WEIGHT = "equal_weight"


class PositionSizer:
    """Position sizing calculator with multiple algorithms."""
    
    def __init__(self, method: PositionSizingMethod = PositionSizingMethod.FIXED_FRACTIONAL):
        """
        Initialize position sizer.
        
        Args:
            method: Position sizing method to use
        """
        self.method = method
        self.risk_per_trade = Decimal("0.02")  # 2% default
        self.max_position_size = Decimal("0.1")  # 10% max position
        self.min_position_size = Decimal("0.001")  # 0.1% min position
        
        # Kelly Criterion parameters
        self.kelly_lookback = 100  # Number of trades for Kelly calculation
        self.kelly_fraction = Decimal("0.25")  # Fraction of Kelly to use
        
        # Volatility-based parameters
        self.volatility_lookback = 20  # Days for volatility calculation
        self.target_volatility = Decimal("0.02")  # 2% target volatility
        
        # Risk parity parameters
        self.risk_parity_target = Decimal("0.01")  # 1% risk per position
    
    def calculate_position_size(self, 
                              account_balance: Decimal,
                              current_price: Decimal,
                              stop_loss_price: Decimal,
                              signal_strength: float = 1.0,
                              volatility: Optional[float] = None,
                              trade_history: Optional[List[Dict]] = None,
                              existing_positions: Optional[List[Position]] = None) -> Decimal:
        """
        Calculate position size based on selected method.
        
        Args:
            account_balance: Current account balance
            current_price: Current price of the asset
            stop_loss_price: Stop loss price
            signal_strength: Signal strength (0-1)
            volatility: Asset volatility (for volatility-based sizing)
            trade_history: Historical trade data (for Kelly Criterion)
            existing_positions: Current open positions
            
        Returns:
            Position size in base currency
        """
        if self.method == PositionSizingMethod.FIXED_FRACTIONAL:
            return self._fixed_fractional_sizing(account_balance, current_price, stop_loss_price, signal_strength)
        
        elif self.method == PositionSizingMethod.KELLY_CRITERION:
            return self._kelly_criterion_sizing(account_balance, current_price, stop_loss_price, 
                                              signal_strength, trade_history)
        
        elif self.method == PositionSizingMethod.VOLATILITY_BASED:
            return self._volatility_based_sizing(account_balance, current_price, stop_loss_price, 
                                               signal_strength, volatility)
        
        elif self.method == PositionSizingMethod.RISK_PARITY:
            return self._risk_parity_sizing(account_balance, current_price, stop_loss_price, 
                                         signal_strength, existing_positions)
        
        elif self.method == PositionSizingMethod.EQUAL_WEIGHT:
            return self._equal_weight_sizing(account_balance, current_price, stop_loss_price, 
                                           signal_strength, existing_positions)
        
        else:
            raise ValueError(f"Unknown position sizing method: {self.method}")
    
    def _fixed_fractional_sizing(self, account_balance: Decimal, current_price: Decimal, 
                                stop_loss_price: Decimal, signal_strength: float) -> Decimal:
        """Fixed fractional position sizing."""
        if stop_loss_price == 0 or current_price == 0:
            return Decimal("0")
        
        # Calculate risk amount
        risk_amount = account_balance * self.risk_per_trade
        
        # Adjust risk based on signal strength
        adjusted_risk = risk_amount * Decimal(str(signal_strength))
        
        # Calculate price difference
        price_diff = abs(current_price - stop_loss_price)
        
        # Calculate position size
        position_size = adjusted_risk / price_diff
        
        # Apply position size limits
        max_position_value = account_balance * self.max_position_size
        max_position_size = max_position_value / current_price
        
        position_size = min(position_size, max_position_size)
        position_size = max(position_size, self.min_position_size)
        
        return position_size
    
    def _kelly_criterion_sizing(self, account_balance: Decimal, current_price: Decimal, 
                               stop_loss_price: Decimal, signal_strength: float, 
                               trade_history: Optional[List[Dict]]) -> Decimal:
        """Kelly Criterion position sizing."""
        if not trade_history or len(trade_history) < 10:
            # Fallback to fixed fractional if insufficient history
            return self._fixed_fractional_sizing(account_balance, current_price, stop_loss_price, signal_strength)
        
        # Calculate Kelly fraction
        kelly_fraction = self._calculate_kelly_fraction(trade_history)
        
        if kelly_fraction <= 0:
            return Decimal("0")
        
        # Apply Kelly fraction and signal strength
        adjusted_kelly = kelly_fraction * self.kelly_fraction * Decimal(str(signal_strength))
        
        # Calculate position size
        risk_amount = account_balance * adjusted_kelly
        price_diff = abs(current_price - stop_loss_price)
        
        if price_diff == 0:
            return Decimal("0")
        
        position_size = risk_amount / price_diff
        
        # Apply position size limits
        max_position_value = account_balance * self.max_position_size
        max_position_size = max_position_value / current_price
        
        position_size = min(position_size, max_position_size)
        position_size = max(position_size, self.min_position_size)
        
        return position_size
    
    def _volatility_based_sizing(self, account_balance: Decimal, current_price: Decimal, 
                               stop_loss_price: Decimal, signal_strength: float, 
                               volatility: Optional[float]) -> Decimal:
        """Volatility-based position sizing."""
        if volatility is None or volatility <= 0:
            # Fallback to fixed fractional if no volatility data
            return self._fixed_fractional_sizing(account_balance, current_price, stop_loss_price, signal_strength)
        
        # Calculate volatility-adjusted position size
        volatility_ratio = self.target_volatility / Decimal(str(volatility))
        
        # Base position size
        base_size = self._fixed_fractional_sizing(account_balance, current_price, stop_loss_price, signal_strength)
        
        # Adjust for volatility
        adjusted_size = base_size * volatility_ratio
        
        # Apply position size limits
        max_position_value = account_balance * self.max_position_size
        max_position_size = max_position_value / current_price
        
        adjusted_size = min(adjusted_size, max_position_size)
        adjusted_size = max(adjusted_size, self.min_position_size)
        
        return adjusted_size
    
    def _risk_parity_sizing(self, account_balance: Decimal, current_price: Decimal, 
                          stop_loss_price: Decimal, signal_strength: float, 
                          existing_positions: Optional[List[Position]]) -> Decimal:
        """Risk parity position sizing."""
        if not existing_positions:
            # First position - use fixed fractional
            return self._fixed_fractional_sizing(account_balance, current_price, stop_loss_price, signal_strength)
        
        # Calculate current risk exposure
        current_risk = sum(abs(float(pos.unrealized_pnl)) for pos in existing_positions)
        
        # Calculate target risk per position
        num_positions = len(existing_positions) + 1
        target_risk_per_position = account_balance * self.risk_parity_target
        
        # Calculate position size to achieve target risk
        price_diff = abs(current_price - stop_loss_price)
        if price_diff == 0:
            return Decimal("0")
        
        position_size = target_risk_per_position / price_diff
        
        # Apply position size limits
        max_position_value = account_balance * self.max_position_size
        max_position_size = max_position_value / current_price
        
        position_size = min(position_size, max_position_size)
        position_size = max(position_size, self.min_position_size)
        
        return position_size
    
    def _equal_weight_sizing(self, account_balance: Decimal, current_price: Decimal, 
                           stop_loss_price: Decimal, signal_strength: float, 
                           existing_positions: Optional[List[Position]]) -> Decimal:
        """Equal weight position sizing."""
        # Calculate number of positions (including new one)
        num_positions = len(existing_positions) + 1 if existing_positions else 1
        
        # Calculate equal weight allocation
        equal_weight = Decimal("1") / Decimal(str(num_positions))
        
        # Calculate position size
        position_value = account_balance * equal_weight * Decimal(str(signal_strength))
        position_size = position_value / current_price
        
        # Apply position size limits
        max_position_value = account_balance * self.max_position_size
        max_position_size = max_position_value / current_price
        
        position_size = min(position_size, max_position_size)
        position_size = max(position_size, self.min_position_size)
        
        return position_size
    
    def _calculate_kelly_fraction(self, trade_history: List[Dict]) -> Decimal:
        """Calculate Kelly fraction from trade history."""
        if len(trade_history) < 10:
            return Decimal("0")
        
        # Extract returns from trade history
        returns = []
        for trade in trade_history[-self.kelly_lookback:]:
            if 'return' in trade:
                returns.append(trade['return'])
            elif 'pnl' in trade and 'entry_price' in trade:
                # Calculate return from PnL and entry price
                pnl = trade['pnl']
                entry_price = trade['entry_price']
                if entry_price != 0:
                    returns.append(pnl / entry_price)
        
        if len(returns) < 5:
            return Decimal("0")
        
        # Calculate win rate and average win/loss
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r < 0]
        
        if not wins or not losses:
            return Decimal("0")
        
        win_rate = len(wins) / len(returns)
        avg_win = sum(wins) / len(wins)
        avg_loss = abs(sum(losses) / len(losses))
        
        if avg_loss == 0:
            return Decimal("0")
        
        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        
        # Ensure Kelly fraction is positive and reasonable
        kelly_fraction = max(Decimal("0"), min(kelly_fraction, Decimal("0.5")))
        
        return kelly_fraction
    
    def set_risk_per_trade(self, risk_per_trade: float):
        """Set risk per trade percentage."""
        self.risk_per_trade = Decimal(str(risk_per_trade))
    
    def set_position_limits(self, min_size: float, max_size: float):
        """Set position size limits."""
        self.min_position_size = Decimal(str(min_size))
        self.max_position_size = Decimal(str(max_size))
    
    def set_kelly_parameters(self, lookback: int, fraction: float):
        """Set Kelly Criterion parameters."""
        self.kelly_lookback = lookback
        self.kelly_fraction = Decimal(str(fraction))
    
    def set_volatility_parameters(self, lookback: int, target_volatility: float):
        """Set volatility-based parameters."""
        self.volatility_lookback = lookback
        self.target_volatility = Decimal(str(target_volatility))
    
    def get_position_size_info(self, account_balance: Decimal, current_price: Decimal, 
                              stop_loss_price: Decimal, signal_strength: float = 1.0) -> Dict[str, Any]:
        """Get detailed position sizing information."""
        position_size = self.calculate_position_size(
            account_balance, current_price, stop_loss_price, signal_strength
        )
        
        position_value = position_size * current_price
        risk_amount = position_size * abs(current_price - stop_loss_price)
        risk_percentage = (risk_amount / account_balance) * 100
        
        return {
            'position_size': float(position_size),
            'position_value': float(position_value),
            'risk_amount': float(risk_amount),
            'risk_percentage': float(risk_percentage),
            'method': self.method.value,
            'signal_strength': signal_strength,
            'account_balance': float(account_balance),
            'current_price': float(current_price),
            'stop_loss_price': float(stop_loss_price)
        }
