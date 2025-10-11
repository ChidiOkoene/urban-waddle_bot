"""
Risk management coordinator.
"""

from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
from datetime import datetime, timedelta
from enum import Enum

from ..core.data_models import Balance, Position, Order, OrderSide, OrderType
from .position_sizer import PositionSizer, PositionSizingMethod


class RiskLevel(str, Enum):
    """Risk levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskManager:
    """Overall risk management coordinator."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize risk manager.
        
        Args:
            config: Risk management configuration
        """
        self.config = config
        
        # Risk limits
        self.max_risk_per_trade = Decimal(str(config.get('max_risk_per_trade', 0.02)))  # 2%
        self.max_total_risk = Decimal(str(config.get('max_total_risk', 0.10)))  # 10%
        self.max_positions = config.get('max_positions', 5)
        self.max_drawdown = Decimal(str(config.get('max_drawdown', 0.15)))  # 15%
        self.correlation_limit = config.get('correlation_limit', 0.7)  # 70%
        
        # Position sizing
        position_sizing_method = config.get('position_sizing', 'fixed_fractional')
        self.position_sizer = PositionSizer(PositionSizingMethod(position_sizing_method))
        
        # Risk tracking
        self.daily_pnl = Decimal("0")
        self.total_pnl = Decimal("0")
        self.max_equity = Decimal("0")
        self.current_drawdown = Decimal("0")
        self.risk_exposure = Decimal("0")
        
        # Trade history for risk calculations
        self.trade_history = []
        self.daily_trades = []
        
        # Risk alerts
        self.risk_alerts = []
        
        # Emergency stop
        self.emergency_stop = False
        self.emergency_stop_reason = None
    
    def check_trade_risk(self, 
                        signal_strength: float,
                        current_price: Decimal,
                        stop_loss_price: Decimal,
                        account_balance: Decimal,
                        existing_positions: List[Position],
                        symbol: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Check if a trade meets risk management criteria.
        
        Args:
            signal_strength: Signal strength (0-1)
            current_price: Current price
            stop_loss_price: Stop loss price
            account_balance: Current account balance
            existing_positions: Current open positions
            symbol: Trading symbol
            
        Returns:
            Tuple of (allowed, reason, risk_info)
        """
        risk_info = {}
        
        # Check emergency stop
        if self.emergency_stop:
            return False, f"Emergency stop active: {self.emergency_stop_reason}", risk_info
        
        # Check maximum positions
        if len(existing_positions) >= self.max_positions:
            return False, f"Maximum positions limit reached ({self.max_positions})", risk_info
        
        # Check drawdown
        if self.current_drawdown > self.max_drawdown:
            return False, f"Maximum drawdown exceeded ({self.current_drawdown:.2%})", risk_info
        
        # Check daily loss limit
        daily_loss_limit = account_balance * Decimal("0.05")  # 5% daily loss limit
        if self.daily_pnl < -daily_loss_limit:
            return False, f"Daily loss limit exceeded ({self.daily_pnl:.2f})", risk_info
        
        # Calculate position size
        position_size = self.position_sizer.calculate_position_size(
            account_balance=account_balance,
            current_price=current_price,
            stop_loss_price=stop_loss_price,
            signal_strength=signal_strength,
            existing_positions=existing_positions
        )
        
        if position_size <= 0:
            return False, "Position size too small", risk_info
        
        # Calculate risk for this trade
        trade_risk = position_size * abs(current_price - stop_loss_price)
        trade_risk_percentage = trade_risk / account_balance
        
        risk_info['position_size'] = float(position_size)
        risk_info['trade_risk'] = float(trade_risk)
        risk_info['trade_risk_percentage'] = float(trade_risk_percentage)
        
        # Check individual trade risk
        if trade_risk_percentage > self.max_risk_per_trade:
            return False, f"Trade risk too high ({trade_risk_percentage:.2%})", risk_info
        
        # Check total risk exposure
        total_risk = self._calculate_total_risk(existing_positions, account_balance)
        new_total_risk = total_risk + trade_risk_percentage
        
        risk_info['total_risk'] = float(total_risk)
        risk_info['new_total_risk'] = float(new_total_risk)
        
        if new_total_risk > self.max_total_risk:
            return False, f"Total risk exposure too high ({new_total_risk:.2%})", risk_info
        
        # Check correlation risk
        if self._check_correlation_risk(symbol, existing_positions):
            return False, f"Correlation risk too high for {symbol}", risk_info
        
        # Check concentration risk
        if self._check_concentration_risk(symbol, existing_positions, position_size, account_balance):
            return False, f"Concentration risk too high for {symbol}", risk_info
        
        # Check volatility risk
        volatility_risk = self._check_volatility_risk(symbol, existing_positions)
        if volatility_risk > 0.1:  # 10% volatility risk limit
            return False, f"Volatility risk too high ({volatility_risk:.2%})", risk_info
        
        risk_info['volatility_risk'] = float(volatility_risk)
        
        return True, "Risk check passed", risk_info
    
    def calculate_position_size(self, 
                               account_balance: Decimal,
                               current_price: Decimal,
                               stop_loss_price: Decimal,
                               signal_strength: float = 1.0,
                               existing_positions: Optional[List[Position]] = None) -> Decimal:
        """Calculate position size using position sizer."""
        return self.position_sizer.calculate_position_size(
            account_balance=account_balance,
            current_price=current_price,
            stop_loss_price=stop_loss_price,
            signal_strength=signal_strength,
            existing_positions=existing_positions or []
        )
    
    def update_pnl(self, pnl: Decimal, trade_info: Optional[Dict] = None):
        """Update PnL and risk metrics."""
        self.daily_pnl += pnl
        self.total_pnl += pnl
        
        # Update maximum equity
        if self.total_pnl > self.max_equity:
            self.max_equity = self.total_pnl
        
        # Calculate current drawdown
        if self.max_equity > 0:
            self.current_drawdown = (self.max_equity - self.total_pnl) / self.max_equity
        
        # Record trade
        if trade_info:
            trade_info['pnl'] = float(pnl)
            trade_info['timestamp'] = datetime.utcnow()
            self.trade_history.append(trade_info)
            self.daily_trades.append(trade_info)
        
        # Check for risk alerts
        self._check_risk_alerts()
    
    def reset_daily_metrics(self):
        """Reset daily metrics."""
        self.daily_pnl = Decimal("0")
        self.daily_trades = []
    
    def _calculate_total_risk(self, positions: List[Position], account_balance: Decimal) -> Decimal:
        """Calculate total risk exposure."""
        if not positions:
            return Decimal("0")
        
        total_risk = Decimal("0")
        for position in positions:
            # Calculate risk as unrealized PnL potential
            position_risk = abs(float(position.unrealized_pnl))
            total_risk += Decimal(str(position_risk))
        
        return total_risk / account_balance
    
    def _check_correlation_risk(self, symbol: str, positions: List[Position]) -> bool:
        """Check correlation risk."""
        if len(positions) < 2:
            return False
        
        # Simple correlation check based on symbol similarity
        # In practice, you would calculate actual correlation coefficients
        similar_symbols = 0
        for position in positions:
            if self._are_symbols_correlated(symbol, position.symbol):
                similar_symbols += 1
        
        correlation_ratio = similar_symbols / len(positions)
        return correlation_ratio > self.correlation_limit
    
    def _are_symbols_correlated(self, symbol1: str, symbol2: str) -> bool:
        """Check if two symbols are correlated."""
        # Simple heuristic: same base currency or same quote currency
        if '/' in symbol1 and '/' in symbol2:
            base1, quote1 = symbol1.split('/')
            base2, quote2 = symbol2.split('/')
            
            # Same base currency
            if base1 == base2:
                return True
            
            # Same quote currency
            if quote1 == quote2:
                return True
        
        return False
    
    def _check_concentration_risk(self, symbol: str, positions: List[Position], 
                                position_size: Decimal, account_balance: Decimal) -> bool:
        """Check concentration risk."""
        if not positions:
            return False
        
        # Calculate position value
        position_value = position_size * account_balance  # Simplified
        
        # Calculate total position value
        total_position_value = sum(abs(float(pos.amount)) for pos in positions)
        
        # Check if new position would create too much concentration
        concentration_ratio = position_value / (total_position_value + position_value)
        return concentration_ratio > 0.3  # 30% concentration limit
    
    def _check_volatility_risk(self, symbol: str, positions: List[Position]) -> float:
        """Check volatility risk."""
        # Simplified volatility risk calculation
        # In practice, you would use actual volatility data
        
        if not positions:
            return 0.0
        
        # Calculate portfolio volatility (simplified)
        total_exposure = sum(abs(float(pos.unrealized_pnl)) for pos in positions)
        
        # Assume 2% daily volatility per position
        portfolio_volatility = len(positions) * 0.02
        
        return min(portfolio_volatility, 0.2)  # Cap at 20%
    
    def _check_risk_alerts(self):
        """Check for risk alerts."""
        alerts = []
        
        # Drawdown alert
        if self.current_drawdown > self.max_drawdown * Decimal("0.8"):  # 80% of max drawdown
            alerts.append({
                'type': 'drawdown_warning',
                'level': RiskLevel.HIGH if self.current_drawdown > self.max_drawdown * Decimal("0.9") else RiskLevel.MEDIUM,
                'message': f"Drawdown approaching limit: {self.current_drawdown:.2%}",
                'timestamp': datetime.utcnow()
            })
        
        # Daily loss alert
        if self.daily_pnl < Decimal("-1000"):  # $1000 daily loss
            alerts.append({
                'type': 'daily_loss',
                'level': RiskLevel.HIGH,
                'message': f"Significant daily loss: {self.daily_pnl:.2f}",
                'timestamp': datetime.utcnow()
            })
        
        # Position count alert
        if len(self.trade_history) > self.max_positions * 2:  # Too many trades
            alerts.append({
                'type': 'overtrading',
                'level': RiskLevel.MEDIUM,
                'message': f"Potential overtrading: {len(self.trade_history)} trades",
                'timestamp': datetime.utcnow()
            })
        
        self.risk_alerts.extend(alerts)
    
    def get_risk_status(self) -> Dict[str, Any]:
        """Get current risk status."""
        return {
            'emergency_stop': self.emergency_stop,
            'emergency_stop_reason': self.emergency_stop_reason,
            'daily_pnl': float(self.daily_pnl),
            'total_pnl': float(self.total_pnl),
            'current_drawdown': float(self.current_drawdown),
            'max_drawdown': float(self.max_drawdown),
            'risk_exposure': float(self.risk_exposure),
            'max_positions': self.max_positions,
            'max_risk_per_trade': float(self.max_risk_per_trade),
            'max_total_risk': float(self.max_total_risk),
            'correlation_limit': self.correlation_limit,
            'position_sizing_method': self.position_sizer.method.value,
            'risk_alerts': self.risk_alerts[-10:],  # Last 10 alerts
            'trade_count': len(self.trade_history),
            'daily_trade_count': len(self.daily_trades)
        }
    
    def set_emergency_stop(self, reason: str):
        """Set emergency stop."""
        self.emergency_stop = True
        self.emergency_stop_reason = reason
        
        # Add emergency stop alert
        self.risk_alerts.append({
            'type': 'emergency_stop',
            'level': RiskLevel.CRITICAL,
            'message': f"Emergency stop activated: {reason}",
            'timestamp': datetime.utcnow()
        })
    
    def clear_emergency_stop(self):
        """Clear emergency stop."""
        self.emergency_stop = False
        self.emergency_stop_reason = None
        
        # Add emergency stop cleared alert
        self.risk_alerts.append({
            'type': 'emergency_stop_cleared',
            'level': RiskLevel.LOW,
            'message': "Emergency stop cleared",
            'timestamp': datetime.utcnow()
        })
    
    def update_risk_limits(self, limits: Dict[str, Any]):
        """Update risk limits."""
        if 'max_risk_per_trade' in limits:
            self.max_risk_per_trade = Decimal(str(limits['max_risk_per_trade']))
        
        if 'max_total_risk' in limits:
            self.max_total_risk = Decimal(str(limits['max_total_risk']))
        
        if 'max_positions' in limits:
            self.max_positions = limits['max_positions']
        
        if 'max_drawdown' in limits:
            self.max_drawdown = Decimal(str(limits['max_drawdown']))
        
        if 'correlation_limit' in limits:
            self.correlation_limit = limits['correlation_limit']
    
    def get_position_sizing_info(self, account_balance: Decimal, current_price: Decimal, 
                                stop_loss_price: Decimal, signal_strength: float = 1.0) -> Dict[str, Any]:
        """Get detailed position sizing information."""
        return self.position_sizer.get_position_size_info(
            account_balance, current_price, stop_loss_price, signal_strength
        )
