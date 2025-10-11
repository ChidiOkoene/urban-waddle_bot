"""
Portfolio management with correlation and risk controls.
"""

from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict

from ..core.data_models import Position, OrderSide, Balance


class PortfolioManager:
    """Portfolio management with correlation and risk controls."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize portfolio manager.
        
        Args:
            config: Portfolio management configuration
        """
        self.config = config
        
        # Portfolio limits
        self.max_positions = config.get('max_positions', 10)
        self.max_correlation = config.get('max_correlation', 0.7)
        self.max_sector_exposure = config.get('max_sector_exposure', 0.3)  # 30%
        self.max_currency_exposure = config.get('max_currency_exposure', 0.5)  # 50%
        
        # Position limits
        self.max_position_size = config.get('max_position_size', 0.2)  # 20%
        self.min_position_size = config.get('min_position_size', 0.01)  # 1%
        
        # Risk limits
        self.max_portfolio_risk = config.get('max_portfolio_risk', 0.15)  # 15%
        self.max_drawdown = config.get('max_drawdown', 0.2)  # 20%
        
        # Portfolio state
        self.positions = {}
        self.correlation_matrix = {}
        self.sector_exposure = defaultdict(Decimal)
        self.currency_exposure = defaultdict(Decimal)
        self.portfolio_value = Decimal("0")
        self.total_risk = Decimal("0")
        
        # Performance tracking
        self.daily_returns = []
        self.portfolio_metrics = {}
        
        # Rebalancing
        self.last_rebalance = None
        self.rebalance_frequency = timedelta(days=7)  # Weekly rebalancing
    
    def add_position(self, position: Position) -> Tuple[bool, str]:
        """
        Add a position to the portfolio.
        
        Args:
            position: Position to add
            
        Returns:
            Tuple of (success, reason)
        """
        # Check maximum positions
        if len(self.positions) >= self.max_positions:
            return False, f"Maximum positions limit reached ({self.max_positions})"
        
        # Check position size
        position_size_ratio = abs(float(position.amount * position.entry_price)) / float(self.portfolio_value)
        if position_size_ratio > self.max_position_size:
            return False, f"Position size too large ({position_size_ratio:.2%})"
        
        if position_size_ratio < self.min_position_size:
            return False, f"Position size too small ({position_size_ratio:.2%})"
        
        # Check correlation
        if self._check_correlation_limit(position):
            return False, f"Correlation limit exceeded for {position.symbol}"
        
        # Check sector exposure
        sector = self._get_sector(position.symbol)
        if self._check_sector_exposure(sector, position):
            return False, f"Sector exposure limit exceeded for {sector}"
        
        # Check currency exposure
        currency = self._get_currency(position.symbol)
        if self._check_currency_exposure(currency, position):
            return False, f"Currency exposure limit exceeded for {currency}"
        
        # Add position
        self.positions[position.id] = position
        self._update_exposures(position)
        
        return True, "Position added successfully"
    
    def remove_position(self, position_id: str):
        """Remove a position from the portfolio."""
        if position_id in self.positions:
            position = self.positions[position_id]
            self._remove_exposures(position)
            del self.positions[position_id]
    
    def update_position(self, position: Position):
        """Update a position in the portfolio."""
        if position.id in self.positions:
            old_position = self.positions[position.id]
            self._remove_exposures(old_position)
            self.positions[position.id] = position
            self._update_exposures(position)
    
    def calculate_portfolio_metrics(self, current_prices: Dict[str, Decimal]) -> Dict[str, Any]:
        """
        Calculate portfolio metrics.
        
        Args:
            current_prices: Current prices for all positions
            
        Returns:
            Portfolio metrics
        """
        if not self.positions:
            return self._get_empty_metrics()
        
        # Calculate portfolio value
        total_value = Decimal("0")
        total_cost = Decimal("0")
        total_pnl = Decimal("0")
        
        position_values = {}
        position_weights = {}
        
        for position_id, position in self.positions.items():
            current_price = current_prices.get(position.symbol, position.current_price)
            position_value = position.amount * current_price
            position_cost = position.amount * position.entry_price
            position_pnl = position_value - position_cost
            
            total_value += position_value
            total_cost += position_cost
            total_pnl += position_pnl
            
            position_values[position_id] = position_value
            position_weights[position_id] = position_value / total_value if total_value > 0 else Decimal("0")
        
        self.portfolio_value = total_value
        
        # Calculate returns
        if total_cost > 0:
            total_return = total_pnl / total_cost
        else:
            total_return = Decimal("0")
        
        # Calculate risk metrics
        portfolio_risk = self._calculate_portfolio_risk(position_values, current_prices)
        sharpe_ratio = self._calculate_sharpe_ratio()
        max_drawdown = self._calculate_max_drawdown()
        
        # Calculate diversification metrics
        diversification_ratio = self._calculate_diversification_ratio(position_weights)
        concentration_ratio = self._calculate_concentration_ratio(position_weights)
        
        # Calculate sector and currency exposure
        sector_exposure = {sector: float(exposure) for sector, exposure in self.sector_exposure.items()}
        currency_exposure = {currency: float(exposure) for currency, exposure in self.currency_exposure.items()}
        
        metrics = {
            'portfolio_value': float(total_value),
            'total_cost': float(total_cost),
            'total_pnl': float(total_pnl),
            'total_return': float(total_return),
            'portfolio_risk': float(portfolio_risk),
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': float(max_drawdown),
            'diversification_ratio': diversification_ratio,
            'concentration_ratio': concentration_ratio,
            'sector_exposure': sector_exposure,
            'currency_exposure': currency_exposure,
            'position_count': len(self.positions),
            'correlation_matrix': self._get_correlation_summary(),
            'rebalance_needed': self._needs_rebalancing(),
            'last_rebalance': self.last_rebalance.isoformat() if self.last_rebalance else None
        }
        
        self.portfolio_metrics = metrics
        return metrics
    
    def rebalance_portfolio(self, target_weights: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """
        Rebalance the portfolio.
        
        Args:
            target_weights: Target weights for positions
            
        Returns:
            List of rebalancing actions
        """
        if not self.positions:
            return []
        
        # Calculate current weights
        current_weights = {}
        total_value = sum(pos.amount * pos.current_price for pos in self.positions.values())
        
        for position_id, position in self.positions.items():
            position_value = position.amount * position.current_price
            current_weights[position_id] = float(position_value / total_value) if total_value > 0 else 0
        
        # Use equal weights if no target weights provided
        if not target_weights:
            target_weights = {pos_id: 1.0 / len(self.positions) for pos_id in self.positions.keys()}
        
        # Calculate rebalancing actions
        actions = []
        rebalance_threshold = 0.05  # 5% threshold
        
        for position_id in self.positions.keys():
            current_weight = current_weights.get(position_id, 0)
            target_weight = target_weights.get(position_id, 0)
            weight_diff = target_weight - current_weight
            
            if abs(weight_diff) > rebalance_threshold:
                actions.append({
                    'position_id': position_id,
                    'action': 'rebalance',
                    'current_weight': current_weight,
                    'target_weight': target_weight,
                    'weight_diff': weight_diff,
                    'adjustment_needed': weight_diff * total_value
                })
        
        self.last_rebalance = datetime.utcnow()
        return actions
    
    def _check_correlation_limit(self, position: Position) -> bool:
        """Check if adding position would exceed correlation limit."""
        if len(self.positions) == 0:
            return False
        
        # Calculate correlation with existing positions
        symbol = position.symbol
        correlations = []
        
        for existing_position in self.positions.values():
            correlation = self._calculate_correlation(symbol, existing_position.symbol)
            correlations.append(correlation)
        
        # Check if any correlation exceeds limit
        return any(corr > self.max_correlation for corr in correlations)
    
    def _check_sector_exposure(self, sector: str, position: Position) -> bool:
        """Check if adding position would exceed sector exposure limit."""
        position_value = abs(position.amount * position.entry_price)
        new_sector_exposure = self.sector_exposure[sector] + position_value
        
        return new_sector_exposure / self.portfolio_value > self.max_sector_exposure
    
    def _check_currency_exposure(self, currency: str, position: Position) -> bool:
        """Check if adding position would exceed currency exposure limit."""
        position_value = abs(position.amount * position.entry_price)
        new_currency_exposure = self.currency_exposure[currency] + position_value
        
        return new_currency_exposure / self.portfolio_value > self.max_currency_exposure
    
    def _get_sector(self, symbol: str) -> str:
        """Get sector for a symbol."""
        # Simple sector classification based on symbol
        if '/' in symbol:
            base, quote = symbol.split('/')
            
            # Crypto sectors
            if base in ['BTC', 'ETH', 'LTC', 'BCH']:
                return 'crypto_major'
            elif base in ['ADA', 'DOT', 'LINK', 'UNI']:
                return 'crypto_alt'
            elif quote in ['USD', 'EUR', 'GBP', 'JPY']:
                return 'forex'
            else:
                return 'crypto_other'
        
        return 'unknown'
    
    def _get_currency(self, symbol: str) -> str:
        """Get currency for a symbol."""
        if '/' in symbol:
            return symbol.split('/')[1]  # Quote currency
        return 'unknown'
    
    def _update_exposures(self, position: Position):
        """Update sector and currency exposures."""
        sector = self._get_sector(position.symbol)
        currency = self._get_currency(position.symbol)
        
        position_value = abs(position.amount * position.entry_price)
        
        self.sector_exposure[sector] += position_value
        self.currency_exposure[currency] += position_value
    
    def _remove_exposures(self, position: Position):
        """Remove sector and currency exposures."""
        sector = self._get_sector(position.symbol)
        currency = self._get_currency(position.symbol)
        
        position_value = abs(position.amount * position.entry_price)
        
        self.sector_exposure[sector] -= position_value
        self.currency_exposure[currency] -= position_value
        
        # Ensure non-negative values
        self.sector_exposure[sector] = max(Decimal("0"), self.sector_exposure[sector])
        self.currency_exposure[currency] = max(Decimal("0"), self.currency_exposure[currency])
    
    def _calculate_correlation(self, symbol1: str, symbol2: str) -> float:
        """Calculate correlation between two symbols."""
        # Simplified correlation calculation
        # In practice, you would use historical price data
        
        if symbol1 == symbol2:
            return 1.0
        
        # Same base currency
        if '/' in symbol1 and '/' in symbol2:
            base1, quote1 = symbol1.split('/')
            base2, quote2 = symbol2.split('/')
            
            if base1 == base2:
                return 0.8
            if quote1 == quote2:
                return 0.6
        
        # Default correlation
        return 0.3
    
    def _calculate_portfolio_risk(self, position_values: Dict[str, Decimal], 
                                current_prices: Dict[str, Decimal]) -> Decimal:
        """Calculate portfolio risk."""
        if not position_values:
            return Decimal("0")
        
        # Simplified risk calculation using position weights
        total_value = sum(position_values.values())
        if total_value == 0:
            return Decimal("0")
        
        # Calculate weighted risk (simplified)
        portfolio_risk = Decimal("0")
        for position_id, position in self.positions.items():
            position_value = position_values[position_id]
            weight = position_value / total_value
            
            # Assume 2% daily volatility per position
            position_risk = weight * Decimal("0.02")
            portfolio_risk += position_risk
        
        return portfolio_risk
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio."""
        if not self.daily_returns or len(self.daily_returns) < 2:
            return 0.0
        
        mean_return = np.mean(self.daily_returns)
        std_return = np.std(self.daily_returns)
        
        if std_return == 0:
            return 0.0
        
        # Assume risk-free rate of 0
        return mean_return / std_return
    
    def _calculate_max_drawdown(self) -> Decimal:
        """Calculate maximum drawdown."""
        if not self.daily_returns:
            return Decimal("0")
        
        cumulative_returns = np.cumprod([1 + r for r in self.daily_returns])
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        
        max_drawdown = abs(np.min(drawdowns))
        return Decimal(str(max_drawdown))
    
    def _calculate_diversification_ratio(self, position_weights: Dict[str, Decimal]) -> float:
        """Calculate diversification ratio."""
        if not position_weights:
            return 0.0
        
        weights = [float(w) for w in position_weights.values()]
        
        # Calculate Herfindahl index
        herfindahl = sum(w**2 for w in weights)
        
        # Diversification ratio is inverse of Herfindahl
        return 1.0 / herfindahl if herfindahl > 0 else 0.0
    
    def _calculate_concentration_ratio(self, position_weights: Dict[str, Decimal]) -> float:
        """Calculate concentration ratio."""
        if not position_weights:
            return 0.0
        
        weights = [float(w) for w in position_weights.values()]
        
        # Sum of squared weights (Herfindahl index)
        return sum(w**2 for w in weights)
    
    def _get_correlation_summary(self) -> Dict[str, Any]:
        """Get correlation matrix summary."""
        if len(self.positions) < 2:
            return {}
        
        symbols = [pos.symbol for pos in self.positions.values()]
        correlations = []
        
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols):
                if i < j:  # Only upper triangle
                    correlation = self._calculate_correlation(symbol1, symbol2)
                    correlations.append(correlation)
        
        if correlations:
            return {
                'average_correlation': np.mean(correlations),
                'max_correlation': np.max(correlations),
                'min_correlation': np.min(correlations)
            }
        
        return {}
    
    def _needs_rebalancing(self) -> bool:
        """Check if portfolio needs rebalancing."""
        if not self.last_rebalance:
            return True
        
        return datetime.utcnow() - self.last_rebalance > self.rebalance_frequency
    
    def _get_empty_metrics(self) -> Dict[str, Any]:
        """Get empty metrics when no positions."""
        return {
            'portfolio_value': 0.0,
            'total_cost': 0.0,
            'total_pnl': 0.0,
            'total_return': 0.0,
            'portfolio_risk': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'diversification_ratio': 0.0,
            'concentration_ratio': 0.0,
            'sector_exposure': {},
            'currency_exposure': {},
            'position_count': 0,
            'correlation_matrix': {},
            'rebalance_needed': False,
            'last_rebalance': None
        }
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get portfolio status."""
        return {
            'position_count': len(self.positions),
            'max_positions': self.max_positions,
            'max_correlation': self.max_correlation,
            'max_sector_exposure': self.max_sector_exposure,
            'max_currency_exposure': self.max_currency_exposure,
            'max_position_size': self.max_position_size,
            'min_position_size': self.min_position_size,
            'max_portfolio_risk': self.max_portfolio_risk,
            'max_drawdown': self.max_drawdown,
            'portfolio_value': float(self.portfolio_value),
            'total_risk': float(self.total_risk),
            'rebalance_needed': self._needs_rebalancing(),
            'last_rebalance': self.last_rebalance.isoformat() if self.last_rebalance else None,
            'positions': {
                pos_id: {
                    'symbol': pos.symbol,
                    'side': pos.side.value,
                    'amount': float(pos.amount),
                    'entry_price': float(pos.entry_price),
                    'current_price': float(pos.current_price),
                    'unrealized_pnl': float(pos.unrealized_pnl)
                }
                for pos_id, pos in self.positions.items()
            }
        }
    
    def update_daily_return(self, daily_return: float):
        """Update daily return for performance tracking."""
        self.daily_returns.append(daily_return)
        
        # Keep only last 252 days (1 year)
        if len(self.daily_returns) > 252:
            self.daily_returns = self.daily_returns[-252:]
