"""
Tests for Risk Management System

This module contains tests for all risk management components,
including position sizing, risk limits, stop-loss management, and portfolio controls.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from src.risk.position_sizer import PositionSizer
from src.risk.risk_manager import RiskManager
from src.risk.stop_loss_manager import StopLossManager
from src.risk.portfolio_manager import PortfolioManager
from src.core.data_models import OrderSide, OrderType, Position, Trade, Balance
from tests.mocks.mock_exchange import MockExchange


class TestPositionSizer:
    """Test cases for position sizing algorithms."""
    
    @pytest.fixture
    def position_sizer(self):
        """Create position sizer instance."""
        return PositionSizer()
    
    @pytest.fixture
    def sample_balance(self):
        """Create sample balance."""
        return Balance(
            asset='USDT',
            free=10000.0,
            used=0.0,
            total=10000.0
        )
    
    def test_fixed_percentage_sizing(self, position_sizer, sample_balance):
        """Test fixed percentage position sizing."""
        # Test 1% risk
        size = position_sizer.calculate_position_size(
            balance=sample_balance,
            risk_percentage=0.01,
            entry_price=100.0,
            stop_loss_price=95.0
        )
        
        # Expected size: 10000 * 0.01 / (100 - 95) = 20
        expected_size = 10000.0 * 0.01 / (100.0 - 95.0)
        assert abs(size - expected_size) < 0.01
    
    def test_kelly_criterion_sizing(self, position_sizer, sample_balance):
        """Test Kelly criterion position sizing."""
        # Test with win rate 60% and avg win/loss ratio 1.5
        size = position_sizer.calculate_position_size(
            balance=sample_balance,
            method='kelly',
            win_rate=0.6,
            avg_win_loss_ratio=1.5,
            entry_price=100.0,
            stop_loss_price=95.0
        )
        
        # Kelly formula: f = (bp - q) / b
        # where b = avg_win_loss_ratio, p = win_rate, q = 1 - win_rate
        b = 1.5
        p = 0.6
        q = 0.4
        kelly_fraction = (b * p - q) / b
        
        expected_size = sample_balance.free * kelly_fraction / (100.0 - 95.0)
        assert abs(size - expected_size) < 0.01
    
    def test_volatility_based_sizing(self, position_sizer, sample_balance):
        """Test volatility-based position sizing."""
        # Test with ATR-based sizing
        size = position_sizer.calculate_position_size(
            balance=sample_balance,
            method='volatility',
            atr=2.0,
            risk_percentage=0.01,
            entry_price=100.0
        )
        
        # Expected size: balance * risk_percentage / (atr * 2)
        expected_size = sample_balance.free * 0.01 / (2.0 * 2.0)
        assert abs(size - expected_size) < 0.01
    
    def test_max_position_size_limit(self, position_sizer, sample_balance):
        """Test maximum position size limit."""
        # Test with very high risk percentage
        size = position_sizer.calculate_position_size(
            balance=sample_balance,
            risk_percentage=0.5,  # 50% risk
            entry_price=100.0,
            stop_loss_price=95.0,
            max_position_size=0.1  # 10% max
        )
        
        # Should be limited to 10% of balance
        max_size = sample_balance.free * 0.1 / 100.0
        assert size <= max_size
    
    def test_minimum_position_size(self, position_sizer, sample_balance):
        """Test minimum position size."""
        # Test with very low risk percentage
        size = position_sizer.calculate_position_size(
            balance=sample_balance,
            risk_percentage=0.001,  # 0.1% risk
            entry_price=100.0,
            stop_loss_price=95.0,
            min_position_size=0.01  # 1% min
        )
        
        # Should be at least 1% of balance
        min_size = sample_balance.free * 0.01 / 100.0
        assert size >= min_size
    
    def test_invalid_parameters(self, position_sizer, sample_balance):
        """Test handling of invalid parameters."""
        # Test with negative risk percentage
        with pytest.raises(ValueError):
            position_sizer.calculate_position_size(
                balance=sample_balance,
                risk_percentage=-0.01,
                entry_price=100.0,
                stop_loss_price=95.0
            )
        
        # Test with stop loss above entry price for long position
        with pytest.raises(ValueError):
            position_sizer.calculate_position_size(
                balance=sample_balance,
                risk_percentage=0.01,
                entry_price=100.0,
                stop_loss_price=105.0
            )


class TestRiskManager:
    """Test cases for risk management."""
    
    @pytest.fixture
    def risk_manager(self):
        """Create risk manager instance."""
        return RiskManager()
    
    @pytest.fixture
    def sample_positions(self):
        """Create sample positions."""
        return [
            Position(
                symbol='BTC/USDT',
                side=OrderSide.BUY,
                size=0.1,
                entry_price=100.0,
                current_price=105.0,
                unrealized_pnl=0.5
            ),
            Position(
                symbol='ETH/USDT',
                side=OrderSide.BUY,
                size=1.0,
                entry_price=200.0,
                current_price=195.0,
                unrealized_pnl=-5.0
            )
        ]
    
    def test_max_drawdown_check(self, risk_manager, sample_positions):
        """Test maximum drawdown check."""
        # Set max drawdown to 5%
        risk_manager.max_drawdown = 0.05
        
        # Test with current drawdown of 3%
        current_drawdown = 0.03
        can_trade = risk_manager.check_max_drawdown(current_drawdown)
        assert can_trade is True
        
        # Test with current drawdown of 6%
        current_drawdown = 0.06
        can_trade = risk_manager.check_max_drawdown(current_drawdown)
        assert can_trade is False
    
    def test_max_positions_check(self, risk_manager, sample_positions):
        """Test maximum positions check."""
        # Set max positions to 2
        risk_manager.max_positions = 2
        
        # Test with 2 positions
        can_trade = risk_manager.check_max_positions(len(sample_positions))
        assert can_trade is True
        
        # Test with 3 positions
        can_trade = risk_manager.check_max_positions(3)
        assert can_trade is False
    
    def test_correlation_check(self, risk_manager, sample_positions):
        """Test correlation-based position limits."""
        # Set max correlation to 0.7
        risk_manager.max_correlation = 0.7
        
        # Test with low correlation
        correlation = 0.5
        can_trade = risk_manager.check_correlation(correlation)
        assert can_trade is True
        
        # Test with high correlation
        correlation = 0.8
        can_trade = risk_manager.check_correlation(correlation)
        assert can_trade is False
    
    def test_risk_per_trade_check(self, risk_manager, sample_positions):
        """Test risk per trade check."""
        # Set max risk per trade to 2%
        risk_manager.max_risk_per_trade = 0.02
        
        # Test with 1% risk
        risk_per_trade = 0.01
        can_trade = risk_manager.check_risk_per_trade(risk_per_trade)
        assert can_trade is True
        
        # Test with 3% risk
        risk_per_trade = 0.03
        can_trade = risk_manager.check_risk_per_trade(risk_per_trade)
        assert can_trade is False
    
    def test_portfolio_risk_check(self, risk_manager, sample_positions):
        """Test portfolio risk check."""
        # Set max portfolio risk to 10%
        risk_manager.max_portfolio_risk = 0.10
        
        # Test with 5% portfolio risk
        portfolio_risk = 0.05
        can_trade = risk_manager.check_portfolio_risk(portfolio_risk)
        assert can_trade is True
        
        # Test with 12% portfolio risk
        portfolio_risk = 0.12
        can_trade = risk_manager.check_portfolio_risk(portfolio_risk)
        assert can_trade is False
    
    def test_emergency_stop(self, risk_manager, sample_positions):
        """Test emergency stop functionality."""
        # Test emergency stop activation
        risk_manager.activate_emergency_stop("Test emergency stop")
        assert risk_manager.emergency_stop_active is True
        assert risk_manager.emergency_stop_reason == "Test emergency stop"
        
        # Test emergency stop deactivation
        risk_manager.deactivate_emergency_stop()
        assert risk_manager.emergency_stop_active is False
        assert risk_manager.emergency_stop_reason is None
    
    def test_risk_limits_validation(self, risk_manager):
        """Test risk limits validation."""
        # Test valid limits
        valid_limits = {
            'max_drawdown': 0.05,
            'max_positions': 5,
            'max_correlation': 0.7,
            'max_risk_per_trade': 0.02,
            'max_portfolio_risk': 0.10
        }
        
        risk_manager.set_risk_limits(valid_limits)
        assert risk_manager.max_drawdown == 0.05
        assert risk_manager.max_positions == 5
        assert risk_manager.max_correlation == 0.7
        assert risk_manager.max_risk_per_trade == 0.02
        assert risk_manager.max_portfolio_risk == 0.10
        
        # Test invalid limits
        invalid_limits = {
            'max_drawdown': -0.05,  # Negative
            'max_positions': 0,  # Zero
            'max_correlation': 1.5,  # > 1
            'max_risk_per_trade': -0.02,  # Negative
            'max_portfolio_risk': 1.5  # > 1
        }
        
        with pytest.raises(ValueError):
            risk_manager.set_risk_limits(invalid_limits)


class TestStopLossManager:
    """Test cases for stop-loss management."""
    
    @pytest.fixture
    def stop_loss_manager(self):
        """Create stop-loss manager instance."""
        return StopLossManager()
    
    @pytest.fixture
    def sample_position(self):
        """Create sample position."""
        return Position(
            symbol='BTC/USDT',
            side=OrderSide.BUY,
            size=0.1,
            entry_price=100.0,
            current_price=105.0,
            unrealized_pnl=0.5
        )
    
    def test_fixed_stop_loss(self, stop_loss_manager, sample_position):
        """Test fixed stop-loss calculation."""
        # Test 5% stop-loss
        stop_loss_price = stop_loss_manager.calculate_fixed_stop_loss(
            position=sample_position,
            stop_loss_percentage=0.05
        )
        
        # Expected: 100 * (1 - 0.05) = 95
        expected_price = 100.0 * (1 - 0.05)
        assert abs(stop_loss_price - expected_price) < 0.01
    
    def test_trailing_stop_loss(self, stop_loss_manager, sample_position):
        """Test trailing stop-loss calculation."""
        # Test 3% trailing stop
        stop_loss_price = stop_loss_manager.calculate_trailing_stop_loss(
            position=sample_position,
            trailing_percentage=0.03,
            highest_price=110.0
        )
        
        # Expected: 110 * (1 - 0.03) = 106.7
        expected_price = 110.0 * (1 - 0.03)
        assert abs(stop_loss_price - expected_price) < 0.01
    
    def test_atr_stop_loss(self, stop_loss_manager, sample_position):
        """Test ATR-based stop-loss calculation."""
        # Test 2x ATR stop-loss
        atr = 2.0
        stop_loss_price = stop_loss_manager.calculate_atr_stop_loss(
            position=sample_position,
            atr=atr,
            atr_multiplier=2.0
        )
        
        # Expected: 100 - (2 * 2) = 96
        expected_price = 100.0 - (2.0 * 2.0)
        assert abs(stop_loss_price - expected_price) < 0.01
    
    def test_time_based_stop_loss(self, stop_loss_manager, sample_position):
        """Test time-based stop-loss calculation."""
        # Test 24-hour time stop
        entry_time = datetime.now() - timedelta(hours=25)
        sample_position.entry_time = entry_time
        
        should_stop = stop_loss_manager.check_time_based_stop_loss(
            position=sample_position,
            max_hold_hours=24
        )
        
        assert should_stop is True
        
        # Test position held for only 12 hours
        entry_time = datetime.now() - timedelta(hours=12)
        sample_position.entry_time = entry_time
        
        should_stop = stop_loss_manager.check_time_based_stop_loss(
            position=sample_position,
            max_hold_hours=24
        )
        
        assert should_stop is False
    
    def test_stop_loss_update(self, stop_loss_manager, sample_position):
        """Test stop-loss update functionality."""
        # Set initial stop-loss
        initial_stop = 95.0
        stop_loss_manager.set_stop_loss(sample_position.symbol, initial_stop)
        
        # Update stop-loss
        new_stop = 98.0
        stop_loss_manager.update_stop_loss(sample_position.symbol, new_stop)
        
        # Check updated stop-loss
        current_stop = stop_loss_manager.get_stop_loss(sample_position.symbol)
        assert current_stop == new_stop
    
    def test_stop_loss_removal(self, stop_loss_manager, sample_position):
        """Test stop-loss removal functionality."""
        # Set stop-loss
        stop_loss_manager.set_stop_loss(sample_position.symbol, 95.0)
        
        # Remove stop-loss
        stop_loss_manager.remove_stop_loss(sample_position.symbol)
        
        # Check stop-loss is removed
        current_stop = stop_loss_manager.get_stop_loss(sample_position.symbol)
        assert current_stop is None


class TestPortfolioManager:
    """Test cases for portfolio management."""
    
    @pytest.fixture
    def portfolio_manager(self):
        """Create portfolio manager instance."""
        return PortfolioManager()
    
    @pytest.fixture
    def sample_positions(self):
        """Create sample positions."""
        return [
            Position(
                symbol='BTC/USDT',
                side=OrderSide.BUY,
                size=0.1,
                entry_price=100.0,
                current_price=105.0,
                unrealized_pnl=0.5
            ),
            Position(
                symbol='ETH/USDT',
                side=OrderSide.BUY,
                size=1.0,
                entry_price=200.0,
                current_price=195.0,
                unrealized_pnl=-5.0
            )
        ]
    
    def test_portfolio_value_calculation(self, portfolio_manager, sample_positions):
        """Test portfolio value calculation."""
        # Mock balance
        balance = Balance(asset='USDT', free=10000.0, used=0.0, total=10000.0)
        
        # Calculate portfolio value
        portfolio_value = portfolio_manager.calculate_portfolio_value(
            balance=balance,
            positions=sample_positions
        )
        
        # Expected: 10000 + 0.5 - 5.0 = 9995.5
        expected_value = 10000.0 + 0.5 - 5.0
        assert abs(portfolio_value - expected_value) < 0.01
    
    def test_portfolio_risk_calculation(self, portfolio_manager, sample_positions):
        """Test portfolio risk calculation."""
        # Calculate portfolio risk
        portfolio_risk = portfolio_manager.calculate_portfolio_risk(
            positions=sample_positions,
            total_value=10000.0
        )
        
        # Risk should be calculated based on position sizes and volatility
        assert portfolio_risk >= 0
        assert portfolio_risk <= 1  # Should be normalized
    
    def test_position_correlation(self, portfolio_manager, sample_positions):
        """Test position correlation calculation."""
        # Mock price data
        btc_prices = [100, 101, 102, 103, 104, 105]
        eth_prices = [200, 201, 202, 203, 204, 205]
        
        correlation = portfolio_manager.calculate_position_correlation(
            symbol1='BTC/USDT',
            symbol2='ETH/USDT',
            prices1=btc_prices,
            prices2=eth_prices
        )
        
        # Should be high correlation for similar price movements
        assert correlation > 0.8
    
    def test_portfolio_rebalancing(self, portfolio_manager, sample_positions):
        """Test portfolio rebalancing logic."""
        # Set target allocation
        target_allocation = {
            'BTC/USDT': 0.6,
            'ETH/USDT': 0.4
        }
        
        # Calculate rebalancing actions
        rebalance_actions = portfolio_manager.calculate_rebalancing_actions(
            positions=sample_positions,
            target_allocation=target_allocation,
            total_value=10000.0
        )
        
        assert isinstance(rebalance_actions, list)
        # Should have some rebalancing actions
        assert len(rebalance_actions) > 0
    
    def test_portfolio_limits_check(self, portfolio_manager, sample_positions):
        """Test portfolio limits check."""
        # Set portfolio limits
        portfolio_limits = {
            'max_single_position': 0.3,
            'max_correlation': 0.7,
            'max_drawdown': 0.05
        }
        
        # Check portfolio limits
        limits_ok = portfolio_manager.check_portfolio_limits(
            positions=sample_positions,
            limits=portfolio_limits,
            total_value=10000.0
        )
        
        assert isinstance(limits_ok, bool)
    
    def test_portfolio_performance_metrics(self, portfolio_manager, sample_positions):
        """Test portfolio performance metrics calculation."""
        # Mock historical data
        historical_data = {
            'BTC/USDT': [100, 101, 102, 103, 104, 105],
            'ETH/USDT': [200, 201, 202, 203, 204, 205]
        }
        
        # Calculate performance metrics
        metrics = portfolio_manager.calculate_performance_metrics(
            positions=sample_positions,
            historical_data=historical_data,
            total_value=10000.0
        )
        
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'volatility' in metrics
        assert 'returns' in metrics
        
        # Check metric values are reasonable
        assert metrics['sharpe_ratio'] is not None
        assert metrics['max_drawdown'] >= 0
        assert metrics['volatility'] >= 0


class TestRiskManagementIntegration:
    """Integration tests for risk management system."""
    
    @pytest.fixture
    def mock_exchange(self):
        """Create mock exchange for testing."""
        return MockExchange(initial_balance=10000.0)
    
    @pytest.mark.asyncio
    async def test_risk_management_workflow(self, mock_exchange):
        """Test complete risk management workflow."""
        await mock_exchange.connect()
        
        # Create risk management components
        position_sizer = PositionSizer()
        risk_manager = RiskManager()
        stop_loss_manager = StopLossManager()
        portfolio_manager = PortfolioManager()
        
        # Set risk limits
        risk_manager.set_risk_limits({
            'max_drawdown': 0.05,
            'max_positions': 5,
            'max_correlation': 0.7,
            'max_risk_per_trade': 0.02,
            'max_portfolio_risk': 0.10
        })
        
        # Get balance
        balance = await mock_exchange.get_balance('USDT')
        
        # Calculate position size
        position_size = position_sizer.calculate_position_size(
            balance=balance,
            risk_percentage=0.01,
            entry_price=100.0,
            stop_loss_price=95.0
        )
        
        # Check risk limits
        can_trade = risk_manager.check_risk_per_trade(0.01)
        assert can_trade is True
        
        can_trade = risk_manager.check_max_positions(0)
        assert can_trade is True
        
        # Place order
        order_id = await mock_exchange.place_order(
            symbol='BTC/USDT',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=position_size
        )
        
        assert order_id is not None
        
        # Set stop-loss
        stop_loss_price = stop_loss_manager.calculate_fixed_stop_loss(
            position=Position(
                symbol='BTC/USDT',
                side=OrderSide.BUY,
                size=position_size,
                entry_price=100.0,
                current_price=100.0,
                unrealized_pnl=0.0
            ),
            stop_loss_percentage=0.05
        )
        
        stop_loss_manager.set_stop_loss('BTC/USDT', stop_loss_price)
        
        # Check stop-loss is set
        current_stop = stop_loss_manager.get_stop_loss('BTC/USDT')
        assert current_stop == stop_loss_price
    
    def test_risk_management_parameter_validation(self):
        """Test parameter validation across risk management components."""
        # Test position sizer
        position_sizer = PositionSizer()
        balance = Balance(asset='USDT', free=10000.0, used=0.0, total=10000.0)
        
        # Valid parameters
        size = position_sizer.calculate_position_size(
            balance=balance,
            risk_percentage=0.01,
            entry_price=100.0,
            stop_loss_price=95.0
        )
        assert size > 0
        
        # Test risk manager
        risk_manager = RiskManager()
        risk_manager.set_risk_limits({
            'max_drawdown': 0.05,
            'max_positions': 5,
            'max_correlation': 0.7,
            'max_risk_per_trade': 0.02,
            'max_portfolio_risk': 0.10
        })
        
        # Test stop-loss manager
        stop_loss_manager = StopLossManager()
        position = Position(
            symbol='BTC/USDT',
            side=OrderSide.BUY,
            size=0.1,
            entry_price=100.0,
            current_price=100.0,
            unrealized_pnl=0.0
        )
        
        stop_price = stop_loss_manager.calculate_fixed_stop_loss(
            position=position,
            stop_loss_percentage=0.05
        )
        assert stop_price < position.entry_price
        
        # Test portfolio manager
        portfolio_manager = PortfolioManager()
        positions = [position]
        
        portfolio_value = portfolio_manager.calculate_portfolio_value(
            balance=balance,
            positions=positions
        )
        assert portfolio_value > 0


if __name__ == "__main__":
    pytest.main([__file__])
