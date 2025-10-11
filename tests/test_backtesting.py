"""
Tests for Backtesting Framework

This module contains tests for the backtesting engine,
performance metrics, strategy comparison, and optimization.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from src.backtesting.backtest_engine import BacktestEngine
from src.backtesting.performance_metrics import PerformanceMetrics
from src.backtesting.strategy_comparison import StrategyComparison
from src.backtesting.optimization_engine import OptimizationEngine
from src.backtesting.report_generator import ReportGenerator
from src.core.data_models import OHLCV, OrderSide, OrderType, Trade, Position
from src.strategies.rsi_macd_strategy import RSIMACDStrategy
from src.strategies.bollinger_mean_reversion import BollingerMeanReversionStrategy
from tests.mocks.mock_exchange import MockExchange


class TestBacktestEngine:
    """Test cases for backtesting engine."""
    
    @pytest.fixture
    def backtest_engine(self):
        """Create backtest engine instance."""
        return BacktestEngine()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        dates = pd.date_range(start='2023-01-01', periods=1000, freq='H')
        prices = 100 + np.cumsum(np.random.randn(1000) * 0.1)
        
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            ohlcv = OHLCV(
                symbol='BTC/USDT',
                timestamp=date,
                open=price,
                high=price * 1.01,
                low=price * 0.99,
                close=price,
                volume=1000
            )
            data.append(ohlcv)
        
        return data
    
    @pytest.fixture
    def sample_strategy(self):
        """Create sample strategy."""
        return RSIMACDStrategy()
    
    def test_backtest_initialization(self, backtest_engine):
        """Test backtest engine initialization."""
        assert backtest_engine.initial_capital == 10000.0
        assert backtest_engine.commission_rate == 0.001
        assert backtest_engine.slippage_rate == 0.0005
    
    def test_backtest_parameters(self, backtest_engine):
        """Test backtest parameter setting."""
        params = {
            'initial_capital': 20000.0,
            'commission_rate': 0.002,
            'slippage_rate': 0.001
        }
        
        backtest_engine.set_parameters(params)
        
        assert backtest_engine.initial_capital == 20000.0
        assert backtest_engine.commission_rate == 0.002
        assert backtest_engine.slippage_rate == 0.001
    
    @pytest.mark.asyncio
    async def test_backtest_execution(self, backtest_engine, sample_data, sample_strategy):
        """Test backtest execution."""
        # Convert to DataFrame
        df = pd.DataFrame([{
            'timestamp': candle.timestamp,
            'open': candle.open,
            'high': candle.high,
            'low': candle.low,
            'close': candle.close,
            'volume': candle.volume
        } for candle in sample_data])
        
        # Run backtest
        results = await backtest_engine.run_backtest(
            strategy=sample_strategy,
            data=df,
            symbol='BTC/USDT',
            timeframe='1h'
        )
        
        assert isinstance(results, dict)
        assert 'trades' in results
        assert 'positions' in results
        assert 'performance' in results
        assert 'equity_curve' in results
        
        # Check results structure
        assert isinstance(results['trades'], list)
        assert isinstance(results['positions'], list)
        assert isinstance(results['performance'], dict)
        assert isinstance(results['equity_curve'], pd.DataFrame)
    
    def test_trade_execution(self, backtest_engine):
        """Test trade execution logic."""
        # Mock order
        order = {
            'symbol': 'BTC/USDT',
            'side': OrderSide.BUY,
            'type': OrderType.MARKET,
            'quantity': 0.1,
            'price': 100.0,
            'timestamp': datetime.now()
        }
        
        # Execute trade
        trade = backtest_engine.execute_trade(order, 100.0)
        
        assert isinstance(trade, Trade)
        assert trade.symbol == 'BTC/USDT'
        assert trade.side == OrderSide.BUY
        assert trade.quantity == 0.1
        assert trade.price == 100.0
    
    def test_position_management(self, backtest_engine):
        """Test position management."""
        # Create position
        position = Position(
            symbol='BTC/USDT',
            side=OrderSide.BUY,
            size=0.1,
            entry_price=100.0,
            current_price=105.0,
            unrealized_pnl=0.5
        )
        
        # Update position
        updated_position = backtest_engine.update_position(position, 110.0)
        
        assert updated_position.current_price == 110.0
        assert updated_position.unrealized_pnl == 1.0  # (110 - 100) * 0.1
    
    def test_commission_calculation(self, backtest_engine):
        """Test commission calculation."""
        # Test commission calculation
        commission = backtest_engine.calculate_commission(1000.0, 0.001)
        
        # Expected: 1000 * 0.001 = 1.0
        assert commission == 1.0
    
    def test_slippage_calculation(self, backtest_engine):
        """Test slippage calculation."""
        # Test slippage calculation
        slippage = backtest_engine.calculate_slippage(100.0, 0.0005)
        
        # Expected: 100 * 0.0005 = 0.05
        assert slippage == 0.05
    
    def test_equity_curve_calculation(self, backtest_engine):
        """Test equity curve calculation."""
        # Mock trades
        trades = [
            Trade(
                symbol='BTC/USDT',
                side=OrderSide.BUY,
                quantity=0.1,
                price=100.0,
                timestamp=datetime.now(),
                commission=0.1
            ),
            Trade(
                symbol='BTC/USDT',
                side=OrderSide.SELL,
                quantity=0.1,
                price=105.0,
                timestamp=datetime.now(),
                commission=0.105
            )
        ]
        
        # Calculate equity curve
        equity_curve = backtest_engine.calculate_equity_curve(trades)
        
        assert isinstance(equity_curve, pd.DataFrame)
        assert 'timestamp' in equity_curve.columns
        assert 'equity' in equity_curve.columns
        assert 'drawdown' in equity_curve.columns


class TestPerformanceMetrics:
    """Test cases for performance metrics calculation."""
    
    @pytest.fixture
    def performance_metrics(self):
        """Create performance metrics instance."""
        return PerformanceMetrics()
    
    @pytest.fixture
    def sample_trades(self):
        """Create sample trades."""
        return [
            Trade(
                symbol='BTC/USDT',
                side=OrderSide.BUY,
                quantity=0.1,
                price=100.0,
                timestamp=datetime.now() - timedelta(hours=2),
                commission=0.1
            ),
            Trade(
                symbol='BTC/USDT',
                side=OrderSide.SELL,
                quantity=0.1,
                price=105.0,
                timestamp=datetime.now() - timedelta(hours=1),
                commission=0.105
            )
        ]
    
    @pytest.fixture
    def sample_equity_curve(self):
        """Create sample equity curve."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='H')
        equity = 10000 + np.cumsum(np.random.randn(100) * 10)
        
        return pd.DataFrame({
            'timestamp': dates,
            'equity': equity,
            'drawdown': np.maximum.accumulate(equity) - equity
        })
    
    def test_total_return_calculation(self, performance_metrics, sample_equity_curve):
        """Test total return calculation."""
        total_return = performance_metrics.calculate_total_return(sample_equity_curve)
        
        # Total return should be calculated correctly
        expected_return = (sample_equity_curve['equity'].iloc[-1] - sample_equity_curve['equity'].iloc[0]) / sample_equity_curve['equity'].iloc[0]
        assert abs(total_return - expected_return) < 0.01
    
    def test_annualized_return_calculation(self, performance_metrics, sample_equity_curve):
        """Test annualized return calculation."""
        annualized_return = performance_metrics.calculate_annualized_return(sample_equity_curve)
        
        # Should be a reasonable annualized return
        assert isinstance(annualized_return, float)
        assert annualized_return > -1.0  # Not -100% or worse
        assert annualized_return < 10.0  # Not 1000% or better
    
    def test_volatility_calculation(self, performance_metrics, sample_equity_curve):
        """Test volatility calculation."""
        volatility = performance_metrics.calculate_volatility(sample_equity_curve)
        
        # Volatility should be positive
        assert volatility > 0
        assert volatility < 1.0  # Should be reasonable
    
    def test_sharpe_ratio_calculation(self, performance_metrics, sample_equity_curve):
        """Test Sharpe ratio calculation."""
        sharpe_ratio = performance_metrics.calculate_sharpe_ratio(sample_equity_curve)
        
        # Sharpe ratio can be negative or positive
        assert isinstance(sharpe_ratio, float)
        assert sharpe_ratio > -10.0  # Not extremely negative
        assert sharpe_ratio < 10.0  # Not extremely positive
    
    def test_sortino_ratio_calculation(self, performance_metrics, sample_equity_curve):
        """Test Sortino ratio calculation."""
        sortino_ratio = performance_metrics.calculate_sortino_ratio(sample_equity_curve)
        
        # Sortino ratio can be negative or positive
        assert isinstance(sortino_ratio, float)
        assert sortino_ratio > -10.0  # Not extremely negative
        assert sortino_ratio < 10.0  # Not extremely positive
    
    def test_max_drawdown_calculation(self, performance_metrics, sample_equity_curve):
        """Test maximum drawdown calculation."""
        max_drawdown = performance_metrics.calculate_max_drawdown(sample_equity_curve)
        
        # Max drawdown should be positive (as a percentage)
        assert max_drawdown >= 0
        assert max_drawdown <= 1.0  # Should be reasonable
    
    def test_calmar_ratio_calculation(self, performance_metrics, sample_equity_curve):
        """Test Calmar ratio calculation."""
        calmar_ratio = performance_metrics.calculate_calmar_ratio(sample_equity_curve)
        
        # Calmar ratio can be negative or positive
        assert isinstance(calmar_ratio, float)
        assert calmar_ratio > -10.0  # Not extremely negative
        assert calmar_ratio < 10.0  # Not extremely positive
    
    def test_win_rate_calculation(self, performance_metrics, sample_trades):
        """Test win rate calculation."""
        win_rate = performance_metrics.calculate_win_rate(sample_trades)
        
        # Win rate should be between 0 and 1
        assert 0 <= win_rate <= 1
    
    def test_profit_factor_calculation(self, performance_metrics, sample_trades):
        """Test profit factor calculation."""
        profit_factor = performance_metrics.calculate_profit_factor(sample_trades)
        
        # Profit factor should be positive
        assert profit_factor > 0
    
    def test_average_trade_duration(self, performance_metrics, sample_trades):
        """Test average trade duration calculation."""
        avg_duration = performance_metrics.calculate_average_trade_duration(sample_trades)
        
        # Should be a timedelta object
        assert isinstance(avg_duration, timedelta)
        assert avg_duration.total_seconds() > 0
    
    def test_comprehensive_metrics(self, performance_metrics, sample_equity_curve, sample_trades):
        """Test comprehensive metrics calculation."""
        metrics = performance_metrics.calculate_comprehensive_metrics(
            equity_curve=sample_equity_curve,
            trades=sample_trades
        )
        
        # Check all metrics are present
        required_metrics = [
            'total_return', 'annualized_return', 'volatility', 'sharpe_ratio',
            'sortino_ratio', 'max_drawdown', 'calmar_ratio', 'win_rate',
            'profit_factor', 'average_trade_duration'
        ]
        
        for metric in required_metrics:
            assert metric in metrics
            assert metrics[metric] is not None


class TestStrategyComparison:
    """Test cases for strategy comparison."""
    
    @pytest.fixture
    def strategy_comparison(self):
        """Create strategy comparison instance."""
        return StrategyComparison()
    
    @pytest.fixture
    def sample_strategies(self):
        """Create sample strategies."""
        return [
            RSIMACDStrategy(),
            BollingerMeanReversionStrategy()
        ]
    
    @pytest.fixture
    def sample_results(self):
        """Create sample backtest results."""
        return {
            'strategy1': {
                'total_return': 0.15,
                'sharpe_ratio': 1.2,
                'max_drawdown': 0.08,
                'win_rate': 0.6
            },
            'strategy2': {
                'total_return': 0.12,
                'sharpe_ratio': 1.0,
                'max_drawdown': 0.06,
                'win_rate': 0.65
            }
        }
    
    def test_strategy_comparison_initialization(self, strategy_comparison):
        """Test strategy comparison initialization."""
        assert strategy_comparison.comparison_metrics == [
            'total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate'
        ]
    
    def test_compare_strategies(self, strategy_comparison, sample_results):
        """Test strategy comparison."""
        comparison = strategy_comparison.compare_strategies(sample_results)
        
        assert isinstance(comparison, pd.DataFrame)
        assert 'strategy1' in comparison.index
        assert 'strategy2' in comparison.index
        
        # Check metrics are present
        for metric in strategy_comparison.comparison_metrics:
            assert metric in comparison.columns
    
    def test_rank_strategies(self, strategy_comparison, sample_results):
        """Test strategy ranking."""
        ranking = strategy_comparison.rank_strategies(sample_results)
        
        assert isinstance(ranking, pd.DataFrame)
        assert 'rank' in ranking.columns
        assert 'score' in ranking.columns
        
        # Check ranking is correct
        assert ranking['rank'].min() == 1
        assert ranking['rank'].max() == len(sample_results)
    
    def test_strategy_correlation(self, strategy_comparison, sample_results):
        """Test strategy correlation calculation."""
        # Mock equity curves
        equity_curves = {
            'strategy1': pd.Series([100, 101, 102, 103, 104, 105]),
            'strategy2': pd.Series([100, 100.5, 101, 101.5, 102, 102.5])
        }
        
        correlation = strategy_comparison.calculate_strategy_correlation(equity_curves)
        
        assert isinstance(correlation, pd.DataFrame)
        assert 'strategy1' in correlation.index
        assert 'strategy2' in correlation.index
        assert 'strategy1' in correlation.columns
        assert 'strategy2' in correlation.columns
    
    def test_risk_adjusted_returns(self, strategy_comparison, sample_results):
        """Test risk-adjusted returns calculation."""
        risk_adjusted = strategy_comparison.calculate_risk_adjusted_returns(sample_results)
        
        assert isinstance(risk_adjusted, pd.DataFrame)
        assert 'sharpe_ratio' in risk_adjusted.columns
        assert 'sortino_ratio' in risk_adjusted.columns
        assert 'calmar_ratio' in risk_adjusted.columns


class TestOptimizationEngine:
    """Test cases for optimization engine."""
    
    @pytest.fixture
    def optimization_engine(self):
        """Create optimization engine instance."""
        return OptimizationEngine()
    
    @pytest.fixture
    def sample_strategy(self):
        """Create sample strategy."""
        return RSIMACDStrategy()
    
    def test_optimization_initialization(self, optimization_engine):
        """Test optimization engine initialization."""
        assert optimization_engine.optimization_methods == [
            'grid_search', 'genetic_algorithm', 'bayesian_optimization'
        ]
    
    def test_parameter_space_definition(self, optimization_engine, sample_strategy):
        """Test parameter space definition."""
        parameter_space = optimization_engine.define_parameter_space(sample_strategy)
        
        assert isinstance(parameter_space, dict)
        assert 'rsi_period' in parameter_space
        assert 'rsi_overbought' in parameter_space
        assert 'rsi_oversold' in parameter_space
    
    def test_grid_search_optimization(self, optimization_engine, sample_strategy):
        """Test grid search optimization."""
        # Mock backtest function
        def mock_backtest(params):
            return {'sharpe_ratio': np.random.random()}
        
        # Run grid search
        best_params = optimization_engine.grid_search_optimization(
            strategy=sample_strategy,
            parameter_space={'rsi_period': [10, 14, 20]},
            backtest_function=mock_backtest,
            max_iterations=10
        )
        
        assert isinstance(best_params, dict)
        assert 'rsi_period' in best_params
    
    def test_genetic_algorithm_optimization(self, optimization_engine, sample_strategy):
        """Test genetic algorithm optimization."""
        # Mock backtest function
        def mock_backtest(params):
            return {'sharpe_ratio': np.random.random()}
        
        # Run genetic algorithm
        best_params = optimization_engine.genetic_algorithm_optimization(
            strategy=sample_strategy,
            parameter_space={'rsi_period': [10, 14, 20]},
            backtest_function=mock_backtest,
            population_size=10,
            generations=5
        )
        
        assert isinstance(best_params, dict)
        assert 'rsi_period' in best_params
    
    def test_bayesian_optimization(self, optimization_engine, sample_strategy):
        """Test Bayesian optimization."""
        # Mock backtest function
        def mock_backtest(params):
            return {'sharpe_ratio': np.random.random()}
        
        # Run Bayesian optimization
        best_params = optimization_engine.bayesian_optimization(
            strategy=sample_strategy,
            parameter_space={'rsi_period': [10, 20]},
            backtest_function=mock_backtest,
            n_iterations=5
        )
        
        assert isinstance(best_params, dict)
        assert 'rsi_period' in best_params
    
    def test_optimization_validation(self, optimization_engine, sample_strategy):
        """Test optimization validation."""
        # Test parameter validation
        valid_params = {'rsi_period': 14, 'rsi_overbought': 70}
        is_valid = optimization_engine.validate_parameters(sample_strategy, valid_params)
        assert is_valid is True
        
        # Test invalid parameters
        invalid_params = {'rsi_period': -1, 'rsi_overbought': 50}
        is_valid = optimization_engine.validate_parameters(sample_strategy, invalid_params)
        assert is_valid is False


class TestReportGenerator:
    """Test cases for report generator."""
    
    @pytest.fixture
    def report_generator(self):
        """Create report generator instance."""
        return ReportGenerator()
    
    @pytest.fixture
    def sample_backtest_results(self):
        """Create sample backtest results."""
        return {
            'trades': [
                Trade(
                    symbol='BTC/USDT',
                    side=OrderSide.BUY,
                    quantity=0.1,
                    price=100.0,
                    timestamp=datetime.now(),
                    commission=0.1
                )
            ],
            'positions': [],
            'performance': {
                'total_return': 0.15,
                'sharpe_ratio': 1.2,
                'max_drawdown': 0.08,
                'win_rate': 0.6
            },
            'equity_curve': pd.DataFrame({
                'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='H'),
                'equity': 10000 + np.cumsum(np.random.randn(100) * 10),
                'drawdown': np.random.rand(100) * 0.1
            })
        }
    
    def test_report_generation(self, report_generator, sample_backtest_results):
        """Test report generation."""
        report = report_generator.generate_report(
            strategy_name='Test Strategy',
            results=sample_backtest_results
        )
        
        assert isinstance(report, dict)
        assert 'summary' in report
        assert 'performance_metrics' in report
        assert 'trades_analysis' in report
        assert 'risk_analysis' in report
    
    def test_html_report_generation(self, report_generator, sample_backtest_results):
        """Test HTML report generation."""
        html_report = report_generator.generate_html_report(
            strategy_name='Test Strategy',
            results=sample_backtest_results
        )
        
        assert isinstance(html_report, str)
        assert '<html>' in html_report
        assert '<body>' in html_report
        assert 'Test Strategy' in html_report
    
    def test_pdf_report_generation(self, report_generator, sample_backtest_results):
        """Test PDF report generation."""
        # This would require additional dependencies
        # For now, just test that the method exists and can be called
        try:
            pdf_report = report_generator.generate_pdf_report(
                strategy_name='Test Strategy',
                results=sample_backtest_results
            )
            assert isinstance(pdf_report, bytes)
        except ImportError:
            # PDF generation requires additional dependencies
            pass
    
    def test_chart_generation(self, report_generator, sample_backtest_results):
        """Test chart generation."""
        charts = report_generator.generate_charts(sample_backtest_results)
        
        assert isinstance(charts, dict)
        assert 'equity_curve' in charts
        assert 'drawdown' in charts
        assert 'returns_distribution' in charts


class TestBacktestingIntegration:
    """Integration tests for backtesting framework."""
    
    @pytest.fixture
    def mock_exchange(self):
        """Create mock exchange for testing."""
        return MockExchange(initial_balance=10000.0)
    
    @pytest.mark.asyncio
    async def test_complete_backtesting_workflow(self, mock_exchange):
        """Test complete backtesting workflow."""
        await mock_exchange.connect()
        
        # Create backtesting components
        backtest_engine = BacktestEngine()
        performance_metrics = PerformanceMetrics()
        strategy_comparison = StrategyComparison()
        optimization_engine = OptimizationEngine()
        report_generator = ReportGenerator()
        
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='H')
        prices = 100 + np.cumsum(np.random.randn(100) * 0.1)
        
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            ohlcv = OHLCV(
                symbol='BTC/USDT',
                timestamp=date,
                open=price,
                high=price * 1.01,
                low=price * 0.99,
                close=price,
                volume=1000
            )
            data.append(ohlcv)
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'timestamp': candle.timestamp,
            'open': candle.open,
            'high': candle.high,
            'low': candle.low,
            'close': candle.close,
            'volume': candle.volume
        } for candle in data])
        
        # Create strategy
        strategy = RSIMACDStrategy()
        
        # Run backtest
        results = await backtest_engine.run_backtest(
            strategy=strategy,
            data=df,
            symbol='BTC/USDT',
            timeframe='1h'
        )
        
        # Calculate performance metrics
        metrics = performance_metrics.calculate_comprehensive_metrics(
            equity_curve=results['equity_curve'],
            trades=results['trades']
        )
        
        # Generate report
        report = report_generator.generate_report(
            strategy_name='RSIMACDStrategy',
            results=results
        )
        
        # Check results
        assert isinstance(results, dict)
        assert isinstance(metrics, dict)
        assert isinstance(report, dict)
        
        # Check that we have some trades
        assert len(results['trades']) > 0
        
        # Check that performance metrics are calculated
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
    
    def test_backtesting_parameter_validation(self):
        """Test parameter validation across backtesting components."""
        # Test backtest engine
        backtest_engine = BacktestEngine()
        backtest_engine.set_parameters({
            'initial_capital': 20000.0,
            'commission_rate': 0.002,
            'slippage_rate': 0.001
        })
        
        assert backtest_engine.initial_capital == 20000.0
        assert backtest_engine.commission_rate == 0.002
        assert backtest_engine.slippage_rate == 0.001
        
        # Test performance metrics
        performance_metrics = PerformanceMetrics()
        
        # Test strategy comparison
        strategy_comparison = StrategyComparison()
        
        # Test optimization engine
        optimization_engine = OptimizationEngine()
        
        # Test report generator
        report_generator = ReportGenerator()
        
        # All components should be properly initialized
        assert backtest_engine is not None
        assert performance_metrics is not None
        assert strategy_comparison is not None
        assert optimization_engine is not None
        assert report_generator is not None


if __name__ == "__main__":
    pytest.main([__file__])
