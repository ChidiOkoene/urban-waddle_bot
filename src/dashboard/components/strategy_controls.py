"""
Strategy Controls Component for Trading Dashboard

This module provides components for configuring and controlling trading strategies,
including parameter adjustment, strategy selection, and real-time monitoring.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import yaml

from ...core.data_models import StrategySignal, OrderSide
from ...strategies.base_strategy import BaseStrategy


class StrategyControls:
    """Component for strategy configuration and control."""
    
    def __init__(self):
        self.strategy_configs = self._load_strategy_configs()
        self.default_params = self._get_default_parameters()
    
    def _load_strategy_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load strategy configurations from file."""
        try:
            config_path = "config/strategy_configs.yaml"
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return self._get_default_strategy_configs()
        except Exception as e:
            st.error(f"Error loading strategy configs: {e}")
            return self._get_default_strategy_configs()
    
    def _get_default_strategy_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get default strategy configurations."""
        return {
            'rsi_macd': {
                'name': 'RSI + MACD Strategy',
                'description': 'Combines RSI and MACD signals for entry/exit',
                'parameters': {
                    'rsi_period': {'type': 'int', 'min': 5, 'max': 50, 'default': 14},
                    'rsi_overbought': {'type': 'float', 'min': 60, 'max': 90, 'default': 70},
                    'rsi_oversold': {'type': 'float', 'min': 10, 'max': 40, 'default': 30},
                    'macd_fast': {'type': 'int', 'min': 5, 'max': 20, 'default': 12},
                    'macd_slow': {'type': 'int', 'min': 20, 'max': 50, 'default': 26},
                    'macd_signal': {'type': 'int', 'min': 5, 'max': 15, 'default': 9}
                }
            },
            'bollinger': {
                'name': 'Bollinger Bands Mean Reversion',
                'description': 'Mean reversion strategy using Bollinger Bands',
                'parameters': {
                    'period': {'type': 'int', 'min': 10, 'max': 50, 'default': 20},
                    'std_dev': {'type': 'float', 'min': 1.0, 'max': 3.0, 'default': 2.0},
                    'entry_threshold': {'type': 'float', 'min': 0.5, 'max': 2.0, 'default': 1.0}
                }
            },
            'ema_crossover': {
                'name': 'EMA Crossover Strategy',
                'description': 'Moving average crossover strategy',
                'parameters': {
                    'fast_period': {'type': 'int', 'min': 5, 'max': 20, 'default': 12},
                    'slow_period': {'type': 'int', 'min': 20, 'max': 50, 'default': 26},
                    'signal_threshold': {'type': 'float', 'min': 0.001, 'max': 0.01, 'default': 0.005}
                }
            },
            'grid_bot': {
                'name': 'Grid Bot Strategy',
                'description': 'Grid trading strategy with multiple levels',
                'parameters': {
                    'grid_levels': {'type': 'int', 'min': 5, 'max': 20, 'default': 10},
                    'grid_spacing': {'type': 'float', 'min': 0.001, 'max': 0.05, 'default': 0.01},
                    'max_position_size': {'type': 'float', 'min': 0.01, 'max': 1.0, 'default': 0.1}
                }
            },
            'dca': {
                'name': 'Dollar Cost Averaging',
                'description': 'DCA strategy with fixed intervals',
                'parameters': {
                    'interval_hours': {'type': 'int', 'min': 1, 'max': 168, 'default': 24},
                    'position_size': {'type': 'float', 'min': 0.01, 'max': 1.0, 'default': 0.05},
                    'max_positions': {'type': 'int', 'min': 1, 'max': 50, 'default': 10}
                }
            }
        }
    
    def _get_default_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get default parameters for each strategy."""
        return {
            'rsi_macd': {
                'rsi_period': 14,
                'rsi_overbought': 70,
                'rsi_oversold': 30,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9
            },
            'bollinger': {
                'period': 20,
                'std_dev': 2.0,
                'entry_threshold': 1.0
            },
            'ema_crossover': {
                'fast_period': 12,
                'slow_period': 26,
                'signal_threshold': 0.005
            },
            'grid_bot': {
                'grid_levels': 10,
                'grid_spacing': 0.01,
                'max_position_size': 0.1
            },
            'dca': {
                'interval_hours': 24,
                'position_size': 0.05,
                'max_positions': 10
            }
        }
    
    def render_strategy_parameters(self, strategy_name: str):
        """
        Render strategy parameter configuration interface.
        
        Args:
            strategy_name: Name of the strategy to configure
        """
        if strategy_name not in self.strategy_configs:
            st.error(f"Strategy '{strategy_name}' not found")
            return
        
        config = self.strategy_configs[strategy_name]
        
        st.subheader(f"Configure {config['name']}")
        st.write(config['description'])
        
        # Get current parameters
        current_params = st.session_state.get(f'{strategy_name}_params', 
                                            self.default_params.get(strategy_name, {}))
        
        # Render parameter controls
        new_params = self._render_parameter_controls(strategy_name, config['parameters'], current_params)
        
        # Update session state
        st.session_state[f'{strategy_name}_params'] = new_params
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save Parameters", key=f"save_{strategy_name}"):
                self._save_strategy_parameters(strategy_name, new_params)
        
        with col2:
            if st.button("🔄 Reset to Defaults", key=f"reset_{strategy_name}"):
                self._reset_to_defaults(strategy_name)
        
        with col3:
            if st.button("📊 Test Strategy", key=f"test_{strategy_name}"):
                self._test_strategy(strategy_name, new_params)
    
    def render_strategy_selection(self):
        """Render strategy selection interface."""
        st.subheader("Strategy Selection")
        
        # Available strategies
        available_strategies = list(self.strategy_configs.keys())
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_strategy = st.selectbox(
                "Select Strategy",
                available_strategies,
                index=0,
                key="strategy_selection"
            )
        
        with col2:
            if st.button("✅ Activate Strategy"):
                self._activate_strategy(selected_strategy)
        
        # Display strategy info
        if selected_strategy in self.strategy_configs:
            config = self.strategy_configs[selected_strategy]
            st.info(f"**{config['name']}**: {config['description']}")
    
    def render_strategy_monitoring(self, strategy_name: str):
        """
        Render strategy monitoring interface.
        
        Args:
            strategy_name: Name of the strategy to monitor
        """
        st.subheader(f"Monitor {strategy_name}")
        
        # Strategy status
        self._render_strategy_status(strategy_name)
        
        # Recent signals
        self._render_recent_signals(strategy_name)
        
        # Performance metrics
        self._render_strategy_performance(strategy_name)
        
        # Risk metrics
        self._render_strategy_risk(strategy_name)
    
    def render_strategy_comparison(self):
        """Render strategy comparison interface."""
        st.subheader("Strategy Comparison")
        
        # Select strategies to compare
        available_strategies = list(self.strategy_configs.keys())
        
        selected_strategies = st.multiselect(
            "Select Strategies to Compare",
            available_strategies,
            default=available_strategies[:3]
        )
        
        if len(selected_strategies) < 2:
            st.warning("Please select at least 2 strategies to compare")
            return
        
        # Comparison metrics
        self._render_comparison_metrics(selected_strategies)
        
        # Comparison charts
        self._render_comparison_charts(selected_strategies)
    
    def _render_parameter_controls(self, 
                                 strategy_name: str, 
                                 parameters: Dict[str, Dict[str, Any]], 
                                 current_params: Dict[str, Any]) -> Dict[str, Any]:
        """Render parameter controls and return updated parameters."""
        new_params = {}
        
        for param_name, param_config in parameters.items():
            param_type = param_config['type']
            min_val = param_config['min']
            max_val = param_config['max']
            default_val = param_config['default']
            
            current_val = current_params.get(param_name, default_val)
            
            if param_type == 'int':
                new_params[param_name] = st.slider(
                    param_name.replace('_', ' ').title(),
                    min_value=min_val,
                    max_value=max_val,
                    value=int(current_val),
                    key=f"{strategy_name}_{param_name}"
                )
            elif param_type == 'float':
                new_params[param_name] = st.slider(
                    param_name.replace('_', ' ').title(),
                    min_value=min_val,
                    max_value=max_val,
                    value=float(current_val),
                    step=0.001,
                    key=f"{strategy_name}_{param_name}"
                )
            elif param_type == 'bool':
                new_params[param_name] = st.checkbox(
                    param_name.replace('_', ' ').title(),
                    value=bool(current_val),
                    key=f"{strategy_name}_{param_name}"
                )
            elif param_type == 'select':
                options = param_config.get('options', [])
                new_params[param_name] = st.selectbox(
                    param_name.replace('_', ' ').title(),
                    options=options,
                    index=options.index(current_val) if current_val in options else 0,
                    key=f"{strategy_name}_{param_name}"
                )
        
        return new_params
    
    def _render_strategy_status(self, strategy_name: str):
        """Render strategy status information."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Status", "🟢 Active")
        with col2:
            st.metric("Signals Today", "5")
        with col3:
            st.metric("Success Rate", "68%")
        with col4:
            st.metric("Last Signal", "2 min ago")
    
    def _render_recent_signals(self, strategy_name: str):
        """Render recent signals table."""
        st.subheader("Recent Signals")
        
        # Sample signal data
        signals_data = [
            {
                'Time': '14:32:15',
                'Symbol': 'BTC/USDT',
                'Signal': 'BUY',
                'Price': '$45,250',
                'Strength': 'Strong',
                'Status': 'Executed'
            },
            {
                'Time': '14:28:42',
                'Symbol': 'ETH/USDT',
                'Signal': 'SELL',
                'Price': '$3,120',
                'Strength': 'Medium',
                'Status': 'Executed'
            },
            {
                'Time': '14:25:18',
                'Symbol': 'BNB/USDT',
                'Signal': 'BUY',
                'Price': '$320',
                'Strength': 'Weak',
                'Status': 'Pending'
            }
        ]
        
        signals_df = pd.DataFrame(signals_data)
        st.dataframe(signals_df, use_container_width=True)
    
    def _render_strategy_performance(self, strategy_name: str):
        """Render strategy performance metrics."""
        st.subheader("Strategy Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Return", "12.5%", "2.1%")
        with col2:
            st.metric("Sharpe Ratio", "1.85", "0.15")
        with col3:
            st.metric("Max Drawdown", "-8.2%", "1.3%")
        with col4:
            st.metric("Win Rate", "68%", "3%")
    
    def _render_strategy_risk(self, strategy_name: str):
        """Render strategy risk metrics."""
        st.subheader("Risk Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("VaR (95%)", "-2.1%")
        with col2:
            st.metric("CVaR (95%)", "-3.2%")
        with col3:
            st.metric("Volatility", "15.8%")
        with col4:
            st.metric("Beta", "0.95")
    
    def _render_comparison_metrics(self, strategies: List[str]):
        """Render comparison metrics table."""
        st.subheader("Comparison Metrics")
        
        comparison_data = []
        for strategy in strategies:
            comparison_data.append({
                'Strategy': strategy,
                'Total Return (%)': f"{np.random.uniform(5, 20):.2f}",
                'Sharpe Ratio': f"{np.random.uniform(0.5, 2.5):.2f}",
                'Max Drawdown (%)': f"{np.random.uniform(-15, -5):.2f}",
                'Win Rate (%)': f"{np.random.uniform(50, 80):.1f}",
                'Volatility (%)': f"{np.random.uniform(10, 25):.1f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
    
    def _render_comparison_charts(self, strategies: List[str]):
        """Render comparison charts."""
        st.subheader("Comparison Charts")
        
        # Sample data for charts
        metrics = ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate']
        
        fig_data = []
        for strategy in strategies:
            for metric in metrics:
                value = np.random.uniform(0, 100) if metric != 'Sharpe Ratio' else np.random.uniform(0, 3)
                fig_data.append({
                    'Strategy': strategy,
                    'Metric': metric,
                    'Value': value
                })
        
        df = pd.DataFrame(fig_data)
        
        # Create bar chart
        import plotly.express as px
        fig = px.bar(df, x='Metric', y='Value', color='Strategy', 
                    title='Strategy Comparison', barmode='group')
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _save_strategy_parameters(self, strategy_name: str, parameters: Dict[str, Any]):
        """Save strategy parameters to file."""
        try:
            # Load existing config
            config_path = "config/strategy_configs.yaml"
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            except FileNotFoundError:
                config = {}
            
            # Update parameters
            if strategy_name not in config:
                config[strategy_name] = {}
            
            config[strategy_name]['parameters'] = parameters
            
            # Save config
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            st.success(f"Parameters saved for {strategy_name}")
            
        except Exception as e:
            st.error(f"Error saving parameters: {e}")
    
    def _reset_to_defaults(self, strategy_name: str):
        """Reset strategy parameters to defaults."""
        if strategy_name in self.default_params:
            st.session_state[f'{strategy_name}_params'] = self.default_params[strategy_name]
            st.success(f"Parameters reset to defaults for {strategy_name}")
        else:
            st.error(f"No default parameters found for {strategy_name}")
    
    def _test_strategy(self, strategy_name: str, parameters: Dict[str, Any]):
        """Test strategy with given parameters."""
        st.info(f"Testing {strategy_name} with parameters: {parameters}")
        
        # This would normally run a backtest or paper trade
        with st.spinner("Running strategy test..."):
            # Simulate test
            import time
            time.sleep(2)
            
            st.success("Strategy test completed!")
            
            # Show test results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Test Return", "8.5%")
            with col2:
                st.metric("Test Sharpe", "1.2")
            with col3:
                st.metric("Test Drawdown", "-5.2%")
    
    def _activate_strategy(self, strategy_name: str):
        """Activate the selected strategy."""
        st.success(f"Strategy '{strategy_name}' activated!")
        
        # This would normally update the trading bot configuration
        st.session_state['active_strategy'] = strategy_name
    
    def render_strategy_optimization(self, strategy_name: str):
        """Render strategy optimization interface."""
        st.subheader(f"Optimize {strategy_name}")
        
        if strategy_name not in self.strategy_configs:
            st.error(f"Strategy '{strategy_name}' not found")
            return
        
        config = self.strategy_configs[strategy_name]
        
        # Optimization parameters
        col1, col2 = st.columns(2)
        
        with col1:
            optimization_method = st.selectbox(
                "Optimization Method",
                ["Grid Search", "Genetic Algorithm", "Bayesian Optimization"],
                key="opt_method"
            )
        
        with col2:
            max_evaluations = st.slider(
                "Max Evaluations",
                min_value=10,
                max_value=1000,
                value=100,
                key="max_eval"
            )
        
        # Objective function
        objective = st.selectbox(
            "Objective Function",
            ["Sharpe Ratio", "Total Return", "Max Drawdown", "Profit Factor"],
            key="objective"
        )
        
        # Parameter ranges
        st.subheader("Parameter Ranges")
        param_ranges = self._render_parameter_ranges(strategy_name, config['parameters'])
        
        # Run optimization
        if st.button("🚀 Run Optimization"):
            self._run_optimization(strategy_name, optimization_method, 
                                max_evaluations, objective, param_ranges)
    
    def _render_parameter_ranges(self, strategy_name: str, parameters: Dict[str, Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
        """Render parameter range controls."""
        ranges = {}
        
        for param_name, param_config in parameters.items():
            col1, col2 = st.columns(2)
            
            with col1:
                min_val = st.number_input(
                    f"{param_name} Min",
                    value=param_config['min'],
                    key=f"opt_{strategy_name}_{param_name}_min"
                )
            
            with col2:
                max_val = st.number_input(
                    f"{param_name} Max",
                    value=param_config['max'],
                    key=f"opt_{strategy_name}_{param_name}_max"
                )
            
            ranges[param_name] = (min_val, max_val)
        
        return ranges
    
    def _run_optimization(self, strategy_name: str, method: str, 
                         max_evaluations: int, objective: str, 
                         param_ranges: Dict[str, Tuple[float, float]]):
        """Run strategy optimization."""
        st.info(f"Running {method} optimization for {strategy_name}")
        
        # This would normally call the optimization engine
        with st.spinner("Optimizing parameters..."):
            import time
            time.sleep(3)
            
            st.success("Optimization completed!")
            
            # Show results
            st.subheader("Optimization Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Best Parameters:**")
                for param, value in param_ranges.items():
                    st.write(f"- {param}: {value[0]:.3f}")
            
            with col2:
                st.write("**Performance:**")
                st.write(f"- {objective}: 1.85")
                st.write("- Total Return: 15.2%")
                st.write("- Max Drawdown: -6.8%")
    
    def get_strategy_parameters(self, strategy_name: str) -> Dict[str, Any]:
        """Get current strategy parameters."""
        return st.session_state.get(f'{strategy_name}_params', 
                                   self.default_params.get(strategy_name, {}))
    
    def update_strategy_parameters(self, strategy_name: str, parameters: Dict[str, Any]):
        """Update strategy parameters in session state."""
        st.session_state[f'{strategy_name}_params'] = parameters
