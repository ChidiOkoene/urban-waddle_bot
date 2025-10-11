"""
Strategy Optimizer implementation using genetic algorithms and grid search.
"""

import random
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from decimal import Decimal
from datetime import datetime, timedelta
import pandas as pd

from .base_strategy import BaseStrategy
from ..core.data_models import OHLCV, StrategySignal


class StrategyOptimizer:
    """Strategy parameter optimization using genetic algorithms and grid search."""
    
    def __init__(self, strategy_class: type, optimization_method: str = 'genetic'):
        """
        Initialize strategy optimizer.
        
        Args:
            strategy_class: Strategy class to optimize
            optimization_method: 'genetic', 'grid', or 'bayesian'
        """
        self.strategy_class = strategy_class
        self.optimization_method = optimization_method
        self.population_size = 50
        self.generations = 100
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.elite_size = 5
        
        # Optimization results
        self.best_parameters = None
        self.best_score = -float('inf')
        self.optimization_history = []
    
    def optimize(self, ohlcv_data: List[OHLCV], fitness_function: Callable, 
                parameter_ranges: Dict[str, Tuple], max_iterations: int = 1000) -> Dict[str, Any]:
        """
        Optimize strategy parameters.
        
        Args:
            ohlcv_data: Historical OHLCV data for optimization
            fitness_function: Function to evaluate strategy performance
            parameter_ranges: Dictionary of parameter ranges
            max_iterations: Maximum optimization iterations
            
        Returns:
            Best parameters found
        """
        if self.optimization_method == 'genetic':
            return self._genetic_optimization(ohlcv_data, fitness_function, parameter_ranges, max_iterations)
        elif self.optimization_method == 'grid':
            return self._grid_search_optimization(ohlcv_data, fitness_function, parameter_ranges)
        elif self.optimization_method == 'bayesian':
            return self._bayesian_optimization(ohlcv_data, fitness_function, parameter_ranges, max_iterations)
        else:
            raise ValueError(f"Unknown optimization method: {self.optimization_method}")
    
    def _genetic_optimization(self, ohlcv_data: List[OHLCV], fitness_function: Callable,
                            parameter_ranges: Dict[str, Tuple], max_iterations: int) -> Dict[str, Any]:
        """Genetic algorithm optimization."""
        # Initialize population
        population = self._initialize_population(parameter_ranges)
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                try:
                    strategy = self.strategy_class(individual)
                    score = fitness_function(strategy, ohlcv_data)
                    fitness_scores.append(score)
                except Exception as e:
                    fitness_scores.append(-float('inf'))
            
            # Track best individual
            best_idx = np.argmax(fitness_scores)
            best_score = fitness_scores[best_idx]
            
            if best_score > self.best_score:
                self.best_score = best_score
                self.best_parameters = population[best_idx].copy()
            
            # Record generation results
            self.optimization_history.append({
                'generation': generation,
                'best_score': best_score,
                'avg_score': np.mean(fitness_scores),
                'best_parameters': population[best_idx].copy()
            })
            
            # Create next generation
            new_population = []
            
            # Elitism: keep best individuals
            elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    offspring1, offspring2 = self._crossover(parent1, parent2, parameter_ranges)
                else:
                    offspring1, offspring2 = parent1.copy(), parent2.copy()
                
                # Mutation
                offspring1 = self._mutate(offspring1, parameter_ranges)
                offspring2 = self._mutate(offspring2, parameter_ranges)
                
                new_population.extend([offspring1, offspring2])
            
            population = new_population[:self.population_size]
        
        return self.best_parameters
    
    def _grid_search_optimization(self, ohlcv_data: List[OHLCV], fitness_function: Callable,
                                parameter_ranges: Dict[str, Tuple]) -> Dict[str, Any]:
        """Grid search optimization."""
        # Generate parameter combinations
        parameter_combinations = self._generate_parameter_combinations(parameter_ranges)
        
        best_score = -float('inf')
        best_parameters = None
        
        for i, params in enumerate(parameter_combinations):
            try:
                strategy = self.strategy_class(params)
                score = fitness_function(strategy, ohlcv_data)
                
                if score > best_score:
                    best_score = score
                    best_parameters = params.copy()
                
                # Record progress
                self.optimization_history.append({
                    'iteration': i,
                    'score': score,
                    'parameters': params.copy()
                })
                
            except Exception as e:
                continue
        
        self.best_score = best_score
        self.best_parameters = best_parameters
        
        return best_parameters
    
    def _bayesian_optimization(self, ohlcv_data: List[OHLCV], fitness_function: Callable,
                             parameter_ranges: Dict[str, Tuple], max_iterations: int) -> Dict[str, Any]:
        """Bayesian optimization (simplified implementation)."""
        # This is a simplified implementation
        # In practice, you would use libraries like scikit-optimize or optuna
        
        best_score = -float('inf')
        best_parameters = None
        
        # Random sampling for initial points
        n_initial = min(20, max_iterations // 2)
        for i in range(n_initial):
            params = self._random_parameters(parameter_ranges)
            try:
                strategy = self.strategy_class(params)
                score = fitness_function(strategy, ohlcv_data)
                
                if score > best_score:
                    best_score = score
                    best_parameters = params.copy()
                
                self.optimization_history.append({
                    'iteration': i,
                    'score': score,
                    'parameters': params.copy()
                })
                
            except Exception as e:
                continue
        
        # Additional random sampling (simplified Bayesian)
        for i in range(n_initial, max_iterations):
            # Add some noise to best parameters for exploration
            params = self._add_noise_to_parameters(best_parameters, parameter_ranges)
            try:
                strategy = self.strategy_class(params)
                score = fitness_function(strategy, ohlcv_data)
                
                if score > best_score:
                    best_score = score
                    best_parameters = params.copy()
                
                self.optimization_history.append({
                    'iteration': i,
                    'score': score,
                    'parameters': params.copy()
                })
                
            except Exception as e:
                continue
        
        self.best_score = best_score
        self.best_parameters = best_parameters
        
        return best_parameters
    
    def _initialize_population(self, parameter_ranges: Dict[str, Tuple]) -> List[Dict[str, Any]]:
        """Initialize population for genetic algorithm."""
        population = []
        for _ in range(self.population_size):
            individual = self._random_parameters(parameter_ranges)
            population.append(individual)
        return population
    
    def _random_parameters(self, parameter_ranges: Dict[str, Tuple]) -> Dict[str, Any]:
        """Generate random parameters within ranges."""
        params = {}
        for param_name, (min_val, max_val, param_type) in parameter_ranges.items():
            if param_type == int:
                params[param_name] = random.randint(min_val, max_val)
            elif param_type == float:
                params[param_name] = random.uniform(min_val, max_val)
            elif param_type == bool:
                params[param_name] = random.choice([True, False])
            elif param_type == str:
                # For string parameters, assume it's from a list
                if isinstance(max_val, list):
                    params[param_name] = random.choice(max_val)
                else:
                    params[param_name] = str(random.uniform(min_val, max_val))
        return params
    
    def _tournament_selection(self, population: List[Dict], fitness_scores: List[float], 
                            tournament_size: int = 3) -> Dict[str, Any]:
        """Tournament selection for genetic algorithm."""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_scores = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_scores)]
        return population[winner_idx]
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any], 
                  parameter_ranges: Dict[str, Tuple]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Crossover operation for genetic algorithm."""
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        
        # Single-point crossover
        crossover_point = random.randint(1, len(parent1) - 1)
        param_names = list(parent1.keys())
        
        for i in range(crossover_point, len(param_names)):
            param_name = param_names[i]
            offspring1[param_name] = parent2[param_name]
            offspring2[param_name] = parent1[param_name]
        
        return offspring1, offspring2
    
    def _mutate(self, individual: Dict[str, Any], parameter_ranges: Dict[str, Tuple]) -> Dict[str, Any]:
        """Mutation operation for genetic algorithm."""
        mutated = individual.copy()
        
        for param_name, (min_val, max_val, param_type) in parameter_ranges.items():
            if random.random() < self.mutation_rate:
                if param_type == int:
                    mutated[param_name] = random.randint(min_val, max_val)
                elif param_type == float:
                    mutated[param_name] = random.uniform(min_val, max_val)
                elif param_type == bool:
                    mutated[param_name] = random.choice([True, False])
                elif param_type == str and isinstance(max_val, list):
                    mutated[param_name] = random.choice(max_val)
        
        return mutated
    
    def _generate_parameter_combinations(self, parameter_ranges: Dict[str, Tuple]) -> List[Dict[str, Any]]:
        """Generate parameter combinations for grid search."""
        import itertools
        
        # Create parameter lists
        param_lists = {}
        for param_name, (min_val, max_val, param_type) in parameter_ranges.items():
            if param_type == int:
                param_lists[param_name] = list(range(min_val, max_val + 1, max(1, (max_val - min_val) // 5)))
            elif param_type == float:
                param_lists[param_name] = np.linspace(min_val, max_val, 5).tolist()
            elif param_type == bool:
                param_lists[param_name] = [True, False]
            elif param_type == str and isinstance(max_val, list):
                param_lists[param_name] = max_val
        
        # Generate combinations
        param_names = list(param_lists.keys())
        param_values = list(param_lists.values())
        
        combinations = []
        for combination in itertools.product(*param_values):
            params = dict(zip(param_names, combination))
            combinations.append(params)
        
        return combinations
    
    def _add_noise_to_parameters(self, parameters: Dict[str, Any], 
                               parameter_ranges: Dict[str, Tuple]) -> Dict[str, Any]:
        """Add noise to parameters for Bayesian optimization."""
        noisy_params = parameters.copy()
        
        for param_name, (min_val, max_val, param_type) in parameter_ranges.items():
            if param_name in noisy_params:
                if param_type == int:
                    noise = random.randint(-max(1, (max_val - min_val) // 10), 
                                          max(1, (max_val - min_val) // 10))
                    noisy_params[param_name] = max(min_val, min(max_val, noisy_params[param_name] + noise))
                elif param_type == float:
                    noise = random.uniform(-(max_val - min_val) * 0.1, (max_val - min_val) * 0.1)
                    noisy_params[param_name] = max(min_val, min(max_val, noisy_params[param_name] + noise))
        
        return noisy_params
    
    def get_optimization_results(self) -> Dict[str, Any]:
        """Get optimization results."""
        return {
            'best_parameters': self.best_parameters,
            'best_score': self.best_score,
            'optimization_history': self.optimization_history,
            'method': self.optimization_method
        }
    
    def plot_optimization_history(self):
        """Plot optimization history."""
        try:
            import matplotlib.pyplot as plt
            
            if not self.optimization_history:
                print("No optimization history available")
                return
            
            if self.optimization_method == 'genetic':
                generations = [entry['generation'] for entry in self.optimization_history]
                best_scores = [entry['best_score'] for entry in self.optimization_history]
                avg_scores = [entry['avg_score'] for entry in self.optimization_history]
                
                plt.figure(figsize=(10, 6))
                plt.plot(generations, best_scores, label='Best Score', color='blue')
                plt.plot(generations, avg_scores, label='Average Score', color='red')
                plt.xlabel('Generation')
                plt.ylabel('Fitness Score')
                plt.title('Genetic Algorithm Optimization Progress')
                plt.legend()
                plt.grid(True)
                plt.show()
            
            else:
                iterations = [entry['iteration'] for entry in self.optimization_history]
                scores = [entry['score'] for entry in self.optimization_history]
                
                plt.figure(figsize=(10, 6))
                plt.plot(iterations, scores, 'o-', color='blue')
                plt.xlabel('Iteration')
                plt.ylabel('Fitness Score')
                plt.title(f'{self.optimization_method.title()} Optimization Progress')
                plt.grid(True)
                plt.show()
                
        except ImportError:
            print("Matplotlib not available for plotting")


# Common fitness functions
def sharpe_ratio_fitness(strategy: BaseStrategy, ohlcv_data: List[OHLCV]) -> float:
    """Calculate Sharpe ratio as fitness function."""
    try:
        # Simulate strategy performance
        returns = []
        for i in range(1, len(ohlcv_data)):
            signal = strategy.analyze(ohlcv_data[:i+1])
            if signal:
                # Simple return calculation
                price_change = (ohlcv_data[i].close - ohlcv_data[i-1].close) / ohlcv_data[i-1].close
                if signal.signal.value == 'buy':
                    returns.append(price_change)
                elif signal.signal.value == 'sell':
                    returns.append(-price_change)
        
        if not returns:
            return 0.0
        
        # Calculate Sharpe ratio
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Assume risk-free rate of 0
        sharpe_ratio = mean_return / std_return
        return sharpe_ratio
        
    except Exception:
        return -float('inf')


def profit_factor_fitness(strategy: BaseStrategy, ohlcv_data: List[OHLCV]) -> float:
    """Calculate profit factor as fitness function."""
    try:
        # Simulate strategy performance
        profits = []
        losses = []
        
        for i in range(1, len(ohlcv_data)):
            signal = strategy.analyze(ohlcv_data[:i+1])
            if signal:
                price_change = (ohlcv_data[i].close - ohlcv_data[i-1].close) / ohlcv_data[i-1].close
                if signal.signal.value == 'buy':
                    if price_change > 0:
                        profits.append(price_change)
                    else:
                        losses.append(abs(price_change))
                elif signal.signal.value == 'sell':
                    if price_change < 0:
                        profits.append(abs(price_change))
                    else:
                        losses.append(price_change)
        
        if not profits and not losses:
            return 0.0
        
        total_profit = sum(profits) if profits else 0
        total_loss = sum(losses) if losses else 0
        
        if total_loss == 0:
            return float('inf') if total_profit > 0 else 0.0
        
        profit_factor = total_profit / total_loss
        return profit_factor
        
    except Exception:
        return 0.0


def win_rate_fitness(strategy: BaseStrategy, ohlcv_data: List[OHLCV]) -> float:
    """Calculate win rate as fitness function."""
    try:
        wins = 0
        total_trades = 0
        
        for i in range(1, len(ohlcv_data)):
            signal = strategy.analyze(ohlcv_data[:i+1])
            if signal:
                price_change = (ohlcv_data[i].close - ohlcv_data[i-1].close) / ohlcv_data[i-1].close
                total_trades += 1
                
                if signal.signal.value == 'buy' and price_change > 0:
                    wins += 1
                elif signal.signal.value == 'sell' and price_change < 0:
                    wins += 1
        
        if total_trades == 0:
            return 0.0
        
        win_rate = wins / total_trades
        return win_rate
        
    except Exception:
        return 0.0
