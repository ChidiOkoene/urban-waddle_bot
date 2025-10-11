"""
Optimization Engine for Trading Strategies

This module provides various optimization algorithms to find the best parameters
for trading strategies using historical data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import itertools
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import warnings
warnings.filterwarnings('ignore')

from .backtest_engine import BacktestEngine
from .performance_metrics import PerformanceMetrics
from ..core.data_models import OHLCV, StrategySignal
from ..strategies.base_strategy import BaseStrategy


@dataclass
class OptimizationResult:
    """Result of strategy optimization."""
    best_params: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]]
    total_evaluations: int
    optimization_time: float
    method: str


class OptimizationMethod(ABC):
    """Abstract base class for optimization methods."""
    
    @abstractmethod
    def optimize(self, 
                 strategy_class: type,
                 param_space: Dict[str, List[Any]],
                 data: List[OHLCV],
                 objective_function: Callable[[Dict[str, Any]], float],
                 max_evaluations: int = 100) -> OptimizationResult:
        """Optimize strategy parameters."""
        pass


class GridSearchOptimizer(OptimizationMethod):
    """Grid search optimization method."""
    
    def optimize(self, 
                 strategy_class: type,
                 param_space: Dict[str, List[Any]],
                 data: List[OHLCV],
                 objective_function: Callable[[Dict[str, Any]], float],
                 max_evaluations: int = 100) -> OptimizationResult:
        """Perform grid search optimization."""
        import time
        start_time = time.time()
        
        # Generate all parameter combinations
        param_names = list(param_space.keys())
        param_values = list(param_space.values())
        
        # Limit combinations if too many
        all_combinations = list(itertools.product(*param_values))
        if len(all_combinations) > max_evaluations:
            # Sample randomly
            np.random.seed(42)
            indices = np.random.choice(len(all_combinations), max_evaluations, replace=False)
            all_combinations = [all_combinations[i] for i in indices]
        
        best_score = -np.inf
        best_params = {}
        optimization_history = []
        
        for i, combination in enumerate(all_combinations):
            params = dict(zip(param_names, combination))
            
            try:
                score = objective_function(params)
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                
                optimization_history.append({
                    'evaluation': i + 1,
                    'params': params.copy(),
                    'score': score,
                    'best_score': best_score
                })
                
            except Exception as e:
                print(f"Error evaluating parameters {params}: {e}")
                continue
        
        optimization_time = time.time() - start_time
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            optimization_history=optimization_history,
            total_evaluations=len(optimization_history),
            optimization_time=optimization_time,
            method="Grid Search"
        )


class GeneticAlgorithmOptimizer(OptimizationMethod):
    """Genetic algorithm optimization method."""
    
    def __init__(self, 
                 population_size: int = 50,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 elite_size: int = 5):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
    
    def optimize(self, 
                 strategy_class: type,
                 param_space: Dict[str, List[Any]],
                 data: List[OHLCV],
                 objective_function: Callable[[Dict[str, Any]], float],
                 max_evaluations: int = 100) -> OptimizationResult:
        """Perform genetic algorithm optimization."""
        import time
        start_time = time.time()
        
        # Convert parameter space to numeric ranges
        param_ranges = self._convert_to_ranges(param_space)
        
        # Initialize population
        population = self._initialize_population(param_ranges)
        
        best_score = -np.inf
        best_params = {}
        optimization_history = []
        
        generation = 0
        evaluations = 0
        
        while evaluations < max_evaluations:
            # Evaluate population
            scores = []
            for individual in population:
                if evaluations >= max_evaluations:
                    break
                
                params = self._decode_individual(individual, param_space)
                
                try:
                    score = objective_function(params)
                    scores.append(score)
                    
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                    
                    optimization_history.append({
                        'evaluation': evaluations + 1,
                        'generation': generation,
                        'params': params.copy(),
                        'score': score,
                        'best_score': best_score
                    })
                    
                    evaluations += 1
                    
                except Exception as e:
                    print(f"Error evaluating parameters {params}: {e}")
                    scores.append(-np.inf)
                    evaluations += 1
            
            if evaluations >= max_evaluations:
                break
            
            # Selection, crossover, and mutation
            population = self._evolve_population(population, scores)
            generation += 1
        
        optimization_time = time.time() - start_time
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            optimization_history=optimization_history,
            total_evaluations=evaluations,
            optimization_time=optimization_time,
            method="Genetic Algorithm"
        )
    
    def _convert_to_ranges(self, param_space: Dict[str, List[Any]]) -> Dict[str, Tuple[float, float]]:
        """Convert parameter space to numeric ranges."""
        ranges = {}
        for param, values in param_space.items():
            if all(isinstance(v, (int, float)) for v in values):
                ranges[param] = (min(values), max(values))
            else:
                # For categorical parameters, use indices
                ranges[param] = (0, len(values) - 1)
        return ranges
    
    def _initialize_population(self, param_ranges: Dict[str, Tuple[float, float]]) -> List[Dict[str, float]]:
        """Initialize random population."""
        population = []
        for _ in range(self.population_size):
            individual = {}
            for param, (min_val, max_val) in param_ranges.items():
                individual[param] = np.random.uniform(min_val, max_val)
            population.append(individual)
        return population
    
    def _decode_individual(self, individual: Dict[str, float], param_space: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Decode individual to parameter values."""
        params = {}
        for param, value in individual.items():
            if param in param_space:
                param_values = param_space[param]
                if all(isinstance(v, (int, float)) for v in param_values):
                    # Numeric parameter
                    params[param] = value
                else:
                    # Categorical parameter
                    idx = int(round(value)) % len(param_values)
                    params[param] = param_values[idx]
        return params
    
    def _evolve_population(self, population: List[Dict[str, float]], scores: List[float]) -> List[Dict[str, float]]:
        """Evolve population through selection, crossover, and mutation."""
        # Sort by score
        sorted_pop = sorted(zip(population, scores), key=lambda x: x[1], reverse=True)
        
        # Elite selection
        elite = [ind for ind, _ in sorted_pop[:self.elite_size]]
        
        # Selection (tournament selection)
        new_population = elite.copy()
        
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_selection(population, scores)
            parent2 = self._tournament_selection(population, scores)
            
            # Crossover
            if np.random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)
            
            new_population.extend([child1, child2])
        
        return new_population[:self.population_size]
    
    def _tournament_selection(self, population: List[Dict[str, float]], scores: List[float], tournament_size: int = 3) -> Dict[str, float]:
        """Tournament selection."""
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_scores = [scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_scores)]
        return population[winner_idx].copy()
    
    def _crossover(self, parent1: Dict[str, float], parent2: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Single-point crossover."""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        for param in parent1.keys():
            if np.random.random() < 0.5:
                child1[param], child2[param] = child2[param], child1[param]
        
        return child1, child2
    
    def _mutate(self, individual: Dict[str, float]) -> Dict[str, float]:
        """Gaussian mutation."""
        mutated = individual.copy()
        
        for param in individual.keys():
            if np.random.random() < self.mutation_rate:
                # Gaussian mutation
                mutated[param] += np.random.normal(0, 0.1)
        
        return mutated


class BayesianOptimizer(OptimizationMethod):
    """Bayesian optimization method."""
    
    def __init__(self, n_initial_points: int = 10):
        self.n_initial_points = n_initial_points
    
    def optimize(self, 
                 strategy_class: type,
                 param_space: Dict[str, List[Any]],
                 data: List[OHLCV],
                 objective_function: Callable[[Dict[str, Any]], float],
                 max_evaluations: int = 100) -> OptimizationResult:
        """Perform Bayesian optimization."""
        import time
        start_time = time.time()
        
        # Convert parameter space to numeric ranges
        param_ranges = self._convert_to_ranges(param_space)
        param_names = list(param_ranges.keys())
        
        # Initialize with random points
        X = []
        y = []
        
        for i in range(min(self.n_initial_points, max_evaluations)):
            params = self._sample_random_params(param_ranges, param_space)
            try:
                score = objective_function(params)
                X.append([params.get(name, 0) for name in param_names])
                y.append(score)
            except Exception as e:
                print(f"Error evaluating parameters {params}: {e}")
                continue
        
        if len(X) == 0:
            raise ValueError("No valid initial evaluations")
        
        X = np.array(X)
        y = np.array(y)
        
        best_score = np.max(y)
        best_params = self._decode_params(X[np.argmax(y)], param_names, param_space)
        optimization_history = []
        
        # Add initial points to history
        for i, (x, score) in enumerate(zip(X, y)):
            params = self._decode_params(x, param_names, param_space)
            optimization_history.append({
                'evaluation': i + 1,
                'params': params,
                'score': score,
                'best_score': best_score
            })
        
        # Bayesian optimization loop
        for i in range(len(X), max_evaluations):
            # Fit Gaussian Process
            kernel = Matern(length_scale=1.0, nu=2.5)
            gp = GaussianProcessRegressor(kernel=kernel, random_state=42)
            gp.fit(X, y)
            
            # Acquisition function (Expected Improvement)
            def acquisition(x):
                x = np.array(x).reshape(1, -1)
                mu, sigma = gp.predict(x, return_std=True)
                improvement = mu - best_score
                z = improvement / (sigma + 1e-9)
                return improvement * (1 + z * (1 + z) / 2)
            
            # Optimize acquisition function
            bounds = [param_ranges[name] for name in param_names]
            result = minimize(lambda x: -acquisition(x), 
                            x0=np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds]),
                            bounds=bounds,
                            method='L-BFGS-B')
            
            if result.success:
                next_params = self._decode_params(result.x, param_names, param_space)
                
                try:
                    score = objective_function(next_params)
                    
                    if score > best_score:
                        best_score = score
                        best_params = next_params.copy()
                    
                    X = np.vstack([X, result.x])
                    y = np.append(y, score)
                    
                    optimization_history.append({
                        'evaluation': i + 1,
                        'params': next_params,
                        'score': score,
                        'best_score': best_score
                    })
                    
                except Exception as e:
                    print(f"Error evaluating parameters {next_params}: {e}")
                    continue
        
        optimization_time = time.time() - start_time
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            optimization_history=optimization_history,
            total_evaluations=len(optimization_history),
            optimization_time=optimization_time,
            method="Bayesian Optimization"
        )
    
    def _convert_to_ranges(self, param_space: Dict[str, List[Any]]) -> Dict[str, Tuple[float, float]]:
        """Convert parameter space to numeric ranges."""
        ranges = {}
        for param, values in param_space.items():
            if all(isinstance(v, (int, float)) for v in values):
                ranges[param] = (min(values), max(values))
            else:
                # For categorical parameters, use indices
                ranges[param] = (0, len(values) - 1)
        return ranges
    
    def _sample_random_params(self, param_ranges: Dict[str, Tuple[float, float]], param_space: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Sample random parameters."""
        params = {}
        for param, (min_val, max_val) in param_ranges.items():
            if param in param_space:
                param_values = param_space[param]
                if all(isinstance(v, (int, float)) for v in param_values):
                    # Numeric parameter
                    params[param] = np.random.uniform(min_val, max_val)
                else:
                    # Categorical parameter
                    idx = np.random.randint(0, len(param_values))
                    params[param] = param_values[idx]
        return params
    
    def _decode_params(self, x: np.ndarray, param_names: List[str], param_space: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Decode numeric parameters to actual values."""
        params = {}
        for i, param in enumerate(param_names):
            if param in param_space:
                param_values = param_space[param]
                if all(isinstance(v, (int, float)) for v in param_values):
                    # Numeric parameter
                    params[param] = x[i]
                else:
                    # Categorical parameter
                    idx = int(round(x[i])) % len(param_values)
                    params[param] = param_values[idx]
        return params


class OptimizationEngine:
    """Main optimization engine that coordinates different optimization methods."""
    
    def __init__(self):
        self.optimizers = {
            'grid_search': GridSearchOptimizer(),
            'genetic_algorithm': GeneticAlgorithmOptimizer(),
            'bayesian': BayesianOptimizer()
        }
    
    def optimize_strategy(self,
                        strategy_class: type,
                        param_space: Dict[str, List[Any]],
                        data: List[OHLCV],
                        method: str = 'bayesian',
                        max_evaluations: int = 100,
                        objective_metric: str = 'sharpe_ratio') -> OptimizationResult:
        """
        Optimize strategy parameters.
        
        Args:
            strategy_class: Strategy class to optimize
            param_space: Parameter space to search
            data: Historical data for backtesting
            method: Optimization method ('grid_search', 'genetic_algorithm', 'bayesian')
            max_evaluations: Maximum number of evaluations
            objective_metric: Metric to optimize ('sharpe_ratio', 'total_return', 'max_drawdown', 'profit_factor')
        
        Returns:
            OptimizationResult with best parameters and performance
        """
        if method not in self.optimizers:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Create objective function
        def objective_function(params: Dict[str, Any]) -> float:
            try:
                # Create strategy instance with parameters
                strategy = strategy_class(**params)
                
                # Run backtest
                engine = BacktestEngine()
                results = engine.run_backtest(strategy, data)
                
                # Calculate objective metric
                metrics = PerformanceMetrics()
                performance = metrics.calculate_metrics(results)
                
                if objective_metric == 'sharpe_ratio':
                    return performance.get('sharpe_ratio', 0)
                elif objective_metric == 'total_return':
                    return performance.get('total_return', 0)
                elif objective_metric == 'max_drawdown':
                    return -performance.get('max_drawdown', 0)  # Minimize drawdown
                elif objective_metric == 'profit_factor':
                    return performance.get('profit_factor', 0)
                else:
                    return performance.get('sharpe_ratio', 0)
                    
            except Exception as e:
                print(f"Error in objective function: {e}")
                return -np.inf
        
        # Run optimization
        optimizer = self.optimizers[method]
        result = optimizer.optimize(
            strategy_class=strategy_class,
            param_space=param_space,
            data=data,
            objective_function=objective_function,
            max_evaluations=max_evaluations
        )
        
        return result
    
    def multi_objective_optimize(self,
                                strategy_class: type,
                                param_space: Dict[str, List[Any]],
                                data: List[OHLCV],
                                objectives: List[str],
                                max_evaluations: int = 100) -> List[OptimizationResult]:
        """
        Multi-objective optimization using different methods for each objective.
        
        Args:
            strategy_class: Strategy class to optimize
            param_space: Parameter space to search
            data: Historical data for backtesting
            objectives: List of objectives to optimize
            max_evaluations: Maximum number of evaluations per objective
        
        Returns:
            List of OptimizationResult for each objective
        """
        results = []
        
        for objective in objectives:
            print(f"Optimizing for objective: {objective}")
            result = self.optimize_strategy(
                strategy_class=strategy_class,
                param_space=param_space,
                data=data,
                method='bayesian',  # Use Bayesian for multi-objective
                max_evaluations=max_evaluations,
                objective_metric=objective
            )
            results.append(result)
        
        return results
    
    def get_optimization_summary(self, result: OptimizationResult) -> Dict[str, Any]:
        """Get summary of optimization results."""
        return {
            'method': result.method,
            'best_params': result.best_params,
            'best_score': result.best_score,
            'total_evaluations': result.total_evaluations,
            'optimization_time': result.optimization_time,
            'convergence': self._analyze_convergence(result.optimization_history)
        }
    
    def _analyze_convergence(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze convergence of optimization."""
        if len(history) < 2:
            return {'converged': False, 'convergence_rate': 0}
        
        scores = [h['score'] for h in history]
        best_scores = [h['best_score'] for h in history]
        
        # Calculate convergence rate
        improvement = best_scores[-1] - best_scores[0]
        convergence_rate = improvement / len(history) if len(history) > 0 else 0
        
        # Check if converged (no improvement in last 20% of evaluations)
        last_20_percent = int(len(history) * 0.2)
        if last_20_percent > 0:
            recent_best = best_scores[-last_20_percent:]
            converged = max(recent_best) - min(recent_best) < 0.01
        else:
            converged = False
        
        return {
            'converged': converged,
            'convergence_rate': convergence_rate,
            'final_improvement': improvement,
            'score_std': np.std(scores),
            'best_score_std': np.std(best_scores)
        }