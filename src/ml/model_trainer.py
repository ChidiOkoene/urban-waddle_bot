"""
Model training for ML-based trading signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import joblib
import os
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

from .feature_engineering import FeatureEngineer
from ..core.data_models import OHLCV


class ModelTrainer:
    """Model training for ML-based trading signals."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model trainer.
        
        Args:
            config: Model training configuration
        """
        self.config = config
        
        # Model parameters
        self.model_type = config.get('model_type', 'classification')  # classification or regression
        self.target_type = config.get('target_type', 'profit')  # profit, direction, return
        self.lookforward_periods = config.get('lookforward_periods', [1, 3, 5, 10])
        self.min_samples = config.get('min_samples', 1000)
        
        # Model selection
        self.models = self._initialize_models()
        self.best_model = None
        self.best_score = -float('inf')
        self.model_scores = {}
        
        # Feature engineering
        self.feature_engineer = FeatureEngineer(config.get('feature_engineering', {}))
        
        # Training data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = []
        
        # Model evaluation
        self.evaluation_metrics = {}
        self.feature_importance = {}
        
        # Model persistence
        self.model_path = Path(config.get('model_path', 'models'))
        self.model_path.mkdir(exist_ok=True)
    
    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize models for training."""
        models = {}
        
        if self.model_type == 'classification':
            models = {
                'random_forest': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                ),
                'logistic_regression': LogisticRegression(
                    random_state=42,
                    max_iter=1000
                ),
                'svm': SVC(
                    kernel='rbf',
                    random_state=42,
                    probability=True
                ),
                'neural_network': MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    random_state=42,
                    max_iter=1000
                ),
                'xgboost': xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    random_state=42,
                    n_jobs=-1
                ),
                'lightgbm': lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=6,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
            }
        else:  # regression
            models = {
                'random_forest': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                ),
                'linear_regression': LinearRegression(),
                'svm': SVR(
                    kernel='rbf'
                ),
                'neural_network': MLPRegressor(
                    hidden_layer_sizes=(100, 50),
                    random_state=42,
                    max_iter=1000
                ),
                'xgboost': xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    random_state=42,
                    n_jobs=-1
                ),
                'lightgbm': lgb.LGBMRegressor(
                    n_estimators=100,
                    max_depth=6,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
            }
        
        return models
    
    def prepare_training_data(self, ohlcv_data: List[OHLCV], 
                            indicators: Optional[Dict] = None,
                            patterns: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from OHLCV data.
        
        Args:
            ohlcv_data: List of OHLCV candles
            indicators: Technical indicators
            patterns: Pattern recognition results
            
        Returns:
            Tuple of (features, targets)
        """
        if len(ohlcv_data) < self.min_samples:
            raise ValueError(f"Insufficient data: {len(ohlcv_data)} < {self.min_samples}")
        
        # Create features
        features_list = []
        targets_list = []
        
        for i in range(max(self.lookback_periods), len(ohlcv_data) - max(self.lookforward_periods)):
            # Get data window
            window_data = ohlcv_data[i-max(self.lookback_periods):i]
            
            # Create features for this window
            try:
                features = self.feature_engineer.create_features(window_data, indicators, patterns)
                if features.size == 0:
                    continue
                
                # Create targets
                targets = self._create_targets(ohlcv_data, i)
                if targets is None:
                    continue
                
                features_list.append(features)
                targets_list.append(targets)
                
            except Exception as e:
                continue
        
        if not features_list:
            raise ValueError("No valid training samples created")
        
        # Convert to numpy arrays
        X = np.array(features_list)
        y = np.array(targets_list)
        
        # Remove any remaining NaN values
        valid_indices = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_indices]
        y = y[valid_indices]
        
        return X, y
    
    def _create_targets(self, ohlcv_data: List[OHLCV], current_index: int) -> Optional[Union[float, int]]:
        """Create target values for training."""
        current_price = float(ohlcv_data[current_index].close)
        
        if self.target_type == 'profit':
            # Target: profit/loss after lookforward periods
            future_prices = []
            for period in self.lookforward_periods:
                if current_index + period < len(ohlcv_data):
                    future_price = float(ohlcv_data[current_index + period].close)
                    future_prices.append(future_price)
            
            if not future_prices:
                return None
            
            # Calculate average return
            returns = [(fp - current_price) / current_price for fp in future_prices]
            avg_return = np.mean(returns)
            
            # Convert to classification if needed
            if self.model_type == 'classification':
                return 1 if avg_return > 0.01 else 0  # 1% threshold
            else:
                return avg_return
        
        elif self.target_type == 'direction':
            # Target: price direction
            if current_index + max(self.lookforward_periods) >= len(ohlcv_data):
                return None
            
            future_price = float(ohlcv_data[current_index + max(self.lookforward_periods)].close)
            return 1 if future_price > current_price else 0
        
        elif self.target_type == 'return':
            # Target: specific return
            if current_index + max(self.lookforward_periods) >= len(ohlcv_data):
                return None
            
            future_price = float(ohlcv_data[current_index + max(self.lookforward_periods)].close)
            return (future_price - current_price) / current_price
        
        return None
    
    def train_models(self, X: np.ndarray, y: np.ndarray, 
                    test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """
        Train all models and select the best one.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Test set size
            random_state: Random state for reproducibility
            
        Returns:
            Training results
        """
        if X.size == 0 or len(y) == 0:
            raise ValueError("Empty training data")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if self.model_type == 'classification' else None
        )
        
        # Fit feature engineer
        self.X_train = self.feature_engineer.fit_transform(self.X_train, self.y_train)
        self.X_test = self.feature_engineer.transform(self.X_test)
        
        # Get feature names
        self.feature_names = self.feature_engineer.get_feature_names()
        
        # Train models
        results = {}
        
        for model_name, model in self.models.items():
            try:
                print(f"Training {model_name}...")
                
                # Train model
                model.fit(self.X_train, self.y_train)
                
                # Evaluate model
                train_score = self._evaluate_model(model, self.X_train, self.y_train)
                test_score = self._evaluate_model(model, self.X_test, self.y_test)
                
                # Cross-validation
                cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
                
                # Store results
                results[model_name] = {
                    'model': model,
                    'train_score': train_score,
                    'test_score': test_score,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'cv_scores': cv_scores
                }
                
                # Update best model
                if test_score > self.best_score:
                    self.best_score = test_score
                    self.best_model = model
                
                print(f"{model_name}: Train={train_score:.4f}, Test={test_score:.4f}, CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}")
                
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        # Get feature importance for best model
        if self.best_model is not None:
            self.feature_importance = self._get_feature_importance(self.best_model)
        
        # Store evaluation metrics
        self.evaluation_metrics = self._calculate_evaluation_metrics()
        
        return results
    
    def _evaluate_model(self, model, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate model performance."""
        if self.model_type == 'classification':
            return model.score(X, y)  # Accuracy
        else:
            y_pred = model.predict(X)
            return r2_score(y, y_pred)
    
    def _get_feature_importance(self, model) -> Dict[str, float]:
        """Get feature importance from model."""
        importance = {}
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            for i, importance_score in enumerate(importances):
                if i < len(self.feature_names):
                    importance[self.feature_names[i]] = importance_score
        
        elif hasattr(model, 'coef_'):
            # For linear models
            coefs = model.coef_
            if coefs.ndim > 1:
                coefs = coefs[0]  # Take first class for multi-class
            
            for i, coef in enumerate(coefs):
                if i < len(self.feature_names):
                    importance[self.feature_names[i]] = abs(coef)
        
        return importance
    
    def _calculate_evaluation_metrics(self) -> Dict[str, Any]:
        """Calculate detailed evaluation metrics."""
        if self.best_model is None or self.X_test is None or self.y_test is None:
            return {}
        
        metrics = {}
        
        # Predictions
        y_pred = self.best_model.predict(self.X_test)
        
        if self.model_type == 'classification':
            # Classification metrics
            metrics['accuracy'] = self.best_model.score(self.X_test, self.y_test)
            metrics['classification_report'] = classification_report(self.y_test, y_pred, output_dict=True)
            metrics['confusion_matrix'] = confusion_matrix(self.y_test, y_pred).tolist()
            
            # Additional metrics
            from sklearn.metrics import precision_score, recall_score, f1_score
            metrics['precision'] = precision_score(self.y_test, y_pred, average='weighted')
            metrics['recall'] = recall_score(self.y_test, y_pred, average='weighted')
            metrics['f1_score'] = f1_score(self.y_test, y_pred, average='weighted')
        
        else:
            # Regression metrics
            metrics['r2_score'] = r2_score(self.y_test, y_pred)
            metrics['mse'] = mean_squared_error(self.y_test, y_pred)
            metrics['rmse'] = np.sqrt(mean_squared_error(self.y_test, y_pred))
            metrics['mae'] = np.mean(np.abs(self.y_test - y_pred))
        
        return metrics
    
    def optimize_hyperparameters(self, model_name: str, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Optimize hyperparameters for a specific model."""
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Define parameter grids
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            'lightgbm': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        }
        
        if model_name not in param_grids:
            return {'error': f'No parameter grid defined for {model_name}'}
        
        # Perform grid search
        model = self.models[model_name]
        param_grid = param_grids[model_name]
        
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='accuracy' if self.model_type == 'classification' else 'r2',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X, y)
        
        # Update model with best parameters
        self.models[model_name] = grid_search.best_estimator_
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def save_model(self, model_name: str = None, filepath: str = None) -> str:
        """Save the best model to disk."""
        if self.best_model is None:
            raise ValueError("No model trained yet")
        
        if model_name is None:
            model_name = f"best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if filepath is None:
            filepath = self.model_path / f"{model_name}.joblib"
        
        # Save model and metadata
        model_data = {
            'model': self.best_model,
            'feature_engineer': self.feature_engineer,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'target_type': self.target_type,
            'evaluation_metrics': self.evaluation_metrics,
            'feature_importance': self.feature_importance,
            'training_date': datetime.now().isoformat(),
            'config': self.config
        }
        
        joblib.dump(model_data, filepath)
        
        return str(filepath)
    
    def load_model(self, filepath: str) -> Dict[str, Any]:
        """Load a trained model from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        # Restore model and components
        self.best_model = model_data['model']
        self.feature_engineer = model_data['feature_engineer']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.target_type = model_data['target_type']
        self.evaluation_metrics = model_data['evaluation_metrics']
        self.feature_importance = model_data['feature_importance']
        
        return model_data
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        return {
            'model_type': self.model_type,
            'target_type': self.target_type,
            'best_model': type(self.best_model).__name__ if self.best_model else None,
            'best_score': self.best_score,
            'feature_count': len(self.feature_names),
            'training_samples': len(self.X_train) if self.X_train is not None else 0,
            'test_samples': len(self.X_test) if self.X_test is not None else 0,
            'evaluation_metrics': self.evaluation_metrics,
            'feature_importance': dict(sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:10])
        }
