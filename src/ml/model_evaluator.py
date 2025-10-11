"""
ML model evaluation and validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import joblib

from .model_trainer import ModelTrainer
from .predictor import MLPredictor
from ..core.data_models import OHLCV


class ModelEvaluator:
    """ML model evaluation and validation."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model evaluator.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
        
        # Evaluation parameters
        self.validation_method = config.get('validation_method', 'time_series_split')
        self.n_splits = config.get('n_splits', 5)
        self.test_size = config.get('test_size', 0.2)
        
        # Performance metrics
        self.evaluation_results = {}
        self.feature_importance_scores = {}
        self.model_comparison = {}
        
        # Visualization settings
        self.plot_style = config.get('plot_style', 'seaborn')
        self.figure_size = config.get('figure_size', (12, 8))
        
        # Set plotting style
        plt.style.use(self.plot_style)
    
    def evaluate_model(self, model_path: str, test_data: List[OHLCV], 
                      model_type: str = 'classification') -> Dict[str, Any]:
        """
        Evaluate a trained model on test data.
        
        Args:
            model_path: Path to trained model
            test_data: Test OHLCV data
            model_type: Type of model (classification or regression)
            
        Returns:
            Evaluation results
        """
        try:
            # Load model
            model_data = joblib.load(model_path)
            model = model_data['model']
            feature_engineer = model_data['feature_engineer']
            
            # Prepare test data
            trainer = ModelTrainer(self.config)
            X_test, y_test = trainer.prepare_training_data(test_data)
            
            if X_test.size == 0 or len(y_test) == 0:
                return {'error': 'No valid test data'}
            
            # Transform features
            X_test_transformed = feature_engineer.transform(X_test)
            
            # Make predictions
            y_pred = model.predict(X_test_transformed)
            
            # Calculate metrics
            if model_type == 'classification':
                metrics = self._calculate_classification_metrics(y_test, y_pred, model, X_test_transformed)
            else:
                metrics = self._calculate_regression_metrics(y_test, y_pred)
            
            # Store results
            self.evaluation_results[model_path] = {
                'metrics': metrics,
                'predictions': y_pred,
                'actual': y_test,
                'model_type': model_type,
                'evaluation_date': datetime.now().isoformat()
            }
            
            return metrics
            
        except Exception as e:
            return {'error': f'Evaluation failed: {str(e)}'}
    
    def cross_validate_model(self, model_path: str, train_data: List[OHLCV], 
                           model_type: str = 'classification') -> Dict[str, Any]:
        """
        Perform cross-validation on a model.
        
        Args:
            model_path: Path to trained model
            train_data: Training OHLCV data
            model_type: Type of model (classification or regression)
            
        Returns:
            Cross-validation results
        """
        try:
            # Load model
            model_data = joblib.load(model_path)
            model = model_data['model']
            feature_engineer = model_data['feature_engineer']
            
            # Prepare training data
            trainer = ModelTrainer(self.config)
            X, y = trainer.prepare_training_data(train_data)
            
            if X.size == 0 or len(y) == 0:
                return {'error': 'No valid training data'}
            
            # Transform features
            X_transformed = feature_engineer.fit_transform(X, y)
            
            # Choose cross-validation strategy
            if self.validation_method == 'time_series_split':
                cv = TimeSeriesSplit(n_splits=self.n_splits)
            else:
                cv = self.n_splits
            
            # Perform cross-validation
            scoring = 'accuracy' if model_type == 'classification' else 'r2'
            cv_scores = cross_val_score(model, X_transformed, y, cv=cv, scoring=scoring)
            
            return {
                'cv_scores': cv_scores.tolist(),
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_min': cv_scores.min(),
                'cv_max': cv_scores.max(),
                'validation_method': self.validation_method,
                'n_splits': self.n_splits
            }
            
        except Exception as e:
            return {'error': f'Cross-validation failed: {str(e)}'}
    
    def compare_models(self, model_paths: List[str], test_data: List[OHLCV]) -> Dict[str, Any]:
        """
        Compare multiple models.
        
        Args:
            model_paths: List of model paths
            test_data: Test OHLCV data
            
        Returns:
            Model comparison results
        """
        comparison_results = {}
        
        for model_path in model_paths:
            try:
                # Load model
                model_data = joblib.load(model_path)
                model_type = model_data.get('model_type', 'classification')
                
                # Evaluate model
                evaluation = self.evaluate_model(model_path, test_data, model_type)
                
                if 'error' not in evaluation:
                    comparison_results[model_path] = evaluation
                
            except Exception as e:
                comparison_results[model_path] = {'error': str(e)}
        
        # Rank models
        if comparison_results:
            ranked_models = self._rank_models(comparison_results)
            comparison_results['ranking'] = ranked_models
        
        self.model_comparison = comparison_results
        return comparison_results
    
    def walk_forward_analysis(self, model_path: str, ohlcv_data: List[OHLCV], 
                            window_size: int = 1000, step_size: int = 100) -> Dict[str, Any]:
        """
        Perform walk-forward analysis.
        
        Args:
            model_path: Path to trained model
            ohlcv_data: Historical OHLCV data
            window_size: Size of training window
            step_size: Step size for moving window
            
        Returns:
            Walk-forward analysis results
        """
        try:
            # Load model
            model_data = joblib.load(model_path)
            model_type = model_data.get('model_type', 'classification')
            
            results = []
            
            # Walk forward through data
            for start_idx in range(window_size, len(ohlcv_data) - step_size, step_size):
                # Training window
                train_data = ohlcv_data[start_idx-window_size:start_idx]
                
                # Test window
                test_data = ohlcv_data[start_idx:start_idx+step_size]
                
                # Prepare data
                trainer = ModelTrainer(self.config)
                X_train, y_train = trainer.prepare_training_data(train_data)
                X_test, y_test = trainer.prepare_training_data(test_data)
                
                if X_train.size == 0 or X_test.size == 0:
                    continue
                
                # Transform features
                feature_engineer = model_data['feature_engineer']
                X_train_transformed = feature_engineer.fit_transform(X_train, y_train)
                X_test_transformed = feature_engineer.transform(X_test)
                
                # Train model (retrain for each window)
                model.fit(X_train_transformed, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_transformed)
                
                # Calculate metrics
                if model_type == 'classification':
                    accuracy = accuracy_score(y_test, y_pred)
                    results.append({
                        'window': start_idx,
                        'accuracy': accuracy,
                        'predictions': y_pred.tolist(),
                        'actual': y_test.tolist()
                    })
                else:
                    r2 = r2_score(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    results.append({
                        'window': start_idx,
                        'r2': r2,
                        'mse': mse,
                        'predictions': y_pred.tolist(),
                        'actual': y_test.tolist()
                    })
            
            # Calculate overall performance
            if results:
                if model_type == 'classification':
                    overall_accuracy = np.mean([r['accuracy'] for r in results])
                    return {
                        'overall_accuracy': overall_accuracy,
                        'window_results': results,
                        'performance_trend': [r['accuracy'] for r in results]
                    }
                else:
                    overall_r2 = np.mean([r['r2'] for r in results])
                    overall_mse = np.mean([r['mse'] for r in results])
                    return {
                        'overall_r2': overall_r2,
                        'overall_mse': overall_mse,
                        'window_results': results,
                        'performance_trend': [r['r2'] for r in results]
                    }
            
            return {'error': 'No valid windows for analysis'}
            
        except Exception as e:
            return {'error': f'Walk-forward analysis failed: {str(e)}'}
    
    def _calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                        model, X_test: np.ndarray) -> Dict[str, Any]:
        """Calculate classification metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': classification_report(y_true, y_pred, output_dict=True)
        }
        
        # Add ROC AUC if model supports probability prediction
        try:
            y_proba = model.predict_proba(X_test)
            if y_proba.shape[1] == 2:  # Binary classification
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
        except:
            pass
        
        return metrics
    
    def _calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate regression metrics."""
        return {
            'r2_score': r2_score(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
    
    def _rank_models(self, comparison_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank models based on performance."""
        rankings = []
        
        for model_path, results in comparison_results.items():
            if 'error' not in results:
                # Determine primary metric based on model type
                if 'accuracy' in results:
                    primary_metric = 'accuracy'
                elif 'r2_score' in results:
                    primary_metric = 'r2_score'
                else:
                    continue
                
                rankings.append({
                    'model_path': model_path,
                    'primary_metric': primary_metric,
                    'score': results[primary_metric],
                    'metrics': results
                })
        
        # Sort by primary metric
        rankings.sort(key=lambda x: x['score'], reverse=True)
        
        return rankings
    
    def plot_performance_metrics(self, model_path: str, save_path: Optional[str] = None):
        """Plot performance metrics."""
        if model_path not in self.evaluation_results:
            print(f"No evaluation results for {model_path}")
            return
        
        results = self.evaluation_results[model_path]
        metrics = results['metrics']
        
        fig, axes = plt.subplots(2, 2, figsize=self.figure_size)
        fig.suptitle(f'Model Performance: {model_path}')
        
        # Classification metrics
        if 'accuracy' in metrics:
            # Confusion Matrix
            cm = np.array(metrics['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
            axes[0, 0].set_title('Confusion Matrix')
            
            # Metrics bar chart
            metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            metric_values = [
                metrics['accuracy'],
                metrics['precision'],
                metrics['recall'],
                metrics['f1_score']
            ]
            axes[0, 1].bar(metric_names, metric_values)
            axes[0, 1].set_title('Classification Metrics')
            axes[0, 1].set_ylim(0, 1)
            
            # Prediction distribution
            axes[1, 0].hist(results['predictions'], bins=2, alpha=0.7, label='Predictions')
            axes[1, 0].hist(results['actual'], bins=2, alpha=0.7, label='Actual')
            axes[1, 0].set_title('Prediction Distribution')
            axes[1, 0].legend()
            
            # ROC Curve (if available)
            if 'roc_auc' in metrics:
                axes[1, 1].text(0.5, 0.5, f'ROC AUC: {metrics["roc_auc"]:.3f}', 
                              ha='center', va='center', fontsize=12)
                axes[1, 1].set_title('ROC AUC Score')
        
        # Regression metrics
        else:
            # Actual vs Predicted scatter plot
            axes[0, 0].scatter(results['actual'], results['predictions'], alpha=0.5)
            axes[0, 0].plot([results['actual'].min(), results['actual'].max()], 
                           [results['actual'].min(), results['actual'].max()], 'r--')
            axes[0, 0].set_xlabel('Actual')
            axes[0, 0].set_ylabel('Predicted')
            axes[0, 0].set_title('Actual vs Predicted')
            
            # Residuals plot
            residuals = results['actual'] - results['predictions']
            axes[0, 1].scatter(results['predictions'], residuals, alpha=0.5)
            axes[0, 1].axhline(y=0, color='r', linestyle='--')
            axes[0, 1].set_xlabel('Predicted')
            axes[0, 1].set_ylabel('Residuals')
            axes[0, 1].set_title('Residuals Plot')
            
            # Metrics bar chart
            metric_names = ['R²', 'MSE', 'RMSE', 'MAE']
            metric_values = [
                metrics['r2_score'],
                metrics['mse'],
                metrics['rmse'],
                metrics['mae']
            ]
            axes[1, 0].bar(metric_names, metric_values)
            axes[1, 0].set_title('Regression Metrics')
            
            # Prediction distribution
            axes[1, 1].hist(results['predictions'], bins=30, alpha=0.7, label='Predictions')
            axes[1, 1].hist(results['actual'], bins=30, alpha=0.7, label='Actual')
            axes[1, 1].set_title('Prediction Distribution')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_model_comparison(self, save_path: Optional[str] = None):
        """Plot model comparison results."""
        if not self.model_comparison:
            print("No model comparison results available")
            return
        
        rankings = self.model_comparison.get('ranking', [])
        if not rankings:
            print("No model rankings available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=self.figure_size)
        fig.suptitle('Model Comparison')
        
        # Model scores
        model_names = [r['model_path'].split('/')[-1] for r in rankings]
        scores = [r['score'] for r in rankings]
        
        axes[0].bar(model_names, scores)
        axes[0].set_title(f'{rankings[0]["primary_metric"].replace("_", " ").title()}')
        axes[0].set_ylabel('Score')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Feature importance comparison (if available)
        feature_importance_data = []
        for ranking in rankings:
            model_path = ranking['model_path']
            if model_path in self.feature_importance_scores:
                feature_importance_data.append(self.feature_importance_scores[model_path])
        
        if feature_importance_data:
            # Plot top 10 features for each model
            for i, importance in enumerate(feature_importance_data[:3]):  # Top 3 models
                top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
                feature_names = [f[0] for f in top_features]
                feature_scores = [f[1] for f in top_features]
                
                axes[1].plot(feature_names, feature_scores, marker='o', label=f'Model {i+1}')
            
            axes[1].set_title('Top 10 Feature Importance')
            axes[1].set_ylabel('Importance Score')
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_evaluation_report(self, model_path: str) -> str:
        """Generate comprehensive evaluation report."""
        if model_path not in self.evaluation_results:
            return f"No evaluation results available for {model_path}"
        
        results = self.evaluation_results[model_path]
        metrics = results['metrics']
        
        report = f"""
# Model Evaluation Report

## Model Information
- **Model Path**: {model_path}
- **Model Type**: {results['model_type']}
- **Evaluation Date**: {results['evaluation_date']}

## Performance Metrics
"""
        
        if 'accuracy' in metrics:
            report += f"""
### Classification Metrics
- **Accuracy**: {metrics['accuracy']:.4f}
- **Precision**: {metrics['precision']:.4f}
- **Recall**: {metrics['recall']:.4f}
- **F1 Score**: {metrics['f1_score']:.4f}
"""
            if 'roc_auc' in metrics:
                report += f"- **ROC AUC**: {metrics['roc_auc']:.4f}\n"
        else:
            report += f"""
### Regression Metrics
- **R² Score**: {metrics['r2_score']:.4f}
- **MSE**: {metrics['mse']:.4f}
- **RMSE**: {metrics['rmse']:.4f}
- **MAE**: {metrics['mae']:.4f}
- **MAPE**: {metrics['mape']:.2f}%
"""
        
        report += f"""
## Data Summary
- **Test Samples**: {len(results['actual'])}
- **Predictions**: {len(results['predictions'])}

## Recommendations
"""
        
        if 'accuracy' in metrics:
            if metrics['accuracy'] > 0.8:
                report += "- Model shows excellent performance\n"
            elif metrics['accuracy'] > 0.7:
                report += "- Model shows good performance\n"
            else:
                report += "- Model performance needs improvement\n"
        else:
            if metrics['r2_score'] > 0.8:
                report += "- Model shows excellent performance\n"
            elif metrics['r2_score'] > 0.6:
                report += "- Model shows good performance\n"
            else:
                report += "- Model performance needs improvement\n"
        
        return report
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations."""
        return {
            'evaluation_results': self.evaluation_results,
            'model_comparison': self.model_comparison,
            'feature_importance_scores': self.feature_importance_scores,
            'total_models_evaluated': len(self.evaluation_results)
        }
