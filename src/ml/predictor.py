"""
ML predictor for real-time trading signals.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
import joblib
import os

from .feature_engineering import FeatureEngineer
from .model_trainer import ModelTrainer
from ..core.data_models import OHLCV, StrategySignal, OrderSide
from ..indicators.technical_indicators import TechnicalIndicators
from ..indicators.pattern_recognition import PatternRecognition


class MLPredictor:
    """ML predictor for real-time trading signals."""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        """
        Initialize ML predictor.
        
        Args:
            model_path: Path to trained model
            config: Predictor configuration
        """
        self.config = config
        self.model_path = model_path
        
        # Load model
        self.model_data = self._load_model()
        self.model = self.model_data['model']
        self.feature_engineer = self.model_data['feature_engineer']
        self.feature_names = self.model_data['feature_names']
        self.model_type = self.model_data['model_type']
        self.target_type = self.model_data['target_type']
        
        # Prediction parameters
        self.prediction_threshold = config.get('prediction_threshold', 0.6)
        self.min_confidence = config.get('min_confidence', 0.7)
        self.max_predictions_per_hour = config.get('max_predictions_per_hour', 10)
        
        # Prediction tracking
        self.prediction_history = []
        self.last_prediction_time = None
        
        # Technical indicators and patterns
        self.indicators = TechnicalIndicators()
        self.patterns = PatternRecognition()
        
        # Performance tracking
        self.prediction_accuracy = 0.0
        self.total_predictions = 0
        self.correct_predictions = 0
    
    def _load_model(self) -> Dict[str, Any]:
        """Load trained model from disk."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        return joblib.load(self.model_path)
    
    def predict(self, ohlcv_data: List[OHLCV], 
                indicators: Optional[Dict] = None,
                patterns: Optional[Dict] = None) -> Optional[StrategySignal]:
        """
        Make prediction and generate trading signal.
        
        Args:
            ohlcv_data: Recent OHLCV data
            indicators: Technical indicators (optional)
            patterns: Pattern recognition results (optional)
            
        Returns:
            StrategySignal or None
        """
        # Check prediction frequency limit
        if not self._can_make_prediction():
            return None
        
        try:
            # Calculate indicators if not provided
            if indicators is None:
                indicators = self.indicators.calculate_all_indicators(ohlcv_data)
            
            # Detect patterns if not provided
            if patterns is None:
                patterns = self.patterns.detect_all_patterns(ohlcv_data)
            
            # Create features
            features = self.feature_engineer.create_features(ohlcv_data, indicators, patterns)
            if features.size == 0:
                return None
            
            # Reshape for single prediction
            features = features.reshape(1, -1)
            
            # Transform features
            features_transformed = self.feature_engineer.transform(features)
            
            # Make prediction
            if self.model_type == 'classification':
                prediction = self.model.predict(features_transformed)[0]
                confidence = self.model.predict_proba(features_transformed)[0].max()
                
                # Check confidence threshold
                if confidence < self.min_confidence:
                    return None
                
                # Convert prediction to signal
                signal_type = OrderSide.BUY if prediction == 1 else OrderSide.SELL
                signal_strength = confidence
                
            else:  # regression
                prediction = self.model.predict(features_transformed)[0]
                
                # Convert regression prediction to signal
                if prediction > self.prediction_threshold:
                    signal_type = OrderSide.BUY
                    signal_strength = min(abs(prediction), 1.0)
                elif prediction < -self.prediction_threshold:
                    signal_type = OrderSide.SELL
                    signal_strength = min(abs(prediction), 1.0)
                else:
                    return None  # No clear signal
            
            # Create strategy signal
            signal = StrategySignal(
                symbol=ohlcv_data[-1].symbol,
                signal=signal_type,
                strength=signal_strength,
                price=ohlcv_data[-1].close,
                strategy="ML_Predictor",
                timeframe=ohlcv_data[-1].timeframe,
                indicators={
                    'ml_prediction': float(prediction),
                    'ml_confidence': float(confidence) if self.model_type == 'classification' else 1.0,
                    'feature_count': len(self.feature_names)
                },
                metadata={
                    'model_type': self.model_type,
                    'target_type': self.target_type,
                    'prediction_threshold': self.prediction_threshold,
                    'min_confidence': self.min_confidence,
                    'model_path': self.model_path
                }
            )
            
            # Record prediction
            self._record_prediction(signal, prediction, confidence if self.model_type == 'classification' else 1.0)
            
            return signal
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    def predict_batch(self, ohlcv_data_list: List[List[OHLCV]]) -> List[Optional[StrategySignal]]:
        """
        Make predictions for multiple symbols.
        
        Args:
            ohlcv_data_list: List of OHLCV data for each symbol
            
        Returns:
            List of StrategySignals
        """
        predictions = []
        
        for ohlcv_data in ohlcv_data_list:
            prediction = self.predict(ohlcv_data)
            predictions.append(prediction)
        
        return predictions
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the model."""
        return self.model_data.get('feature_importance', {})
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_type': self.model_type,
            'target_type': self.target_type,
            'model_path': self.model_path,
            'feature_count': len(self.feature_names),
            'training_date': self.model_data.get('training_date'),
            'evaluation_metrics': self.model_data.get('evaluation_metrics', {}),
            'prediction_threshold': self.prediction_threshold,
            'min_confidence': self.min_confidence
        }
    
    def update_prediction_accuracy(self, actual_outcome: bool):
        """Update prediction accuracy with actual outcome."""
        self.total_predictions += 1
        
        if actual_outcome:
            self.correct_predictions += 1
        
        self.prediction_accuracy = self.correct_predictions / self.total_predictions
    
    def _can_make_prediction(self) -> bool:
        """Check if prediction can be made based on frequency limits."""
        current_time = datetime.utcnow()
        
        # Check hourly limit
        if self.last_prediction_time:
            time_diff = current_time - self.last_prediction_time
            if time_diff.total_seconds() < 3600:  # Less than 1 hour
                # Count predictions in last hour
                recent_predictions = [
                    p for p in self.prediction_history
                    if (current_time - p['timestamp']).total_seconds() < 3600
                ]
                
                if len(recent_predictions) >= self.max_predictions_per_hour:
                    return False
        
        return True
    
    def _record_prediction(self, signal: StrategySignal, prediction: Union[float, int], confidence: float):
        """Record prediction for tracking."""
        self.last_prediction_time = datetime.utcnow()
        
        prediction_record = {
            'timestamp': self.last_prediction_time,
            'symbol': signal.symbol,
            'signal': signal.signal.value,
            'strength': signal.strength,
            'prediction': prediction,
            'confidence': confidence,
            'price': float(signal.price)
        }
        
        self.prediction_history.append(prediction_record)
        
        # Keep only last 1000 predictions
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-1000:]
    
    def get_prediction_stats(self) -> Dict[str, Any]:
        """Get prediction statistics."""
        if not self.prediction_history:
            return {
                'total_predictions': 0,
                'prediction_accuracy': 0.0,
                'avg_confidence': 0.0,
                'recent_predictions': []
            }
        
        recent_predictions = self.prediction_history[-10:]  # Last 10 predictions
        avg_confidence = np.mean([p['confidence'] for p in self.prediction_history])
        
        return {
            'total_predictions': len(self.prediction_history),
            'prediction_accuracy': self.prediction_accuracy,
            'avg_confidence': avg_confidence,
            'recent_predictions': recent_predictions,
            'hourly_limit': self.max_predictions_per_hour,
            'predictions_last_hour': len([
                p for p in self.prediction_history
                if (datetime.utcnow() - p['timestamp']).total_seconds() < 3600
            ])
        }
    
    def retrain_model(self, new_data: List[OHLCV], retrain_config: Dict[str, Any]) -> str:
        """Retrain model with new data."""
        # Initialize trainer
        trainer = ModelTrainer(retrain_config)
        
        # Prepare training data
        X, y = trainer.prepare_training_data(new_data)
        
        # Train models
        results = trainer.train_models(X, y)
        
        # Save new model
        new_model_path = trainer.save_model()
        
        # Update current model
        self.model_path = new_model_path
        self.model_data = self._load_model()
        self.model = self.model_data['model']
        self.feature_engineer = self.model_data['feature_engineer']
        
        return new_model_path
    
    def validate_prediction(self, ohlcv_data: List[OHLCV], 
                          future_periods: int = 5) -> Dict[str, Any]:
        """
        Validate prediction against future price movements.
        
        Args:
            ohlcv_data: Historical OHLCV data
            future_periods: Number of periods to look ahead
            
        Returns:
            Validation results
        """
        if len(ohlcv_data) < future_periods + 10:
            return {'error': 'Insufficient data for validation'}
        
        # Make prediction
        prediction_signal = self.predict(ohlcv_data[:-future_periods])
        
        if not prediction_signal:
            return {'error': 'No prediction generated'}
        
        # Check actual outcome
        current_price = float(ohlcv_data[-future_periods-1].close)
        future_price = float(ohlcv_data[-1].close)
        
        actual_return = (future_price - current_price) / current_price
        
        # Determine if prediction was correct
        if prediction_signal.signal == OrderSide.BUY:
            correct = actual_return > 0.01  # 1% threshold
        else:  # SELL
            correct = actual_return < -0.01  # 1% threshold
        
        return {
            'prediction': prediction_signal.signal.value,
            'prediction_strength': prediction_signal.strength,
            'actual_return': actual_return,
            'correct': correct,
            'current_price': current_price,
            'future_price': future_price,
            'periods_ahead': future_periods
        }
