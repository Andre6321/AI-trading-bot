"""
Machine Learning trading strategy using XGBoost model.
"""

import logging
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


logger = logging.getLogger(__name__)


class MLStrategy:
    """
    Trading strategy based on XGBoost machine learning model.
    
    Loads a trained XGBoost model and generates trading signals based on
    predicted price direction.
    """
    
    def __init__(
        self,
        model_path: str,
        metadata_path: Optional[str] = None,
        confidence_threshold: float = 0.6,
        use_trend_filter: bool = True
    ):
        """
        Initialize ML strategy.
        
        Args:
            model_path: Path to trained XGBoost model (.pkl file)
            metadata_path: Path to model metadata JSON (optional)
            confidence_threshold: Minimum prediction probability for trade signal
            use_trend_filter: Whether to filter signals by trend direction
        """
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.confidence_threshold = confidence_threshold
        self.use_trend_filter = use_trend_filter
        
        # Load model
        self.model = self._load_model()
        self.metadata = self._load_metadata()
        self.expected_features = self.metadata.get("features", [])
        
        logger.info(
            f"MLStrategy initialized: model={self.model_path.name}, "
            f"features={len(self.expected_features)}, threshold={confidence_threshold}"
        )
    
    def _load_model(self):
        """Load trained XGBoost model from file."""
        try:
            with open(self.model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Model loaded from {self.model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load model metadata from JSON file."""
        if not self.metadata_path or not self.metadata_path.exists():
            logger.warning("No metadata file found")
            return {}
        
        try:
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Metadata loaded from {self.metadata_path}")
            return metadata
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return {}
    
    def prepare_features(self, features: Dict[str, float]) -> pd.DataFrame:
        """
        Prepare features for model prediction.
        
        Args:
            features: Dictionary of feature names and values
            
        Returns:
            DataFrame with features in correct order
        """
        # If we have expected features from metadata, ensure correct order
        if self.expected_features:
            # Check for missing features
            missing = set(self.expected_features) - set(features.keys())
            if missing:
                logger.warning(f"Missing features: {missing}")
                # Fill missing features with 0
                for feat in missing:
                    features[feat] = 0.0
            
            # Create DataFrame with features in correct order
            feature_values = [features.get(feat, 0.0) for feat in self.expected_features]
            df = pd.DataFrame([feature_values], columns=self.expected_features)
        else:
            # No metadata, use features as provided
            df = pd.DataFrame([features])
        
        return df
    
    def predict(self, features: Dict[str, float]) -> Tuple[int, float]:
        """
        Make prediction using ML model.
        
        Args:
            features: Dictionary of feature names and values
            
        Returns:
            Tuple of (prediction, confidence)
            - prediction: 0 (down) or 1 (up)
            - confidence: Probability of predicted class
        """
        # Prepare features
        X = self.prepare_features(features)
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        
        # Get prediction probability
        probabilities = self.model.predict_proba(X)[0]
        confidence = probabilities[prediction]
        
        logger.debug(
            f"Prediction: {prediction} (confidence: {confidence:.3f}) | "
            f"Probabilities: Down={probabilities[0]:.3f}, Up={probabilities[1]:.3f}"
        )
        
        return int(prediction), float(confidence)
    
    def generate_signal(
        self,
        features: Dict[str, float],
        current_position: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate trading signal based on ML prediction.
        
        Args:
            features: Dictionary of feature names and values
            current_position: Current position side ("Buy", "Sell", or None)
            
        Returns:
            Signal dictionary with action, confidence, and reasoning
        """
        # Get prediction
        prediction, confidence = self.predict(features)
        
        # Initialize signal
        signal = {
            "action": "HOLD",
            "side": None,
            "confidence": confidence,
            "prediction": "UP" if prediction == 1 else "DOWN",
            "reasoning": []
        }
        
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            signal["reasoning"].append(f"Low confidence: {confidence:.3f} < {self.confidence_threshold}")
            return signal
        
        # Apply trend filter if enabled
        if self.use_trend_filter:
            # Use EMA trend as filter
            ema_short = features.get("ema_20", 0)
            ema_long = features.get("ema_50", 0)
            
            if ema_short > 0 and ema_long > 0:
                trend_is_up = ema_short > ema_long
                
                # Only take trades aligned with trend
                if prediction == 1 and not trend_is_up:
                    signal["reasoning"].append("UP prediction but trend is DOWN (EMA filter)")
                    return signal
                elif prediction == 0 and trend_is_up:
                    signal["reasoning"].append("DOWN prediction but trend is UP (EMA filter)")
                    return signal
        
        # Generate signal based on prediction
        if prediction == 1:  # Bullish prediction
            if current_position == "Sell":
                signal["action"] = "CLOSE"
                signal["reasoning"].append("Close short position (bullish signal)")
            elif current_position is None:
                signal["action"] = "ENTER"
                signal["side"] = "Buy"
                signal["reasoning"].append(f"Enter long position (confidence: {confidence:.3f})")
        
        elif prediction == 0:  # Bearish prediction
            if current_position == "Buy":
                signal["action"] = "CLOSE"
                signal["reasoning"].append("Close long position (bearish signal)")
            elif current_position is None:
                # Note: Uncomment below to enable short trading
                # signal["action"] = "ENTER"
                # signal["side"] = "Sell"
                # signal["reasoning"].append(f"Enter short position (confidence: {confidence:.3f})")
                signal["reasoning"].append("Bearish signal but shorting disabled")
        
        # Log signal
        if signal["action"] != "HOLD":
            logger.info(
                f"SIGNAL: {signal['action']} {signal.get('side', '')} | "
                f"Prediction: {signal['prediction']} @ {confidence:.3f} | "
                f"Reason: {', '.join(signal['reasoning'])}"
            )
        
        return signal
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get strategy information and configuration.
        
        Returns:
            Strategy info dictionary
        """
        info = {
            "strategy_name": "XGBoost ML Strategy",
            "model_path": str(self.model_path),
            "confidence_threshold": self.confidence_threshold,
            "use_trend_filter": self.use_trend_filter,
            "expected_features": len(self.expected_features)
        }
        
        # Add metadata if available
        if self.metadata:
            info["training_info"] = {
                "training_date": self.metadata.get("training_date"),
                "test_accuracy": self.metadata.get("test_accuracy"),
                "test_precision": self.metadata.get("test_precision"),
                "test_roc_auc": self.metadata.get("test_roc_auc")
            }
        
        return info
    
    def validate_features(self, features: Dict[str, float]) -> Tuple[bool, Optional[str]]:
        """
        Validate that all required features are present.
        
        Args:
            features: Dictionary of feature names and values
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.expected_features:
            return True, None
        
        missing = set(self.expected_features) - set(features.keys())
        
        if missing:
            return False, f"Missing required features: {missing}"
        
        # Check for NaN values
        nan_features = [k for k, v in features.items() if pd.isna(v)]
        if nan_features:
            return False, f"NaN values in features: {nan_features}"
        
        return True, None
