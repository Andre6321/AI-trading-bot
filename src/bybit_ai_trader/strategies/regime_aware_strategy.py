"""
Regime-Aware Machine Learning trading strategy.

Uses HMM to detect market regime (bull/bear/sideways) and selects the 
appropriate regime-specific ensemble model for predictions.
"""

import logging
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger(__name__)


class RegimeAwareMLStrategy:
    """
    Trading strategy that detects market regime and uses regime-specific models.
    
    Automatically switches between bull/bear/sideways models based on HMM detection.
    """
    
    def __init__(
        self,
        regime_models_dir: str = "models",
        hmm_model_path: str = "models/market_regime_hmm.pkl",
        confidence_threshold: float = 0.6,
        use_trend_filter: bool = True,
        regime_switch_delay: int = 3  # Hours before switching regime
    ):
        """
        Initialize regime-aware ML strategy.
        
        Args:
            regime_models_dir: Directory containing regime-specific models
            hmm_model_path: Path to trained HMM model
            confidence_threshold: Minimum prediction probability for trade signal
            use_trend_filter: Whether to filter signals by trend direction
            regime_switch_delay: Minimum hours in regime before switching
        """
        self.models_dir = Path(regime_models_dir)
        self.hmm_model_path = Path(hmm_model_path)
        self.confidence_threshold = confidence_threshold
        self.use_trend_filter = use_trend_filter
        self.regime_switch_delay = regime_switch_delay
        
        # Load HMM model for regime detection
        self.hmm_model, self.hmm_scaler = self._load_hmm_model()
        
        # Load regime-specific ensemble models
        self.regime_models = self._load_regime_models()
        self.regime_metadata = self._load_regime_metadata()
        
        # Track current regime
        self.current_regime = None
        self.regime_confidence = 0.0
        self.regime_history = []  # Track recent regime predictions
        
        logger.info(
            f"RegimeAwareMLStrategy initialized: "
            f"regimes={list(self.regime_models.keys())}, "
            f"threshold={confidence_threshold}"
        )
    
    def _load_hmm_model(self) -> Tuple[hmm.GaussianHMM, StandardScaler]:
        """Load trained HMM model and scaler."""
        try:
            with open(self.hmm_model_path, 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, tuple):
                hmm_model, scaler = data
            else:
                # Old format - just the model
                hmm_model = data
                logger.error("HMM scaler not found! Model requires scaler for feature scaling.")
                raise ValueError("HMM model file missing scaler. Please retrain or provide scaler.")
            
            logger.info(f"HMM model loaded from {self.hmm_model_path}")
            return hmm_model, scaler
            
        except Exception as e:
            logger.error(f"Failed to load HMM model: {e}")
            raise
    
    def _load_regime_models(self) -> Dict[str, Any]:
        """Load regime-specific ensemble models."""
        models = {}
        regimes = ['bear', 'sideways', 'bull']
        
        for regime in regimes:
            model_path = self.models_dir / f'ensemble_{regime}_regime.pkl'
            
            if not model_path.exists():
                logger.warning(f"Regime model not found: {model_path}")
                continue
            
            try:
                with open(model_path, 'rb') as f:
                    models[regime] = pickle.load(f)
                logger.info(f"Loaded {regime} regime model")
            except Exception as e:
                logger.error(f"Failed to load {regime} model: {e}")
        
        if not models:
            raise ValueError("No regime models loaded!")
        
        return models
    
    def _load_regime_metadata(self) -> Dict[str, Dict]:
        """Load metadata for regime models."""
        metadata = {}
        regimes = ['bear', 'sideways', 'bull']
        
        for regime in regimes:
            metadata_path = self.models_dir / f'ensemble_{regime}_regime_metadata.json'
            
            if not metadata_path.exists():
                continue
            
            try:
                with open(metadata_path, 'r') as f:
                    metadata[regime] = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load {regime} metadata: {e}")
        
        return metadata
    
    def calculate_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate features for regime detection from recent price data.
        
        Args:
            df: DataFrame with recent OHLCV data (at least 168 hours)
            
        Returns:
            DataFrame with regime features
        """
        features = pd.DataFrame(index=df.index)
        
        # Returns at different timeframes
        features['ret_1h'] = df['close'].pct_change(1)
        features['ret_4h'] = df['close'].pct_change(4)
        features['ret_24h'] = df['close'].pct_change(24)
        features['ret_7d'] = df['close'].pct_change(168) if len(df) >= 168 else df['close'].pct_change(len(df)//2)
        
        # Volatility
        features['vol_24h'] = features['ret_1h'].rolling(24).std()
        features['vol_7d'] = features['ret_1h'].rolling(168 if len(df) >= 168 else len(df)//2).std()
        
        # Trend strength
        ma_fast = df['close'].rolling(min(20, len(df)//4)).mean()
        ma_slow = df['close'].rolling(min(50, len(df)//2)).mean()
        features['ma_diff'] = (ma_fast - ma_slow) / ma_slow
        
        # RSI
        features['rsi'] = self._calculate_rsi(df['close'], 14)
        features['momentum_24h'] = df['close'] / df['close'].shift(24) - 1
        
        # Volume
        if 'volume' in df.columns:
            features['vol_change'] = df['volume'].pct_change(24)
            features['vol_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Price range
        if 'high' in df.columns and 'low' in df.columns:
            features['hl_range'] = (df['high'] - df['low']) / df['close']
        
        # Trend direction
        features['price_vs_ma50'] = (df['close'] - df['close'].rolling(50).mean()) / df['close'].rolling(50).mean()
        
        # Clean up
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.ffill().bfill().fillna(0)
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def detect_regime(self, recent_data: pd.DataFrame) -> Tuple[str, float]:
        """
        Detect current market regime using HMM.
        
        Args:
            recent_data: Recent OHLCV data (at least 168 hours recommended)
            
        Returns:
            Tuple of (regime_name, confidence)
        """
        # Validate data length
        if len(recent_data) < 50:
            logger.warning(f"Insufficient data for regime detection: {len(recent_data)} < 50 hours")
            return self.current_regime or 'sideways', 0.5
        
        # Calculate regime features
        regime_features = self.calculate_regime_features(recent_data)
        
        if len(regime_features) == 0:
            logger.warning("Not enough data for regime detection after feature calculation")
            return self.current_regime or 'sideways', 0.5
        
        # Get most recent features
        latest_features = regime_features.iloc[-1:].values
        
        # Scale features
        features_scaled = self.hmm_scaler.transform(latest_features)
        
        # Predict regime
        regime_state = self.hmm_model.predict(features_scaled)[0]
        
        # Get state probabilities
        log_prob, posteriors = self.hmm_model.score_samples(features_scaled)
        confidence = posteriors[0, regime_state]
        
        # Map state to regime name based on characteristics
        regime_names = self._map_state_to_regime(regime_state, recent_data)
        
        # Update regime history
        self.regime_history.append(regime_names)
        if len(self.regime_history) > self.regime_switch_delay:
            self.regime_history.pop(0)
        
        # Only switch regime if consistent over delay period
        if len(self.regime_history) >= self.regime_switch_delay:
            most_common = max(set(self.regime_history), key=self.regime_history.count)
            if most_common != self.current_regime:
                logger.info(f"Regime switch: {self.current_regime} → {most_common}")
            self.current_regime = most_common
        else:
            self.current_regime = regime_names
        
        self.regime_confidence = confidence
        
        return self.current_regime, confidence
    
    def _map_state_to_regime(self, state: int, df: pd.DataFrame) -> str:
        """
        Map HMM state directly to regime name.
        
        The HMM was trained to identify 3 states based on returns, volatility,
        trend strength, momentum, and volume. States are sorted by average return:
        - State 0 (lowest return) = BEAR
        - State 1 (middle return) = SIDEWAYS  
        - State 2 (highest return) = BULL
        
        Args:
            state: HMM state (0, 1, or 2)
            df: Recent price data (unused, kept for compatibility)
            
        Returns:
            Regime name: 'bear', 'sideways', or 'bull'
        """
        regime_names = {
            0: 'bear',
            1: 'sideways',
            2: 'bull'
        }
        return regime_names.get(state, 'sideways')
    
    def prepare_features(self, features: Dict[str, float], regime: str) -> pd.DataFrame:
        """
        Prepare features for regime-specific model prediction.
        
        Args:
            features: Dictionary of feature names and values
            regime: Current market regime
            
        Returns:
            DataFrame with features in correct order
        """
        # Get expected features for this regime
        expected_features = self.regime_metadata.get(regime, {}).get('features', [])
        
        if expected_features:
            # Check for missing features
            missing = set(expected_features) - set(features.keys())
            if missing:
                logger.warning(f"Missing features for {regime}: {missing}")
                for feat in missing:
                    features[feat] = 0.0
            
            # Create DataFrame with features in correct order
            feature_values = [features.get(feat, 0.0) for feat in expected_features]
            df = pd.DataFrame([feature_values], columns=expected_features)
        else:
            # No metadata, use features as provided
            df = pd.DataFrame([features])
        
        return df
    
    def predict(
        self,
        features: Dict[str, float],
        recent_data: pd.DataFrame
    ) -> Tuple[int, float, str]:
        """
        Make prediction using regime-aware model selection.
        
        Args:
            features: Dictionary of feature names and values
            recent_data: Recent OHLCV data for regime detection
            
        Returns:
            Tuple of (prediction, confidence, regime)
            - prediction: 0 (down) or 1 (up)
            - confidence: Probability of predicted class
            - regime: Detected market regime
        """
        # Detect current regime
        regime, regime_conf = self.detect_regime(recent_data)
        
        # Get appropriate model
        model = self.regime_models.get(regime)
        if model is None:
            logger.warning(f"No model for regime {regime}, using sideways")
            regime = 'sideways'
            model = self.regime_models['sideways']
        
        # Prepare features
        X = self.prepare_features(features, regime)
        
        # Validate features
        if X.shape[1] == 0:
            logger.error(f"No valid features prepared for {regime} regime")
            return 0, 0.5, regime
        
        # Make prediction
        try:
            prediction = model.predict(X)[0]
        except Exception as e:
            logger.error(f"Prediction failed for {regime} regime: {e}")
            return 0, 0.5, regime
        
        # Get prediction probability
        probabilities = model.predict_proba(X)[0]
        confidence = probabilities[prediction]
        
        logger.debug(
            f"Regime: {regime.upper()} (conf: {regime_conf:.3f}) | "
            f"Prediction: {prediction} (conf: {confidence:.3f}) | "
            f"Probs: Down={probabilities[0]:.3f}, Up={probabilities[1]:.3f}"
        )
        
        return int(prediction), float(confidence), regime
    
    def generate_signal(
        self,
        features: Dict[str, float],
        recent_data: pd.DataFrame,
        current_position: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate trading signal based on regime-aware ML prediction.
        
        Args:
            features: Dictionary of feature names and values
            recent_data: Recent OHLCV data for regime detection
            current_position: Current position side ("Buy", "Sell", or None)
            
        Returns:
            Signal dictionary with action, confidence, regime, and reasoning
        """
        # Get prediction
        prediction, confidence, regime = self.predict(features, recent_data)
        
        # Initialize signal
        signal = {
            "action": "HOLD",
            "side": None,
            "confidence": confidence,
            "prediction": "UP" if prediction == 1 else "DOWN",
            "regime": regime,
            "regime_confidence": self.regime_confidence,
            "reasoning": []
        }
        
        # Add regime info
        signal["reasoning"].append(f"Regime: {regime.upper()} (conf: {self.regime_confidence:.3f})")
        
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            signal["reasoning"].append(f"Low confidence: {confidence:.3f} < {self.confidence_threshold}")
            return signal
        
        # Apply trend filter if enabled
        if self.use_trend_filter:
            ma_short = features.get("ma_20", 0)
            ma_long = features.get("ma_50", 0)
            
            if ma_short > 0 and ma_long > 0:
                trend_is_up = ma_short > ma_long
                
                # Only take trades aligned with trend
                if prediction == 1 and not trend_is_up:
                    signal["reasoning"].append("UP prediction but trend is DOWN (MA filter)")
                    return signal
                elif prediction == 0 and trend_is_up:
                    signal["reasoning"].append("DOWN prediction but trend is UP (MA filter)")
                    return signal
        
        # Generate signal based on prediction
        if prediction == 1:  # Bullish prediction
            if current_position == "Sell":
                signal["action"] = "CLOSE"
                signal["reasoning"].append(f"Close short ({regime} regime, conf: {confidence:.3f})")
            elif current_position is None:
                signal["action"] = "ENTER"
                signal["side"] = "Buy"
                signal["reasoning"].append(f"Enter long ({regime} regime, conf: {confidence:.3f})")
        
        elif prediction == 0:  # Bearish prediction
            if current_position == "Buy":
                signal["action"] = "CLOSE"
                signal["reasoning"].append(f"Close long ({regime} regime, conf: {confidence:.3f})")
            elif current_position is None:
                # Note: Uncomment to enable short trading
                # signal["action"] = "ENTER"
                # signal["side"] = "Sell"
                # signal["reasoning"].append(f"Enter short ({regime} regime, conf: {confidence:.3f})")
                signal["reasoning"].append(f"Bearish signal ({regime} regime) but shorting disabled")
        
        # Log signal
        if signal["action"] != "HOLD":
            logger.info(
                f"SIGNAL [{regime.upper()}]: {signal['action']} {signal.get('side', '')} | "
                f"Pred: {signal['prediction']} @ {confidence:.3f} | "
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
            "strategy_name": "Regime-Aware ML Strategy",
            "current_regime": self.current_regime,
            "regime_confidence": self.regime_confidence,
            "confidence_threshold": self.confidence_threshold,
            "use_trend_filter": self.use_trend_filter,
            "regime_switch_delay": self.regime_switch_delay,
            "available_regimes": list(self.regime_models.keys())
        }
        
        # Add regime model performance
        regime_performance = {}
        for regime, metadata in self.regime_metadata.items():
            # Try both 'ensemble' and 'ensemble_metrics' keys for compatibility
            ensemble_data = metadata.get('ensemble_metrics') or metadata.get('ensemble', {})
            if ensemble_data:
                regime_performance[regime] = {
                    "roc_auc": ensemble_data.get('roc_auc'),
                    "accuracy": ensemble_data.get('accuracy'),
                    "f1_score": ensemble_data.get('f1')
                }
        
        info["regime_performance"] = regime_performance
        
        return info
