"""
XGBoost model training script for BTCUSDT price direction prediction.

This script:
1. Loads processed features from data/processed/btcusdt_1h_features.parquet
2. Creates ML dataset with 4-hour prediction horizon
3. Splits data chronologically (70% train, 15% validation, 15% test)
4. Trains XGBoost binary classifier
5. Evaluates on test set and prints metrics
6. Saves trained model and feature names
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb

# Add project root to path to import our modules
import sys
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.bybit_ai_trader.research.ml_dataset import build_ml_dataset


def main():
    print("🚀 Starting XGBoost model training for BTCUSDT...")
    
    # Setup paths
    features_path = project_root / "data" / "processed" / "btcusdt_1h_features.parquet"
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "xgb_btcusdt_h4.pkl"
    features_path_json = models_dir / "xgb_btcusdt_h4_features.json"
    
    print(f"📂 Loading features from: {features_path}")
    
    # Load features
    try:
        features_df = pd.read_parquet(features_path)
        print(f"   Loaded {features_df.shape[0]:,} rows x {features_df.shape[1]} columns")
        print(f"   Date range: {features_df.index.min()} to {features_df.index.max()}")
    except FileNotFoundError:
        print(f"❌ Features file not found: {features_path}")
        print("   Please run scripts/build_features.py first to generate features")
        return
    except Exception as e:
        print(f"❌ Error loading features: {e}")
        return
    
    # Build ML dataset
    print(f"\n🎯 Building ML dataset with 4-hour horizon...")
    try:
        X, y = build_ml_dataset(features_df, horizon_hours=4)
        print(f"   ML dataset shape: X={X.shape}, y={y.shape}")
    except Exception as e:
        print(f"❌ Error building ML dataset: {e}")
        return
    
    # Sort by index to ensure chronological order
    print(f"\n📊 Sorting by timestamp and creating chronological splits...")
    X = X.sort_index()
    y = y.sort_index()
    
    # Chronological splits
    n_samples = len(X)
    train_end = int(n_samples * 0.70)
    valid_end = int(n_samples * 0.85)  # 70% + 15% = 85%
    
    # Create splits
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    
    X_valid = X.iloc[train_end:valid_end]
    y_valid = y.iloc[train_end:valid_end]
    
    X_test = X.iloc[valid_end:]
    y_test = y.iloc[valid_end:]
    
    print(f"   Train: {len(X_train):,} samples ({len(X_train)/n_samples:.1%}) - {X_train.index.min()} to {X_train.index.max()}")
    print(f"   Valid: {len(X_valid):,} samples ({len(X_valid)/n_samples:.1%}) - {X_valid.index.min()} to {X_valid.index.max()}")
    print(f"   Test:  {len(X_test):,} samples ({len(X_test)/n_samples:.1%}) - {X_test.index.min()} to {X_test.index.max()}")
    
    # Check label distribution
    print(f"\n   Label distributions:")
    print(f"   Train positive rate: {y_train.mean():.1%}")
    print(f"   Valid positive rate: {y_valid.mean():.1%}")
    print(f"   Test positive rate:  {y_test.mean():.1%}")
    
    # Train XGBoost model
    print(f"\n🤖 Training XGBoost binary classifier...")
    
    # XGBoost parameters
    xgb_params = {
        'objective': 'binary:logistic',
        'max_depth': 4,
        'n_estimators': 300,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'eval_metric': 'logloss',
        'early_stopping_rounds': 50,
        'verbose': False
    }
    
    print(f"   Parameters: {xgb_params}")
    
    # Initialize and train model
    model = xgb.XGBClassifier(**xgb_params)
    
    # Train with early stopping using validation set
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        verbose=50  # Print every 50 rounds
    )
    
    print(f"   Training completed!")
    print(f"   Best iteration: {model.best_iteration}")
    print(f"   Best score: {model.best_score:.4f}")
    
    # Make predictions on test set
    print(f"\n📈 Evaluating on test set...")
    
    # Probabilities and predictions
    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    roc_auc = roc_auc_score(y_test, y_test_proba)
    
    # Print metrics
    print(f"\n📊 Test Set Performance Metrics:")
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy:.1%})")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   ROC AUC:   {roc_auc:.4f}")
    
    # Detailed classification report
    print(f"\n   Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=['Down', 'Up']))
    
    # Confusion matrix
    print(f"\n   Confusion Matrix:")
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"                 Predicted")
    print(f"   Actual    Down    Up")
    print(f"   Down    {cm[0,0]:6d} {cm[0,1]:6d}")
    print(f"   Up      {cm[1,0]:6d} {cm[1,1]:6d}")
    
    # Feature importance
    print(f"\n🔍 Top 10 Feature Importances:")
    feature_names = X.columns.tolist()
    importances = model.feature_importances_
    
    # Sort features by importance
    feature_importance = list(zip(feature_names, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    for i, (feature, importance) in enumerate(feature_importance[:10]):
        print(f"   {i+1:2d}. {feature:<25} {importance:.4f}")
    
    # Save trained model
    print(f"\n💾 Saving model and feature names...")
    
    try:
        # Save model using joblib
        joblib.dump(model, model_path)
        print(f"   Model saved to: {model_path}")
        
        # Save feature names as JSON
        feature_info = {
            'feature_names': feature_names,
            'n_features': len(feature_names),
            'model_params': xgb_params,
            'training_info': {
                'train_samples': len(X_train),
                'valid_samples': len(X_valid),
                'test_samples': len(X_test),
                'horizon_hours': 4,
                'test_accuracy': float(accuracy),
                'test_roc_auc': float(roc_auc),
                'best_iteration': int(model.best_iteration),
                'training_date': pd.Timestamp.now().isoformat()
            }
        }
        
        with open(features_path_json, 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        print(f"   Feature info saved to: {features_path_json}")
        
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return
    
    print(f"\n✅ XGBoost training completed successfully!")
    print(f"   Model performance: Accuracy={accuracy:.1%}, ROC AUC={roc_auc:.4f}")
    print(f"   Model saved for production use")


if __name__ == "__main__":
    main()