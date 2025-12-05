"""
Advanced model training with walk-forward validation and model comparison.
Tests XGBoost, LightGBM, and ensemble approaches.
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# ML libraries
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import SMOTE

# Import enhanced features
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from models.ml_dataset import build_ml_dataset
from bybit_ai_trader.research.features_enhanced import build_enhanced_features


def walk_forward_validation(
    df: pd.DataFrame,
    n_splits: int = 5,
    test_size_months: int = 1
) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Create walk-forward validation splits.
    
    Args:
        df: Full dataset
        n_splits: Number of splits
        test_size_months: Size of test set in months
        
    Returns:
        List of (train, val, test) splits
    """
    print(f"\n📊 Creating {n_splits} walk-forward splits...")
    
    splits = []
    step_size = len(df) // (n_splits + 1)
    
    for i in range(n_splits):
        # Expanding window: use all data up to split point
        train_end = (i + 1) * step_size
        val_end = min(train_end + step_size // 2, len(df))
        test_end = min(val_end + step_size // 2, len(df))
        
        train_data = df.iloc[:train_end].copy()
        val_data = df.iloc[train_end:val_end].copy()
        test_data = df.iloc[val_end:test_end].copy()
        
        print(f"   Split {i+1}: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        splits.append((train_data, val_data, test_data))
    
    return splits


def train_xgboost(X_train, y_train, X_val, y_val) -> xgb.XGBClassifier:
    """Train XGBoost model with class weights."""
    print("   Training XGBoost...")
    
    # Calculate class weight for imbalanced data
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
    print(f"      Class imbalance ratio: {scale_pos_weight:.2f}:1")
    
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        max_depth=5,
        n_estimators=500,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,  # Handle class imbalance
        random_state=42,
        eval_metric='logloss',
        early_stopping_rounds=50
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    return model


def train_lightgbm(X_train, y_train, X_val, y_val) -> lgb.LGBMClassifier:
    """Train LightGBM model with class weights."""
    print("   Training LightGBM...")
    
    # Use class_weight='balanced' for automatic balancing
    model = lgb.LGBMClassifier(
        objective='binary',
        max_depth=5,
        n_estimators=500,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight='balanced',  # Handle class imbalance
        random_state=42,
        verbose=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    return model


def evaluate_model(model, X_test, y_test, model_name: str) -> Dict[str, float]:
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }
    
    print(f"   {model_name} Performance:")
    print(f"      Accuracy:  {metrics['accuracy']:.4f}")
    print(f"      Precision: {metrics['precision']:.4f}")
    print(f"      Recall:    {metrics['recall']:.4f}")
    print(f"      ROC AUC:   {metrics['roc_auc']:.4f}")
    
    return metrics


def create_ensemble(xgb_model, lgb_model) -> VotingClassifier:
    """Create ensemble of already-trained models."""
    print("   Creating ensemble model...")
    
    from sklearn.preprocessing import LabelEncoder
    
    # Use already-fitted models
    ensemble = VotingClassifier(
        estimators=[('xgb', xgb_model), ('lgb', lgb_model)],
        voting='soft'  # Use predicted probabilities
    )
    
    # Set fitted=True to skip refitting
    ensemble.estimators_ = [xgb_model, lgb_model]
    ensemble.named_estimators_ = {'xgb': xgb_model, 'lgb': lgb_model}
    ensemble.classes_ = np.array([0, 1])
    
    # Create proper label encoder
    le = LabelEncoder()
    le.fit([0, 1])
    ensemble.le_ = le
    
    return ensemble


def main():
    """Main training pipeline with advanced features and validation."""
    print("=" * 60)
    print("🚀 ADVANCED MODEL TRAINING")
    print("=" * 60)
    
    # Load data
    print("\n📥 Loading data...")
    data_path = Path("data/processed/btcusdt_1h_full_history_enhanced.parquet")
    
    if not data_path.exists():
        print("❌ Full history features not found. Run build_full_history_features.py first.")
        return
    
    df = pd.read_parquet(data_path)
    print(f"✅ Loaded {len(df)} rows with {len(df.columns)} features")
    print(f"   Date range: {df.index[0] if len(df) > 0 else 'N/A'} to {df.index[-1] if len(df) > 0 else 'N/A'}")
    
    # Create ML dataset
    print("\n📊 Creating ML dataset...")
    X, y, index = build_ml_dataset(
        df,
        horizon_hours=4,
        threshold=0.01
    )
    
    print(f"✅ ML dataset: {len(X)} samples, {len(X.columns)} features")
    
    # Combine for walk-forward validation
    ml_df = X.copy()
    ml_df['target'] = y
    
    feature_cols = X.columns.tolist()
    
    # Walk-forward validation
    splits = walk_forward_validation(ml_df, n_splits=5)
    
    all_results = []
    best_model = None
    best_score = 0
    
    for split_idx, (train_df, val_df, test_df) in enumerate(splits):
        print(f"\n{'='*60}")
        print(f"📈 SPLIT {split_idx + 1}/{len(splits)}")
        print(f"{'='*60}")
        
        # Prepare data
        X_train = train_df[feature_cols]
        y_train = train_df['target']
        X_val = val_df[feature_cols]
        y_val = val_df['target']
        X_test = test_df[feature_cols]
        y_test = test_df['target']
        
        # Apply SMOTE to balance training data
        print(f"   Original training set: {len(y_train)} samples")
        print(f"      Class 0: {(y_train==0).sum()}, Class 1: {(y_train==1).sum()}")
        
        if (y_train == 1).sum() > 5:  # Only if we have enough positive samples
            smote = SMOTE(random_state=42, k_neighbors=min(5, (y_train==1).sum()-1))
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            print(f"   After SMOTE: {len(y_train_balanced)} samples")
            print(f"      Class 0: {(y_train_balanced==0).sum()}, Class 1: {(y_train_balanced==1).sum()}")
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
            print("   Skipping SMOTE (insufficient positive samples)")
        
        # Train models with balanced data
        xgb_model = train_xgboost(X_train_balanced, y_train_balanced, X_val, y_val)
        lgb_model = train_lightgbm(X_train_balanced, y_train_balanced, X_val, y_val)
        
        # Evaluate
        xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
        lgb_metrics = evaluate_model(lgb_model, X_test, y_test, "LightGBM")
        
        # Create ensemble
        ensemble = create_ensemble(xgb_model, lgb_model)
        ensemble_metrics = evaluate_model(ensemble, X_test, y_test, "Ensemble")
        
        # Track best model
        if ensemble_metrics['roc_auc'] > best_score:
            best_score = ensemble_metrics['roc_auc']
            best_model = ensemble
        
        all_results.append({
            'split': split_idx + 1,
            'xgb': xgb_metrics,
            'lgb': lgb_metrics,
            'ensemble': ensemble_metrics
        })
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 OVERALL RESULTS")
    print("=" * 60)
    
    for model_name in ['xgb', 'lgb', 'ensemble']:
        accuracies = [r[model_name]['accuracy'] for r in all_results]
        roc_aucs = [r[model_name]['roc_auc'] for r in all_results]
        
        print(f"\n{model_name.upper()}:")
        print(f"   Avg Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"   Avg ROC AUC:  {np.mean(roc_aucs):.4f} ± {np.std(roc_aucs):.4f}")
    
    # Save best model
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    
    model_path = output_dir / "ensemble_btcusdt_h4_enhanced.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    
    print(f"\n✅ Best model saved to {model_path}")
    
    # Save metadata
    metadata = {
        "model_type": "ensemble",
        "features": feature_cols,
        "n_features": len(feature_cols),
        "training_date": datetime.now().isoformat(),
        "n_splits": len(splits),
        "avg_accuracy": float(np.mean([r['ensemble']['accuracy'] for r in all_results])),
        "avg_roc_auc": float(np.mean([r['ensemble']['roc_auc'] for r in all_results])),
        "all_results": all_results
    }
    
    metadata_path = output_dir / "ensemble_btcusdt_h4_enhanced_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"✅ Metadata saved to {metadata_path}")
    print("\n🎉 Training complete!")


if __name__ == "__main__":
    main()
