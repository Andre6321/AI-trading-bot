"""
Train separate ML models for each market regime (bull/bear/sideways).
This should improve prediction accuracy by specializing each model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report
)
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import VotingClassifier

def load_regime_data():
    """Load data with regime labels."""
    print("\n📥 Loading regime-labeled data...")
    df = pd.read_parquet('data/processed/btcusdt_1h_with_regimes.parquet')
    print(f"   Loaded {len(df):,} samples")
    
    # Filter to samples with valid target
    df = df[df['target_4h'].notna()].copy()
    print(f"   Valid samples: {len(df):,}")
    
    return df

def prepare_regime_dataset(df, regime_name):
    """Prepare training data for specific regime."""
    print(f"\n🔧 Preparing {regime_name} regime dataset...")
    
    # Filter to specific regime
    regime_df = df[df['regime_name'] == regime_name].copy()
    print(f"   Regime samples: {len(regime_df):,}")
    
    # Feature columns (exclude target and metadata)
    feature_cols = [col for col in regime_df.columns 
                   if col not in ['target_4h', 'regime', 'regime_name'] 
                   and not col.startswith('future_')]
    
    X = regime_df[feature_cols]
    y = regime_df['target_4h']
    
    # Check class distribution
    print(f"   Class distribution:")
    print(f"     UP (1):   {(y==1).sum():>6} ({(y==1).sum()/len(y)*100:.1f}%)")
    print(f"     DOWN (0): {(y==0).sum():>6} ({(y==0).sum()/len(y)*100:.1f}%)")
    
    return X, y, feature_cols

def train_xgboost_regime(X_train, X_test, y_train, y_test, regime_name):
    """Train XGBoost for specific regime."""
    print(f"\n🌲 Training XGBoost for {regime_name}...")
    
    # Use optimized hyperparameters from Optuna
    params = {
        'max_depth': 8,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'random_state': 42,
        'tree_method': 'hist'
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"   ROC AUC:   {auc:.4f}")
    print(f"   Accuracy:  {acc:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    
    return model, auc

def train_lightgbm_regime(X_train, X_test, y_train, y_test, regime_name):
    """Train LightGBM for specific regime."""
    print(f"\n💡 Training LightGBM for {regime_name}...")
    
    params = {
        'num_leaves': 63,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'objective': 'binary',
        'metric': 'auc',
        'random_state': 42,
        'verbose': -1
    }
    
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.log_evaluation(0)]
    )
    
    # Evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"   ROC AUC:   {auc:.4f}")
    print(f"   Accuracy:  {acc:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    
    return model, auc

def train_catboost_regime(X_train, X_test, y_train, y_test, regime_name):
    """Train CatBoost for specific regime."""
    print(f"\n🐱 Training CatBoost for {regime_name}...")
    
    params = {
        'depth': 6,
        'learning_rate': 0.05,
        'iterations': 500,
        'l2_leaf_reg': 3,
        'random_state': 42,
        'verbose': False,
        'task_type': 'CPU'
    }
    
    model = cb.CatBoostClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        verbose=False
    )
    
    # Evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"   ROC AUC:   {auc:.4f}")
    print(f"   Accuracy:  {acc:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    
    return model, auc

def create_regime_ensemble(xgb_model, lgb_model, cat_model):
    """Create voting ensemble of 3 models."""
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', xgb_model),
            ('lgb', lgb_model),
            ('cat', cat_model)
        ],
        voting='soft',
        weights=[1, 1, 1]
    )
    return ensemble

def evaluate_regime_ensemble(ensemble, X_test, y_test, regime_name):
    """Evaluate ensemble performance."""
    print(f"\n🎯 Evaluating {regime_name} Ensemble...")
    
    y_pred_proba = ensemble.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"   ROC AUC:   {auc:.4f} ⭐")
    print(f"   Accuracy:  {acc:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    
    return {
        'roc_auc': auc,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def train_regime_specific_models():
    """Main training loop for regime-specific models."""
    print("="*70)
    print("🎯 REGIME-SPECIFIC MODEL TRAINING")
    print("="*70)
    
    # Load data
    df = load_regime_data()
    
    # Train models for each regime
    regimes = ['BEAR', 'SIDEWAYS', 'BULL']
    all_results = {}
    
    for regime in regimes:
        print(f"\n{'='*70}")
        print(f"📊 TRAINING MODELS FOR {regime} MARKET")
        print(f"{'='*70}")
        
        # Prepare regime-specific dataset
        X, y, feature_cols = prepare_regime_dataset(df, regime)
        
        # Skip if not enough samples
        if len(X) < 100:
            print(f"   ⚠️ Not enough samples for {regime}, skipping...")
            continue
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n   Train: {len(X_train):,} samples")
        print(f"   Test:  {len(X_test):,} samples")
        
        # Train individual models
        xgb_model, xgb_auc = train_xgboost_regime(X_train, X_test, y_train, y_test, regime)
        lgb_model, lgb_auc = train_lightgbm_regime(X_train, X_test, y_train, y_test, regime)
        cat_model, cat_auc = train_catboost_regime(X_train, X_test, y_train, y_test, regime)
        
        # Create ensemble
        print(f"\n🔗 Creating {regime} ensemble...")
        ensemble = create_regime_ensemble(xgb_model, lgb_model, cat_model)
        
        # Fit ensemble (required for VotingClassifier)
        ensemble.fit(X_train, y_train)
        
        # Evaluate ensemble
        metrics = evaluate_regime_ensemble(ensemble, X_test, y_test, regime)
        
        # Save models
        regime_lower = regime.lower()
        model_dir = Path('models')
        model_dir.mkdir(exist_ok=True)
        
        print(f"\n💾 Saving {regime} models...")
        
        # Save ensemble
        ensemble_path = model_dir / f'ensemble_{regime_lower}_regime.pkl'
        joblib.dump(ensemble, ensemble_path)
        print(f"   Ensemble: {ensemble_path}")
        
        # Save individual models
        joblib.dump(xgb_model, model_dir / f'xgb_{regime_lower}_regime.pkl')
        joblib.dump(lgb_model, model_dir / f'lgb_{regime_lower}_regime.pkl')
        joblib.dump(cat_model, model_dir / f'cat_{regime_lower}_regime.pkl')
        
        # Save metadata
        metadata = {
            'regime': regime,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features': feature_cols,
            'xgb_auc': float(xgb_auc),
            'lgb_auc': float(lgb_auc),
            'cat_auc': float(cat_auc),
            'ensemble_metrics': {k: float(v) for k, v in metrics.items()},
            'class_distribution': {
                'train_up': int((y_train==1).sum()),
                'train_down': int((y_train==0).sum()),
                'test_up': int((y_test==1).sum()),
                'test_down': int((y_test==0).sum())
            }
        }
        
        metadata_path = model_dir / f'ensemble_{regime_lower}_regime_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"   Metadata: {metadata_path}")
        
        all_results[regime] = metrics
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 TRAINING SUMMARY")
    print(f"{'='*70}")
    
    for regime, metrics in all_results.items():
        print(f"\n{regime}:")
        print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
    
    print(f"\n{'='*70}")
    print("✅ REGIME-SPECIFIC TRAINING COMPLETE!")
    print(f"{'='*70}")
    print("\nNext steps:")
    print("1. Compare regime-specific vs unified model performance")
    print("2. Implement regime-aware trading strategy")
    print("3. Backtest with dynamic regime detection")
    print("4. Consider regime transition handling")

if __name__ == '__main__':
    train_regime_specific_models()
