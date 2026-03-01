"""
Retrain regime-specific models with order book features.

This script:
1. Loads data with regime labels + order book features
2. Trains separate models for each regime (BEAR, SIDEWAYS, BULL)
3. Uses ensemble of XGBoost + LightGBM + CatBoost
4. Saves models with order book feature enhancements

Expected improvement: +5-10% win rate from order book signals
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb
import lightgbm as lgb
try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("⚠️  CatBoost not available, using XGBoost + LightGBM ensemble")


def load_data_with_orderbook():
    """Load regime-labeled data with order book features."""
    print("\n📥 Loading data with order book features...")
    
    # Need to combine regime labels with OB features
    regimes_df = pd.read_parquet('data/processed/btcusdt_1h_with_regimes.parquet')
    ob_df = pd.read_parquet('data/processed/btcusdt_1h_with_orderbook.parquet')
    
    print(f"   Regimes: {len(regimes_df):,} rows")
    print(f"   Order book: {len(ob_df):,} rows")
    
    # Merge on timestamp if available, otherwise use index
    if 'timestamp' in regimes_df.columns and 'timestamp' in ob_df.columns:
        df = pd.merge(regimes_df, ob_df, on='timestamp', how='inner', suffixes=('', '_ob'))
    else:
        # Merge on index
        df = regimes_df.join(ob_df, rsuffix='_ob')
    
    print(f"   Merged: {len(df):,} rows")
    
    # Filter to valid targets
    df = df[df['target_4h'].notna()].copy()
    print(f"   Valid targets: {len(df):,} rows")
    
    return df


def train_regime_model(df, regime_name):
    """Train ensemble model for specific regime with order book features."""
    print(f"\n{'='*60}")
    print(f"🎯 Training {regime_name.upper()} Regime Model")
    print(f"{'='*60}")
    
    # Filter to regime
    regime_df = df[df['regime_name'] == regime_name].copy()
    print(f"\nRegime samples: {len(regime_df):,}")
    
    # Feature columns
    exclude_cols = [
        'target_4h', 'regime', 'regime_name', 
        'timestamp', 'timestamp_ob'
    ] + [col for col in regime_df.columns if col.startswith('future_')]
    
    feature_cols = [col for col in regime_df.columns if col not in exclude_cols]
    
    # Count OB features
    ob_features = [col for col in feature_cols if 'ob_' in col]
    print(f"Total features: {len(feature_cols)} ({len(ob_features)} order book)")
    
    X = regime_df[feature_cols].fillna(0)
    
    # Convert any object columns to numeric
    for col in X.columns:
        if X[col].dtype == 'object':
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
            except:
                # If conversion fails, drop the column
                print(f"   Warning: Dropping non-numeric column: {col}")
                X = X.drop(columns=[col])
    
    y = regime_df['target_4h']
    
    # Class distribution
    print(f"\nClass distribution:")
    print(f"  UP (1):   {(y==1).sum():>6} ({(y==1).sum()/len(y)*100:.1f}%)")
    print(f"  DOWN (0): {(y==0).sum():>6} ({(y==0).sum()/len(y)*100:.1f}%)")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")
    
    # Train ensemble models
    print(f"\n🌲 Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        max_depth=8,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric='auc',
        random_state=42,
        tree_method='hist'
    )
    xgb_model.fit(X_train, y_train, verbose=False)
    xgb_auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])
    print(f"   XGBoost ROC AUC: {xgb_auc:.4f}")
    
    print(f"\n💡 Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        max_depth=8,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary',
        metric='auc',
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    lgb_auc = roc_auc_score(y_test, lgb_model.predict_proba(X_test)[:, 1])
    print(f"   LightGBM ROC AUC: {lgb_auc:.4f}")
    
    # Train CatBoost if available
    cat_model = None
    cat_auc = None
    if HAS_CATBOOST:
        print(f"\n🐈 Training CatBoost...")
        cat_model = cb.CatBoostClassifier(
            depth=8,
            learning_rate=0.05,
            iterations=300,
            subsample=0.8,
            random_state=42,
            verbose=False
        )
        cat_model.fit(X_train, y_train)
        cat_auc = roc_auc_score(y_test, cat_model.predict_proba(X_test)[:, 1])
        print(f"   CatBoost ROC AUC: {cat_auc:.4f}")
    
    # Ensemble predictions
    print(f"\n🎼 Creating ensemble...")
    if HAS_CATBOOST and cat_model is not None:
        ensemble_proba = (
            xgb_model.predict_proba(X_test)[:, 1] * 0.33 +
            lgb_model.predict_proba(X_test)[:, 1] * 0.33 +
            cat_model.predict_proba(X_test)[:, 1] * 0.34
        )
    else:
        ensemble_proba = (
            xgb_model.predict_proba(X_test)[:, 1] * 0.5 +
            lgb_model.predict_proba(X_test)[:, 1] * 0.5
        )
    ensemble_auc = roc_auc_score(y_test, ensemble_proba)
    ensemble_acc = accuracy_score(y_test, (ensemble_proba >= 0.5).astype(int))
    
    print(f"   Ensemble ROC AUC: {ensemble_auc:.4f} ⭐")
    print(f"   Ensemble Accuracy: {ensemble_acc:.4f}")
    
    # Save models
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    regime_lower = regime_name.lower()
    
    joblib.dump(xgb_model, models_dir / f'xgb_{regime_lower}_with_ob.pkl')
    joblib.dump(lgb_model, models_dir / f'lgb_{regime_lower}_with_ob.pkl')
    if HAS_CATBOOST and cat_model is not None:
        joblib.dump(cat_model, models_dir / f'cat_{regime_lower}_with_ob.pkl')
    
    # Save metadata
    metadata = {
        'regime': regime_name,
        'samples': len(regime_df),
        'features': len(feature_cols),
        'ob_features': len(ob_features),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'xgb_auc': float(xgb_auc),
        'lgb_auc': float(lgb_auc),
        'cat_auc': float(cat_auc) if cat_auc is not None else None,
        'ensemble_auc': float(ensemble_auc),
        'ensemble_accuracy': float(ensemble_acc),
        'feature_names': feature_cols,
        'has_catboost': HAS_CATBOOST
    }
    
    with open(models_dir / f'ensemble_{regime_lower}_with_ob_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n💾 Saved models to models/ directory")
    
    return ensemble_auc, metadata


def main():
    """Main training pipeline."""
    print("🚀 Training Regime-Specific Models with Order Book Features")
    print("="*60)
    
    try:
        # Load data
        df = load_data_with_orderbook()
        
        # Train each regime
        results = {}
        
        for regime_name in ['BEAR', 'SIDEWAYS', 'BULL']:
            auc, metadata = train_regime_model(df, regime_name)
            results[regime_name] = auc
        
        # Summary
        print(f"\n{'='*60}")
        print("📊 TRAINING SUMMARY")
        print(f"{'='*60}")
        
        for regime, auc in results.items():
            print(f"{regime:>10}: {auc:.4f} ROC AUC")
        
        avg_auc = np.mean(list(results.values()))
        print(f"\n{'Average':>10}: {avg_auc:.4f} ROC AUC")
        
        # Compare with previous models
        print(f"\n💡 Comparing with previous regime models...")
        try:
            prev_results = {}
            for regime in ['bear', 'sideways', 'bull']:
                meta_path = Path(f'models/ensemble_{regime}_regime_metadata.json')
                if meta_path.exists():
                    with open(meta_path) as f:
                        meta = json.load(f)
                        prev_results[regime.upper()] = meta.get('ensemble_auc', 0)
            
            if prev_results:
                print(f"\n   Previous Results (without OB features):")
                for regime, auc in prev_results.items():
                    print(f"   {regime:>10}: {auc:.4f}")
                
                prev_avg = np.mean(list(prev_results.values()))
                print(f"   {'Average':>10}: {prev_avg:.4f}")
                
                improvement = (avg_auc - prev_avg) * 100
                print(f"\n   🎯 Improvement: {improvement:+.2f}% ROC AUC")
                
                if improvement > 0:
                    print(f"   ✅ Order book features improved performance!")
                else:
                    print(f"   ⚠️  No improvement detected")
        except Exception as e:
            print(f"   ℹ️  Could not compare with previous models: {e}")
        
        print(f"\n✅ All regime models trained successfully!")
        print(f"\n💡 Next steps:")
        print(f"   1. Backtest with new OB-enhanced models")
        print(f"   2. Update live trading to use real-time order book")
        print(f"   3. Monitor improvement in win rate")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
