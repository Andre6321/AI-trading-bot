"""
Hyperparameter optimization using Optuna for XGBoost, LightGBM, and CatBoost.
Automatically finds the best hyperparameters to maximize ROC AUC.
Creates 3-model ensemble for maximum prediction accuracy.
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
import catboost as cb
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import SMOTE

# Import enhanced features
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from models.ml_dataset import build_ml_dataset


def create_walk_forward_splits(
    df: pd.DataFrame,
    n_splits: int = 5
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Create walk-forward validation splits."""
    print(f"\n📊 Creating {n_splits} walk-forward splits...")
    
    splits = []
    step_size = len(df) // (n_splits + 1)
    
    for i in range(n_splits):
        train_end = (i + 1) * step_size
        test_end = min(train_end + step_size // 2, len(df))
        
        train_data = df.iloc[:train_end].copy()
        test_data = df.iloc[train_end:test_end].copy()
        
        print(f"   Split {i+1}: Train={len(train_data)}, Test={len(test_data)}")
        splits.append((train_data, test_data))
    
    return splits


def objective_xgboost(trial: optuna.Trial, X_train, y_train, X_val, y_val) -> float:
    """Optuna objective function for XGBoost hyperparameter optimization."""
    
    # Calculate class weight
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
    
    # Suggest hyperparameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': 42,
        'scale_pos_weight': scale_pos_weight,
        
        # Optuna-optimized parameters
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
    }
    
    # Train model
    model = xgb.XGBClassifier(**params, early_stopping_rounds=50)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Evaluate
    y_proba = model.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, y_proba)
    
    return roc_auc


def objective_lightgbm(trial: optuna.Trial, X_train, y_train, X_val, y_val) -> float:
    """Optuna objective function for LightGBM hyperparameter optimization."""
    
    # Suggest hyperparameters
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'random_state': 42,
        'class_weight': 'balanced',
        'verbose': -1,
        
        # Optuna-optimized parameters
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
    }
    
    # Train model
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    # Evaluate
    y_proba = model.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, y_proba)
    
    return roc_auc


def optimize_xgboost(X_train, y_train, X_val, y_val, n_trials: int = 100) -> Dict[str, Any]:
    """Optimize XGBoost hyperparameters using Optuna."""
    print("\n🔍 Optimizing XGBoost hyperparameters...")
    print(f"   Running {n_trials} trials...")
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        study_name='xgboost_optimization'
    )
    
    study.optimize(
        lambda trial: objective_xgboost(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    print(f"\n✅ XGBoost Optimization Complete!")
    print(f"   Best ROC AUC: {study.best_value:.4f}")
    print(f"   Best params: {study.best_params}")
    
    return study.best_params


def optimize_lightgbm(X_train, y_train, X_val, y_val, n_trials: int = 100) -> Dict[str, Any]:
    """Optimize LightGBM hyperparameters using Optuna."""
    print("\n🔍 Optimizing LightGBM hyperparameters...")
    print(f"   Running {n_trials} trials...")
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        study_name='lightgbm_optimization'
    )
    
    study.optimize(
        lambda trial: objective_lightgbm(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    print(f"\n✅ LightGBM Optimization Complete!")
    print(f"   Best ROC AUC: {study.best_value:.4f}")
    print(f"   Best params: {study.best_params}")
    
    return study.best_params


def objective_catboost(trial: optuna.Trial, X_train, y_train, X_val, y_val) -> float:
    """Optuna objective function for CatBoost hyperparameter optimization."""
    
    # Calculate class weight
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
    
    # Suggest hyperparameters
    params = {
        'loss_function': 'Logloss',
        'random_seed': 42,
        'verbose': False,
        'scale_pos_weight': scale_pos_weight,
        
        # Optuna-optimized parameters
        'depth': trial.suggest_int('depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'iterations': trial.suggest_int('iterations', 100, 1000, step=100),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 0.0, 10.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
    }
    
    # Train model
    model = cb.CatBoostClassifier(**params, early_stopping_rounds=50)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Evaluate
    y_proba = model.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, y_proba)
    
    return roc_auc


def optimize_catboost(X_train, y_train, X_val, y_val, n_trials: int = 100) -> Dict[str, Any]:
    """Optimize CatBoost hyperparameters using Optuna."""
    print("\n🔍 Optimizing CatBoost hyperparameters...")
    print(f"   Running {n_trials} trials...")
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        study_name='catboost_optimization'
    )
    
    study.optimize(
        lambda trial: objective_catboost(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    print(f"\n✅ CatBoost Optimization Complete!")
    print(f"   Best ROC AUC: {study.best_value:.4f}")
    print(f"   Best params: {study.best_params}")
    
    return study.best_params


def train_final_model_xgb(X_train, y_train, X_val, y_val, best_params: Dict) -> xgb.XGBClassifier:
    """Train final XGBoost model with best hyperparameters."""
    print("\n🚀 Training final XGBoost model with optimized hyperparameters...")
    
    # Calculate class weight
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
    
    params = {
        **best_params,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': 42,
        'scale_pos_weight': scale_pos_weight
    }
    
    model = xgb.XGBClassifier(**params, early_stopping_rounds=50)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Evaluate
    y_proba = model.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, y_proba)
    print(f"   Final XGBoost ROC AUC: {roc_auc:.4f}")
    
    return model


def train_final_model_lgb(X_train, y_train, X_val, y_val, best_params: Dict) -> lgb.LGBMClassifier:
    """Train final LightGBM model with best hyperparameters."""
    print("\n🚀 Training final LightGBM model with optimized hyperparameters...")
    
    params = {
        **best_params,
        'objective': 'binary',
        'metric': 'binary_logloss',
        'random_state': 42,
        'class_weight': 'balanced',
        'verbose': -1
    }
    
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    # Evaluate
    y_proba = model.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, y_proba)
    print(f"   Final LightGBM ROC AUC: {roc_auc:.4f}")
    
    return model


def train_final_model_cat(X_train, y_train, X_val, y_val, best_params: Dict) -> cb.CatBoostClassifier:
    """Train final CatBoost model with best hyperparameters."""
    print("\n🚀 Training final CatBoost model with optimized hyperparameters...")
    
    # Calculate class weight
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
    
    params = {
        **best_params,
        'loss_function': 'Logloss',
        'random_seed': 42,
        'verbose': False,
        'scale_pos_weight': scale_pos_weight
    }
    
    model = cb.CatBoostClassifier(**params, early_stopping_rounds=50)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Evaluate
    y_proba = model.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, y_proba)
    print(f"   Final CatBoost ROC AUC: {roc_auc:.4f}")
    
    return model


def create_ensemble(xgb_model, lgb_model, cat_model) -> VotingClassifier:
    """Create ensemble of already-trained models (XGBoost + LightGBM + CatBoost)."""
    print("\n🎯 Creating 3-model optimized ensemble...")
    
    from sklearn.preprocessing import LabelEncoder
    
    ensemble = VotingClassifier(
        estimators=[('xgb', xgb_model), ('lgb', lgb_model), ('cat', cat_model)],
        voting='soft'
    )
    
    # Set fitted=True to skip refitting
    ensemble.estimators_ = [xgb_model, lgb_model, cat_model]
    ensemble.named_estimators_ = {'xgb': xgb_model, 'lgb': lgb_model, 'cat': cat_model}
    ensemble.classes_ = np.array([0, 1])
    
    # Create proper label encoder
    le = LabelEncoder()
    le.fit([0, 1])
    ensemble.le_ = le
    
    return ensemble


def main():
    """Main training pipeline with Optuna optimization for 3-model ensemble."""
    print("=" * 70)
    print("🎯 HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
    print("   XGBoost + LightGBM + CatBoost Ensemble")
    print("=" * 70)
    
    # Load data
    print("\n📥 Loading data...")
    data_path = Path("data/processed/btcusdt_1h_full_history_enhanced.parquet")
    
    if not data_path.exists():
        print("❌ Full history features not found. Run build_full_history_features.py first.")
        return
    
    df = pd.read_parquet(data_path)
    print(f"✅ Loaded {len(df)} rows with {len(df.columns)} features")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")
    
    # Create ML dataset
    print("\n📊 Creating ML dataset...")
    X, y, index = build_ml_dataset(
        df,
        horizon_hours=4,
        threshold=0.01
    )
    
    print(f"✅ ML dataset: {len(X)} samples, {len(X.columns)} features")
    print(f"   Class distribution: 0={((y==0).sum())} ({(y==0).mean()*100:.1f}%), 1={((y==1).sum())} ({(y==1).mean()*100:.1f}%)")
    
    feature_cols = X.columns.tolist()
    
    # Combine for walk-forward validation
    ml_df = X.copy()
    ml_df['target'] = y
    
    # Create splits
    splits = create_walk_forward_splits(ml_df, n_splits=5)
    
    # Use first split for optimization (to save time)
    print("\n" + "=" * 70)
    print("🔬 PHASE 1: HYPERPARAMETER OPTIMIZATION (using first split)")
    print("=" * 70)
    
    train_df, val_df = splits[0]
    
    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_val = val_df[feature_cols]
    y_val = val_df['target']
    
    # Apply SMOTE
    print(f"\n📊 Applying SMOTE to balance training data...")
    print(f"   Original: {(y_train==0).sum()} negative, {(y_train==1).sum()} positive")
    
    if (y_train == 1).sum() > 5:
        smote = SMOTE(random_state=42, k_neighbors=min(5, (y_train==1).sum()-1))
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        print(f"   After SMOTE: {(y_train_balanced==0).sum()} negative, {(y_train_balanced==1).sum()} positive")
    else:
        X_train_balanced, y_train_balanced = X_train, y_train
    
    # Optimize all three models
    xgb_best_params = optimize_xgboost(X_train_balanced, y_train_balanced, X_val, y_val, n_trials=100)
    lgb_best_params = optimize_lightgbm(X_train_balanced, y_train_balanced, X_val, y_val, n_trials=100)
    cat_best_params = optimize_catboost(X_train_balanced, y_train_balanced, X_val, y_val, n_trials=100)
    
    # Train final models with best hyperparameters on ALL data
    print("\n" + "=" * 70)
    print("🚀 PHASE 2: TRAINING FINAL MODELS WITH OPTIMIZED HYPERPARAMETERS")
    print("=" * 70)
    
    # Use all data for final training
    all_train_idx = int(len(ml_df) * 0.9)
    train_df_final = ml_df.iloc[:all_train_idx]
    val_df_final = ml_df.iloc[all_train_idx:]
    
    X_train_final = train_df_final[feature_cols]
    y_train_final = train_df_final['target']
    X_val_final = val_df_final[feature_cols]
    y_val_final = val_df_final['target']
    
    # Apply SMOTE to final training set
    if (y_train_final == 1).sum() > 5:
        smote = SMOTE(random_state=42, k_neighbors=min(5, (y_train_final==1).sum()-1))
        X_train_final_balanced, y_train_final_balanced = smote.fit_resample(X_train_final, y_train_final)
    else:
        X_train_final_balanced, y_train_final_balanced = X_train_final, y_train_final
    
    # Train final models
    xgb_model = train_final_model_xgb(X_train_final_balanced, y_train_final_balanced, X_val_final, y_val_final, xgb_best_params)
    lgb_model = train_final_model_lgb(X_train_final_balanced, y_train_final_balanced, X_val_final, y_val_final, lgb_best_params)
    cat_model = train_final_model_cat(X_train_final_balanced, y_train_final_balanced, X_val_final, y_val_final, cat_best_params)
    
    # Create 3-model ensemble
    ensemble = create_ensemble(xgb_model, lgb_model, cat_model)
    
    # Evaluate ensemble
    y_proba_ensemble = ensemble.predict_proba(X_val_final)[:, 1]
    ensemble_roc_auc = roc_auc_score(y_val_final, y_proba_ensemble)
    print(f"\n🎯 Final Ensemble ROC AUC: {ensemble_roc_auc:.4f}")
    
    # Save model
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    
    model_path = output_dir / "ensemble_btcusdt_h4_optuna_optimized.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(ensemble, f)
    
    print(f"\n✅ Optimized model saved to {model_path}")
    
    # Save metadata
    metadata = {
        "model_type": "ensemble_3model_optuna_optimized",
        "models": ["XGBoost", "LightGBM", "CatBoost"],
        "features": feature_cols,
        "n_features": len(feature_cols),
        "training_date": datetime.now().isoformat(),
        "optimization_trials": 300,  # 100 each
        "xgb_best_params": xgb_best_params,
        "lgb_best_params": lgb_best_params,
        "cat_best_params": cat_best_params,
        "final_roc_auc": float(ensemble_roc_auc),
        "data_samples": len(ml_df),
        "class_distribution": {
            "negative": int((y==0).sum()),
            "positive": int((y==1).sum())
        }
    }
    
    metadata_path = output_dir / "ensemble_btcusdt_h4_optuna_optimized_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"✅ Metadata saved to {metadata_path}")
    
    print("\n" + "=" * 70)
    print("🎉 OPTIMIZATION COMPLETE!")
    print("=" * 70)
    print(f"\n📊 RESULTS SUMMARY:")
    print(f"   XGBoost Best Params: {xgb_best_params}")
    print(f"   LightGBM Best Params: {lgb_best_params}")
    print(f"   CatBoost Best Params: {cat_best_params}")
    print(f"   Final 3-Model Ensemble ROC AUC: {ensemble_roc_auc:.4f}")
    print(f"\n💡 To use this model, update your .env file:")
    print(f"   MODEL_PATH=models/ensemble_btcusdt_h4_optuna_optimized.pkl")
    print(f"   MODEL_FEATURES_PATH=models/ensemble_btcusdt_h4_optuna_optimized_metadata.json")


if __name__ == "__main__":
    main()
