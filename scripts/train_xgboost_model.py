"""
XGBoost Model Training Pipeline for Bitcoin Price Prediction.

This script:
1. Loads processed features from data/processed/btcusdt_1h_features.parquet
2. Builds ML dataset with 4-hour prediction horizon
3. Creates time-based splits (70% train, 15% valid, 15% test)
4. Trains and optimizes XGBoost binary classifier
5. Evaluates model performance with comprehensive metrics
6. Saves trained model to models/xgb_h4.pkl

Usage: python scripts/train_xgboost_model.py
"""
import os
import sys
from pathlib import Path
import pickle
import warnings
from datetime import datetime

# Add src to path for imports
script_dir = Path(__file__).parent
src_dir = script_dir.parent / "src"
sys.path.insert(0, str(src_dir))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ML imports
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix,
    classification_report
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import ParameterGrid

# Our modules
from models.ml_dataset import build_ml_dataset


def create_time_splits(X: pd.DataFrame, y: pd.Series, 
                      train_pct: float = 0.70, 
                      valid_pct: float = 0.15, 
                      test_pct: float = 0.15) -> dict:
    """
    Create time-based splits respecting temporal order.
    
    Args:
        X: Features DataFrame
        y: Target Series
        train_pct: Training set percentage
        valid_pct: Validation set percentage  
        test_pct: Test set percentage
    
    Returns:
        Dictionary with train/valid/test splits
    """
    assert abs(train_pct + valid_pct + test_pct - 1.0) < 0.001, "Percentages must sum to 1"
    
    n_samples = len(X)
    train_end = int(n_samples * train_pct)
    valid_end = int(n_samples * (train_pct + valid_pct))
    
    # Create splits maintaining temporal order
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    
    X_valid = X.iloc[train_end:valid_end]
    y_valid = y.iloc[train_end:valid_end]
    
    X_test = X.iloc[valid_end:]
    y_test = y.iloc[valid_end:]
    
    print(f"Data splits created:")
    print(f"  Train: {len(X_train):,} samples ({len(X_train)/n_samples:.1%})")
    print(f"  Valid: {len(X_valid):,} samples ({len(X_valid)/n_samples:.1%})")
    print(f"  Test:  {len(X_test):,} samples ({len(X_test)/n_samples:.1%})")
    
    # Check label distribution
    for split_name, y_split in [("Train", y_train), ("Valid", y_valid), ("Test", y_test)]:
        pos_rate = y_split.mean()
        print(f"  {split_name} positive rate: {pos_rate:.1%}")
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_valid': X_valid, 'y_valid': y_valid,
        'X_test': X_test, 'y_test': y_test
    }


def optimize_xgboost_hyperparameters(X_train: pd.DataFrame, y_train: pd.Series,
                                   X_valid: pd.DataFrame, y_valid: pd.Series,
                                   param_grid: dict = None) -> dict:
    """
    Optimize XGBoost hyperparameters using validation set.
    
    Args:
        X_train, y_train: Training data
        X_valid, y_valid: Validation data  
        param_grid: Parameter grid to search
    
    Returns:
        Best parameters dictionary
    """
    if param_grid is None:
        param_grid = {
            'max_depth': [3, 4, 5, 6],
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.05, 0.1, 0.15],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }
    
    print(f"Optimizing hyperparameters with {len(ParameterGrid(param_grid))} combinations...")
    
    best_score = 0
    best_params = None
    results = []
    
    for i, params in enumerate(ParameterGrid(param_grid)):
        # Train model with current parameters
        model = xgb.XGBClassifier(
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss',
            **params
        )
        
        # Fit with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
            early_stopping_rounds=20
        )
        
        # Evaluate on validation set
        y_valid_pred = model.predict_proba(X_valid)[:, 1]
        score = roc_auc_score(y_valid, y_valid_pred)
        
        results.append({
            'params': params.copy(),
            'roc_auc': score,
            'n_estimators_best': model.best_iteration + 1
        })
        
        if score > best_score:
            best_score = score
            best_params = params.copy()
            # Update n_estimators to best iteration
            best_params['n_estimators'] = model.best_iteration + 1
        
        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{len(ParameterGrid(param_grid))} combinations...")
    
    print(f"Best validation ROC AUC: {best_score:.4f}")
    print(f"Best parameters: {best_params}")
    
    return best_params, results


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, 
                  model_name: str = "XGBoost") -> dict:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained model
        X_test, y_test: Test data
        model_name: Name for reporting
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
    }
    
    # Print detailed evaluation
    print(f"\n{'='*60}")
    print(f"  {model_name.upper()} MODEL EVALUATION")
    print(f"{'='*60}")
    
    print(f"\n📊 CLASSIFICATION METRICS:")
    print(f"   Accuracy:     {metrics['accuracy']:.4f}")
    print(f"   Precision:    {metrics['precision']:.4f}")
    print(f"   Recall:       {metrics['recall']:.4f}")
    print(f"   F1-Score:     {metrics['f1']:.4f}")
    print(f"   ROC AUC:      {metrics['roc_auc']:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n📋 CONFUSION MATRIX:")
    print(f"   True Neg:  {tn:,}")
    print(f"   False Pos: {fp:,}")
    print(f"   False Neg: {fn:,}")
    print(f"   True Pos:  {tp:,}")
    
    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"\n📈 ADDITIONAL METRICS:")
    print(f"   Specificity:  {specificity:.4f}")
    print(f"   Sensitivity:  {sensitivity:.4f}")
    print(f"   Pos Rate:     {y_test.mean():.4f}")
    print(f"   Pred Pos Rate: {y_pred.mean():.4f}")
    
    return {
        **metrics,
        'confusion_matrix': cm,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }


def plot_evaluation_charts(y_test: pd.Series, y_pred_proba: np.ndarray, 
                          output_dir: Path, model_name: str = "XGBoost"):
    """
    Create evaluation plots.
    
    Args:
        y_test: True labels
        y_pred_proba: Predicted probabilities
        output_dir: Directory to save plots
        model_name: Model name for titles
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    ax1.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title(f'{model_name} - ROC Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    ax2.plot(recall, precision, linewidth=2)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title(f'{model_name} - Precision-Recall Curve')
    ax2.grid(True, alpha=0.3)
    
    # Prediction Distribution
    ax3.hist(y_pred_proba[y_test == 0], bins=50, alpha=0.7, label='Class 0', density=True)
    ax3.hist(y_pred_proba[y_test == 1], bins=50, alpha=0.7, label='Class 1', density=True)
    ax3.set_xlabel('Predicted Probability')
    ax3.set_ylabel('Density')
    ax3.set_title(f'{model_name} - Prediction Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Calibration Plot
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, y_pred_proba, n_bins=10
    )
    ax4.plot(mean_predicted_value, fraction_of_positives, 's-', linewidth=2, label='Model')
    ax4.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect Calibration')
    ax4.set_xlabel('Mean Predicted Probability')
    ax4.set_ylabel('Fraction of Positives')
    ax4.set_title(f'{model_name} - Calibration Plot')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / f'{model_name.lower()}_evaluation.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Evaluation plots saved to: {plot_path}")
    plt.close()


def plot_feature_importance(model, feature_names: list, output_dir: Path, 
                           model_name: str = "XGBoost", top_n: int = 20):
    """
    Plot feature importance.
    
    Args:
        model: Trained XGBoost model
        feature_names: List of feature names
        output_dir: Directory to save plot
        model_name: Model name for title
        top_n: Number of top features to show
    """
    # Get feature importance
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Plot top features
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(top_n)
    
    plt.barh(range(len(top_features)), top_features['importance'], color='skyblue')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'{model_name} - Top {top_n} Feature Importance')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / f'{model_name.lower()}_feature_importance.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Feature importance plot saved to: {plot_path}")
    plt.close()
    
    # Print top features
    print(f"\n🎯 TOP {min(10, len(feature_importance))} FEATURES:")
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
        print(f"   {i+1:2d}. {row['feature']:<25} {row['importance']:.4f}")


def main():
    """Main training pipeline."""
    print("🚀 XGBoost Model Training Pipeline")
    print("=" * 50)
    
    # Define paths
    features_path = script_dir.parent / "data" / "processed" / "btcusdt_1h_features.parquet"
    models_dir = script_dir.parent / "models"
    outputs_dir = script_dir.parent / "outputs"
    
    # Ensure directories exist
    models_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)
    
    # Check if features file exists
    if not features_path.exists():
        print(f"❌ Features file not found: {features_path}")
        print("   Please run 'python scripts/build_dataset.py' first.")
        return 1
    
    try:
        # Load features
        print(f"📥 Loading features from {features_path.name}...")
        df = pd.read_parquet(features_path)
        print(f"   Loaded {len(df):,} rows with {len(df.columns)} columns")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Build ML dataset
        print(f"\n🎯 Building ML dataset with 4-hour horizon...")
        X, y, aligned_index = build_ml_dataset(
            df=df,
            horizon_hours=4,
            threshold=0.01,  # 1% threshold for positive labels
            validate_features=True
        )
        
        # Create time-based splits
        print(f"\n📊 Creating time-based data splits...")
        splits = create_time_splits(X, y, train_pct=0.70, valid_pct=0.15, test_pct=0.15)
        
        # Optimize hyperparameters
        print(f"\n🔧 Optimizing XGBoost hyperparameters...")
        best_params, optimization_results = optimize_xgboost_hyperparameters(
            splits['X_train'], splits['y_train'],
            splits['X_valid'], splits['y_valid']
        )
        
        # Train final model with best parameters
        print(f"\n🤖 Training final XGBoost model...")
        final_model = xgb.XGBClassifier(
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss',
            **best_params
        )
        
        # Combine train and validation for final training
        X_train_final = pd.concat([splits['X_train'], splits['X_valid']])
        y_train_final = pd.concat([splits['y_train'], splits['y_valid']])
        
        final_model.fit(X_train_final, y_train_final, verbose=False)
        
        # Evaluate model
        print(f"\n📈 Evaluating model performance...")
        eval_results = evaluate_model(final_model, splits['X_test'], splits['y_test'])
        
        # Create evaluation plots
        print(f"\n📊 Creating evaluation visualizations...")
        plot_evaluation_charts(
            splits['y_test'], eval_results['y_pred_proba'], 
            outputs_dir, "XGBoost"
        )
        
        # Plot feature importance
        plot_feature_importance(
            final_model, X.columns.tolist(), 
            outputs_dir, "XGBoost"
        )
        
        # Save model and metadata
        model_path = models_dir / "xgb_h4.pkl"
        model_metadata = {
            'model': final_model,
            'feature_columns': X.columns.tolist(),
            'best_params': best_params,
            'evaluation_metrics': eval_results,
            'training_info': {
                'horizon_hours': 4,
                'threshold': 0.01,
                'train_samples': len(X_train_final),
                'test_samples': len(splits['X_test']),
                'train_date_range': (df['timestamp'].iloc[0], df['timestamp'].iloc[len(X_train_final)-1]),
                'test_date_range': (df['timestamp'].iloc[-len(splits['X_test']):].iloc[0], df['timestamp'].iloc[-1]),
                'training_date': datetime.now().isoformat()
            }
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_metadata, f)
        
        print(f"\n💾 Model saved to: {model_path}")
        
        # Final summary
        print(f"\n✅ Training completed successfully!")
        print(f"   Model: XGBoost Binary Classifier")
        print(f"   Features: {len(X.columns)} features")
        print(f"   Training samples: {len(X_train_final):,}")
        print(f"   Test ROC AUC: {eval_results['roc_auc']:.4f}")
        print(f"   Test Accuracy: {eval_results['accuracy']:.4f}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    exit_code = main()
    sys.exit(exit_code)
