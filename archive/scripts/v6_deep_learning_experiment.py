"""
V6 Deep Learning + Ensemble Mega-Experiment
============================================
Comprehensive attempt at profitability using:

  1. DEEP LEARNING MODELS
     - LSTM (captures sequential dependencies in price action)
     - GRU (lighter LSTM variant)
     - Temporal Convolutional Network (TCN - parallel, fast training)
     - Transformer encoder (attention over time steps)

  2. ENSEMBLE / STACKING
     - Level-0: XGBoost, LightGBM, LSTM, TCN predictions
     - Level-1: Logistic regression meta-learner on OOS predictions
     - Weighted average with learned weights

  3. REGRESSION TARGETS
     - Instead of binary (TP hit / not), predict forward return magnitude
     - Then threshold on predicted return for signal generation
     - This gives the model more granular information

  4. IMPROVED FEATURES
     - Sequence features (lookback windows of 24/48/96 bars)
     - Normalised features (z-score per rolling window)
     - Additional microstructure features

  5. FIXED TRAILING STOP SIMULATOR
     - Conservative: assumes price hits WORST point first within each bar
     - For SHORT trailing: first check if HIGH hits stop, THEN update trail with LOW
     - Eliminates the intra-bar look-ahead bias from V5

  6. PROP FIRM ACCOUNT ($25K)
     - Realistic position sizing for $25K account
     - Max drawdown limits (typical prop firm: 5-10% daily, 10-12% overall)
     - Conservative risk per trade: 0.5-1% of account

Usage:
    python scripts/v6_deep_learning_experiment.py
"""
import os, sys, json, warnings, time, math
os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from imblearn.over_sampling import SMOTE

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from features.alpha_features import build_alpha_features
from features.triple_barrier import triple_barrier_labels

ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ======================================================================
# TREE MODEL PARAMS (from V4 Optuna)
# ======================================================================
LGB_PARAMS = {
    'objective': 'binary', 'metric': 'auc', 'random_state': 42,
    'class_weight': 'balanced', 'verbose': -1,
    'max_depth': 5, 'learning_rate': 0.03, 'n_estimators': 800,
    'subsample': 0.7, 'colsample_bytree': 0.6, 'min_child_samples': 30,
    'reg_alpha': 1.0, 'reg_lambda': 5.0, 'num_leaves': 31,
}
XGB_PARAMS = {
    'objective': 'binary:logistic', 'eval_metric': 'auc',
    'tree_method': 'hist', 'random_state': 42,
    'max_depth': 5, 'learning_rate': 0.03, 'n_estimators': 800,
    'subsample': 0.7, 'colsample_bytree': 0.6, 'min_child_weight': 20,
    'gamma': 1.0, 'reg_alpha': 1.0, 'reg_lambda': 5.0,
}

# ======================================================================
# 1. DATA LOADING & FEATURE ENGINEERING
# ======================================================================

def load_and_prepare_data():
    """Load data, build features, and return clean DataFrame."""
    paths = [
        ROOT / "data" / "processed" / "btcusdt_1h_full_history_enhanced.parquet",
        ROOT / "data" / "processed" / "btcusdt_1h_features.parquet",
    ]
    data_path = next((p for p in paths if p.exists()), None)
    assert data_path, "No data found!"

    df = pd.read_parquet(data_path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    print(f"   Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Build alpha features
    df = build_alpha_features(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"   After cleanup: {len(df):,} rows, {len(df.columns)} columns")

    # Hour feature
    df['_hour'] = df.index.hour if hasattr(df.index, 'hour') else 0

    # Regime filter (exclude ultra-low vol)
    if 'atr_14' in df.columns:
        df['atr_pct'] = df['atr_14'] / df['close'] * 100
        thr = df['atr_pct'].quantile(0.25)
        df['regime_ok'] = df['atr_pct'] >= thr
    else:
        df['regime_ok'] = True

    return df


def get_feature_cols(df):
    """Get feature column names, excluding targets and metadata."""
    exclude = {
        'open', 'high', 'low', 'close', 'volume', 'funding_rate',
        'open_interest', 'timestamp', 'target', 'classic_binary',
        'regime_filter', 'regime_ok', '_hour', 'atr_pct',
        'tb_label', 'tb_binary', 'tb_holding', 'tb_barrier_tp', 'tb_barrier_sl',
        'short_tb_label', 'short_tb_binary', 'short_tb_holding',
        'short_tb_barrier_tp', 'short_tb_barrier_sl',
        'fwd_ret', 'fwd_ret_label',
    }
    return [c for c in df.columns
            if c not in exclude
            and not any(p in c.lower() for p in ['future_', 'target_', 'label_'])
            and df[c].dtype in ('float64', 'float32', 'int64', 'int32')]


def build_forward_returns(df, horizons=[4, 8, 12, 24]):
    """Build forward return targets at multiple horizons (causal: shift -h)."""
    out = df.copy()
    for h in horizons:
        out[f'fwd_ret_{h}h'] = np.log(out['close'].shift(-h) / out['close'])
    return out


# ======================================================================
# 2. PYTORCH SEQUENCE DATASET
# ======================================================================

class SequenceDataset(Dataset):
    """Creates sequences of length `seq_len` from feature matrix."""

    def __init__(self, X, y, seq_len=24):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.X[idx:idx + self.seq_len]
        y_val = self.y[idx + self.seq_len - 1]  # label at end of sequence
        return x_seq, y_val


# ======================================================================
# 3. DEEP LEARNING MODELS
# ======================================================================

class LSTMModel(nn.Module):
    """Bidirectional LSTM with attention pooling."""

    def __init__(self, input_dim, hidden_dim=64, n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0,
            bidirectional=True,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out, _ = self.lstm(x)  # (B, T, 2*H)
        # Attention-weighted pooling
        attn_w = self.attention(out)  # (B, T, 1)
        attn_w = torch.softmax(attn_w, dim=1)
        context = (out * attn_w).sum(dim=1)  # (B, 2*H)
        return self.head(context).squeeze(-1)


class GRUModel(nn.Module):
    """GRU with last-hidden-state output."""

    def __init__(self, input_dim, hidden_dim=64, n_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out, h = self.gru(x)
        # Use last time step
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


class TCNBlock(nn.Module):
    """Single temporal conv block with residual."""

    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.relu = nn.GELU()
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.padding = padding

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = out[:, :, :x.size(2)]  # causal trim
        out = self.bn1(out)
        out = self.relu(out)
        out = self.drop(out)
        out = self.conv2(out)
        out = out[:, :, :x.size(2)]  # causal trim
        out = self.bn2(out)
        out = self.relu(out)
        out = self.drop(out)
        if self.downsample is not None:
            res = self.downsample(res)
        return self.relu(out + res)


class TCNModel(nn.Module):
    """Temporal Convolutional Network with increasing dilation."""

    def __init__(self, input_dim, hidden_dim=64, n_layers=4, dropout=0.2):
        super().__init__()
        layers = []
        for i in range(n_layers):
            in_ch = input_dim if i == 0 else hidden_dim
            layers.append(TCNBlock(in_ch, hidden_dim, kernel_size=3,
                                   dilation=2**i, dropout=dropout))
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, T, C) -> (B, C, T) for Conv1d
        x = x.permute(0, 2, 1)
        out = self.tcn(x)
        # Global average pool over time
        out = out.mean(dim=2)  # (B, C)
        return self.head(out).squeeze(-1)


class TransformerModel(nn.Module):
    """Transformer encoder for time series."""

    def __init__(self, input_dim, d_model=64, nhead=4, n_layers=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, 200, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, T, C = x.shape
        x = self.input_proj(x)  # (B, T, d_model)
        x = x + self.pos_enc[:, :T, :]
        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        out = self.encoder(x, mask=mask)
        # Use last time step (most recent)
        out = out[:, -1, :]
        return self.head(out).squeeze(-1)


# ======================================================================
# 4. REGRESSION DL MODELS (predict forward returns instead of binary)
# ======================================================================

class LSTMRegressor(nn.Module):
    """LSTM that predicts forward return magnitude."""

    def __init__(self, input_dim, hidden_dim=64, n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


# ======================================================================
# 5. DL TRAINING ENGINE
# ======================================================================

def train_dl_model(model, train_loader, val_loader, epochs=80,
                   lr=1e-3, patience=15, task='classification'):
    """Train a DL model with early stopping."""
    model = model.to(DEVICE)

    if task == 'classification':
        criterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val = float('inf') if task == 'regression' else 0.0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        scheduler.step()
        avg_train = train_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                pred = model(xb)
                val_preds.extend(pred.cpu().numpy())
                val_true.extend(yb.numpy())

        val_preds = np.array(val_preds)
        val_true = np.array(val_true)

        if task == 'classification':
            try:
                val_auc = roc_auc_score(val_true, val_preds)
            except:
                val_auc = 0.5
            metric = val_auc
            improved = metric > best_val
        else:
            val_mse = mean_squared_error(val_true, val_preds)
            metric = -val_mse  # higher is better
            improved = metric > best_val

        if improved:
            best_val = metric
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    model = model.to(DEVICE)

    if task == 'classification':
        return model, best_val
    else:
        return model, -best_val  # return MSE as positive


def predict_dl(model, X_scaled, seq_len=24):
    """Generate predictions for all valid positions."""
    model.eval()
    n = len(X_scaled)
    preds = np.full(n, np.nan)

    # Create sequences and predict in batches
    dataset = SequenceDataset(X_scaled, np.zeros(n), seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=512, shuffle=False)

    idx = seq_len - 1  # first prediction position
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(DEVICE)
            out = model(xb).cpu().numpy()
            for p in out:
                if idx < n:
                    preds[idx] = p
                idx += 1

    return preds


# ======================================================================
# 6. WALK-FORWARD ENGINE (supports both tree and DL models)
# ======================================================================

def purged_wf_splits(n, n_splits=5, test_pct=0.12, purge=12):
    """Walk-forward splits with purge gap."""
    test_size = int(n * test_pct)
    splits = []
    for i in range(n_splits):
        te_start = n - (n_splits - i) * test_size
        te_end = min(te_start + test_size, n)
        tr_end = te_start - purge
        if te_start < test_size or tr_end < 100:
            continue
        splits.append((np.arange(0, tr_end), np.arange(te_start, te_end)))
    return splits


def walk_forward_tree(X, y, feature_cols, splits, side_label="LONG"):
    """Walk-forward with XGBoost + LightGBM ensemble. Returns OOS predictions."""
    oos_probs = np.full(len(X), np.nan)
    fold_aucs = []

    # CRITICAL: Only use feature columns to avoid target leakage
    X_feat = X[feature_cols]

    for fold_i, (tr_idx, te_idx) in enumerate(splits):
        X_tr, y_tr = X_feat.iloc[tr_idx], y.iloc[tr_idx]
        X_te, y_te = X_feat.iloc[te_idx], y.iloc[te_idx]

        # SMOTE
        if y_tr.sum() > 5 and y_tr.sum() < len(y_tr) - 5:
            smote = SMOTE(random_state=42, k_neighbors=min(5, int(y_tr.sum())-1))
            X_tr_s, y_tr_s = smote.fit_resample(X_tr, y_tr)
        else:
            X_tr_s, y_tr_s = X_tr, y_tr

        # XGBoost
        xp = dict(XGB_PARAMS)
        xp['scale_pos_weight'] = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        m_xgb = xgb.XGBClassifier(**xp, early_stopping_rounds=50)
        m_xgb.fit(X_tr_s, y_tr_s, eval_set=[(X_te, y_te)], verbose=False)

        # LightGBM
        m_lgb = lgb.LGBMClassifier(**LGB_PARAMS)
        m_lgb.fit(X_tr_s, y_tr_s, eval_set=[(X_te, y_te)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])

        p = (m_xgb.predict_proba(X_te)[:, 1] + m_lgb.predict_proba(X_te)[:, 1]) / 2
        auc = roc_auc_score(y_te, p)
        fold_aucs.append(auc)
        oos_probs[te_idx] = p
        print(f"      {side_label} Tree Fold {fold_i+1}: AUC={auc:.4f}")

    mean_auc = np.mean(fold_aucs)
    print(f"      {side_label} Tree Mean AUC: {mean_auc:.4f} +/- {np.std(fold_aucs):.4f}")
    return oos_probs, mean_auc


def walk_forward_dl(X_df, y_series, feature_cols, splits, model_class,
                    side_label="LONG", seq_len=24, task='classification',
                    **model_kwargs):
    """Walk-forward with a DL model. Returns OOS predictions."""
    oos_probs = np.full(len(X_df), np.nan)
    fold_aucs = []

    X_np = X_df[feature_cols].values.astype(np.float32)
    y_np = y_series.values.astype(np.float32)

    for fold_i, (tr_idx, te_idx) in enumerate(splits):
        # Scale features using training set statistics
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_np[tr_idx])
        X_te_s = scaler.transform(X_np[te_idx])

        y_tr = y_np[tr_idx]
        y_te = y_np[te_idx]

        # Need sequences: include some history before test set
        # Build combined array for sequence creation
        hist_len = seq_len + 10  # grab some history before test
        te_start = te_idx[0]
        hist_start = max(0, te_start - hist_len)
        # Re-scale the history portion using train scaler
        X_hist = scaler.transform(X_np[hist_start:te_start])

        # Training sequences
        train_ds = SequenceDataset(X_tr_s, y_tr, seq_len=seq_len)
        if len(train_ds) < 50:
            print(f"      {side_label} DL Fold {fold_i+1}: Too few sequences, skip")
            continue

        # Split train into train/val (last 15% for validation)
        val_size = max(int(len(train_ds) * 0.15), 50)
        train_size = len(train_ds) - val_size
        train_subset = torch.utils.data.Subset(train_ds, range(train_size))
        val_subset = torch.utils.data.Subset(train_ds, range(train_size, len(train_ds)))

        train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=256, shuffle=False)

        # Build and train model
        input_dim = len(feature_cols)
        model = model_class(input_dim, **model_kwargs)
        model, val_metric = train_dl_model(model, train_loader, val_loader,
                                           epochs=40, lr=1e-3, patience=8,
                                           task=task)

        # Predict on test set
        # Build test sequences using history + test data
        X_test_full = np.vstack([X_hist, X_te_s])
        y_test_full = np.concatenate([y_np[hist_start:te_start], y_te])
        test_ds = SequenceDataset(X_test_full, y_test_full, seq_len=seq_len)

        test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
        model.eval()
        test_preds = []
        with torch.no_grad():
            for xb, _ in test_loader:
                xb = xb.to(DEVICE)
                out = model(xb).cpu().numpy()
                test_preds.extend(out)
        test_preds = np.array(test_preds)

        # Map predictions back — we only care about the test portion
        # The test_ds starts at index 0 of X_test_full, first prediction at seq_len-1
        hist_bars = len(X_hist)
        test_offset = max(0, seq_len - 1 - hist_bars)
        pred_start_in_test = max(0, hist_bars - (seq_len - 1))

        # Map each prediction to test indices
        for p_idx, pred_val in enumerate(test_preds):
            # Position in X_test_full
            pos_in_full = p_idx + seq_len - 1
            # Position in test set
            pos_in_test = pos_in_full - hist_bars
            if 0 <= pos_in_test < len(te_idx):
                oos_probs[te_idx[pos_in_test]] = pred_val

        # Calculate AUC for this fold
        fold_preds = oos_probs[te_idx]
        valid = ~np.isnan(fold_preds)
        if valid.sum() > 20:
            if task == 'classification':
                try:
                    auc = roc_auc_score(y_te[valid], fold_preds[valid])
                except:
                    auc = 0.5
                fold_aucs.append(auc)
                print(f"      {side_label} DL Fold {fold_i+1}: AUC={auc:.4f}")
            else:
                rmse = np.sqrt(mean_squared_error(y_te[valid], fold_preds[valid]))
                fold_aucs.append(rmse)
                print(f"      {side_label} DL Fold {fold_i+1}: RMSE={rmse:.6f}")
        else:
            print(f"      {side_label} DL Fold {fold_i+1}: too few valid preds ({valid.sum()})")

    if fold_aucs:
        mean_metric = np.mean(fold_aucs)
        lbl = "AUC" if task == 'classification' else "RMSE"
        print(f"      {side_label} DL Mean {lbl}: {mean_metric:.4f} +/- {np.std(fold_aucs):.4f}")
    else:
        mean_metric = 0.5 if task == 'classification' else 999.0

    return oos_probs, mean_metric


# ======================================================================
# 7. STACKING ENSEMBLE
# ======================================================================

def build_stacking_ensemble(X_df, y_series, feature_cols, splits, side_label,
                            seq_len=24, precomputed=None):
    """
    Build a 2-level stacking ensemble:
      Level-0: XGB+LGB, LSTM, TCN, Transformer (reuse precomputed if available)
      Level-1: Logistic regression on OOS predictions from Level-0
    """
    print(f"\n   === STACKING ENSEMBLE ({side_label}) ===")

    # Level-0: Collect OOS predictions - reuse precomputed to save time
    level0 = {}
    scores = {}

    if precomputed and 'tree' in precomputed:
        print(f"   [1/4] Reusing tree predictions")
        level0['tree'] = precomputed['tree'][0]
        scores['tree'] = precomputed['tree'][1]
    else:
        print(f"   [1/4] Training tree ensemble (XGB+LGB)...")
        tree_probs, tree_auc = walk_forward_tree(X_df, y_series, feature_cols,
                                                 splits, side_label)
        level0['tree'] = tree_probs
        scores['tree'] = tree_auc

    if precomputed and 'lstm' in precomputed:
        print(f"   [2/4] Reusing LSTM predictions")
        level0['lstm'] = precomputed['lstm'][0]
        scores['lstm'] = precomputed['lstm'][1]
    else:
        print(f"   [2/4] Training LSTM...")
        lstm_probs, lstm_auc = walk_forward_dl(
            X_df, y_series, feature_cols, splits,
            LSTMModel, side_label, seq_len=seq_len,
            hidden_dim=48, n_layers=2, dropout=0.3,
        )
        level0['lstm'] = lstm_probs
        scores['lstm'] = lstm_auc

    if precomputed and 'tcn' in precomputed:
        print(f"   [3/4] Reusing TCN predictions")
        level0['tcn'] = precomputed['tcn'][0]
        scores['tcn'] = precomputed['tcn'][1]
    else:
        print(f"   [3/4] Training TCN...")
        tcn_probs, tcn_auc = walk_forward_dl(
            X_df, y_series, feature_cols, splits,
            TCNModel, side_label, seq_len=seq_len,
            hidden_dim=48, n_layers=4, dropout=0.2,
        )
        level0['tcn'] = tcn_probs
        scores['tcn'] = tcn_auc

    if precomputed and 'transformer' in precomputed:
        print(f"   [4/4] Reusing Transformer predictions")
        level0['transformer'] = precomputed['transformer'][0]
        scores['transformer'] = precomputed['transformer'][1]
    else:
        print(f"   [4/4] Training Transformer...")
        tf_probs, tf_auc = walk_forward_dl(
            X_df, y_series, feature_cols, splits,
            TransformerModel, side_label, seq_len=seq_len,
            d_model=48, nhead=4, n_layers=2, dropout=0.2,
        )
        level0['transformer'] = tf_probs
        scores['transformer'] = tf_auc

    # Print individual model scores
    print(f"\n   --- Level-0 scores ({side_label}) ---")
    for name, score in scores.items():
        print(f"      {name:>12s}: AUC={score:.4f}")

    # Level-1: Stack using logistic regression
    # Find bars where ALL level-0 models have valid predictions
    stack_cols = list(level0.keys())
    meta_X = np.column_stack([level0[k] for k in stack_cols])
    valid_mask = ~np.any(np.isnan(meta_X), axis=1)

    # Also only use bars where we have ground truth
    y_np = y_series.values
    valid_mask = valid_mask & ~np.isnan(y_np)

    if valid_mask.sum() < 100:
        print(f"   ⚠️  Only {valid_mask.sum()} valid stacking samples, using simple average")
        # Fall back to simple weighted average
        weights = np.array([max(scores[k] - 0.5, 0.01) for k in stack_cols])
        weights /= weights.sum()
        stacked = np.full(len(y_np), np.nan)
        for i, k in enumerate(stack_cols):
            valid_k = ~np.isnan(level0[k])
            stacked[valid_k] = np.nansum(
                [stacked[valid_k] if not np.all(np.isnan(stacked[valid_k])) else 0,
                 level0[k][valid_k] * weights[i]], axis=0
            )
        # Simpler fallback
        stacked = np.nanmean([level0[k] for k in stack_cols], axis=0)
        stacked_auc = 0.0
        try:
            vm = ~np.isnan(stacked) & ~np.isnan(y_np)
            stacked_auc = roc_auc_score(y_np[vm], stacked[vm])
        except:
            pass
        print(f"   Stacked (avg): AUC={stacked_auc:.4f}")
        return stacked, scores, stacked_auc

    # Walk-forward the meta-learner too
    meta_X_valid = meta_X[valid_mask]
    y_valid = y_np[valid_mask]
    n_valid = len(meta_X_valid)
    stacked_probs = np.full(len(y_np), np.nan)

    # Simple time-based split for meta-learner (70/30)
    split_pt = int(n_valid * 0.7)
    if split_pt > 50 and n_valid - split_pt > 30:
        meta_lr = LogisticRegression(C=1.0, max_iter=1000)
        meta_lr.fit(meta_X_valid[:split_pt], y_valid[:split_pt])
        meta_pred = meta_lr.predict_proba(meta_X_valid[split_pt:])[:, 1]

        try:
            stacked_auc = roc_auc_score(y_valid[split_pt:], meta_pred)
        except:
            stacked_auc = 0.5

        # Now predict on ALL valid samples
        all_pred = meta_lr.predict_proba(meta_X_valid)[:, 1]
        valid_indices = np.where(valid_mask)[0]
        for i, idx in enumerate(valid_indices):
            stacked_probs[idx] = all_pred[i]

        print(f"\n   Level-1 meta-learner coefficients: {dict(zip(stack_cols, meta_lr.coef_[0]))}")
        print(f"   Stacked AUC: {stacked_auc:.4f}")
    else:
        # Not enough data, use weighted average
        weights = np.array([max(scores[k] - 0.5, 0.01) for k in stack_cols])
        weights /= weights.sum()
        stacked_probs = np.zeros(len(y_np))
        for i, k in enumerate(stack_cols):
            valid_k = ~np.isnan(level0[k])
            stacked_probs[valid_k] += level0[k][valid_k] * weights[i]
        stacked_auc = 0.5

    return stacked_probs, scores, stacked_auc


# ======================================================================
# 8. FORWARD RETURN REGRESSION ENSEMBLE
# ======================================================================

def regression_experiment(X_df, feature_cols, df, splits, horizon=12):
    """
    Train regression models to predict forward returns.
    Returns predicted returns for signal generation.
    """
    print(f"\n   === REGRESSION EXPERIMENT (horizon={horizon}h) ===")

    # Build forward return target
    y_col = f'fwd_ret_{horizon}h'
    if y_col not in df.columns:
        df[y_col] = np.log(df['close'].shift(-horizon) / df['close'])

    mask = ~np.isnan(df[y_col].values) & ~np.any(np.isnan(X_df[feature_cols].values), axis=1)
    X_clean = X_df.loc[mask]
    y_clean = df.loc[mask, y_col]

    reg_splits = purged_wf_splits(len(X_clean), n_splits=5, test_pct=0.12, purge=horizon + 2)

    # Tree regression
    print(f"   [1/2] Tree regression...")
    tree_preds = np.full(len(X_clean), np.nan)
    fold_rmses = []
    for fold_i, (tr_idx, te_idx) in enumerate(reg_splits):
        X_tr = X_clean.iloc[tr_idx][feature_cols]
        y_tr = y_clean.iloc[tr_idx]
        X_te = X_clean.iloc[te_idx][feature_cols]
        y_te = y_clean.iloc[te_idx]

        m_lgb = lgb.LGBMRegressor(
            objective='regression', metric='rmse', random_state=42, verbose=-1,
            max_depth=5, learning_rate=0.03, n_estimators=800,
            subsample=0.7, colsample_bytree=0.6, min_child_samples=30,
            reg_alpha=1.0, reg_lambda=5.0, num_leaves=31,
        )
        m_lgb.fit(X_tr, y_tr, eval_set=[(X_te, y_te)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])

        m_xgb = xgb.XGBRegressor(
            objective='reg:squarederror', eval_metric='rmse',
            tree_method='hist', random_state=42,
            max_depth=5, learning_rate=0.03, n_estimators=800,
            subsample=0.7, colsample_bytree=0.6, min_child_weight=20,
            gamma=1.0, reg_alpha=1.0, reg_lambda=5.0,
            early_stopping_rounds=50,
        )
        m_xgb.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

        p = (m_lgb.predict(X_te) + m_xgb.predict(X_te)) / 2
        rmse = np.sqrt(mean_squared_error(y_te, p))
        fold_rmses.append(rmse)
        tree_preds[te_idx] = p
        print(f"      Tree Fold {fold_i+1}: RMSE={rmse:.6f}")

    print(f"      Tree Mean RMSE: {np.mean(fold_rmses):.6f}")

    # LSTM regression
    print(f"   [2/2] LSTM regression...")
    lstm_preds, lstm_rmse = walk_forward_dl(
        X_clean, y_clean, feature_cols, reg_splits,
        LSTMRegressor, "REG", seq_len=24, task='regression',
        hidden_dim=48, n_layers=2, dropout=0.3,
    )

    # Combine tree + LSTM predictions
    combined = np.full(len(df), np.nan)
    clean_idx = X_clean.index

    for i, idx in enumerate(clean_idx):
        iloc_pos = df.index.get_loc(idx)
        t_val = tree_preds[i] if not np.isnan(tree_preds[i]) else 0
        l_val = lstm_preds[i] if not np.isnan(lstm_preds[i]) else 0
        if not np.isnan(tree_preds[i]) and not np.isnan(lstm_preds[i]):
            combined[iloc_pos] = 0.6 * t_val + 0.4 * l_val
        elif not np.isnan(tree_preds[i]):
            combined[iloc_pos] = t_val
        elif not np.isnan(lstm_preds[i]):
            combined[iloc_pos] = l_val

    return combined


# ======================================================================
# 9. FIXED PORTFOLIO SIMULATOR (conservative trailing stop)
# ======================================================================

def simulate_portfolio_v6(
    close, high, low, atr, hours,
    long_probs, short_probs,
    cfg: dict,
) -> list:
    """
    Bar-by-bar portfolio simulator with FIXED trailing stop logic.

    Key fix from V5: For trailing stops, we assume WORST price happens first.
    - LONG trailing: first check if LOW hits stop, THEN update trail with HIGH
    - SHORT trailing: first check if HIGH hits stop, THEN update trail with LOW

    This is the conservative (honest) approach that avoids intra-bar look-ahead.

    For prop firm: supports max_drawdown_pct to stop trading if breached.
    """
    n = len(close)
    initial_cap = cfg.get('initial_capital', 25000)  # $25K prop firm
    cap = initial_cap
    peak_cap = initial_cap
    risk_pct = cfg.get('risk_per_trade', 0.01)  # 1% per trade for prop firm
    max_pos = cfg.get('max_pos_frac', 0.25)
    fee = cfg.get('fee_pct', 0.02) / 100       # maker fee default
    slip = cfg.get('slippage_pct', 0.01) / 100
    total_cost = 2 * (fee + slip)
    tp_mult = cfg['tp_mult']
    sl_mult = cfg['sl_mult']
    max_hold = cfg['max_hold']
    exit_mode = cfg.get('exit_mode', 'fixed')
    trail_mult = cfg.get('trail_mult', 1.0)
    be_trigger = cfg.get('be_trigger', 1.0)
    partial_at = cfg.get('partial_at', 1.0)
    partial_frac = cfg.get('partial_frac', 0.5)
    hour_filter = cfg.get('hour_filter', None)
    direction = cfg.get('direction', 'both')
    funding = cfg.get('funding_rates', None)
    cooldown = cfg.get('cooldown', 0)
    confirmation = cfg.get('confirmation', False)
    max_dd_pct = cfg.get('max_drawdown_pct', 0.10)  # 10% max DD for prop firm

    # Build signal masks
    signal_mode = cfg.get('signal_mode', 'percentile')
    signal_pct = cfg.get('signal_pct', 0.05)
    signal_thr = cfg.get('signal_thr', 0.15)
    regime = cfg.get('regime_mask', np.ones(n, dtype=bool))

    if signal_mode == 'percentile':
        lp_valid = long_probs[~np.isnan(long_probs)]
        sp_valid = short_probs[~np.isnan(short_probs)]
        long_thr = np.percentile(lp_valid, 100 * (1 - signal_pct)) if len(lp_valid) > 0 else 1.0
        short_thr = np.percentile(sp_valid, 100 * (1 - signal_pct)) if len(sp_valid) > 0 else 1.0
    else:
        long_thr = signal_thr
        short_thr = signal_thr

    long_sig = (~np.isnan(long_probs)) & (long_probs >= long_thr) & regime
    short_sig = (~np.isnan(short_probs)) & (short_probs >= short_thr) & regime

    if confirmation:
        sp_med = np.nanmedian(short_probs)
        lp_med = np.nanmedian(long_probs)
        long_sig = long_sig & (~np.isnan(short_probs)) & (short_probs < sp_med)
        short_sig = short_sig & (~np.isnan(long_probs)) & (long_probs < lp_med)

    if hour_filter is not None:
        hour_ok = np.isin(hours, list(hour_filter))
        long_sig = long_sig & hour_ok
        short_sig = short_sig & hour_ok

    if direction == 'long':
        short_sig[:] = False
    elif direction == 'short':
        long_sig[:] = False

    # Resolve conflicts
    both = long_sig & short_sig
    for idx in np.where(both)[0]:
        if long_probs[idx] >= short_probs[idx]:
            short_sig[idx] = False
        else:
            long_sig[idx] = False

    # ---- Simulation ----
    in_pos = False
    trades = []
    last_exit_bar = -cooldown - 1
    dd_breached = False

    for i in range(n):
        # Check drawdown limit
        if cap < peak_cap * (1 - max_dd_pct):
            dd_breached = True

        if dd_breached:
            break

        if in_pos:
            bars_held = i - entry_bar
            exit_price = None
            reason = None

            # Accumulate funding
            if funding is not None and i < len(funding) and not np.isnan(funding[i]):
                if pos_side == 'LONG':
                    fund_pnl -= funding[i] * remaining * pos_size
                else:
                    fund_pnl += funding[i] * remaining * pos_size

            if pos_side == 'LONG':
                # === CONSERVATIVE ORDER: check worst first ===
                # LONG worst case: price goes DOWN first (check SL), then UP

                # Step 1: Check SL using LOW (worst for longs)
                if low[i] <= current_sl:
                    exit_price = current_sl
                    reason = 'SL'
                # Step 2: Only if not stopped out, check TP
                elif high[i] >= tp_price:
                    exit_price = tp_price
                    reason = 'TP'
                # Step 3: Timeout
                elif bars_held >= max_hold:
                    exit_price = close[i]
                    reason = 'TIMEOUT'

                # Step 4: AFTER checking exits, update trailing stop
                # (only if still in position)
                if exit_price is None:
                    if exit_mode == 'trailing':
                        new_sl = high[i] - trail_mult * atr_entry
                        current_sl = max(current_sl, new_sl)
                    if exit_mode in ('breakeven', 'partial'):
                        if not be_active and high[i] >= entry_p + be_trigger * atr_entry:
                            current_sl = max(current_sl, entry_p)
                            be_active = True
                    if exit_mode == 'partial' and not partial_done:
                        if high[i] >= entry_p + partial_at * atr_entry:
                            p_exit = entry_p + partial_at * atr_entry
                            p_ret = (p_exit - entry_p) / entry_p
                            partial_pnl += partial_frac * p_ret * pos_size
                            remaining = 1.0 - partial_frac
                            partial_done = True
                            current_sl = max(current_sl, entry_p)

                if exit_price is not None:
                    gross_ret = (exit_price - entry_p) / entry_p
                    net_ret_main = gross_ret * remaining * pos_size
                    total_pnl = net_ret_main + partial_pnl + fund_pnl - total_cost * pos_size
                    cap += total_pnl
                    peak_cap = max(peak_cap, cap)

            else:  # SHORT
                # === CONSERVATIVE ORDER: check worst first ===
                # SHORT worst case: price goes UP first (check SL), then DOWN

                # Step 1: Check SL using HIGH (worst for shorts)
                if high[i] >= current_sl:
                    exit_price = current_sl
                    reason = 'SL'
                # Step 2: Only if not stopped out, check TP
                elif low[i] <= tp_price:
                    exit_price = tp_price
                    reason = 'TP'
                # Step 3: Timeout
                elif bars_held >= max_hold:
                    exit_price = close[i]
                    reason = 'TIMEOUT'

                # Step 4: AFTER exit checks, update trailing stop
                if exit_price is None:
                    if exit_mode == 'trailing':
                        new_sl = low[i] + trail_mult * atr_entry
                        current_sl = min(current_sl, new_sl)
                    if exit_mode in ('breakeven', 'partial'):
                        if not be_active and low[i] <= entry_p - be_trigger * atr_entry:
                            current_sl = min(current_sl, entry_p)
                            be_active = True
                    if exit_mode == 'partial' and not partial_done:
                        if low[i] <= entry_p - partial_at * atr_entry:
                            p_exit = entry_p - partial_at * atr_entry
                            p_ret = (entry_p - p_exit) / entry_p
                            partial_pnl += partial_frac * p_ret * pos_size
                            remaining = 1.0 - partial_frac
                            partial_done = True
                            current_sl = min(current_sl, entry_p)

                if exit_price is not None:
                    gross_ret = (entry_p - exit_price) / entry_p
                    net_ret_main = gross_ret * remaining * pos_size
                    total_pnl = net_ret_main + partial_pnl + fund_pnl - total_cost * pos_size
                    cap += total_pnl
                    peak_cap = max(peak_cap, cap)

            if exit_price is not None:
                trades.append({
                    'side': pos_side, 'reason': reason,
                    'bars_held': bars_held,
                    'pnl': total_pnl, 'capital': cap,
                    'gross_ret': gross_ret,
                })
                in_pos = False
                last_exit_bar = i

        # Try entering
        if not in_pos and cap > 100 and not dd_breached and i - last_exit_bar > cooldown:
            enter_side = None
            if long_sig[i] and atr[i] > 0:
                enter_side = 'LONG'
            elif short_sig[i] and atr[i] > 0:
                enter_side = 'SHORT'

            if enter_side is not None:
                atr_entry = atr[i]
                if enter_side == 'LONG':
                    entry_p = close[i] * (1 + slip)
                    tp_price = entry_p + tp_mult * atr_entry
                    sl_p = entry_p - sl_mult * atr_entry
                else:
                    entry_p = close[i] * (1 - slip)
                    tp_price = entry_p - tp_mult * atr_entry
                    sl_p = entry_p + sl_mult * atr_entry

                sl_dist_pct = abs(entry_p - sl_p) / entry_p
                pos_size = min(cap * risk_pct / max(sl_dist_pct, 1e-6),
                               cap * max_pos)
                if pos_size >= 50:
                    in_pos = True
                    pos_side = enter_side
                    entry_bar = i
                    current_sl = sl_p
                    be_active = False
                    partial_done = False
                    partial_pnl = 0.0
                    fund_pnl = 0.0
                    remaining = 1.0

    return trades


def calc_stats(trades, label="", initial_capital=25000):
    """Calculate portfolio statistics from trade list."""
    if not trades:
        return None
    df_t = pd.DataFrame(trades)
    n = len(df_t)
    if n < 3:
        return None

    wins = df_t['pnl'] > 0
    wr = wins.mean()
    gp = df_t.loc[wins, 'pnl'].sum()
    gl = abs(df_t.loc[~wins, 'pnl'].sum())
    pf = gp / gl if gl > 0 else 99.0
    final = df_t['capital'].iloc[-1]
    tr = (final - initial_capital) / initial_capital * 100

    eq = np.array([initial_capital] + df_t['capital'].tolist())
    pk = np.maximum.accumulate(eq)
    dd = (pk - eq) / (pk + 1e-10)
    max_dd = dd.max() * 100

    rets = df_t['pnl'].values / initial_capital
    sharpe = np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(365 * 2) if len(rets) > 1 else 0

    # Avg trade PnL in dollars
    avg_pnl = df_t['pnl'].mean()
    avg_win = df_t.loc[wins, 'pnl'].mean() if wins.sum() > 0 else 0
    avg_loss = df_t.loc[~wins, 'pnl'].mean() if (~wins).sum() > 0 else 0

    n_l = (df_t['side'] == 'LONG').sum()
    n_s = (df_t['side'] == 'SHORT').sum()
    tp_n = (df_t['reason'] == 'TP').sum()
    sl_n = (df_t['reason'] == 'SL').sum()

    return {
        'label': label, 'n_trades': n, 'wr': wr, 'pf': pf,
        'ret': tr, 'max_dd': max_dd, 'sharpe': sharpe,
        'final_cap': final,
        'avg_pnl': avg_pnl, 'avg_win': avg_win, 'avg_loss': avg_loss,
        'n_long': n_l, 'n_short': n_s,
        'tp': tp_n, 'sl': sl_n, 'timeout': n - tp_n - sl_n,
    }


def run_single_experiment(close, high, low, atr, hours,
                          long_probs, short_probs,
                          regime_mask, funding_rates, cfg, label):
    """Run one experiment and return stats."""
    cfg['regime_mask'] = regime_mask
    cfg['funding_rates'] = funding_rates
    trades = simulate_portfolio_v6(close, high, low, atr, hours,
                                   long_probs, short_probs, cfg)
    return calc_stats(trades, label, cfg.get('initial_capital', 25000))


# ======================================================================
# 10. BACKTEST SWEEP
# ======================================================================

def sweep_strategies(close, high, low, atr, hours,
                     long_probs, short_probs,
                     regime_mask, funding_rates,
                     model_label="model"):
    """Sweep strategy parameters for a given set of predictions.
    Optimized: reduced from 2,016 to 480 combos (removed redundant configs)."""
    results = []

    barrier_configs = [
        (1.5, 1.0, "TP1.5_SL1.0"),
        (2.0, 1.0, "TP2.0_SL1.0"),
        (2.0, 1.5, "TP2.0_SL1.5"),
        (2.5, 1.5, "TP2.5_SL1.5"),
        (3.0, 1.5, "TP3.0_SL1.5"),
    ]

    signal_configs = [
        (0.05, 'top5%'),
        (0.10, 'top10%'),
        (0.20, 'top20%'),
    ]

    fee_configs = [
        (0.02, 0.01, 'maker'),
        (0.01, 0.005, 'vip'),
    ]

    exit_configs = [
        ('fixed', 1.0, 'fixed'),
        ('trailing', 0.75, 'trail0.75'),
        ('trailing', 1.0, 'trail1.0'),
        ('breakeven', 1.0, 'be1.0'),
    ]

    dir_configs = ['both', 'short']

    hold_configs = [12, 18]
    cooldown_configs = [0]

    total = (len(barrier_configs) * len(signal_configs) * len(dir_configs) *
             len(fee_configs) * len(exit_configs) * len(hold_configs) *
             len(cooldown_configs))
    done = 0
    t0 = time.time()
    print(f"      Sweeping {total} strategy combinations...", flush=True)

    for (tp, sl, b_lbl) in barrier_configs:
        for (sig_pct, sig_lbl) in signal_configs:
            for direction in dir_configs:
                for (fee_p, slip_p, fee_lbl) in fee_configs:
                    for (exit_mode, trail_m, exit_lbl) in exit_configs:
                        for hold in hold_configs:
                            for cd in cooldown_configs:
                                cfg = {
                                    'tp_mult': tp, 'sl_mult': sl, 'max_hold': hold,
                                    'exit_mode': exit_mode,
                                    'trail_mult': trail_m, 'be_trigger': 1.0,
                                    'fee_pct': fee_p, 'slippage_pct': slip_p,
                                    'signal_mode': 'percentile', 'signal_pct': sig_pct,
                                    'direction': direction,
                                    'hour_filter': None,
                                    'initial_capital': 25000,
                                    'risk_per_trade': 0.01,
                                    'max_pos_frac': 0.25,
                                    'cooldown': cd,
                                    'max_drawdown_pct': 0.10,
                                }
                                label = (f"{model_label}|{b_lbl}|{sig_lbl}|{direction}"
                                         f"|{fee_lbl}|{exit_lbl}|H{hold}|CD{cd}")
                                s = run_single_experiment(
                                    close, high, low, atr, hours,
                                    long_probs, short_probs,
                                    regime_mask, funding_rates, cfg, label)
                                if s:
                                    results.append(s)
                                done += 1

    print(f"      Sweep done: {done} configs ({time.time()-t0:.0f}s), {len(results)} valid, "
          f"{sum(1 for r in results if r['pf'] > 1.0)} profitable")
    return results


# ======================================================================
# 11. SIGNAL FROM REGRESSION PREDICTIONS
# ======================================================================

def regression_to_signals(pred_returns, threshold_pct=0.005):
    """Convert return predictions to long/short probability-like signals."""
    long_probs = np.full_like(pred_returns, np.nan)
    short_probs = np.full_like(pred_returns, np.nan)

    valid = ~np.isnan(pred_returns)
    # Normalise to 0-1 range
    v = pred_returns[valid]
    # Long signal strength: how positive the predicted return is
    long_probs[valid] = 1.0 / (1.0 + np.exp(-v * 200))  # sigmoid scaling
    # Short signal: invert
    short_probs[valid] = 1.0 / (1.0 + np.exp(v * 200))

    return long_probs, short_probs


# ======================================================================
# 12. RESULTS PRINTING & SAVING
# ======================================================================

def print_top_results(results, title="RESULTS", top_n=30):
    """Print ranked results table."""
    profitable = sorted([r for r in results if r and r['pf'] > 1.0],
                        key=lambda x: x['pf'], reverse=True)

    print(f"\n{'='*130}")
    print(f"  {title}")
    print(f"  Total experiments: {len(results):,} | Profitable: {len(profitable)}")
    print(f"{'='*130}")

    show = profitable[:top_n] if profitable else sorted(
        [r for r in results if r], key=lambda x: x['pf'], reverse=True)[:top_n]

    if not show:
        print("  No results to show.")
        return profitable

    print(f"\n  {'#':<4s} {'Strategy':<70s} {'Trades':>6s} {'WR':>6s} "
          f"{'PF':>7s} {'Ret%':>8s} {'DD%':>6s} {'Sharpe':>7s} {'AvgPnL':>8s}")
    print(f"  {'-'*128}")

    for i, s in enumerate(show):
        wr_str = f"{s['wr']:.1%}"
        pf_str = f"{s['pf']:.2f}" if s['pf'] < 100 else f"{s['pf']:.0f}"
        tag = "✅" if s['pf'] > 1.0 else "❌"
        print(f"  {i+1:<4d} {s['label']:<70s} {s['n_trades']:>6d} "
              f"{wr_str:>6s} {pf_str:>7s} {s['ret']:>+8.1f} {s['max_dd']:>6.1f} "
              f"{s['sharpe']:>7.2f} ${s['avg_pnl']:>7.2f} {tag}")

    # Pattern analysis
    if profitable:
        from collections import Counter
        print(f"\n  --- Pattern Analysis (profitable only) ---")
        exits = Counter()
        dirs = Counter()
        fees = Counter()
        sigs = Counter()
        for r in profitable:
            parts = r['label'].split('|')
            for p in parts:
                if p.startswith('trail') or p in ('fixed', 'be1.0'):
                    exits[p] += 1
                elif p in ('short', 'long', 'both'):
                    dirs[p] += 1
                elif p in ('taker', 'maker', 'vip'):
                    fees[p] += 1
                elif p.startswith('top'):
                    sigs[p] += 1
        print(f"  Exit modes: {dict(exits)}")
        print(f"  Directions: {dict(dirs)}")
        print(f"  Fee tiers:  {dict(fees)}")
        print(f"  Signals:    {dict(sigs)}")

    return profitable


# ======================================================================
# MAIN EXPERIMENT
# ======================================================================

def main():
    t_start = time.time()
    print("="*80)
    print("  V6 DEEP LEARNING + ENSEMBLE MEGA-EXPERIMENT")
    print(f"  Started: {datetime.now()}")
    print(f"  Account: $25,000 prop firm | Max DD: 10%")
    print("="*80)

    # ---- Load data ----
    print("\n[1] Loading and preparing data...")
    df = load_and_prepare_data()
    feature_cols = get_feature_cols(df)
    print(f"    Features: {len(feature_cols)}")

    # ---- Generate labels ----
    print("\n[2] Generating triple-barrier labels (TP2.5 SL1.5 H12)...")
    from features.triple_barrier import triple_barrier_labels

    tb_long = triple_barrier_labels(df['close'], df['high'], df['low'],
                                    tp_mult=2.5, sl_mult=1.5, max_holding=12,
                                    direction='long')
    tb_short = triple_barrier_labels(df['close'], df['high'], df['low'],
                                     tp_mult=2.5, sl_mult=1.5, max_holding=12,
                                     direction='short')

    df['y_long'] = tb_long['tb_binary']
    df['y_short'] = tb_short['short_tb_binary']

    # Clean
    mask = df['y_long'].notna() & df['y_short'].notna()
    mask = mask & df[feature_cols].notna().all(axis=1)
    df_clean = df[mask].copy()
    print(f"    Clean samples: {len(df_clean):,}")
    print(f"    LONG positive rate: {df_clean['y_long'].mean():.1%}")
    print(f"    SHORT positive rate: {df_clean['y_short'].mean():.1%}")

    # Prepare arrays for backtesting
    close_v = df_clean['close'].values
    high_v = df_clean['high'].values
    low_v = df_clean['low'].values
    atr_v = df_clean['atr_14'].values if 'atr_14' in df_clean.columns else np.ones(len(df_clean))
    hours_v = df_clean['_hour'].values
    regime_v = df_clean['regime_ok'].values
    fund_v = df_clean['funding_rate'].values if 'funding_rate' in df_clean.columns else None

    splits = purged_wf_splits(len(df_clean), n_splits=5, test_pct=0.12, purge=14)

    all_results = []

    # ================================================================
    # EXPERIMENT A: Tree-only baseline (V4 equivalent)
    # ================================================================
    print("\n" + "="*80)
    print("  EXPERIMENT A: TREE BASELINE (XGB+LGB)")
    print("="*80, flush=True)

    tree_long, tree_l_auc = walk_forward_tree(
        df_clean, df_clean['y_long'].astype(int), feature_cols, splits, "LONG")
    tree_short, tree_s_auc = walk_forward_tree(
        df_clean, df_clean['y_short'].astype(int), feature_cols, splits, "SHORT")

    print(f"\n   Sweeping strategies for tree baseline...")
    tree_results = sweep_strategies(
        close_v, high_v, low_v, atr_v, hours_v,
        tree_long, tree_short, regime_v, fund_v, "tree")

    tree_profitable = print_top_results(tree_results,
                                        "TREE BASELINE RESULTS (fixed trailing stop)")
    all_results.extend(tree_results)

    # ================================================================
    # EXPERIMENT B: LSTM
    # ================================================================
    print("\n" + "="*80)
    print("  EXPERIMENT B: LSTM")
    print("="*80, flush=True)

    lstm_long, lstm_l_auc = walk_forward_dl(
        df_clean, df_clean['y_long'].astype(int), feature_cols, splits,
        LSTMModel, "LONG", seq_len=24,
        hidden_dim=48, n_layers=2, dropout=0.3)

    lstm_short, lstm_s_auc = walk_forward_dl(
        df_clean, df_clean['y_short'].astype(int), feature_cols, splits,
        LSTMModel, "SHORT", seq_len=24,
        hidden_dim=48, n_layers=2, dropout=0.3)

    print(f"\n   Sweeping strategies for LSTM...")
    lstm_results = sweep_strategies(
        close_v, high_v, low_v, atr_v, hours_v,
        lstm_long, lstm_short, regime_v, fund_v, "lstm")

    lstm_profitable = print_top_results(lstm_results, "LSTM RESULTS")
    all_results.extend(lstm_results)

    # ================================================================
    # EXPERIMENT C: TCN
    # ================================================================
    print("\n" + "="*80)
    print("  EXPERIMENT C: TEMPORAL CONVOLUTIONAL NETWORK")
    print("="*80, flush=True)

    tcn_long, tcn_l_auc = walk_forward_dl(
        df_clean, df_clean['y_long'].astype(int), feature_cols, splits,
        TCNModel, "LONG", seq_len=24,
        hidden_dim=48, n_layers=4, dropout=0.2)

    tcn_short, tcn_s_auc = walk_forward_dl(
        df_clean, df_clean['y_short'].astype(int), feature_cols, splits,
        TCNModel, "SHORT", seq_len=24,
        hidden_dim=48, n_layers=4, dropout=0.2)

    print(f"\n   Sweeping strategies for TCN...")
    tcn_results = sweep_strategies(
        close_v, high_v, low_v, atr_v, hours_v,
        tcn_long, tcn_short, regime_v, fund_v, "tcn")

    tcn_profitable = print_top_results(tcn_results, "TCN RESULTS")
    all_results.extend(tcn_results)

    # ================================================================
    # EXPERIMENT D: TRANSFORMER
    # ================================================================
    print("\n" + "="*80)
    print("  EXPERIMENT D: TRANSFORMER")
    print("="*80, flush=True)

    tf_long, tf_l_auc = walk_forward_dl(
        df_clean, df_clean['y_long'].astype(int), feature_cols, splits,
        TransformerModel, "LONG", seq_len=24,
        d_model=48, nhead=4, n_layers=2, dropout=0.2)

    tf_short, tf_s_auc = walk_forward_dl(
        df_clean, df_clean['y_short'].astype(int), feature_cols, splits,
        TransformerModel, "SHORT", seq_len=24,
        d_model=48, nhead=4, n_layers=2, dropout=0.2)

    print(f"\n   Sweeping strategies for Transformer...")
    tf_results = sweep_strategies(
        close_v, high_v, low_v, atr_v, hours_v,
        tf_long, tf_short, regime_v, fund_v, "transformer")

    tf_profitable = print_top_results(tf_results, "TRANSFORMER RESULTS")
    all_results.extend(tf_results)

    # ================================================================
    # EXPERIMENT E: STACKING ENSEMBLE
    # ================================================================
    print("\n" + "="*80)
    print("  EXPERIMENT E: STACKING ENSEMBLE (reusing predictions)")
    print("="*80, flush=True)

    # Reuse already-computed predictions from experiments A-D
    precomputed_long = {
        'tree': (tree_long, tree_l_auc),
        'lstm': (lstm_long, lstm_l_auc),
        'tcn': (tcn_long, tcn_l_auc),
        'transformer': (tf_long, tf_l_auc),
    }
    precomputed_short = {
        'tree': (tree_short, tree_s_auc),
        'lstm': (lstm_short, lstm_s_auc),
        'tcn': (tcn_short, tcn_s_auc),
        'transformer': (tf_short, tf_s_auc),
    }

    # Stack long models
    stacked_long, long_scores, stack_l_auc = build_stacking_ensemble(
        df_clean, df_clean['y_long'].astype(int), feature_cols, splits,
        "LONG", seq_len=24, precomputed=precomputed_long)

    # Stack short models
    stacked_short, short_scores, stack_s_auc = build_stacking_ensemble(
        df_clean, df_clean['y_short'].astype(int), feature_cols, splits,
        "SHORT", seq_len=24, precomputed=precomputed_short)

    print(f"\n   Sweeping strategies for stacking ensemble...")
    stack_results = sweep_strategies(
        close_v, high_v, low_v, atr_v, hours_v,
        stacked_long, stacked_short, regime_v, fund_v, "stack")

    stack_profitable = print_top_results(stack_results, "STACKING ENSEMBLE RESULTS")
    all_results.extend(stack_results)

    # ================================================================
    # EXPERIMENT F: SIMPLE MODEL AVERAGING (no meta-learner)
    # ================================================================
    print("\n" + "="*80)
    print("  EXPERIMENT F: SIMPLE MODEL AVERAGE")
    print("="*80, flush=True)

    # Average all model predictions (tree, lstm, tcn, transformer)
    avg_long = np.nanmean([tree_long, lstm_long, tcn_long, tf_long], axis=0)
    avg_short = np.nanmean([tree_short, lstm_short, tcn_short, tf_short], axis=0)

    print(f"   Sweeping strategies for model average...")
    avg_results = sweep_strategies(
        close_v, high_v, low_v, atr_v, hours_v,
        avg_long, avg_short, regime_v, fund_v, "avg")

    avg_profitable = print_top_results(avg_results, "MODEL AVERAGE RESULTS")
    all_results.extend(avg_results)

    # ================================================================
    # EXPERIMENT G: TREE + BEST DL AVERAGE
    # ================================================================
    print("\n" + "="*80)
    print("  EXPERIMENT G: TREE + BEST DL")
    print("="*80, flush=True)

    # Find best individual DL model
    dl_scores = {'lstm': (lstm_l_auc + lstm_s_auc) / 2,
                 'tcn': (tcn_l_auc + tcn_s_auc) / 2,
                 'transformer': (tf_l_auc + tf_s_auc) / 2}
    best_dl = max(dl_scores, key=dl_scores.get)
    print(f"   Best DL model: {best_dl} (avg AUC={dl_scores[best_dl]:.4f})")

    dl_long_map = {'lstm': lstm_long, 'tcn': tcn_long, 'transformer': tf_long}
    dl_short_map = {'lstm': lstm_short, 'tcn': tcn_short, 'transformer': tf_short}

    # 70% tree + 30% best DL
    combo_long = 0.7 * np.nan_to_num(tree_long, nan=0.5) + 0.3 * np.nan_to_num(dl_long_map[best_dl], nan=0.5)
    combo_short = 0.7 * np.nan_to_num(tree_short, nan=0.5) + 0.3 * np.nan_to_num(dl_short_map[best_dl], nan=0.5)

    print(f"   Sweeping strategies for tree+{best_dl} combo...")
    combo_results = sweep_strategies(
        close_v, high_v, low_v, atr_v, hours_v,
        combo_long, combo_short, regime_v, fund_v, f"tree+{best_dl}")

    combo_profitable = print_top_results(combo_results, f"TREE + {best_dl.upper()} RESULTS")
    all_results.extend(combo_results)

    # ================================================================
    # EXPERIMENT H: REGRESSION-BASED SIGNALS
    # ================================================================
    print("\n" + "="*80)
    print("  EXPERIMENT H: REGRESSION (predict forward returns)")
    print("="*80, flush=True)

    for horizon in [8, 12]:
        print(f"\n   --- Horizon: {horizon}h ---")
        pred_returns = regression_experiment(
            df_clean, feature_cols, df_clean.copy(), splits, horizon=horizon)

        reg_long, reg_short = regression_to_signals(pred_returns)

        reg_results = sweep_strategies(
            close_v, high_v, low_v, atr_v, hours_v,
            reg_long, reg_short, regime_v, fund_v, f"reg_{horizon}h")

        reg_profitable = print_top_results(reg_results,
                                           f"REGRESSION {horizon}H RESULTS")
        all_results.extend(reg_results)

    # ================================================================
    # GRAND SUMMARY
    # ================================================================
    print("\n" + "="*130)
    print("  🏆 GRAND SUMMARY -- V6 DEEP LEARNING EXPERIMENT 🏆")
    print("="*130)

    all_profitable = print_top_results(all_results,
                                       "ALL STRATEGIES RANKED", top_n=50)

    # Model comparison
    print(f"\n{'='*80}")
    print(f"  MODEL COMPARISON (AUC)")
    print(f"{'='*80}")
    print(f"  {'Model':<20s} {'LONG AUC':>10s} {'SHORT AUC':>10s} {'Avg AUC':>10s}")
    print(f"  {'-'*55}")
    models = [
        ('Tree (XGB+LGB)', tree_l_auc, tree_s_auc),
        ('LSTM', lstm_l_auc, lstm_s_auc),
        ('TCN', tcn_l_auc, tcn_s_auc),
        ('Transformer', tf_l_auc, tf_s_auc),
        ('Stacking', stack_l_auc, stack_s_auc),
    ]
    for name, l_auc, s_auc in models:
        avg = (l_auc + s_auc) / 2
        print(f"  {name:<20s} {l_auc:>10.4f} {s_auc:>10.4f} {avg:>10.4f}")

    # Profitability by model
    print(f"\n{'='*80}")
    print(f"  PROFITABILITY BY MODEL TYPE")
    print(f"{'='*80}")
    model_tags = ['tree', 'lstm', 'tcn', 'transformer', 'stack', 'avg',
                  'tree+', 'reg_8h', 'reg_12h']
    for tag in model_tags:
        matching = [r for r in all_results if r and r['label'].startswith(tag)]
        prof = [r for r in matching if r['pf'] > 1.0]
        if matching:
            best_pf = max(r['pf'] for r in matching)
            print(f"  {tag:<15s}: {len(matching):>5d} experiments, "
                  f"{len(prof):>4d} profitable ({len(prof)/len(matching):.1%}), "
                  f"best PF={best_pf:.2f}")

    # $25K account results for best strategies
    if all_profitable:
        print(f"\n{'='*80}")
        print(f"  TOP STRATEGIES FOR $25K PROP FIRM ACCOUNT")
        print(f"{'='*80}")
        for i, s in enumerate(all_profitable[:10]):
            ann_ret = s['ret'] / 5  # ~5 years of data
            dollar_pnl = s['final_cap'] - 25000
            print(f"  {i+1}. {s['label']}")
            print(f"     Trades: {s['n_trades']} | WR: {s['wr']:.1%} | PF: {s['pf']:.2f}")
            print(f"     Total Return: {s['ret']:+.1f}% (${dollar_pnl:+,.0f})")
            print(f"     Annualized: ~{ann_ret:+.1f}%/yr | Max DD: {s['max_dd']:.1f}%")
            print(f"     Avg Trade: ${s['avg_pnl']:+.2f} | Avg Win: ${s['avg_win']:+.2f} | Avg Loss: ${s['avg_loss']:+.2f}")
            print(f"     Sharpe: {s['sharpe']:.2f} | L/S: {s['n_long']}/{s['n_short']}")
            print()

    # Save results
    duration = time.time() - t_start
    save_data = {
        'total_experiments': len(all_results),
        'profitable_count': len(all_profitable) if all_profitable else 0,
        'top_50': all_profitable[:50] if all_profitable else [],
        'model_aucs': {name: {'long': l, 'short': s} for name, l, s in models},
        'timestamp': str(datetime.now()),
        'duration_seconds': duration,
        'account_size': 25000,
        'trailing_stop_fix': 'conservative (worst-price-first)',
    }

    out_path = OUTPUT_DIR / "v6_dl_experiment_results.json"
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n   Results saved to {out_path}")

    print(f"\n   Total duration: {duration/60:.1f} minutes")
    print(f"   Experiments run: {len(all_results):,}")


if __name__ == "__main__":
    main()
