"""
Inference helper for the V2 alpha ensemble model.

Usage:
    from features.predict_v2 import load_model_v2, predict_v2

    model = load_model_v2()
    prob, calibrated_prob, confidence_tier = predict_v2(model, feature_df)
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any


def load_model_v2(model_path: str = None) -> Dict[str, Any]:
    """
    Load the V2 alpha ensemble model artifact.

    Returns a dict with keys:
        xgb_model, lgb_model, cat_model, calibrator, feature_cols
    """
    if model_path is None:
        model_path = Path(__file__).parent.parent.parent / "models" / "ensemble_v2_alpha.pkl"
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run scripts/train_v2_alpha.py first."
        )

    with open(model_path, 'rb') as f:
        artifact = pickle.load(f)

    print(f"✅ Loaded V2 model ({len(artifact['feature_cols'])} features)")
    return artifact


def predict_v2(
    artifact: Dict[str, Any],
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate predictions from a feature DataFrame.

    Parameters
    ----------
    artifact : dict from load_model_v2()
    df : pd.DataFrame with feature columns matching artifact['feature_cols']

    Returns
    -------
    raw_prob : np.ndarray  — raw ensemble probability (avg of 3 models)
    cal_prob : np.ndarray  — isotonic-calibrated probability
    tier     : np.ndarray  — confidence tier (0=low, 1=medium, 2=high)
    """
    feature_cols = artifact['feature_cols']

    # Check for missing features
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {len(missing)} features: {missing[:10]}...")

    X = df[feature_cols].values

    # 3-model ensemble average
    p_xgb = artifact['xgb_model'].predict_proba(X)[:, 1]
    p_lgb = artifact['lgb_model'].predict_proba(X)[:, 1]
    p_cat = artifact['cat_model'].predict_proba(X)[:, 1]
    raw_prob = (p_xgb + p_lgb + p_cat) / 3.0

    # Calibrate
    calibrator = artifact['calibrator']
    cal_prob = calibrator.predict(raw_prob)

    # Confidence tiers based on calibrated probability distance from 0.5
    distance = np.abs(cal_prob - 0.5)
    tier = np.zeros(len(cal_prob), dtype=int)
    tier[distance > 0.10] = 1   # medium confidence
    tier[distance > 0.20] = 2   # high confidence

    return raw_prob, cal_prob, tier


def get_signal(
    artifact: Dict[str, Any],
    df: pd.DataFrame,
    min_confidence_tier: int = 1,
) -> Dict[str, Any]:
    """
    Generate a trading signal from the latest row of a feature DataFrame.

    Returns
    -------
    dict with keys: action, side, confidence, raw_prob, cal_prob, tier, reasoning
    """
    raw_prob, cal_prob, tier = predict_v2(artifact, df.tail(1))

    raw_p = float(raw_prob[0])
    cal_p = float(cal_prob[0])
    t = int(tier[0])

    signal = {
        "action": "HOLD",
        "side": None,
        "confidence": cal_p,
        "raw_prob": raw_p,
        "cal_prob": cal_p,
        "tier": t,
        "reasoning": [],
    }

    if t < min_confidence_tier:
        signal["reasoning"].append(f"Low confidence (tier {t} < required {min_confidence_tier})")
        return signal

    if cal_p > 0.5:
        signal["action"] = "ENTER"
        signal["side"] = "Buy"
        signal["reasoning"].append(f"Bullish: P(TP hit)={cal_p:.3f}, tier={t}")
    else:
        signal["action"] = "ENTER"
        signal["side"] = "Sell"
        signal["reasoning"].append(f"Bearish: P(TP hit)={cal_p:.3f}, tier={t}")

    return signal
