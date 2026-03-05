#!/usr/bin/env python3
"""
CHAOS V2 Evaluation Module
===========================
Out-of-sample validation, cost-adjusted scoring, and model ranking.

Evaluates all completed V1.0 models by:
1. Loading each model checkpoint (.joblib or .pt)
2. Running inference on OOS data (2023-01-01 onwards)
3. Computing raw PF, cost-adjusted PF, overfit ratio
4. Sub-period consistency checks (walk-forward lite)
5. Outputting ranked scorecard CSVs

Run with: python chaos_v2_evaluation.py [--cost-profile retail|raw_vps|ibkr|all]
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import sys
import gc
import json
import time
import argparse
import joblib
import warnings
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict, Optional, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

# Detect environment
if os.path.exists('/content/drive'):
    BASE_DIR = Path("/content/drive/MyDrive/chaos_v1.0")
else:
    BASE_DIR = Path(r"G:\My Drive\chaos_v1.0")

FEATURES_DIR = BASE_DIR / "features"
MODELS_DIR = BASE_DIR / "models"
PARAMS_DIR = MODELS_DIR / "optuna_params"

# Output files
OOS_OUTPUT_FILE = MODELS_DIR / "oos_validation.csv"
RANKING_OUTPUT_FILE = MODELS_DIR / "model_ranking.csv"
EVAL_CHECKPOINT_FILE = MODELS_DIR / "eval_checkpoint.json"

# Column names — must match training scripts exactly
TARGET_COL = 'target_3class_8'
RETURNS_COL = 'target_return_8'
FEATURE_FILE_PATTERN = "{pair}_{tf}_features.parquet"

# Execution lag — must match training scripts
EXECUTION_LAG_BARS = 1

# Trade frequency minimum
MIN_TRADE_RATIO = 0.10

# OOS cutoff date
OOS_CUTOFF_DATE = '2023-01-01'

# Dead zone timeframes — auto-skip
SKIP_TIMEFRAMES = ['W1', 'MN1']

# Inference batch size for neural networks
VAL_INFERENCE_BATCH_SIZE = 2048

# Grid
ALL_PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD',
             'USDCHF', 'NZDUSD', 'EURJPY', 'GBPJPY']

ALL_TIMEFRAMES = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1', 'MN1']

ALL_BRAINS = [
    # Gradient Boosting (6)
    'lgb_optuna', 'xgb_optuna', 'cat_optuna',
    'lgb_v2_optuna', 'xgb_v2_optuna', 'cat_v2_optuna',
    # TabNet & MLP (2)
    'tabnet_optuna', 'mlp_optuna',
    # RNN/Transformer (3)
    'lstm_optuna', 'gru_optuna', 'transformer_optuna',
    # CNN-based (3)
    'cnn1d_optuna', 'tcn_optuna', 'wavenet_optuna',
    # Attention & Residual (3)
    'attention_net_optuna', 'residual_mlp_optuna', 'ensemble_nn_optuna',
    # Time Series Specialized (2)
    'nbeats_optuna', 'tft_optuna',
    # Tree Ensembles (2) — local track
    'rf_optuna', 'et_optuna',
]

# Neural network brain names (require .pt loading + architecture reconstruction)
NN_BRAINS = [
    'mlp_optuna', 'lstm_optuna', 'gru_optuna', 'transformer_optuna',
    'cnn1d_optuna', 'tcn_optuna', 'wavenet_optuna',
    'attention_net_optuna', 'residual_mlp_optuna', 'ensemble_nn_optuna',
    'nbeats_optuna', 'tft_optuna',
]

# Sub-period windows for consistency checks
SUB_PERIODS = [
    ('2023-01-01', '2023-06-30', 'H1_2023'),
    ('2023-07-01', '2023-12-31', 'H2_2023'),
    ('2024-01-01', '2024-06-30', 'H1_2024'),
    ('2024-07-01', '2024-12-31', 'H2_2024'),
    ('2025-01-01', '2025-06-30', 'H1_2025'),
    ('2025-07-01', '2025-12-31', 'H2_2025'),
]


# =============================================================================
# TRANSACTION COST TABLES
# =============================================================================

# Retail broker cost profile (original)
RETAIL_COSTS = {
    'EURUSD': {'spread_pips': 0.8, 'slippage_pips': 0.3, 'pip_value': 0.0001},
    'GBPUSD': {'spread_pips': 1.0, 'slippage_pips': 0.3, 'pip_value': 0.0001},
    'USDJPY': {'spread_pips': 0.8, 'slippage_pips': 0.3, 'pip_value': 0.01},
    'USDCHF': {'spread_pips': 1.2, 'slippage_pips': 0.4, 'pip_value': 0.0001},
    'USDCAD': {'spread_pips': 1.2, 'slippage_pips': 0.4, 'pip_value': 0.0001},
    'AUDUSD': {'spread_pips': 1.0, 'slippage_pips': 0.3, 'pip_value': 0.0001},
    'NZDUSD': {'spread_pips': 1.5, 'slippage_pips': 0.4, 'pip_value': 0.0001},
    'EURJPY': {'spread_pips': 1.2, 'slippage_pips': 0.4, 'pip_value': 0.01},
    'GBPJPY': {'spread_pips': 1.5, 'slippage_pips': 0.5, 'pip_value': 0.01},
}

# Raw Pricing VPS cost profile — Forex.com/GAIN Capital Raw Pricing account
# Commission: $7 per $100K per leg ($14 RT = 1.4 pips on standard pairs)
# Actual commission in pips: $14 / (100K * pip_value) but normalised to 0.7 pips
# per leg × 2 legs = 1.4 pips RT. We store 0.7 as the per-leg commission in pips
# but get_cost_per_trade sums spread + commission + slippage for total RT cost.
# Execution: London VPS, 3.5ms ping, ~0.03 pip slippage.
RAW_VPS_COSTS = {
    'EURUSD': {'spread_pips': 0.2, 'commission_pips': 0.7, 'slippage_pips': 0.03, 'pip_value': 0.0001},
    'GBPUSD': {'spread_pips': 0.4, 'commission_pips': 0.7, 'slippage_pips': 0.03, 'pip_value': 0.0001},
    'USDJPY': {'spread_pips': 0.2, 'commission_pips': 0.7, 'slippage_pips': 0.03, 'pip_value': 0.01},
    'USDCHF': {'spread_pips': 0.5, 'commission_pips': 0.7, 'slippage_pips': 0.03, 'pip_value': 0.0001},
    'USDCAD': {'spread_pips': 0.5, 'commission_pips': 0.7, 'slippage_pips': 0.03, 'pip_value': 0.0001},
    'AUDUSD': {'spread_pips': 0.3, 'commission_pips': 0.7, 'slippage_pips': 0.03, 'pip_value': 0.0001},
    'NZDUSD': {'spread_pips': 0.6, 'commission_pips': 0.7, 'slippage_pips': 0.03, 'pip_value': 0.0001},
    'EURJPY': {'spread_pips': 0.5, 'commission_pips': 0.7, 'slippage_pips': 0.03, 'pip_value': 0.01},
    'GBPJPY': {'spread_pips': 0.8, 'commission_pips': 0.7, 'slippage_pips': 0.03, 'pip_value': 0.01},
}
# Total RT cost (pips): EURUSD=0.93, GBPUSD=1.13, USDJPY=0.93, USDCHF=1.23,
# USDCAD=1.23, AUDUSD=1.03, NZDUSD=1.33, EURJPY=1.23, GBPJPY=1.53

# Interactive Brokers cost profile — IBKR Pro, true ECN/interbank
# Commission: 0.2 bps of trade value (~$2-4/side/lot) ≈ 0.2-0.4 pip equivalent
# Execution: Equinix NY4, ~0.02 pip slippage
# True interbank spreads, no dealing desk
IBKR_COSTS = {
    'EURUSD': {'spread_pips': 0.1, 'commission_pips': 0.4, 'slippage_pips': 0.02, 'pip_value': 0.0001},
    'GBPUSD': {'spread_pips': 0.2, 'commission_pips': 0.4, 'slippage_pips': 0.02, 'pip_value': 0.0001},
    'USDJPY': {'spread_pips': 0.2, 'commission_pips': 0.4, 'slippage_pips': 0.02, 'pip_value': 0.01},
    'USDCHF': {'spread_pips': 0.3, 'commission_pips': 0.4, 'slippage_pips': 0.02, 'pip_value': 0.0001},
    'USDCAD': {'spread_pips': 0.3, 'commission_pips': 0.4, 'slippage_pips': 0.02, 'pip_value': 0.0001},
    'AUDUSD': {'spread_pips': 0.2, 'commission_pips': 0.4, 'slippage_pips': 0.02, 'pip_value': 0.0001},
    'NZDUSD': {'spread_pips': 0.4, 'commission_pips': 0.4, 'slippage_pips': 0.02, 'pip_value': 0.0001},
    'EURJPY': {'spread_pips': 0.3, 'commission_pips': 0.4, 'slippage_pips': 0.02, 'pip_value': 0.01},
    'GBPJPY': {'spread_pips': 0.5, 'commission_pips': 0.4, 'slippage_pips': 0.02, 'pip_value': 0.01},
}
# Total RT cost (pips): EURUSD=0.52, GBPUSD=0.62, USDJPY=0.62, USDCHF=0.72,
# USDCAD=0.72, AUDUSD=0.62, NZDUSD=0.82, EURJPY=0.72, GBPJPY=0.92

COST_PROFILES = {
    'retail': RETAIL_COSTS,
    'raw_vps': RAW_VPS_COSTS,
    'ibkr': IBKR_COSTS,
}

# Backward-compatible alias — used by existing code paths
COST_TABLE = RETAIL_COSTS


def get_cost_per_trade(pair: str, cost_profile: str = 'retail') -> float:
    """Get round-trip cost in decimal return units for a pair.

    For retail: spread + slippage (no explicit commission).
    For raw_vps/ibkr: spread + commission + slippage.
    """
    table = COST_PROFILES[cost_profile]
    cost = table[pair]
    total_pips = cost['spread_pips'] + cost.get('commission_pips', 0.0) + cost['slippage_pips']
    return total_pips * cost['pip_value']


# =============================================================================
# IMPORT NN ARCHITECTURE CLASSES FROM TRAINING SCRIPT
# =============================================================================
sys.path.insert(0, str(BASE_DIR))

try:
    from chaos_gpu_training import (
        MLPClassifier, LSTMClassifier, GRUClassifier, TransformerClassifier,
        CNN1DClassifier, TCNClassifier, WaveNetClassifier,
        AttentionNetClassifier, ResidualMLPClassifier, EnsembleNNClassifier,
        NBeatsClassifier, TemporalFusionTransformer,
        get_feature_columns,
    )
    NN_CLASSES_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Cannot import NN classes from chaos_gpu_training: {e}")
    print("  Neural network models (.pt) will be skipped.")
    NN_CLASSES_AVAILABLE = False

    def get_feature_columns(df):
        """Fallback if import fails."""
        exclude_patterns = ['target_', 'return', 'Open', 'High', 'Low', 'Close',
                            'Volume', 'timestamp', 'date', 'pair', 'symbol', 'tf',
                            'bar_time', 'Bid_', 'Ask_', 'Spread_', 'Unnamed']
        feature_cols = []
        for col in df.columns:
            if any(p in col for p in exclude_patterns):
                continue
            if df[col].dtype not in ['int64', 'int32', 'float64', 'float32', 'Float64']:
                continue
            feature_cols.append(col)
        return feature_cols


# =============================================================================
# NN MODEL RECONSTRUCTION
# =============================================================================
def reconstruct_nn_model(brain: str, n_features: int, hidden_size: int,
                         dropout: float) -> nn.Module:
    """
    Reconstruct NN architecture from brain name and hyperparameters.
    Mirrors chaos_gpu_training.py lines 1566-1608 exactly.
    """
    if not NN_CLASSES_AVAILABLE:
        raise RuntimeError("NN classes not available — cannot reconstruct model")

    if brain == 'mlp_optuna':
        return MLPClassifier(n_features, 3,
                             [hidden_size, hidden_size // 2, hidden_size // 4],
                             dropout)
    elif brain == 'lstm_optuna':
        return LSTMClassifier(n_features, 3, hidden_size, 2, dropout)
    elif brain == 'gru_optuna':
        return GRUClassifier(n_features, 3, hidden_size, 2, dropout)
    elif brain == 'transformer_optuna':
        return TransformerClassifier(n_features, 3, hidden_size, 4, 2, dropout)
    elif brain == 'cnn1d_optuna':
        return CNN1DClassifier(n_features, 3, [32, 64, 64], 3, dropout)
    elif brain == 'tcn_optuna':
        return TCNClassifier(n_features, 3, [32, 64], 3, dropout)
    elif brain == 'wavenet_optuna':
        return WaveNetClassifier(n_features, 3, 64, 3, 2, dropout)
    elif brain == 'attention_net_optuna':
        return AttentionNetClassifier(n_features, 3, hidden_size, 4, dropout)
    elif brain == 'residual_mlp_optuna':
        return ResidualMLPClassifier(n_features, 3, hidden_size, 3, dropout)
    elif brain == 'ensemble_nn_optuna':
        return EnsembleNNClassifier(n_features, 3, 64, dropout)
    elif brain == 'nbeats_optuna':
        return NBeatsClassifier(n_features, 3, hidden_size, 4, dropout)
    elif brain == 'tft_optuna':
        return TemporalFusionTransformer(n_features, 3, hidden_size, 4, 2, dropout)
    else:
        raise ValueError(f"Unknown NN brain: {brain}")


# =============================================================================
# FEATURE LOADING — DATE-BASED OOS SPLIT
# =============================================================================
def load_and_split_features(pair: str, tf: str) -> Dict[str, Any]:
    """
    Load feature parquet and split by date for OOS evaluation.

    Returns dict with X_oos, y_oos, returns_oos, oos_index, n_features.
    """
    filepath = FEATURES_DIR / FEATURE_FILE_PATTERN.format(pair=pair, tf=tf)

    if not filepath.exists():
        raise FileNotFoundError(f"Feature file not found: {filepath}")

    df = pd.read_parquet(filepath)

    # Validate columns
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in {filepath}")
    if RETURNS_COL not in df.columns:
        raise ValueError(f"Returns column '{RETURNS_COL}' not found in {filepath}")

    # Apply execution lag to returns
    df['returns_lagged'] = df[RETURNS_COL].shift(-EXECUTION_LAG_BARS)
    df = df.dropna(subset=['returns_lagged'])

    feature_cols = get_feature_columns(df)

    # Prepare arrays
    X = df[feature_cols].values.astype('float32')
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    y = df[TARGET_COL].values.copy()
    y = np.nan_to_num(y, nan=0.0)
    if y.min() < 0:
        y = y + 1
    y = y.astype('int64')

    returns = df['returns_lagged'].values.copy()
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

    # Date-based OOS split
    oos_mask = df.index >= OOS_CUTOFF_DATE
    is_mask = ~oos_mask

    n_is = int(is_mask.sum())
    n_oos = int(oos_mask.sum())

    return {
        'X_is': X[is_mask],
        'y_is': y[is_mask],
        'returns_is': returns[is_mask],
        'X_oos': X[oos_mask],
        'y_oos': y[oos_mask],
        'returns_oos': returns[oos_mask],
        'oos_index': df.index[oos_mask],
        'feature_cols': feature_cols,
        'n_features': len(feature_cols),
        'n_is': n_is,
        'n_oos': n_oos,
    }


# =============================================================================
# MODEL LOADING
# =============================================================================
def load_joblib_model(filepath: Path) -> Tuple[Any, Optional[StandardScaler], Optional[float]]:
    """Load a .joblib model (RF, ET, LGB, XGB, CatBoost, TabNet)."""
    checkpoint = joblib.load(filepath)
    model = checkpoint['model']
    scaler = checkpoint.get('scaler', None)
    train_pf = checkpoint.get('best_pf', None)
    return model, scaler, train_pf


def load_pt_model(filepath: Path, brain: str, n_features: int,
                  ) -> Tuple[nn.Module, Optional[StandardScaler], Optional[float]]:
    """Load a .pt model by reconstructing architecture and loading state_dict."""
    checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)

    scaler = checkpoint.get('scaler', None)
    train_pf = checkpoint.get('best_pf', None)
    pair = checkpoint.get('pair', '')
    tf = checkpoint.get('tf', '')

    # Load Optuna params for hidden_size and dropout
    params_file = PARAMS_DIR / f"{pair}_{tf}_{brain}_params.json"
    if params_file.exists():
        with open(params_file) as f:
            params = json.load(f)
        hidden_size = params['best_params'].get('hidden_size', 128)
        dropout = params['best_params'].get('dropout', 0.3)
    else:
        # Fallback defaults
        hidden_size = 128
        dropout = 0.3

    # Reconstruct architecture and load weights
    model = reconstruct_nn_model(brain, n_features, hidden_size, dropout)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, scaler, train_pf


# =============================================================================
# PREDICTION PIPELINE
# =============================================================================
def predict_joblib(model, scaler: Optional[StandardScaler],
                   X: np.ndarray) -> np.ndarray:
    """Get predictions from a joblib model."""
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X
    return model.predict(X_scaled)


def batched_nn_inference(model: nn.Module, X_tensor: torch.Tensor,
                         batch_size: int = VAL_INFERENCE_BATCH_SIZE) -> torch.Tensor:
    """Batched inference to prevent OOM on large datasets."""
    model.eval()
    all_outputs = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i + batch_size].to(next(model.parameters()).device)
            outputs = model(batch)
            all_outputs.append(outputs.cpu())
    return torch.cat(all_outputs, dim=0)


def predict_pytorch(model: nn.Module, scaler: Optional[StandardScaler],
                    X: np.ndarray, device: torch.device) -> np.ndarray:
    """Get predictions from a PyTorch model."""
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X

    model = model.to(device)
    model.eval()

    X_tensor = torch.FloatTensor(X_scaled)
    outputs = batched_nn_inference(model, X_tensor, VAL_INFERENCE_BATCH_SIZE)
    _, predictions = torch.max(outputs, 1)

    model.cpu()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return predictions.numpy()


# =============================================================================
# POSITION CONVERSION — matches training script exactly
# =============================================================================
def convert_to_positions(predictions: np.ndarray) -> np.ndarray:
    """Convert predictions [0,1,2] to positions [-1,0,+1]."""
    predictions = np.asarray(predictions).flatten()
    if len(predictions) == 0:
        return np.array([], dtype=np.float64)
    return predictions.astype(np.float64) - 1.0


# =============================================================================
# PROFIT FACTOR — RAW (matches V1 training exactly)
# =============================================================================
def calculate_raw_pf(predictions: np.ndarray, returns: np.ndarray) -> Tuple[float, Dict]:
    """
    Calculate raw profit factor — same formula as V1 training.
    Returns (pf, stats_dict).
    """
    predictions = np.asarray(predictions).flatten()
    returns = np.asarray(returns).flatten()

    if len(predictions) == 0:
        return 1.0, {'trade_ratio': 0.0, 'n_trades': 0, 'n_total': 0}

    positions = convert_to_positions(predictions)

    n_trades = int(np.sum(positions != 0))
    n_total = len(positions)
    trade_ratio = n_trades / n_total

    # Trade frequency penalty
    if trade_ratio < MIN_TRADE_RATIO:
        penalty = trade_ratio / MIN_TRADE_RATIO
        return 0.5 * penalty, {
            'trade_ratio': trade_ratio, 'n_trades': n_trades,
            'n_total': n_total, 'penalized': True,
            'n_long': int(np.sum(positions == 1)),
            'n_short': int(np.sum(positions == -1)),
            'n_flat': int(np.sum(positions == 0)),
        }

    pnl = positions * returns
    gross_profit = np.sum(pnl[pnl > 0])
    gross_loss = np.abs(np.sum(pnl[pnl < 0]))

    if gross_loss < 1e-10:
        pf = 1.0 if gross_profit < 1e-10 else 99.99
    else:
        pf = gross_profit / gross_loss

    return pf, {
        'trade_ratio': trade_ratio, 'n_trades': n_trades, 'n_total': n_total,
        'penalized': False,
        'n_long': int(np.sum(positions == 1)),
        'n_short': int(np.sum(positions == -1)),
        'n_flat': int(np.sum(positions == 0)),
        'gross_profit': float(gross_profit),
        'gross_loss': float(gross_loss),
    }


# =============================================================================
# PROFIT FACTOR — COST-ADJUSTED
# =============================================================================
def calculate_cost_adjusted_pf(predictions: np.ndarray, returns: np.ndarray,
                               pair: str,
                               cost_profile: str = 'retail') -> Tuple[float, Dict]:
    """
    Calculate cost-adjusted PF with proper half-RT cost model.

    Cost logic: each trade execution (buy or sell) costs half a round-trip.
    - Flat → Position: 1 execution (entry) = 0.5 RT
    - Position → Flat: 1 execution (exit) = 0.5 RT
    - Position → Opposite: 2 executions (exit + entry) = 1.0 RT
    - Same position maintained: 0 cost

    Returns (adj_pf, cost_stats).
    """
    predictions = np.asarray(predictions).flatten()
    returns = np.asarray(returns).flatten()

    if len(predictions) == 0:
        return 1.0, {'num_position_changes': 0, 'total_cost': 0.0}

    positions = convert_to_positions(predictions)

    n_trades = int(np.sum(positions != 0))
    n_total = len(positions)
    trade_ratio = n_trades / n_total

    # Trade frequency penalty
    if trade_ratio < MIN_TRADE_RATIO:
        penalty = trade_ratio / MIN_TRADE_RATIO
        return 0.5 * penalty, {
            'num_position_changes': 0, 'total_cost': 0.0,
            'cost_per_trade': get_cost_per_trade(pair, cost_profile),
        }

    # Count trade executions (half round-trips) — vectorized
    # Each execution costs RT/2
    rt_cost = get_cost_per_trade(pair, cost_profile)
    half_rt = rt_cost / 2.0

    # Prepend a zero (start flat) to detect first-bar entry
    prev_positions = np.concatenate([[0.0], positions[:-1]])
    changed = positions != prev_positions

    # At each change: exit old (if non-flat) + enter new (if non-flat)
    exits = changed & (prev_positions != 0)   # closing old position
    entries = changed & (positions != 0)       # opening new position

    cost_per_bar = exits.astype(np.float64) * half_rt + entries.astype(np.float64) * half_rt
    n_executions = int(exits.sum() + entries.sum())

    num_changes = int(changed.sum())

    # PnL with cost deduction
    pnl = positions * returns
    adjusted_pnl = pnl - cost_per_bar

    gross_profit = np.sum(adjusted_pnl[adjusted_pnl > 0])
    gross_loss = np.abs(np.sum(adjusted_pnl[adjusted_pnl < 0]))

    if gross_loss < 1e-10:
        adj_pf = 1.0 if gross_profit < 1e-10 else 99.99
    else:
        adj_pf = gross_profit / gross_loss

    total_cost = float(np.sum(cost_per_bar))

    return adj_pf, {
        'num_position_changes': num_changes,
        'n_executions': n_executions,
        'total_cost': total_cost,
        'cost_per_trade_rt': rt_cost,
    }


# =============================================================================
# SUB-PERIOD CONSISTENCY (WALK-FORWARD LITE)
# =============================================================================
def calculate_sub_period_consistency(predictions: np.ndarray, returns: np.ndarray,
                                    oos_index: pd.DatetimeIndex,
                                    pair: str,
                                    cost_profile: str = 'retail') -> Dict[str, Any]:
    """
    Split OOS into half-year sub-periods and check PF consistency.
    Returns dict with per-period PFs and aggregate stats.
    """
    period_pfs = []
    period_details = {}

    for start, end, label in SUB_PERIODS:
        mask = (oos_index >= start) & (oos_index <= end)
        n_bars = int(mask.sum())
        if n_bars < 20:
            continue

        sub_preds = predictions[mask]
        sub_returns = returns[mask]

        sub_pf, _ = calculate_cost_adjusted_pf(sub_preds, sub_returns, pair,
                                               cost_profile)
        period_pfs.append(sub_pf)
        period_details[f'pf_{label}'] = round(sub_pf, 4)

    if len(period_pfs) >= 2:
        pf_mean = float(np.mean(period_pfs))
        pf_std = float(np.std(period_pfs))
        pf_min = float(np.min(period_pfs))
        pf_max = float(np.max(period_pfs))
        pf_cv = pf_std / (pf_mean + 1e-6)
        consistency_score = pf_mean / (pf_std + 1e-6)
    elif len(period_pfs) == 1:
        pf_mean = period_pfs[0]
        pf_std = 0.0
        pf_min = pf_mean
        pf_max = pf_mean
        pf_cv = 0.0
        consistency_score = 0.0
    else:
        pf_mean = 0.0
        pf_std = 0.0
        pf_min = 0.0
        pf_max = 0.0
        pf_cv = 0.0
        consistency_score = 0.0

    return {
        'sub_pf_mean': round(pf_mean, 4),
        'sub_pf_std': round(pf_std, 4),
        'sub_pf_min': round(pf_min, 4),
        'sub_pf_max': round(pf_max, 4),
        'sub_pf_cv': round(pf_cv, 4),
        'consistency_score': round(consistency_score, 4),
        'n_sub_periods': len(period_pfs),
        **period_details,
    }


# =============================================================================
# SINGLE MODEL EVALUATION
# =============================================================================
def evaluate_single_model(pair: str, tf: str, brain: str,
                          data: Dict[str, Any],
                          device: torch.device,
                          cost_profile: str = 'retail') -> Optional[Dict[str, Any]]:
    """
    Evaluate one model on OOS data.
    Returns result dict or None if model doesn't exist or fails.
    """
    model_key = f"{pair}_{tf}_{brain}"
    is_nn = brain in NN_BRAINS

    # Determine file path
    if is_nn:
        filepath = MODELS_DIR / f"{model_key}.pt"
    else:
        filepath = MODELS_DIR / f"{model_key}.joblib"

    if not filepath.exists():
        return None  # Model not yet trained

    X_oos = data['X_oos']
    returns_oos = data['returns_oos']
    oos_index = data['oos_index']
    n_features = data['n_features']

    if len(X_oos) < 50:
        return None  # Too little OOS data

    try:
        # Load model
        if is_nn:
            if not NN_CLASSES_AVAILABLE:
                return None
            model, scaler, train_pf = load_pt_model(filepath, brain, n_features)
            predictions = predict_pytorch(model, scaler, X_oos, device)
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
        else:
            model, scaler, train_pf = load_joblib_model(filepath)
            predictions = predict_joblib(model, scaler, X_oos)
            del model
            gc.collect()

        # Raw PF
        raw_pf, raw_stats = calculate_raw_pf(predictions, returns_oos)

        # Cost-adjusted PF
        adj_pf, cost_stats = calculate_cost_adjusted_pf(
            predictions, returns_oos, pair, cost_profile
        )

        # Overfit ratio
        if train_pf and train_pf > 0 and adj_pf > 0:
            overfit_ratio = train_pf / adj_pf
        else:
            overfit_ratio = None

        # Sub-period consistency
        sub_results = calculate_sub_period_consistency(
            predictions, returns_oos, oos_index, pair, cost_profile
        )

        # Flagging
        is_overfit = (overfit_ratio is not None and overfit_ratio > 2.0)
        is_unstable = (sub_results['sub_pf_cv'] > 0.5 or
                       sub_results['sub_pf_min'] < 0.8)
        passed_all = (adj_pf > 1.0 and not is_overfit and not is_unstable)

        return {
            'model_key': model_key,
            'pair': pair,
            'timeframe': tf,
            'brain': brain,
            'train_pf': round(train_pf, 4) if train_pf else None,
            'oos_raw_pf': round(raw_pf, 4),
            'oos_adj_pf': round(adj_pf, 4),
            'overfit_ratio': round(overfit_ratio, 4) if overfit_ratio else None,
            'n_oos_bars': len(X_oos),
            'num_trades': raw_stats.get('n_trades', 0),
            'trade_ratio': round(raw_stats.get('trade_ratio', 0), 4),
            'n_long': raw_stats.get('n_long', 0),
            'n_short': raw_stats.get('n_short', 0),
            'n_flat': raw_stats.get('n_flat', 0),
            'num_position_changes': cost_stats.get('num_position_changes', 0),
            'total_cost': round(cost_stats.get('total_cost', 0), 6),
            **sub_results,
            'is_overfit': is_overfit,
            'is_unstable': is_unstable,
            'passed_all': passed_all,
        }

    except Exception as e:
        print(f"  ERROR evaluating {model_key}: {e}")
        return None


# =============================================================================
# CHECKPOINT SYSTEM
# =============================================================================
def load_eval_checkpoint() -> Dict:
    """Load evaluation checkpoint for resumability."""
    if EVAL_CHECKPOINT_FILE.exists():
        with open(EVAL_CHECKPOINT_FILE) as f:
            return json.load(f)
    return {'completed': [], 'started': None, 'last_updated': None}


def save_eval_checkpoint(completed_list: List[str]):
    """Save evaluation checkpoint."""
    with open(EVAL_CHECKPOINT_FILE, 'w') as f:
        json.dump({
            'completed': completed_list,
            'started': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
        }, f, indent=2)


# =============================================================================
# WALK-FORWARD RE-TRAINING PLACEHOLDER (V3)
# =============================================================================
def walk_forward_retrain(pair: str, tf: str, brain: str, data: Dict,
                         windows: List, **kwargs):
    """
    PLACEHOLDER: True walk-forward validation with re-training.
    Not implemented in V2. Reserved for V3.

    Future design:
        for window in rolling_windows:
            model = retrain_model(brain, pair, tf, data[:window.train_end])
            pf = evaluate_on_window(model, data[window.test_start:window.test_end])
            results.append(pf)

    Requires extracting training logic into callable functions
    and significant compute time (~50x current evaluation).
    """
    raise NotImplementedError(
        "Walk-forward re-training not yet implemented. "
        "Use sub-period consistency checks as a lightweight alternative."
    )


# =============================================================================
# PROFILE-AWARE FILE PATHS
# =============================================================================
def get_output_paths(cost_profile: str) -> Dict[str, Path]:
    """Get output file paths for a given cost profile."""
    if cost_profile == 'retail':
        # Original paths — backward compatible
        return {
            'oos': MODELS_DIR / "oos_validation.csv",
            'ranking': MODELS_DIR / "model_ranking.csv",
            'checkpoint': MODELS_DIR / "eval_checkpoint.json",
        }
    else:
        suffix = f"_{cost_profile}"
        return {
            'oos': MODELS_DIR / f"oos_validation{suffix}.csv",
            'ranking': MODELS_DIR / f"model_ranking{suffix}.csv",
            'checkpoint': MODELS_DIR / f"eval_checkpoint{suffix}.json",
        }


def load_profile_checkpoint(checkpoint_file: Path) -> Dict:
    """Load evaluation checkpoint for a specific profile."""
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            return json.load(f)
    return {'completed': [], 'started': None, 'last_updated': None}


def save_profile_checkpoint(checkpoint_file: Path, completed_list: List[str]):
    """Save evaluation checkpoint for a specific profile."""
    with open(checkpoint_file, 'w') as f:
        json.dump({
            'completed': completed_list,
            'started': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
        }, f, indent=2)


# =============================================================================
# COST PROFILE COMPARISON
# =============================================================================
def generate_cost_comparison():
    """
    Generate cost_profile_comparison.csv by joining all available profile results.
    Identifies models that flip from FAIL (retail) to PASS under cheaper profiles.
    """
    retail_paths = get_output_paths('retail')
    raw_vps_paths = get_output_paths('raw_vps')
    ibkr_paths = get_output_paths('ibkr')
    comparison_file = MODELS_DIR / "cost_profile_comparison.csv"

    if not retail_paths['oos'].exists():
        print("Cannot generate comparison: retail results not found.")
        return

    retail_df = pd.read_csv(retail_paths['oos'])

    # Start with retail base columns
    merged = retail_df[['model_key', 'pair', 'timeframe', 'brain',
                         'oos_adj_pf', 'passed_all']].copy()
    merged = merged.rename(columns={
        'oos_adj_pf': 'retail_adj_pf',
        'passed_all': 'retail_passed',
    })

    has_raw_vps = raw_vps_paths['oos'].exists()
    has_ibkr = ibkr_paths['oos'].exists()

    if not has_raw_vps and not has_ibkr:
        print("Cannot generate comparison: need at least one non-retail profile.")
        return

    # Merge raw_vps if available
    if has_raw_vps:
        raw_vps_df = pd.read_csv(raw_vps_paths['oos'])
        merged = merged.merge(
            raw_vps_df[['model_key', 'oos_adj_pf', 'passed_all']].rename(
                columns={'oos_adj_pf': 'raw_vps_adj_pf',
                         'passed_all': 'raw_vps_passed'}),
            on='model_key', how='left'
        )
        merged['flipped_raw_vps'] = (
            (~merged['retail_passed']) & merged['raw_vps_passed'].fillna(False)
        )
        merged['pf_improvement_raw_vps'] = (
            merged['raw_vps_adj_pf'] - merged['retail_adj_pf']
        )

    # Merge ibkr if available
    if has_ibkr:
        ibkr_df = pd.read_csv(ibkr_paths['oos'])
        merged = merged.merge(
            ibkr_df[['model_key', 'oos_adj_pf', 'passed_all']].rename(
                columns={'oos_adj_pf': 'ibkr_adj_pf',
                         'passed_all': 'ibkr_passed'}),
            on='model_key', how='left'
        )
        merged['flipped_ibkr'] = (
            (~merged['retail_passed']) & merged['ibkr_passed'].fillna(False)
        )
        merged['pf_improvement_ibkr'] = (
            merged['ibkr_adj_pf'] - merged['retail_adj_pf']
        )

    # Sort: ibkr flips first (if present), then raw_vps flips, then by best improvement
    sort_cols = []
    if has_ibkr:
        sort_cols.append('flipped_ibkr')
    if has_raw_vps:
        sort_cols.append('flipped_raw_vps')
    if has_ibkr:
        sort_cols.append('pf_improvement_ibkr')
    elif has_raw_vps:
        sort_cols.append('pf_improvement_raw_vps')

    if sort_cols:
        merged = merged.sort_values(
            sort_cols, ascending=[False] * len(sort_cols))

    merged.to_csv(comparison_file, index=False)

    # --- Print summary ---
    print(f"\n{'=' * 70}")
    print(f"COST PROFILE COMPARISON")
    print(f"{'=' * 70}")
    print(f"Models in retail baseline: {len(merged)}")
    n_retail_pass = int(merged['retail_passed'].sum())
    print(f"  Retail passed: {n_retail_pass}")

    if has_raw_vps:
        n_rvps = int(merged['raw_vps_passed'].sum())
        n_flip_rvps = int(merged['flipped_raw_vps'].sum())
        print(f"  Raw VPS passed: {n_rvps}  |  flipped: {n_flip_rvps}")
        print(f"  Mean PF improvement (raw_vps vs retail): "
              f"{merged['pf_improvement_raw_vps'].mean():.4f}")

    if has_ibkr:
        n_ibkr = int(merged['ibkr_passed'].sum())
        n_flip_ibkr = int(merged['flipped_ibkr'].sum())
        print(f"  IBKR passed: {n_ibkr}  |  flipped: {n_flip_ibkr}")
        print(f"  Mean PF improvement (ibkr vs retail): "
              f"{merged['pf_improvement_ibkr'].mean():.4f}")

    print(f"Saved: {comparison_file}")

    # Print flipped models for each profile
    if has_raw_vps and int(merged['flipped_raw_vps'].sum()) > 0:
        print(f"\n--- FLIPPED MODELS: raw_vps (failed retail, passed raw_vps) ---")
        flipped = merged[merged['flipped_raw_vps']]
        for _, row in flipped.iterrows():
            print(f"  {row['model_key']:<35s} "
                  f"retail={row['retail_adj_pf']:.3f} "
                  f"raw_vps={row['raw_vps_adj_pf']:.3f} "
                  f"(+{row['pf_improvement_raw_vps']:.3f})")

    if has_ibkr and int(merged['flipped_ibkr'].sum()) > 0:
        print(f"\n--- FLIPPED MODELS: ibkr (failed retail, passed IBKR) ---")
        flipped = merged[merged['flipped_ibkr']]
        for _, row in flipped.iterrows():
            ibkr_pf = row['ibkr_adj_pf']
            imp = row['pf_improvement_ibkr']
            print(f"  {row['model_key']:<35s} "
                  f"retail={row['retail_adj_pf']:.3f} "
                  f"ibkr={ibkr_pf:.3f} "
                  f"(+{imp:.3f})")


# =============================================================================
# RUN EVALUATION FOR ONE PROFILE
# =============================================================================
def run_evaluation(cost_profile: str):
    """Run full evaluation loop for a single cost profile."""
    paths = get_output_paths(cost_profile)

    print(f"\n{'=' * 70}")
    print(f"CHAOS V2 EVALUATION — [{cost_profile.upper()}] cost profile")
    print(f"Started: {datetime.now()}")
    print(f"OOS cutoff: {OOS_CUTOFF_DATE}")
    print(f"Skipping timeframes: {SKIP_TIMEFRAMES}")
    print(f"Output: {paths['oos'].name}")
    print(f"{'=' * 70}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load checkpoint
    checkpoint = load_profile_checkpoint(paths['checkpoint'])
    completed = set(checkpoint['completed'])
    print(f"Previously evaluated: {len(completed)} models")

    # Load existing results if resuming
    results = []
    if paths['oos'].exists() and len(completed) > 0:
        existing_df = pd.read_csv(paths['oos'])
        results = existing_df.to_dict('records')
        print(f"Loaded {len(results)} existing results from {paths['oos'].name}")

    total_evaluated = 0
    total_passed = 0
    total_skipped = 0
    start_time = time.time()

    # Outer loop: pair x timeframe (load features once per combo)
    for pair in ALL_PAIRS:
        for tf in ALL_TIMEFRAMES:
            if tf in SKIP_TIMEFRAMES:
                continue

            # Check if any brains need evaluation for this pair/tf
            pending_brains = [
                b for b in ALL_BRAINS
                if f"{pair}_{tf}_{b}" not in completed
            ]
            if not pending_brains:
                total_skipped += len(ALL_BRAINS)
                continue

            print(f"\n--- {pair}_{tf} ({len(pending_brains)} brains pending) ---")

            # Load features once for this pair/tf
            try:
                data = load_and_split_features(pair, tf)
                print(f"  Features: {data['n_features']} cols, "
                      f"IS={data['n_is']:,} bars, OOS={data['n_oos']:,} bars")
            except Exception as e:
                print(f"  SKIP: Cannot load features: {e}")
                continue

            # Inner loop: evaluate all brains against cached features
            for brain in ALL_BRAINS:
                model_key = f"{pair}_{tf}_{brain}"

                if model_key in completed:
                    continue

                result = evaluate_single_model(pair, tf, brain, data, device,
                                               cost_profile)

                if result is not None:
                    results.append(result)
                    completed.add(model_key)

                    total_evaluated += 1
                    if result.get('passed_all', False):
                        total_passed += 1

                    status = "PASS" if result['passed_all'] else "FAIL"
                    or_str = (f"OR={result['overfit_ratio']:.2f}"
                              if result['overfit_ratio'] else "OR=N/A")
                    print(f"  {brain}: adj_PF={result['oos_adj_pf']:.3f} "
                          f"{or_str} TR={result['trade_ratio']:.1%} [{status}]")

                    # Save checkpoint periodically (every model)
                    save_profile_checkpoint(paths['checkpoint'], list(completed))

            # Release feature data
            del data
            gc.collect()

    # Generate output CSVs
    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"[{cost_profile.upper()}] EVALUATION COMPLETE")
    print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Evaluated: {total_evaluated} models")
    print(f"Passed: {total_passed}")
    print(f"{'=' * 70}")

    if results:
        df = pd.DataFrame(results)

        # Ensure flag columns exist
        if 'is_overfit' not in df.columns:
            df['is_overfit'] = False
        if 'is_unstable' not in df.columns:
            df['is_unstable'] = False
        if 'passed_all' not in df.columns:
            df['passed_all'] = False

        # Save full results
        df.to_csv(paths['oos'], index=False)
        print(f"\nSaved: {paths['oos']} ({len(df)} models)")

        # Summary stats
        print(f"\n--- SUMMARY [{cost_profile.upper()}] ---")
        print(f"Total models evaluated: {len(df)}")
        print(f"Passed all criteria: {df['passed_all'].sum()}")
        print(f"Overfit (ratio > 2.0): {df['is_overfit'].sum()}")
        print(f"Unstable (CV > 0.5): {df['is_unstable'].sum()}")

        if 'oos_adj_pf' in df.columns:
            print(f"\nCost-adjusted PF distribution:")
            print(f"  Mean: {df['oos_adj_pf'].mean():.3f}")
            print(f"  Median: {df['oos_adj_pf'].median():.3f}")
            print(f"  Min: {df['oos_adj_pf'].min():.3f}")
            print(f"  Max: {df['oos_adj_pf'].max():.3f}")

        # Generate ranking
        passing = df[df['passed_all']].copy()
        if len(passing) > 0:
            or_clipped = passing['overfit_ratio'].fillna(99).clip(lower=0.1)
            passing['composite_score'] = (
                passing['oos_adj_pf'] * 0.5 +
                passing['consistency_score'] * 0.3 +
                (1.0 / (or_clipped + 0.1)) * 0.2
            )
            passing = passing.sort_values('composite_score', ascending=False)
            passing['rank'] = range(1, len(passing) + 1)

            rank_cols = ['rank', 'model_key', 'pair', 'timeframe', 'brain',
                         'train_pf', 'oos_raw_pf', 'oos_adj_pf',
                         'overfit_ratio', 'consistency_score', 'composite_score',
                         'trade_ratio', 'n_oos_bars', 'sub_pf_min', 'sub_pf_max']
            # Only include columns that exist
            rank_cols = [c for c in rank_cols if c in passing.columns]
            passing[rank_cols].to_csv(paths['ranking'], index=False)
            print(f"\nSaved: {paths['ranking']} ({len(passing)} models passing)")

            print(f"\n--- TOP 20 MODELS [{cost_profile.upper()}] ---")
            for _, row in passing.head(20).iterrows():
                print(f"  #{int(row['rank']):>3d} {row['model_key']:<35s} "
                      f"adj_PF={row['oos_adj_pf']:.3f} "
                      f"OR={row['overfit_ratio']:.2f} "
                      f"CS={row['composite_score']:.3f}")
        else:
            print(f"\nWARNING: No models passed all criteria [{cost_profile}].")
    else:
        print("\nNo models evaluated. Check that model files exist.")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='CHAOS V2 Evaluation Module')
    parser.add_argument('--cost-profile', type=str, default='retail',
                        choices=['retail', 'raw_vps', 'ibkr', 'all'],
                        help='Cost profile to evaluate: retail (default), '
                             'raw_vps, ibkr, or all (runs all three then '
                             'generates comparison)')
    args = parser.parse_args()

    if args.cost_profile == 'all':
        profiles_to_run = ['retail', 'raw_vps', 'ibkr']
    else:
        profiles_to_run = [args.cost_profile]

    for profile in profiles_to_run:
        run_evaluation(profile)

    # Generate comparison if retail + at least one other profile exist
    retail_oos = get_output_paths('retail')['oos']
    raw_vps_oos = get_output_paths('raw_vps')['oos']
    ibkr_oos = get_output_paths('ibkr')['oos']
    if retail_oos.exists() and (raw_vps_oos.exists() or ibkr_oos.exists()):
        generate_cost_comparison()


if __name__ == '__main__':
    main()
