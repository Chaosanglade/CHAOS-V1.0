"""
CHAOS V1.0 — V2 Retraining Script (Anti-Flat Constraints)

Retrains all v1 brains that failed the audit (89-100% FLAT) using
v2-style hyperparameter ranges proven to prevent flat collapse.

Changes from original training:
  - V2 hyperparameter ranges for ALL brains (higher reg, slower LR)
  - MIN_TRADE_RATIO raised from 10% to 20%
  - Entropy penalty: single-class > 80% → PF * 0.1
  - Post-training eligibility gate (flat <= 70%, directional >= 30%)
  - NEW: RF_v2 and ET_v2 with shallow trees + high min_samples_leaf

Runs on local GPU (RTX 4060) or Colab A100.

Usage:
    python chaos_v2_retrain.py                              # All 9 pairs x 2 TFs x 9 brains
    python chaos_v2_retrain.py --pairs EURUSD GBPUSD        # Specific pairs
    python chaos_v2_retrain.py --brains lgb_optuna rf_v2    # Specific brains
    python chaos_v2_retrain.py --resume                     # Resume from checkpoint
"""
import os
import sys
import json
import time
import argparse
import warnings
import gc

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import optuna
from pathlib import Path
from datetime import datetime
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Config ──
PROJECT_ROOT = Path(os.environ.get('CHAOS_BASE_DIR', '.'))
FEATURES_DIR = PROJECT_ROOT / 'features'
OUTPUT_DIR = PROJECT_ROOT / 'models' / 'v2_retrained'
# Checkpoint on LOCAL disk (not Google Drive) to avoid Errno 22 on sync
CHECKPOINT_DIR = Path(os.path.expanduser('~/chaos_checkpoints'))

RANDOM_SEED = 42
TARGET_COL = 'target_3class_8'
RETURNS_COL = 'target_return_8'
EXECUTION_LAG_BARS = 1
MIN_TRADE_RATIO = 0.20  # Raised from 0.10 to 0.20
OPTUNA_TRIALS = 250

ALL_PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD',
             'USDCHF', 'NZDUSD', 'EURJPY', 'GBPJPY']
TRADE_TFS = ['H1', 'M30']

BRAINS_TO_RETRAIN = [
    'lgb_optuna', 'xgb_optuna', 'cat_optuna',
    'rf_v2_optuna', 'et_v2_optuna',
    'lgb_v2_optuna', 'xgb_v2_optuna', 'cat_v2_optuna',
    'mlp_optuna', 'transformer_optuna',
]

# Estimated time per brain per block (minutes)
BRAIN_TIMES = {
    'lgb_optuna': 8, 'lgb_v2_optuna': 10, 'xgb_optuna': 12, 'xgb_v2_optuna': 15,
    'cat_optuna': 15, 'cat_v2_optuna': 18, 'rf_v2_optuna': 5, 'et_v2_optuna': 5,
    'mlp_optuna': 20, 'transformer_optuna': 25,
}

np.random.seed(RANDOM_SEED)

EXCLUDE_SUBS = ['target_', 'return', 'Open', 'High', 'Low', 'Close', 'Volume',
                'timestamp', 'date', 'pair', 'symbol', 'tf', 'bar_time',
                'Bid_', 'Ask_', 'Spread_', 'Unnamed']


# ═══════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════

def load_data(pair, tf):
    """Load features, split 80/20 chronologically, apply execution lag."""
    parquet = FEATURES_DIR / f'{pair}_{tf}_features.parquet'
    if not parquet.exists():
        raise FileNotFoundError(f'{parquet}')

    df = pd.read_parquet(str(parquet))
    feature_cols = [c for c in df.columns if not any(ex in c for ex in EXCLUDE_SUBS)]

    if TARGET_COL not in df.columns or RETURNS_COL not in df.columns:
        raise ValueError(f'Missing target columns in {pair}_{tf}')

    # Execution lag on returns
    df['returns_lagged'] = df[RETURNS_COL].shift(-EXECUTION_LAG_BARS)
    df = df.dropna(subset=['returns_lagged'])

    X = np.nan_to_num(df[feature_cols].values.astype('float32'), nan=0, posinf=0, neginf=0)
    y = df[TARGET_COL].values.copy()
    y = np.nan_to_num(y, nan=0).astype('int64')
    if y.min() < 0:
        y = y + 1
    returns = np.nan_to_num(df['returns_lagged'].values, nan=0, posinf=0, neginf=0)

    split = int(len(X) * 0.8)
    return (X[:split], y[:split], X[split:], y[split:], returns[split:], len(feature_cols))


# ═══════════════════════════════════════════════════════════════════════
# Profit Factor + Anti-Flat Penalty
# ═══════════════════════════════════════════════════════════════════════

def calculate_pf_with_penalty(predictions, returns):
    """
    PF calculation with anti-flat constraints:
    1. Trade ratio < 20% → PF scaled down to 0-0.5
    2. Single class > 80% → PF * 0.1
    3. Single class > 70% → PF * 0.5
    """
    predictions = np.asarray(predictions).flatten()
    returns = np.asarray(returns).flatten()

    if len(predictions) == 0:
        return 0.0

    positions = predictions.astype(np.float64) - 1.0  # 0→-1, 1→0, 2→+1

    n_trades = int(np.sum(positions != 0))
    trade_ratio = n_trades / len(positions)

    # Penalty 1: Low trade ratio
    if trade_ratio < MIN_TRADE_RATIO:
        return 0.5 * (trade_ratio / MIN_TRADE_RATIO)

    # Calculate raw PF
    pnl = positions * returns
    gross_profit = np.sum(pnl[pnl > 0])
    gross_loss = np.abs(np.sum(pnl[pnl < 0]))

    if gross_loss < 1e-10:
        pf = 99.99 if gross_profit > 1e-10 else 1.0
    else:
        pf = gross_profit / gross_loss

    # Penalty 2: Class dominance (anti-flat / anti-single-class)
    counts = np.bincount(predictions.astype(int), minlength=3)
    max_pct = counts.max() / max(counts.sum(), 1)
    if max_pct > 0.80:
        pf = pf * 0.1  # Severe penalty
    elif max_pct > 0.70:
        pf = pf * 0.5  # Moderate penalty

    return pf


# ═══════════════════════════════════════════════════════════════════════
# Eligibility Gate (post-training check)
# ═══════════════════════════════════════════════════════════════════════

def check_eligibility(model, X_val, y_val, returns_val, brain_name):
    """Check if model passes the 5-layer audit gate."""
    if hasattr(model, 'predict'):
        preds = model.predict(X_val)
    else:
        preds = np.argmax(model.predict_proba(X_val), axis=1)

    counts = Counter(preds.astype(int))
    n = len(preds)
    flat_pct = counts.get(1, 0) / n * 100
    long_pct = counts.get(2, 0) / n * 100
    short_pct = counts.get(0, 0) / n * 100
    directional = long_pct + short_pct

    # Entropy
    p = np.array([short_pct, flat_pct, long_pct]) / 100 + 1e-10
    entropy = -np.sum(p * np.log2(p))

    # PF on validation
    pf = calculate_pf_with_penalty(preds, returns_val)

    # Trade count
    trades = sum(1 for i in range(len(preds)) if preds[i] != 1 and (i == 0 or preds[i-1] == 1))

    eligible = (flat_pct <= 70 and directional >= 30 and trades >= 10)

    return {
        'eligible': eligible,
        'flat_pct': round(flat_pct, 1),
        'long_pct': round(long_pct, 1),
        'short_pct': round(short_pct, 1),
        'entropy': round(entropy, 3),
        'pf': round(pf, 3),
        'trades': trades,
    }


# ═══════════════════════════════════════════════════════════════════════
# Brain Objective Functions (all use v2-style ranges)
# ═══════════════════════════════════════════════════════════════════════

def create_lgb_objective(X_train, y_train, X_val, y_val, returns_val):
    from lightgbm import LGBMClassifier

    def objective(trial):
        params = {
            'objective': 'multiclass', 'num_class': 3, 'boosting_type': 'gbdt',
            'n_estimators': trial.suggest_int('n_estimators', 200, 800),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'num_leaves': trial.suggest_int('num_leaves', 10, 50),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 200),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 100.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 100.0, log=True),
            'class_weight': 'balanced',
            'random_state': RANDOM_SEED, 'verbosity': -1, 'n_jobs': -1,
        }
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return calculate_pf_with_penalty(preds, returns_val)
    return objective


def create_xgb_objective(X_train, y_train, X_val, y_val, returns_val):
    from xgboost import XGBClassifier
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    weight_map = {int(c): float(w) for c, w in zip(classes, weights)}
    sample_weights = np.array([weight_map[int(y)] for y in y_train])

    def objective(trial):
        use_gpu = trial.suggest_categorical('use_gpu', [True, False])
        params = {
            'objective': 'multi:softmax', 'num_class': 3,
            'tree_method': 'hist',
            'device': 'cuda' if use_gpu else 'cpu',
            'n_estimators': trial.suggest_int('n_estimators', 200, 800),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 2, 6),
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 50),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 100.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 100.0, log=True),
            'random_state': RANDOM_SEED, 'verbosity': 0, 'n_jobs': -1,
        }
        try:
            model = XGBClassifier(**params)
            model.fit(X_train, y_train, sample_weight=sample_weights)
        except Exception:
            params['device'] = 'cpu'
            model = XGBClassifier(**params)
            model.fit(X_train, y_train, sample_weight=sample_weights)
        preds = model.predict(X_val)
        return calculate_pf_with_penalty(preds, returns_val)
    return objective


def create_cat_objective(X_train, y_train, X_val, y_val, returns_val):
    from catboost import CatBoostClassifier
    # CatBoost chokes on numpy int64 — convert everything to native Python types
    X_train_f64 = X_train.astype(np.float64)
    X_val_f64 = X_val.astype(np.float64)
    y_train_list = [int(y) for y in y_train]
    y_val_list = [int(y) for y in y_val]

    def objective(trial):
        use_gpu = trial.suggest_categorical('use_gpu', [True, False])
        params = {
            'iterations': trial.suggest_int('iterations', 200, 800),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'depth': trial.suggest_int('depth', 3, 6),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 100.0, log=True),
            'auto_class_weights': 'Balanced',
            'task_type': 'GPU' if use_gpu else 'CPU',
            'random_seed': RANDOM_SEED, 'verbose': False,
            'thread_count': -1,
        }
        if use_gpu:
            params['devices'] = '0'
        try:
            model = CatBoostClassifier(**params)
            model.fit(X_train_f64, y_train_list)
        except Exception:
            params['task_type'] = 'CPU'
            params.pop('devices', None)
            model = CatBoostClassifier(**params)
            model.fit(X_train_f64, y_train_list)
        preds = model.predict(X_val_f64).flatten().astype(int)
        return calculate_pf_with_penalty(preds, returns_val)
    return objective


def create_rf_v2_objective(X_train, y_train, X_val, y_val, returns_val):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'min_samples_split': trial.suggest_int('min_samples_split', 10, 50),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 50),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'class_weight': 'balanced',
            'n_jobs': -1, 'random_state': RANDOM_SEED,
        }
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return calculate_pf_with_penalty(preds, returns_val)
    return objective


def create_et_v2_objective(X_train, y_train, X_val, y_val, returns_val):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'min_samples_split': trial.suggest_int('min_samples_split', 10, 50),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 50),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'class_weight': 'balanced',
            'n_jobs': -1, 'random_state': RANDOM_SEED,
        }
        model = ExtraTreesClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return calculate_pf_with_penalty(preds, returns_val)
    return objective


def _train_pytorch_model(model, X_train, y_train, X_val, y_val, lr, weight_decay,
                         batch_size, epochs, patience=15):
    """Train a PyTorch model with balanced class weights + early stopping."""
    import torch
    import torch.nn as nn

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Balanced class weights
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_v = torch.tensor(y_val, dtype=torch.long).to(device)

    best_loss = float('inf')
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(X_t))
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(X_t), batch_size):
            idx = indices[start:start + batch_size]
            out = model(X_t[idx])
            loss = criterion(out, y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(X_v)
            val_loss = criterion(val_out, y_v).item()
        scheduler.step(val_loss)

        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model.cpu()


class _MLPClassifier(object):
    """Sklearn-like wrapper around a PyTorch MLP for predict/predict_proba."""
    def __init__(self, net, n_features):
        self.net = net
        self.n_features = n_features

    def predict_proba(self, X):
        import torch
        with torch.no_grad():
            out = self.net(torch.tensor(X[:, :self.n_features], dtype=torch.float32))
            return torch.softmax(out, dim=1).numpy()

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


class _TransformerClassifier(object):
    """Sklearn-like wrapper around a PyTorch Transformer."""
    def __init__(self, net, n_features):
        self.net = net
        self.n_features = n_features

    def predict_proba(self, X):
        import torch
        with torch.no_grad():
            out = self.net(torch.tensor(X[:, :self.n_features], dtype=torch.float32))
            return torch.softmax(out, dim=1).numpy()

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def create_mlp_objective(X_train, y_train, X_val, y_val, returns_val):
    import torch
    import torch.nn as nn
    n_features = X_train.shape[1]

    def objective(trial):
        n_layers = trial.suggest_int('n_layers', 2, 4)
        hidden_size = trial.suggest_int('hidden_size', 64, 512, step=64)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        wd = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024])
        epochs = trial.suggest_int('epochs', 50, 200, step=25)

        layers = []
        in_dim = n_features
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_dim, hidden_size), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = hidden_size
        layers.append(nn.Linear(in_dim, 3))
        net = nn.Sequential(*layers)

        net = _train_pytorch_model(net, X_train, y_train, X_val, y_val,
                                    lr=lr, weight_decay=wd, batch_size=batch_size, epochs=epochs)

        wrapper = _MLPClassifier(net, n_features)
        preds = wrapper.predict(X_val)
        return calculate_pf_with_penalty(preds, returns_val)
    return objective


def create_transformer_objective(X_train, y_train, X_val, y_val, returns_val):
    import torch
    import torch.nn as nn
    n_features = X_train.shape[1]

    class _TransformerNet(nn.Module):
        def __init__(self, d_model, n_heads, n_layers, dropout):
            super().__init__()
            self.proj = nn.Linear(n_features, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                dropout=dropout, batch_first=True)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.head = nn.Linear(d_model, 3)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            # Treat feature vector as single-token sequence
            x = self.proj(x).unsqueeze(1)  # (batch, 1, d_model)
            x = self.encoder(x)
            x = x.squeeze(1)  # (batch, d_model)
            return self.head(self.dropout(x))

    def objective(trial):
        d_model_raw = trial.suggest_int('d_model', 64, 256, step=64)
        n_heads = trial.suggest_categorical('n_heads', [2, 4, 8])
        # d_model must be divisible by n_heads
        d_model = max(n_heads, (d_model_raw // n_heads) * n_heads)
        n_layers = trial.suggest_int('n_layers', 1, 4)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        wd = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [256, 512])
        epochs = trial.suggest_int('epochs', 50, 150, step=25)

        net = _TransformerNet(d_model, n_heads, n_layers, dropout)
        net = _train_pytorch_model(net, X_train, y_train, X_val, y_val,
                                    lr=lr, weight_decay=wd, batch_size=batch_size, epochs=epochs)

        wrapper = _TransformerClassifier(net, n_features)
        preds = wrapper.predict(X_val)
        return calculate_pf_with_penalty(preds, returns_val)
    return objective


OBJECTIVE_MAP = {
    'lgb_optuna': create_lgb_objective,
    'lgb_v2_optuna': create_lgb_objective,
    'xgb_optuna': create_xgb_objective,
    'xgb_v2_optuna': create_xgb_objective,
    'cat_optuna': create_cat_objective,
    'cat_v2_optuna': create_cat_objective,
    'rf_v2_optuna': create_rf_v2_objective,
    'et_v2_optuna': create_et_v2_objective,
    'mlp_optuna': create_mlp_objective,
    'transformer_optuna': create_transformer_objective,
}


# ═══════════════════════════════════════════════════════════════════════
# Final Model Training (rebuild best params)
# ═══════════════════════════════════════════════════════════════════════

def train_final_model(brain, best_params, X_train, y_train):
    """Rebuild model with best Optuna params."""
    # Clean Optuna-only params
    p = {k: v for k, v in best_params.items() if k != 'use_gpu'}

    if 'lgb' in brain:
        from lightgbm import LGBMClassifier
        p.update({'objective': 'multiclass', 'num_class': 3, 'boosting_type': 'gbdt',
                  'class_weight': 'balanced', 'random_state': RANDOM_SEED,
                  'verbosity': -1, 'n_jobs': -1})
        model = LGBMClassifier(**p)
        model.fit(X_train, y_train)

    elif 'xgb' in brain:
        from xgboost import XGBClassifier
        p.update({'objective': 'multi:softmax', 'num_class': 3, 'tree_method': 'hist',
                  'device': 'cpu', 'random_state': RANDOM_SEED, 'verbosity': 0, 'n_jobs': -1})
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        weight_map = {int(c): float(w) for c, w in zip(classes, weights)}
        sw = np.array([weight_map[int(y)] for y in y_train])
        model = XGBClassifier(**p)
        model.fit(X_train, y_train, sample_weight=sw)

    elif 'cat' in brain:
        from catboost import CatBoostClassifier
        if 'iterations' not in p:
            p['iterations'] = p.pop('n_estimators', 500)
        p.update({'auto_class_weights': 'Balanced', 'task_type': 'CPU',
                  'random_seed': RANDOM_SEED, 'verbose': False, 'thread_count': -1})
        model = CatBoostClassifier(**p)
        model.fit(X_train.astype(np.float64), [int(y) for y in y_train])

    elif 'rf' in brain:
        p.update({'class_weight': 'balanced', 'n_jobs': -1, 'random_state': RANDOM_SEED})
        model = RandomForestClassifier(**p)
        model.fit(X_train, y_train)

    elif 'et' in brain:
        p.update({'class_weight': 'balanced', 'n_jobs': -1, 'random_state': RANDOM_SEED})
        model = ExtraTreesClassifier(**p)
        model.fit(X_train, y_train)

    elif 'mlp' in brain:
        import torch
        import torch.nn as nn
        n_features = X_train.shape[1]
        n_layers = p.get('n_layers', 3)
        hidden_size = p.get('hidden_size', 256)
        dropout = p.get('dropout', 0.3)
        lr = p.get('lr', 1e-3)
        wd = p.get('weight_decay', 1e-4)
        batch_size = p.get('batch_size', 512)
        epochs = p.get('epochs', 100)

        layers = []
        in_dim = n_features
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_dim, hidden_size), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = hidden_size
        layers.append(nn.Linear(in_dim, 3))
        net = nn.Sequential(*layers)

        net = _train_pytorch_model(net, X_train, y_train,
                                    X_train[-len(X_train)//5:], y_train[-len(y_train)//5:],
                                    lr=lr, weight_decay=wd, batch_size=batch_size, epochs=epochs)
        model = _MLPClassifier(net, n_features)

    elif 'transformer' in brain:
        import torch
        import torch.nn as nn
        n_features = X_train.shape[1]
        d_model_raw = p.get('d_model', 128)
        n_heads = p.get('n_heads', 4)
        d_model = max(n_heads, (d_model_raw // n_heads) * n_heads)
        n_layers_t = p.get('n_layers', 2)
        dropout = p.get('dropout', 0.3)
        lr = p.get('lr', 1e-3)
        wd = p.get('weight_decay', 1e-4)
        batch_size = p.get('batch_size', 512)
        epochs = p.get('epochs', 100)

        class _TNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(n_features, d_model)
                enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                         dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
                self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers_t)
                self.head = nn.Linear(d_model, 3)
                self.drop = nn.Dropout(dropout)
            def forward(self, x):
                x = self.proj(x).unsqueeze(1)
                x = self.encoder(x).squeeze(1)
                return self.head(self.drop(x))

        net = _TNet()
        net = _train_pytorch_model(net, X_train, y_train,
                                    X_train[-len(X_train)//5:], y_train[-len(y_train)//5:],
                                    lr=lr, weight_decay=wd, batch_size=batch_size, epochs=epochs)
        model = _TransformerClassifier(net, n_features)

    else:
        raise ValueError(f'Unknown brain type: {brain}')

    return model


# ═══════════════════════════════════════════════════════════════════════
# Main Training Loop
# ═══════════════════════════════════════════════════════════════════════

def run_retraining(pairs=None, tfs=None, brains=None, resume=False):
    pairs = pairs or ALL_PAIRS
    tfs = tfs or TRADE_TFS
    brains = brains or BRAINS_TO_RETRAIN

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Checkpoint tracking
    checkpoint_file = CHECKPOINT_DIR / 'progress.json'
    completed = set()
    if resume and checkpoint_file.exists():
        with open(checkpoint_file) as f:
            completed = set(json.load(f).get('completed', []))
        print(f'Resuming: {len(completed)} models already completed')

    total_jobs = len(pairs) * len(tfs) * len(brains)
    total_eligible = 0
    total_failed = 0
    results = []
    job_idx = 0
    t_total = time.time()

    # Estimate time
    est_minutes = sum(BRAIN_TIMES.get(b, 10) for b in brains) * len(pairs) * len(tfs)
    print(f'Retraining plan: {len(pairs)} pairs x {len(tfs)} TFs x {len(brains)} brains = {total_jobs} models')
    print(f'Estimated time: {est_minutes/60:.1f} hours ({est_minutes} minutes)')
    print(f'Output: {OUTPUT_DIR}')
    print(f'MIN_TRADE_RATIO: {MIN_TRADE_RATIO} (was 0.10)')
    print(f'Optuna trials: {OPTUNA_TRIALS}')
    print()

    for pair in pairs:
        for tf in tfs:
            # Load data once per block
            try:
                X_train, y_train, X_val, y_val, returns_val, n_features = load_data(pair, tf)
            except Exception as e:
                print(f'SKIP {pair}_{tf}: {e}')
                continue

            print(f'\n{"="*70}')
            print(f'{pair}_{tf}: {len(X_train):,} train, {len(X_val):,} val, {n_features} features')
            print(f'{"="*70}')

            for brain in brains:
                job_idx += 1
                key = f'{pair}_{tf}_{brain}'

                if key in completed:
                    print(f'  [{job_idx}/{total_jobs}] {brain:<25} SKIP (already completed)')
                    continue

                if brain not in OBJECTIVE_MAP:
                    print(f'  [{job_idx}/{total_jobs}] {brain:<25} SKIP (no objective function)')
                    continue

                t0 = time.time()
                print(f'  [{job_idx}/{total_jobs}] {brain:<25}', end=' ', flush=True)

                try:
                    # Run Optuna
                    obj_fn = OBJECTIVE_MAP[brain](X_train, y_train, X_val, y_val, returns_val)
                    study = optuna.create_study(direction='maximize',
                                                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
                    study.optimize(obj_fn, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

                    best_pf = study.best_value
                    best_params = study.best_params

                    # Train final model with best params
                    model = train_final_model(brain, best_params, X_train, y_train)

                    # Eligibility gate
                    elig = check_eligibility(model, X_val, y_val, returns_val, brain)
                    elapsed = time.time() - t0

                    if elig['eligible']:
                        # Save model (joblib)
                        save_path = OUTPUT_DIR / f'{pair}_{tf}_{brain}.joblib'
                        joblib.dump({'model': model, 'params': best_params,
                                     'brain': brain, 'pair': pair, 'tf': tf,
                                     'eligibility': elig, 'best_pf': best_pf}, str(save_path))

                        # ONNX export for neural net models
                        if hasattr(model, 'net'):
                            try:
                                import torch
                                onnx_path = OUTPUT_DIR / f'{pair}_{tf}_{brain}.onnx'
                                dummy = torch.randn(1, n_features, dtype=torch.float32)
                                model.net.eval()
                                torch.onnx.export(
                                    model.net, dummy, str(onnx_path),
                                    input_names=['features'], output_names=['logits'],
                                    dynamic_axes={'features': {0: 'batch'}, 'logits': {0: 'batch'}},
                                    opset_version=17)
                                print(f'       ONNX exported: {onnx_path.name}')
                            except Exception as onnx_err:
                                print(f'       ONNX export failed: {onnx_err}')

                        total_eligible += 1
                        print(f'PASS  PF={best_pf:.2f}  F={elig["flat_pct"]}% L={elig["long_pct"]}% '
                              f'S={elig["short_pct"]}%  H={elig["entropy"]}  T={elig["trades"]}  '
                              f'({elapsed:.0f}s)')
                    else:
                        total_failed += 1
                        print(f'FAIL  PF={best_pf:.2f}  F={elig["flat_pct"]}% L={elig["long_pct"]}% '
                              f'S={elig["short_pct"]}%  H={elig["entropy"]}  T={elig["trades"]}  '
                              f'({elapsed:.0f}s)')

                    results.append({
                        'pair': pair, 'tf': tf, 'brain': brain,
                        'best_pf': round(best_pf, 3), **elig, 'elapsed_sec': round(elapsed, 1),
                    })

                except Exception as e:
                    elapsed = time.time() - t0
                    total_failed += 1
                    print(f'ERROR ({elapsed:.0f}s): {str(e)[:60]}')
                    results.append({
                        'pair': pair, 'tf': tf, 'brain': brain,
                        'error': str(e)[:100], 'elapsed_sec': round(elapsed, 1),
                    })

                # Update checkpoint
                completed.add(key)
                with open(checkpoint_file, 'w') as f:
                    json.dump({'completed': list(completed),
                               'timestamp': datetime.now().isoformat()}, f)

                gc.collect()

            del X_train, y_train, X_val, y_val, returns_val
            gc.collect()

    total_time = time.time() - t_total

    # Save results
    results_path = OUTPUT_DIR / 'retrain_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_time_sec': round(total_time),
            'total_jobs': total_jobs,
            'eligible': total_eligible,
            'failed': total_failed,
            'results': results,
        }, f, indent=2, default=str)

    print(f'\n{"="*70}')
    print(f'RETRAINING COMPLETE')
    print(f'  Total time: {total_time/3600:.1f} hours')
    print(f'  Eligible: {total_eligible}/{job_idx}')
    print(f'  Failed: {total_failed}/{job_idx}')
    print(f'  Results: {results_path}')
    print(f'  Models: {OUTPUT_DIR}/')
    print(f'{"="*70}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CHAOS V1.0 — V2 Retraining')
    parser.add_argument('--pairs', nargs='+', default=None)
    parser.add_argument('--tfs', nargs='+', default=None)
    parser.add_argument('--brains', nargs='+', default=None)
    parser.add_argument('--trials', type=int, default=OPTUNA_TRIALS)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    if args.trials != OPTUNA_TRIALS:
        OPTUNA_TRIALS = args.trials
        print(f'Optuna trials set to {OPTUNA_TRIALS}')

    run_retraining(pairs=args.pairs, tfs=args.tfs, brains=args.brains, resume=args.resume)
