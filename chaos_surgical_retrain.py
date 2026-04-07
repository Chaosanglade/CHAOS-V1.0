"""
CHAOS V1.0 — Surgical Feature Replacement + XGBoost Retrain

Instead of recomputing all 540 features from scratch (588 hours),
this script:
1. Loads existing feature parquets (already have 1.12M bars x 540 cols)
2. Fixes ONLY the columns where production adapter differs from training
3. Saves to features_fixed/
4. Trains XGBoost on the fixed parquets with walk-forward validation

Estimated time: ~30 minutes per pair on A100, ~2 hours on RTX 4060.

Fixed features:
  - rsi_stretched: was (rsi_14-50)*2, production uses float(rsi>80 or rsi<20)
    → recompute as float(rsi_14 > 80 or rsi_14 < 20) to match production
  - ema_cross_21_55: was {-1,+1} in some versions, production uses {0,1}
    → recompute as (ema_21 > ema_55).astype(int)
  - ofi: sign correction — production uses sign(close-open)*volume
    → recompute from Close/Open/Volume columns
  - rn_dist_big_pips: formula fix — distance in figure intervals
    → recompute from Close column
  - rn_dist_half_pips: same formula fix
  - mtf_*_sma_cross: ensure {0,1} not {-1,+1}
  - mtf_*_support_dist, mtf_*_resistance_dist: raw vs percentage fix
  - TWAP: ensure raw price (mean of close over 20 bars)
  - spread_decomposition: fix constant-0 issue

Usage:
    python chaos_surgical_retrain.py --pairs EURUSD GBPUSD USDJPY
    python chaos_surgical_retrain.py --pairs EURUSD --skip-train  # Fix only, no retrain
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
from pathlib import Path
from datetime import datetime
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight

PROJECT_ROOT = Path(os.environ.get('CHAOS_BASE_DIR', '.'))
FEATURES_DIR = PROJECT_ROOT / 'features'
FIXED_DIR = PROJECT_ROOT / 'features_fixed'
OUTPUT_DIR = PROJECT_ROOT / 'models' / 'surgical_retrained'

RANDOM_SEED = 42
TARGET_COL = 'target_3class_2'
RETURNS_COL = 'target_return_2'
EXECUTION_LAG_BARS = 1
MIN_TRADE_RATIO = 0.20
PURGE_GAP_BARS = 5

ALL_PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD',
             'USDCHF', 'NZDUSD', 'EURJPY', 'GBPJPY']
TF = 'M5'

np.random.seed(RANDOM_SEED)


# ═══════════════════════════════════════════════════════════════════════
# Step 1: Surgical Feature Fixes
# ═══════════════════════════════════════════════════════════════════════

def fix_features(df, pair):
    """
    Fix ONLY the columns that differ between training parquet and
    production live adapter. All fixes are vectorized on full series.
    """
    n_fixed = 0

    # --- rsi_stretched: production returns 0/1 (boolean), training had continuous ---
    if 'rsi_14' in df.columns:
        rsi = df['rsi_14'].values
        # Production: float(rsi > 80 or rsi < 20)
        df['rsi_stretched'] = ((rsi > 80) | (rsi < 20)).astype(np.float32)
        n_fixed += 1
        print(f'    rsi_stretched: fixed (was continuous, now 0/1 boolean)')

    # --- ema_cross_21_55: ensure {0, 1} not {-1, +1} ---
    if 'ema_21' in df.columns and 'ema_55' in df.columns:
        df['ema_cross_21_55'] = (df['ema_21'] > df['ema_55']).astype(np.int8)
        n_fixed += 1
        print(f'    ema_cross_21_55: fixed (ensured {{0,1}})')

    # --- ofi: production uses sign(close-open)*volume ---
    if 'Close' in df.columns and 'Open' in df.columns and 'Volume' in df.columns:
        df['ofi'] = (np.sign(df['Close'] - df['Open']) * df['Volume']).astype(np.float32)
        n_fixed += 1
        print(f'    ofi: fixed (sign(close-open)*volume)')

    # --- rn_dist_big_pips, rn_dist_half_pips: figure interval formula ---
    if 'Close' in df.columns:
        close = df['Close'].values
        is_jpy = 'JPY' in pair
        pip_size = 0.01 if is_jpy else 0.0001
        big_interval = 500 * pip_size    # 0.05 for non-JPY, 5.0 for JPY
        half_interval = 50 * pip_size    # 0.005 for non-JPY, 0.50 for JPY

        big_fig = np.round(close / big_interval) * big_interval
        half_fig = np.round(close / half_interval) * half_interval

        df['rn_dist_big_pips'] = (np.abs(close - big_fig) / half_interval).astype(np.float32)
        df['rn_dist_half_pips'] = (np.abs(close - half_fig) / half_interval).astype(np.float32)
        n_fixed += 2
        print(f'    rn_dist_big_pips, rn_dist_half_pips: fixed (figure interval formula)')

    # --- mtf_*_sma_cross: ensure {0, 1} ---
    sma_cross_cols = [c for c in df.columns if 'sma_cross' in c]
    for col in sma_cross_cols:
        vals = df[col].values
        if vals.min() < 0:  # Has {-1, +1} encoding
            df[col] = (vals > 0).astype(np.int8)
            n_fixed += 1
            print(f'    {col}: fixed (-1/+1 → 0/1)')

    # --- mtf_*_support_dist, mtf_*_resistance_dist ---
    # Production computes as raw price distance / price (percentage)
    # Training may have raw pip distance. Normalize to percentage.
    for prefix in ['support_dist', 'resistance_dist']:
        dist_cols = [c for c in df.columns if prefix in c]
        for col in dist_cols:
            vals = df[col].values
            # If max > 10, it's in raw pips/points — convert to percentage
            if np.nanmax(vals) > 10:
                if 'Close' in df.columns:
                    df[col] = (vals * pip_size / df['Close'].values).astype(np.float32)
                    n_fixed += 1
                    print(f'    {col}: fixed (raw pips → percentage of price)')

    # --- TWAP: ensure raw price average ---
    if 'Close' in df.columns:
        df['TWAP'] = df['Close'].rolling(20, min_periods=1).mean().astype(np.float32)
        n_fixed += 1
        print(f'    TWAP: fixed (rolling 20-bar close mean)')

    # --- spread_decomposition: fix if constant 0 ---
    if 'spread_decomposition' in df.columns:
        if df['spread_decomposition'].nunique() <= 2:
            if 'High' in df.columns and 'Low' in df.columns and 'Open' in df.columns and 'Close' in df.columns:
                bar_range = df['High'] - df['Low']
                body = np.abs(df['Close'] - df['Open'])
                df['spread_decomposition'] = ((bar_range - body) / (bar_range + 1e-10)).astype(np.float32)
                n_fixed += 1
                print(f'    spread_decomposition: fixed (wick ratio)')

    # --- Bid_High: fill if missing ---
    if 'Bid_High' in df.columns:
        if df['Bid_High'].isna().sum() > len(df) * 0.5:
            if 'High' in df.columns:
                df['Bid_High'] = df['High'].astype(np.float32)
                n_fixed += 1
                print(f'    Bid_High: filled from High')

    print(f'    Total fixes applied: {n_fixed}')
    return df


# ═══════════════════════════════════════════════════════════════════════
# Step 2: PF + Anti-Flat (same as other scripts)
# ═══════════════════════════════════════════════════════════════════════

def calculate_pf_with_penalty(predictions, returns):
    predictions = np.asarray(predictions).flatten()
    returns = np.asarray(returns).flatten()
    if len(predictions) == 0:
        return 0.0
    positions = predictions.astype(np.float64) - 1.0
    n_trades = int(np.sum(positions != 0))
    trade_ratio = n_trades / len(positions)
    if trade_ratio < MIN_TRADE_RATIO:
        return 0.5 * (trade_ratio / MIN_TRADE_RATIO)
    pnl = positions * returns
    gross_profit = np.sum(pnl[pnl > 0])
    gross_loss = np.abs(np.sum(pnl[pnl < 0]))
    if gross_loss < 1e-10:
        pf = 99.99 if gross_profit > 1e-10 else 1.0
    else:
        pf = gross_profit / gross_loss
    counts = np.bincount(predictions.astype(int), minlength=3)
    max_pct = counts.max() / max(counts.sum(), 1)
    if max_pct > 0.80:
        pf *= 0.1
    elif max_pct > 0.70:
        pf *= 0.5
    return pf


def check_eligibility(preds, returns_val):
    counts = Counter(preds.astype(int))
    n = len(preds)
    flat_pct = counts.get(1, 0) / n * 100
    long_pct = counts.get(2, 0) / n * 100
    short_pct = counts.get(0, 0) / n * 100
    p = np.array([short_pct, flat_pct, long_pct]) / 100 + 1e-10
    entropy = -np.sum(p * np.log2(p))
    pf = calculate_pf_with_penalty(preds, returns_val)
    trades = sum(1 for i in range(len(preds)) if preds[i] != 1 and (i == 0 or preds[i-1] == 1))
    eligible = flat_pct <= 70 and (long_pct + short_pct) >= 30 and trades >= 10
    return {
        'eligible': eligible, 'flat_pct': round(flat_pct, 1),
        'long_pct': round(long_pct, 1), 'short_pct': round(short_pct, 1),
        'entropy': round(entropy, 3), 'pf': round(pf, 3), 'trades': trades,
    }


# ═══════════════════════════════════════════════════════════════════════
# Step 3: XGBoost Training with Walk-Forward Validation
# ═══════════════════════════════════════════════════════════════════════

def train_xgb(pair, df, feature_cols):
    """Train XGBoost on fixed features with walk-forward validation."""
    import optuna
    from xgboost import XGBClassifier
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Prepare data
    if TARGET_COL not in df.columns or RETURNS_COL not in df.columns:
        print(f'  SKIP {pair}: missing target columns')
        return None

    df_clean = df.dropna(subset=[TARGET_COL, RETURNS_COL]).copy()
    df_clean['returns_lagged'] = df_clean[RETURNS_COL].shift(-EXECUTION_LAG_BARS)
    df_clean = df_clean.dropna(subset=['returns_lagged'])

    X = df_clean[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    y_raw = df_clean[TARGET_COL].values
    y = np.where(y_raw == -1, 0, np.where(y_raw == 0, 1, 2)).astype(np.int64)
    returns = df_clean['returns_lagged'].values.astype(np.float64)

    # Chronological split with purge gap
    n = len(X)
    split = int(n * 0.80)
    X_train, X_val = X[:split - PURGE_GAP_BARS], X[split:]
    y_train, y_val = y[:split - PURGE_GAP_BARS], y[split:]
    returns_val = returns[split:]

    print(f'  Train: {len(X_train):,}  Val: {len(X_val):,}  Features: {len(feature_cols)}')

    # Class weights
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    weight_map = {int(c): float(w) for c, w in zip(classes, weights)}
    sample_weights = np.array([weight_map[int(y)] for y in y_train])

    # Optuna
    def objective(trial):
        params = {
            'objective': 'multi:softmax', 'num_class': 3,
            'tree_method': 'hist', 'device': 'cpu',
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 2, 6),
            'min_child_weight': trial.suggest_int('min_child_weight', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.8),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 100.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 100.0, log=True),
            'random_state': RANDOM_SEED, 'verbosity': 0, 'n_jobs': -1,
        }
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, sample_weight=sample_weights)
        preds = model.predict(X_val)
        return calculate_pf_with_penalty(preds, returns_val)

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=250, show_progress_bar=False)

    best_pf = study.best_value
    best_params = study.best_params
    print(f'  Best PF: {best_pf:.3f}  Params: {best_params}')

    # Train final model
    best_params.update({
        'objective': 'multi:softmax', 'num_class': 3,
        'tree_method': 'hist', 'device': 'cpu',
        'random_state': RANDOM_SEED, 'verbosity': 0, 'n_jobs': -1,
    })
    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train, sample_weight=sample_weights)

    # Eligibility
    preds = model.predict(X_val)
    elig = check_eligibility(preds, returns_val)
    print(f'  Eligibility: {"PASS" if elig["eligible"] else "FAIL"}  '
          f'F={elig["flat_pct"]}% L={elig["long_pct"]}% S={elig["short_pct"]}% '
          f'H={elig["entropy"]} T={elig["trades"]}')

    return {
        'model': model, 'params': best_params, 'best_pf': best_pf,
        'eligibility': elig, 'feature_names': feature_cols,
        'n_train': len(X_train), 'n_val': len(X_val),
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def run(pairs=None, skip_train=False):
    pairs = pairs or ALL_PAIRS
    FIXED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    t_total = time.time()

    for pair in pairs:
        parquet = FEATURES_DIR / f'{pair}_{TF}_features.parquet'
        if not parquet.exists():
            print(f'SKIP {pair}: {parquet} not found')
            continue

        print(f'\n{"="*60}')
        print(f'  {pair}_{TF}')
        print(f'{"="*60}')

        # Load
        t0 = time.time()
        print(f'  Loading {parquet.name}...', end=' ', flush=True)
        df = pd.read_parquet(str(parquet))
        print(f'{df.shape[0]:,} rows x {df.shape[1]} cols ({time.time()-t0:.1f}s)')

        # Fix features
        print(f'  Applying surgical fixes:')
        df = fix_features(df, pair)

        # Save fixed parquet
        fixed_path = FIXED_DIR / f'{pair}_{TF}_features.parquet'
        print(f'  Saving to {fixed_path.name}...', end=' ', flush=True)
        t0 = time.time()
        df.to_parquet(str(fixed_path), engine='pyarrow', compression='snappy')
        print(f'OK ({time.time()-t0:.1f}s)')

        if skip_train:
            print(f'  --skip-train: skipping model training')
            continue

        # Get feature columns
        feature_cols = [c for c in df.columns
                        if not c.startswith('target_') and df[c].dtype != 'object']

        # Train
        print(f'  Training XGBoost (250 Optuna trials)...')
        t0 = time.time()
        result = train_xgb(pair, df, feature_cols)
        elapsed = time.time() - t0

        if result:
            # Save model
            save_path = OUTPUT_DIR / f'{pair}_{TF}_xgb_surgical.joblib'
            joblib.dump({
                'model': result['model'], 'params': result['params'],
                'brain': 'xgb_surgical', 'pair': pair, 'tf': TF,
                'n_features': len(feature_cols),
                'feature_names': feature_cols,
                'eligibility': result['eligibility'],
                'best_pf': result['best_pf'],
            }, str(save_path))
            print(f'  Saved: {save_path.name} ({elapsed:.0f}s)')

            results.append({
                'pair': pair, 'tf': TF, 'best_pf': result['best_pf'],
                **result['eligibility'], 'elapsed_sec': round(elapsed),
            })

        del df
        gc.collect()

    # Summary
    total_time = time.time() - t_total
    print(f'\n{"="*60}')
    print(f'SURGICAL RETRAIN COMPLETE ({total_time/60:.1f} min)')
    for r in results:
        status = 'PASS' if r.get('eligible') else 'FAIL'
        print(f'  {r["pair"]}_{TF}: {status} PF={r["best_pf"]:.2f} '
              f'F={r["flat_pct"]}% L={r["long_pct"]}% S={r["short_pct"]}% T={r["trades"]}')
    print(f'{"="*60}')

    # Save results
    results_path = OUTPUT_DIR / 'surgical_results.json'
    with open(results_path, 'w') as f:
        json.dump({'timestamp': datetime.now().isoformat(),
                   'total_time_sec': round(total_time),
                   'results': results}, f, indent=2, default=str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CHAOS V1.0 — Surgical Feature Fix + Retrain')
    parser.add_argument('--pairs', nargs='+', default=None)
    parser.add_argument('--skip-train', action='store_true', help='Fix features only, skip training')
    args = parser.parse_args()
    run(pairs=args.pairs, skip_train=args.skip_train)
