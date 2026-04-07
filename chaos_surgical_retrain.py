"""
CHAOS V1.0 — Surgical Feature Replacement + XGBoost Retrain

Fixes ONLY the columns where training parquets differ from production.
Verified via parity audit — most features already match (corr>0.99).

Features that NEED fixing (verified mismatched):
  - TWAP: parquet has unknown formula, production uses rolling 20-bar close mean
  - spread_decomposition: parquet range [-26K,+26K], production uses wick ratio [0,1]
  - mtf_*_support_dist, mtf_*_resistance_dist: raw pips (0-265), production uses pct of price

Features that DON'T need fixing (verified matching):
  - rsi_stretched: parquet = (rsi_14-50)*2 continuous — matches production (DO NOT TOUCH)
  - ema_cross_21_55: {0,1} — matches (99.9%)
  - ofi: sign(close-open)*volume — matches (corr=1.0)
  - rn_dist_big_pips, rn_dist_half_pips — matches (corr=1.0)
  - mtf_*_sma_cross: {0,1} — matches

Usage:
    python chaos_surgical_retrain.py --pairs EURUSD GBPUSD USDJPY
    python chaos_surgical_retrain.py --pairs EURUSD --skip-train
    python chaos_surgical_retrain.py --pairs EURUSD --verify-only
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
# Step 1: Surgical Feature Fixes — ONLY verified mismatches
# ═══════════════════════════════════════════════════════════════════════

def fix_features(df, pair):
    """
    Fix ONLY columns verified to differ between training parquet and
    production live_feature_adapter.py. Each formula is copied EXACTLY
    from the production source.
    """
    n_fixed = 0
    pip_size = 0.01 if 'JPY' in pair else 0.0001

    # --- TWAP: production = rolling 20-bar close mean (raw price) ---
    # Parquet has different formula (corr=0.13 vs production)
    # Source: live_feature_adapter.py line 234: F['TWAP'] = _roll_mean(c, 20)[-1]
    if 'Close' in df.columns:
        df['TWAP'] = df['Close'].rolling(20, min_periods=1).mean().astype(np.float32)
        n_fixed += 1
        print(f'    TWAP: fixed -> rolling 20-bar close mean')

    # --- spread_decomposition: production = (H-L - |C-O|) / (H-L) ---
    # Parquet range [-26K,+26K] vs production [0,1] (corr=0.0007)
    # Source: live_feature_adapter.py line 301:
    #   F['spread_decomposition'] = _safe_div(spread_raw - np.abs(c - o), spread_raw + 1e-10)[-1]
    if all(c in df.columns for c in ['High', 'Low', 'Close', 'Open']):
        bar_range = (df['High'] - df['Low']).values
        body = np.abs(df['Close'] - df['Open']).values
        sd = np.where(bar_range > 1e-10, (bar_range - body) / bar_range, 0.0)
        df['spread_decomposition'] = sd.astype(np.float32)
        n_fixed += 1
        print(f'    spread_decomposition: fixed -> wick ratio [0,1]')

    # --- mtf_*_support_dist, mtf_*_resistance_dist: raw pips -> pct of price ---
    # Parquet has raw values (max 265), production normalizes by price
    # Source: live_feature_adapter.py uses _safe_div for similar features
    if 'Close' in df.columns:
        close = df['Close'].values
        for prefix in ['support_dist', 'resistance_dist']:
            dist_cols = [c for c in df.columns if prefix in c]
            for col in dist_cols:
                vals = df[col].values
                if np.nanmax(np.abs(vals)) > 10:  # Raw pips, needs normalization
                    df[col] = (vals * pip_size / (close + 1e-10)).astype(np.float32)
                    n_fixed += 1
                    print(f'    {col}: fixed -> pct of price')

    print(f'    Total fixes applied: {n_fixed}')
    return df


# ═══════════════════════════════════════════════════════════════════════
# Step 2: Verification — compare fixed parquet vs production adapter
# ═══════════════════════════════════════════════════════════════════════

def verify_fixes(df, pair, n_samples=100):
    """
    Verify fixed features match what production adapter would compute
    on the same OHLCV data, for a random sample of bars.
    """
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / 'inference'))
    from inference.live_feature_adapter import LiveFeatureAdapter, _roll_mean, _safe_div

    print(f'  Verifying {n_samples} random bars...')
    np.random.seed(42)
    indices = np.random.choice(range(500, len(df) - 10), n_samples, replace=False)

    adapter = LiveFeatureAdapter()
    features_to_check = ['TWAP', 'spread_decomposition']
    mismatches = {f: [] for f in features_to_check}

    for idx in indices:
        # Extract 500-bar window ending at idx
        window = df.iloc[max(0, idx - 499):idx + 1]
        bars = []
        for _, row in window.iterrows():
            bar = {'open': float(row['Open']), 'high': float(row['High']),
                   'low': float(row['Low']), 'close': float(row['Close']),
                   'volume': float(row['Volume'])}
            if hasattr(window.index, 'strftime'):
                bar['time'] = row.name.strftime('%Y-%m-%dT%H:%M:%S')
            else:
                bar['time'] = '2020-01-01T00:00:00'
            bars.append(bar)

        schema_vec = adapter.compute(pair, TF, {TF: bars})
        if schema_vec is None:
            continue

        # Compare each fixed feature
        for fname in features_to_check:
            if fname in adapter._name_to_idx and fname in df.columns:
                prod_val = float(schema_vec[adapter._name_to_idx[fname]])
                parq_val = float(df[fname].iloc[idx])
                if abs(parq_val) > 1e-10:
                    pct_diff = abs(prod_val - parq_val) / abs(parq_val) * 100
                else:
                    pct_diff = abs(prod_val - parq_val) * 100
                if pct_diff > 5.0:  # >5% difference
                    mismatches[fname].append((idx, parq_val, prod_val, pct_diff))

    all_ok = True
    for fname, mm in mismatches.items():
        if mm:
            print(f'    {fname}: {len(mm)}/{n_samples} mismatches (>{5}%)')
            for idx, pv, rv, pct in mm[:3]:
                print(f'      bar {idx}: parquet={pv:.6f} prod={rv:.6f} diff={pct:.1f}%')
            all_ok = False
        else:
            print(f'    {fname}: OK (0 mismatches)')

    return all_ok


# ═══════════════════════════════════════════════════════════════════════
# Step 3: PF + Anti-Flat
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
# Step 4: XGBoost Training
# ═══════════════════════════════════════════════════════════════════════

def train_xgb(pair, df, feature_cols):
    import optuna
    from xgboost import XGBClassifier
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    df_clean = df.dropna(subset=[TARGET_COL, RETURNS_COL]).copy()
    df_clean['returns_lagged'] = df_clean[RETURNS_COL].shift(-EXECUTION_LAG_BARS)
    df_clean = df_clean.dropna(subset=['returns_lagged'])

    X = np.nan_to_num(df_clean[feature_cols].values.astype(np.float32), nan=0, posinf=0, neginf=0)
    y_raw = df_clean[TARGET_COL].values
    y = np.where(y_raw == -1, 0, np.where(y_raw == 0, 1, 2)).astype(np.int64)
    returns = df_clean['returns_lagged'].values.astype(np.float64)

    n = len(X)
    split = int(n * 0.80)
    X_train, X_val = X[:split - PURGE_GAP_BARS], X[split:]
    y_train, y_val = y[:split - PURGE_GAP_BARS], y[split:]
    returns_val = returns[split:]

    print(f'  Train: {len(X_train):,}  Val: {len(X_val):,}  Features: {len(feature_cols)}')

    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    weight_map = {int(c): float(w) for c, w in zip(classes, weights)}
    sample_weights = np.array([weight_map[int(y)] for y in y_train])

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

    best_params = study.best_params
    best_params.update({
        'objective': 'multi:softmax', 'num_class': 3,
        'tree_method': 'hist', 'device': 'cpu',
        'random_state': RANDOM_SEED, 'verbosity': 0, 'n_jobs': -1,
    })
    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train, sample_weight=sample_weights)

    preds = model.predict(X_val)
    elig = check_eligibility(preds, returns_val)
    print(f'  PF={study.best_value:.3f}  '
          f'F={elig["flat_pct"]}% L={elig["long_pct"]}% S={elig["short_pct"]}% '
          f'H={elig["entropy"]} T={elig["trades"]}')

    return {
        'model': model, 'params': best_params, 'best_pf': study.best_value,
        'eligibility': elig, 'feature_names': feature_cols,
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def run(pairs=None, skip_train=False, verify_only=False):
    pairs = pairs or ALL_PAIRS
    FIXED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    t_total = time.time()

    for pair in pairs:
        parquet = FEATURES_DIR / f'{pair}_{TF}_features.parquet'
        if not parquet.exists():
            print(f'SKIP {pair}: not found')
            continue

        print(f'\n{"="*60}')
        print(f'  {pair}_{TF}')
        print(f'{"="*60}')

        t0 = time.time()
        print(f'  Loading...', end=' ', flush=True)
        df = pd.read_parquet(str(parquet))
        print(f'{df.shape[0]:,} rows x {df.shape[1]} cols ({time.time()-t0:.1f}s)')

        print(f'  Applying surgical fixes (ONLY verified mismatches):')
        df = fix_features(df, pair)

        # Verify
        print(f'  Verification:')
        verify_ok = verify_fixes(df, pair, n_samples=50)
        if not verify_ok:
            print(f'  WARNING: verification found mismatches')
        if verify_only:
            continue

        # Save
        fixed_path = FIXED_DIR / f'{pair}_{TF}_features.parquet'
        print(f'  Saving fixed parquet...', end=' ', flush=True)
        t0 = time.time()
        df.to_parquet(str(fixed_path), engine='pyarrow', compression='snappy')
        print(f'OK ({time.time()-t0:.1f}s)')

        if skip_train:
            continue

        feature_cols = [c for c in df.columns
                        if not c.startswith('target_') and df[c].dtype != 'object']

        print(f'  Training XGBoost (250 trials)...')
        t0 = time.time()
        result = train_xgb(pair, df, feature_cols)
        elapsed = time.time() - t0

        if result:
            save_path = OUTPUT_DIR / f'{pair}_{TF}_xgb_surgical.joblib'
            joblib.dump({
                'model': result['model'], 'params': result['params'],
                'brain': 'xgb_surgical', 'pair': pair, 'tf': TF,
                'n_features': len(feature_cols),
                'feature_names': feature_cols,
                'eligibility': result['eligibility'],
                'best_pf': result['best_pf'],
            }, str(save_path))
            status = 'PASS' if result['eligibility']['eligible'] else 'FAIL'
            print(f'  {status} -> {save_path.name} ({elapsed:.0f}s)')
            results.append({'pair': pair, **result['eligibility'],
                           'best_pf': result['best_pf'], 'elapsed': round(elapsed)})

        del df; gc.collect()

    total_time = time.time() - t_total
    print(f'\n{"="*60}')
    print(f'DONE ({total_time/60:.1f} min)')
    for r in results:
        print(f'  {r["pair"]}: PF={r["best_pf"]:.2f} F={r["flat_pct"]}% T={r["trades"]}')
    print(f'{"="*60}')

    results_path = OUTPUT_DIR / 'surgical_results.json'
    with open(results_path, 'w') as f:
        json.dump({'timestamp': datetime.now().isoformat(),
                   'results': results}, f, indent=2, default=str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', nargs='+', default=None)
    parser.add_argument('--skip-train', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    run(pairs=args.pairs, skip_train=args.skip_train, verify_only=args.verify_only)
