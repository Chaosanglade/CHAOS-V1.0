"""
CHAOS V1.0 — M5 Tree Model Training (Phase 2 Step 1)

Trains LGB, XGB, RF, ET on M5 features for all 9 pairs.
Uses 368 features per pair (all non-target/OHLCV columns from parquet).
Same anti-flat pipeline as v2 retraining (MIN_TRADE_RATIO 0.20,
class penalty, eligibility gate).

Target: target_3class_2 (2-bar / 10-minute forward return) encoded
{-1=SHORT, 0=FLAT, 1=LONG} → remapped to {0, 1, 2} for sklearn.

Usage:
    python chaos_m5_train.py                          # All 9 pairs x 4 brains
    python chaos_m5_train.py --pairs EURUSD GBPUSD    # Specific pairs
    python chaos_m5_train.py --brains lgb xgb         # Specific brains
    python chaos_m5_train.py --resume                  # Resume from checkpoint
    python chaos_m5_train.py --trials 50               # Quick test
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

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Config ──
PROJECT_ROOT = Path(os.environ.get('CHAOS_BASE_DIR', '.'))
FEATURES_DIR = PROJECT_ROOT / 'features'
OUTPUT_DIR = PROJECT_ROOT / 'models' / 'm5_retrained'
CHECKPOINT_DIR = OUTPUT_DIR / 'checkpoints'
SCHEMA_DIR = PROJECT_ROOT / 'schema'

RANDOM_SEED = 42
TARGET_COL = 'target_3class_2'     # 2-bar (10 min) forward return class
RETURNS_COL = 'target_return_2'    # Matching return for PF calculation
EXECUTION_LAG_BARS = 1             # 1 bar lag on returns
MIN_TRADE_RATIO = 0.20             # Minimum 20% non-FLAT predictions
OPTUNA_TRIALS = 250
PURGE_GAP_BARS = 5                 # Purged gap between train/val to prevent leakage

ALL_PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD',
             'USDCHF', 'NZDUSD', 'EURJPY', 'GBPJPY']
TF = 'M5'

BRAINS = ['lgb', 'xgb', 'rf', 'et']

EXCLUDE_SUBS = ['target_', 'return', 'Open', 'High', 'Low', 'Close', 'Volume',
                'timestamp', 'date', 'pair', 'symbol', 'tf', 'bar_time',
                'Bid_', 'Ask_', 'Spread_', 'Unnamed']

np.random.seed(RANDOM_SEED)

# Estimated time per brain per pair (minutes on RTX 4060 / 8-core CPU)
BRAIN_TIMES = {'lgb': 15, 'xgb': 20, 'rf': 8, 'et': 8}


# ═══════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════

def load_data(pair):
    """Load M5 features with purged chronological split."""
    parquet = FEATURES_DIR / f'{pair}_{TF}_features.parquet'
    if not parquet.exists():
        raise FileNotFoundError(f'{parquet}')

    print(f'  Loading {parquet.name}...', end=' ', flush=True)
    t0 = time.time()
    df = pd.read_parquet(str(parquet))
    print(f'{df.shape[0]:,} rows x {df.shape[1]} cols ({time.time()-t0:.1f}s)')

    feature_cols = [c for c in df.columns if not any(ex in c for ex in EXCLUDE_SUBS)]

    if TARGET_COL not in df.columns:
        raise ValueError(f'Missing {TARGET_COL} in {pair}_{TF}')
    if RETURNS_COL not in df.columns:
        raise ValueError(f'Missing {RETURNS_COL} in {pair}_{TF}')

    # Apply execution lag on returns
    df['returns_lagged'] = df[RETURNS_COL].shift(-EXECUTION_LAG_BARS)
    df = df.dropna(subset=['returns_lagged', TARGET_COL])

    X = df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    # Remap target: {-1, 0, 1} → {0, 1, 2} (SHORT=0, FLAT=1, LONG=2)
    y_raw = df[TARGET_COL].values
    y = np.where(y_raw == -1, 0, np.where(y_raw == 0, 1, 2)).astype(np.int64)

    returns = df['returns_lagged'].values.astype(np.float64)
    returns = np.nan_to_num(returns, nan=0, posinf=0, neginf=0)

    # Chronological split: 80% train, purge gap, 20% val
    n = len(X)
    split = int(n * 0.80)
    X_train = X[:split - PURGE_GAP_BARS]
    y_train = y[:split - PURGE_GAP_BARS]
    X_val = X[split:]
    y_val = y[split:]
    returns_val = returns[split:]

    # Class distribution
    dist = Counter(y_train)
    total = len(y_train)
    print(f'  Train: {len(X_train):,} rows, Val: {len(X_val):,} rows, Features: {len(feature_cols)}')
    print(f'  Classes: SHORT={dist[0]/total*100:.1f}% FLAT={dist[1]/total*100:.1f}% LONG={dist[2]/total*100:.1f}%')

    return X_train, y_train, X_val, y_val, returns_val, feature_cols


# ═══════════════════════════════════════════════════════════════════════
# PF + Anti-Flat Penalty (same as v2)
# ═══════════════════════════════════════════════════════════════════════

def calculate_pf_with_penalty(predictions, returns):
    predictions = np.asarray(predictions).flatten()
    returns = np.asarray(returns).flatten()
    if len(predictions) == 0:
        return 0.0

    positions = predictions.astype(np.float64) - 1.0  # 0→-1, 1→0, 2→+1
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
# Optuna Objectives — v2 anti-flat ranges for M5
# ═══════════════════════════════════════════════════════════════════════

def create_lgb_objective(X_train, y_train, X_val, returns_val):
    from lightgbm import LGBMClassifier

    def objective(trial):
        params = {
            'objective': 'multiclass', 'num_class': 3, 'boosting_type': 'gbdt',
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'num_leaves': trial.suggest_int('num_leaves', 10, 60),
            'min_child_samples': trial.suggest_int('min_child_samples', 50, 500),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.8),
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


def create_xgb_objective(X_train, y_train, X_val, returns_val):
    from xgboost import XGBClassifier
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
    return objective


def create_rf_objective(X_train, y_train, X_val, returns_val):
    from sklearn.ensemble import RandomForestClassifier

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 600),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 20, 100),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 20, 100),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3]),
            'class_weight': 'balanced',
            'n_jobs': -1, 'random_state': RANDOM_SEED,
        }
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return calculate_pf_with_penalty(preds, returns_val)
    return objective


def create_et_objective(X_train, y_train, X_val, returns_val):
    from sklearn.ensemble import ExtraTreesClassifier

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 600),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 20, 100),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 20, 100),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3]),
            'class_weight': 'balanced',
            'n_jobs': -1, 'random_state': RANDOM_SEED,
        }
        model = ExtraTreesClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return calculate_pf_with_penalty(preds, returns_val)
    return objective


OBJECTIVE_MAP = {
    'lgb': create_lgb_objective,
    'xgb': create_xgb_objective,
    'rf': create_rf_objective,
    'et': create_et_objective,
}


# ═══════════════════════════════════════════════════════════════════════
# Final Model Training
# ═══════════════════════════════════════════════════════════════════════

def train_final_model(brain, best_params, X_train, y_train):
    p = {k: v for k, v in best_params.items()}

    if brain == 'lgb':
        from lightgbm import LGBMClassifier
        p.update({'objective': 'multiclass', 'num_class': 3, 'boosting_type': 'gbdt',
                  'class_weight': 'balanced', 'random_state': RANDOM_SEED,
                  'verbosity': -1, 'n_jobs': -1})
        model = LGBMClassifier(**p)
        model.fit(X_train, y_train)

    elif brain == 'xgb':
        from xgboost import XGBClassifier
        p.update({'objective': 'multi:softmax', 'num_class': 3, 'tree_method': 'hist',
                  'device': 'cpu', 'random_state': RANDOM_SEED, 'verbosity': 0, 'n_jobs': -1})
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        wm = {int(c): float(w) for c, w in zip(classes, weights)}
        sw = np.array([wm[int(y)] for y in y_train])
        model = XGBClassifier(**p)
        model.fit(X_train, y_train, sample_weight=sw)

    elif brain == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        p.update({'class_weight': 'balanced', 'n_jobs': -1, 'random_state': RANDOM_SEED})
        model = RandomForestClassifier(**p)
        model.fit(X_train, y_train)

    elif brain == 'et':
        from sklearn.ensemble import ExtraTreesClassifier
        p.update({'class_weight': 'balanced', 'n_jobs': -1, 'random_state': RANDOM_SEED})
        model = ExtraTreesClassifier(**p)
        model.fit(X_train, y_train)

    else:
        raise ValueError(f'Unknown brain: {brain}')

    return model


def export_onnx(model, brain, n_features, save_path):
    """Export tree model to ONNX."""
    try:
        if brain == 'lgb':
            from onnxmltools import convert_lightgbm
            from onnxmltools.utils import FloatTensorType
            onnx_model = convert_lightgbm(model, initial_types=[('features', FloatTensorType([None, n_features]))])
            import onnx
            onnx.save_model(onnx_model, str(save_path))
        elif brain == 'xgb':
            from onnxmltools import convert_xgboost
            from onnxmltools.utils import FloatTensorType
            onnx_model = convert_xgboost(model, initial_types=[('features', FloatTensorType([None, n_features]))])
            import onnx
            onnx.save_model(onnx_model, str(save_path))
        else:
            # RF/ET: skl2onnx
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            onnx_model = convert_sklearn(model, initial_types=[('features', FloatTensorType([None, n_features]))])
            import onnx
            onnx.save_model(onnx_model, str(save_path))
        return True
    except Exception as e:
        print(f'       ONNX export failed: {e}')
        return False


# ═══════════════════════════════════════════════════════════════════════
# Feature Schema Export
# ═══════════════════════════════════════════════════════════════════════

def save_feature_schema(feature_cols, pair):
    """Save M5 feature schema if not already saved."""
    schema_path = SCHEMA_DIR / 'm5_feature_schema.json'
    col_order_path = SCHEMA_DIR / 'feature_columns_by_pair_tf.json'

    # Feature schema
    if not schema_path.exists():
        schema = {
            'timeframe': 'M5',
            'n_features': len(feature_cols),
            'features': [{'index': i, 'name': name} for i, name in enumerate(feature_cols)],
        }
        with open(schema_path, 'w') as f:
            json.dump(schema, f, indent=2)
        print(f'  Saved M5 feature schema: {len(feature_cols)} features')

    # Column order per pair+TF
    if col_order_path.exists():
        with open(col_order_path) as f:
            orders = json.load(f)
    else:
        orders = {}
    orders[f'{pair}_M5'] = feature_cols
    with open(col_order_path, 'w') as f:
        json.dump(orders, f)


# ═══════════════════════════════════════════════════════════════════════
# Main Training Loop
# ═══════════════════════════════════════════════════════════════════════

def run_training(pairs=None, brains=None, trials=None, resume=False):
    pairs = pairs or ALL_PAIRS
    brains = brains or BRAINS
    trials = trials or OPTUNA_TRIALS

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    SCHEMA_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_file = CHECKPOINT_DIR / 'progress.json'
    completed = set()
    if resume and checkpoint_file.exists():
        with open(checkpoint_file) as f:
            completed = set(json.load(f).get('completed', []))
        print(f'Resuming: {len(completed)} models already completed')

    total_jobs = len(pairs) * len(brains)
    total_eligible = 0
    total_failed = 0
    results = []
    job_idx = 0
    t_total = time.time()

    est_minutes = sum(BRAIN_TIMES.get(b, 15) for b in brains) * len(pairs)
    print(f'M5 Training: {len(pairs)} pairs x {len(brains)} brains = {total_jobs} models')
    print(f'Estimated time: {est_minutes/60:.1f} hours')
    print(f'Target: {TARGET_COL}, Returns: {RETURNS_COL}')
    print(f'MIN_TRADE_RATIO: {MIN_TRADE_RATIO}, Optuna trials: {trials}')
    print(f'Output: {OUTPUT_DIR}')
    print()

    for pair in pairs:
        try:
            X_train, y_train, X_val, y_val, returns_val, feature_cols = load_data(pair)
        except Exception as e:
            print(f'SKIP {pair}: {e}')
            continue

        # Save feature schema + column order
        save_feature_schema(feature_cols, pair)

        for brain in brains:
            job_idx += 1
            key = f'{pair}_M5_{brain}'

            if key in completed:
                print(f'  [{job_idx}/{total_jobs}] {brain:<6} SKIP (already completed)')
                continue

            t0 = time.time()
            print(f'  [{job_idx}/{total_jobs}] {brain:<6}', end=' ', flush=True)

            try:
                obj_fn = OBJECTIVE_MAP[brain](X_train, y_train, X_val, returns_val)
                study = optuna.create_study(direction='maximize',
                                            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
                study.optimize(obj_fn, n_trials=trials, show_progress_bar=False)

                best_pf = study.best_value
                best_params = study.best_params

                # Train final model
                model = train_final_model(brain, best_params, X_train, y_train)

                # Eligibility check
                preds = model.predict(X_val)
                elig = check_eligibility(preds, returns_val)
                elapsed = time.time() - t0

                # Save model
                save_path = OUTPUT_DIR / f'{pair}_M5_{brain}.joblib'
                joblib.dump({'model': model, 'params': best_params,
                             'brain': brain, 'pair': pair, 'tf': 'M5',
                             'n_features': len(feature_cols),
                             'eligibility': elig, 'best_pf': best_pf}, str(save_path))

                # ONNX export
                onnx_path = OUTPUT_DIR / f'{pair}_M5_{brain}.onnx'
                onnx_ok = export_onnx(model, brain, len(feature_cols), onnx_path)

                if elig['eligible']:
                    total_eligible += 1
                    status = 'PASS'
                else:
                    total_failed += 1
                    status = 'FAIL'

                print(f'{status}  PF={best_pf:.2f}  F={elig["flat_pct"]}% L={elig["long_pct"]}% '
                      f'S={elig["short_pct"]}%  H={elig["entropy"]}  T={elig["trades"]}  '
                      f'ONNX={"OK" if onnx_ok else "FAIL"}  ({elapsed:.0f}s)')

                results.append({
                    'pair': pair, 'tf': 'M5', 'brain': brain,
                    'best_pf': round(best_pf, 3), **elig,
                    'onnx': onnx_ok, 'elapsed_sec': round(elapsed, 1),
                })

            except Exception as e:
                elapsed = time.time() - t0
                total_failed += 1
                print(f'ERROR ({elapsed:.0f}s): {str(e)[:60]}')
                results.append({
                    'pair': pair, 'tf': 'M5', 'brain': brain,
                    'error': str(e)[:100], 'elapsed_sec': round(elapsed, 1),
                })

            # Checkpoint
            completed.add(key)
            with open(checkpoint_file, 'w') as f:
                json.dump({'completed': list(completed),
                           'timestamp': datetime.now().isoformat()}, f)
            gc.collect()

        del X_train, y_train, X_val, y_val, returns_val
        gc.collect()

    total_time = time.time() - t_total

    # Save results
    results_path = OUTPUT_DIR / 'm5_train_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_time_sec': round(total_time),
            'eligible': total_eligible,
            'failed': total_failed,
            'results': results,
        }, f, indent=2, default=str)

    print(f'\n{"="*70}')
    print(f'M5 TRAINING COMPLETE')
    print(f'  Total time: {total_time/3600:.1f} hours')
    print(f'  Eligible: {total_eligible}/{job_idx}')
    print(f'  Failed/Ineligible: {total_failed}/{job_idx}')
    print(f'  Results: {results_path}')
    print(f'  Models: {OUTPUT_DIR}/')
    print(f'{"="*70}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CHAOS V1.0 — M5 Tree Model Training')
    parser.add_argument('--pairs', nargs='+', default=None)
    parser.add_argument('--brains', nargs='+', default=None)
    parser.add_argument('--trials', type=int, default=OPTUNA_TRIALS)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run_training(pairs=args.pairs, brains=args.brains, trials=args.trials, resume=args.resume)
