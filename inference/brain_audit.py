"""
CHAOS V1.0 — Brain Eligibility Audit (Colab Team 5-Layer Framework)

Runs on the LOCAL machine where training parquets are available.
Tests every active brain across all production pair+TF combos.

Layer 1: Feature Parity — live adapter vs training parquet distributions
Layer 2: Export Parity — native model vs ONNX predictions
Layer 3: Directional Coverage — class balance and entropy
Layer 4: Trade Viability — signal frequency on validation set

Usage:
    cd "G:/My Drive/chaos_v1.0"
    python -u inference/brain_audit.py [--pairs EURUSD GBPUSD] [--tfs H1 M30]
"""
import os, sys, json, glob, time, argparse
os.environ['ORT_LOG_SEVERITY_LEVEL'] = '3'
os.environ['CHAOS_BASE_DIR'] = os.getcwd()
sys.path.insert(0, '.')
sys.path.insert(0, 'inference')

import numpy as np
import pandas as pd
import joblib
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get('CHAOS_BASE_DIR', os.getcwd()))

try:
    import onnxruntime as ort
    ort.set_default_logger_severity(3)
except ImportError:
    ort = None

# Exclusion substrings for identifying feature columns
EXCLUDE_SUBS = ['target_', 'return', 'Open', 'High', 'Low', 'Close', 'Volume',
                'timestamp', 'date', 'pair', 'symbol', 'tf', 'bar_time',
                'Bid_', 'Ask_', 'Spread_', 'Unnamed']

ALL_PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD',
             'USDCHF', 'NZDUSD', 'EURJPY', 'GBPJPY']
PROD_TFS = ['H1', 'M30']

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def get_feature_cols(df):
    return [c for c in df.columns if not any(ex in c for ex in EXCLUDE_SUBS)]


def load_quarantine():
    with open(PROJECT_ROOT / 'replay' / 'config' / 'brain_quarantine.json') as f:
        cfg = json.load(f)
    q = set(cfg.get('global_quarantine', []))
    cond = cfg.get('conditional_quarantine', {})
    return q, cond


def find_models(pair, tf):
    """Return dict {brain_name: {'onnx': path, 'joblib': path, 'pt': path}}.
    Searches models/v2_retrained/ first (preferred), then models/."""
    models = {}
    search_dirs = [PROJECT_ROOT / 'models' / 'v2_retrained', PROJECT_ROOT / 'models']
    for mdir in search_dirs:
        if not mdir.exists():
            continue
        for ext in ['onnx', 'joblib', 'pt']:
            for path in glob.glob(str(mdir / f'{pair}_{tf}_*.{ext}')):
                basename = os.path.basename(path)
                parts = basename.replace(f'.{ext}', '').split('_')
                brain = '_'.join(parts[2:])
                if brain not in models:
                    models[brain] = {}
                # Prefer v2_retrained over original
                if ext not in models[brain] or 'v2_retrained' in str(mdir):
                    models[brain][ext] = path
    return models


def onnx_predict(session, X):
    """Run ONNX model, return (preds, probs) arrays."""
    inp = session.get_inputs()[0]
    results = session.run(None, {inp.name: X.astype(np.float32)})
    preds = np.array(results[0]).flatten()
    if len(results) >= 2:
        probs_raw = results[1]
        if isinstance(probs_raw, list) and isinstance(probs_raw[0], dict):
            probs = np.array([[p.get(0, 0), p.get(1, 0), p.get(2, 0)] for p in probs_raw])
        else:
            probs = np.array(probs_raw)
    else:
        probs = np.array(results[0])
        if probs.ndim == 2:
            preds = np.argmax(probs, axis=1)
    if probs.ndim == 2 and probs.shape[1] != 3:
        probs = probs[:, :3]
    # Softmax if needed
    if probs.ndim == 2 and not np.allclose(probs.sum(axis=1), 1.0, atol=0.05):
        exp_p = np.exp(probs - probs.max(axis=1, keepdims=True))
        probs = exp_p / exp_p.sum(axis=1, keepdims=True)
    return preds.astype(int), probs


def native_predict(model_path, X):
    """Run native (joblib/pt) model, return (preds, probs)."""
    if model_path.endswith('.joblib'):
        loaded = joblib.load(model_path)
        model = loaded if not isinstance(loaded, dict) else loaded.get('model', loaded.get('estimator', loaded))
        scaler = loaded.get('scaler') if isinstance(loaded, dict) else None
        Xs = scaler.transform(X) if scaler is not None else X
        probs = model.predict_proba(Xs.astype(np.float64))
        preds = np.argmax(probs, axis=1)
        return preds, probs
    elif model_path.endswith('.pt'):
        try:
            import torch
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            # Can't easily reconstruct PyTorch model without architecture info
            return None, None
        except Exception:
            return None, None
    return None, None


# ──────────────────────────────────────────────────────────────────────
# Layer 1: Feature Parity
# ──────────────────────────────────────────────────────────────────────

def test_feature_parity(pair, tf):
    """Compare live adapter output vs training parquet feature distributions."""
    from inference.live_feature_adapter import LiveFeatureAdapter

    parquet = PROJECT_ROOT / 'features' / f'{pair}_{tf}_features.parquet'
    ohlcv_path = PROJECT_ROOT / 'ohlcv_data' / pair / f'{pair}_{tf}.parquet'
    if not parquet.exists() or not ohlcv_path.exists():
        return {'status': 'SKIP', 'reason': 'missing_data'}

    train = pd.read_parquet(str(parquet))
    feat_cols = get_feature_cols(train)
    train_recent = train.tail(5000)

    # Load schema
    with open(PROJECT_ROOT / 'schema' / 'feature_schema.json') as f:
        schema_names = [e['name'] for e in json.load(f)['features']]

    # Compute live features from last 500 OHLCV bars
    ohlcv = pd.read_parquet(str(ohlcv_path))
    if isinstance(ohlcv.index, pd.DatetimeIndex):
        ohlcv['time'] = ohlcv.index.strftime('%Y-%m-%dT%H:%M:%S')
    bars = [{'time': str(r.get('time', '')),
             'open': float(r['Open']), 'high': float(r['High']),
             'low': float(r['Low']), 'close': float(r['Close']),
             'volume': float(r['Volume'])}
            for _, r in ohlcv.tail(500).iterrows()]

    adapter = LiveFeatureAdapter()
    live_vec = adapter.compute(pair, tf, {tf: bars})
    if live_vec is None:
        return {'status': 'FAIL', 'reason': 'adapter_returned_none'}

    # Compare
    ok = severe = oor = 0
    for i, name in enumerate(schema_names):
        if name not in train_recent.columns:
            continue
        col = train_recent[name].replace([np.inf, -np.inf], np.nan).dropna()
        if len(col) < 10:
            continue
        live_val = float(live_vec[i])
        t_mean = col.mean()
        t_std = col.std() + 1e-15
        z = abs(live_val - t_mean) / t_std
        if z > 10:
            severe += 1
        elif col.min() <= live_val <= col.max():
            ok += 1
        else:
            oor += 1

    total = ok + severe + oor
    pct = ok / max(total, 1) * 100
    return {
        'status': 'PASS' if pct >= 85 else 'WARN' if pct >= 70 else 'FAIL',
        'ok': ok, 'severe_z10': severe, 'out_of_range': oor,
        'pct_ok': round(pct, 1)
    }


# ──────────────────────────────────────────────────────────────────────
# Layer 2: Export Parity
# ──────────────────────────────────────────────────────────────────────

def test_export_parity(brain, model_files, X_test):
    """Compare native vs ONNX predictions on the same 100 rows."""
    onnx_path = model_files.get('onnx')
    native_path = model_files.get('joblib') or model_files.get('pt')

    if not onnx_path or not native_path:
        return {'status': 'SKIP', 'reason': 'missing_artifact'}

    if not os.path.exists(onnx_path) or not os.path.exists(native_path):
        return {'status': 'SKIP', 'reason': 'file_not_found'}

    try:
        # ONNX
        sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        n_feat = sess.get_inputs()[0].shape[1] if len(sess.get_inputs()[0].shape) > 1 else X_test.shape[1]
        X = X_test[:, :n_feat].astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        onnx_preds, onnx_probs = onnx_predict(sess, X)
        del sess

        # Native
        native_preds, native_probs = native_predict(native_path, X[:, :n_feat])
        if native_preds is None:
            return {'status': 'SKIP', 'reason': 'native_load_failed'}

        # Compare
        n = min(len(onnx_preds), len(native_preds))
        match_pct = (onnx_preds[:n] == native_preds[:n]).sum() / n * 100

        if onnx_probs.ndim == 2 and native_probs.ndim == 2:
            m = min(onnx_probs.shape[0], native_probs.shape[0])
            max_diff = np.max(np.abs(onnx_probs[:m] - native_probs[:m]))
        else:
            max_diff = -1

        return {
            'status': 'PASS' if match_pct >= 99 else 'WARN' if match_pct >= 90 else 'FAIL',
            'class_match_pct': round(match_pct, 1),
            'max_prob_diff': round(float(max_diff), 4) if max_diff >= 0 else None
        }

    except Exception as e:
        return {'status': 'ERROR', 'reason': str(e)[:80]}


# ──────────────────────────────────────────────────────────────────────
# Layer 3: Directional Coverage
# ──────────────────────────────────────────────────────────────────────

def test_directional_coverage(brain, model_files, X_test):
    """Run on 500 rows, report class distribution and entropy."""
    onnx_path = model_files.get('onnx')
    joblib_path = model_files.get('joblib')

    try:
        if onnx_path and os.path.exists(onnx_path):
            sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            n_feat = sess.get_inputs()[0].shape[1] if len(sess.get_inputs()[0].shape) > 1 else X_test.shape[1]
            X = np.nan_to_num(X_test[:, :n_feat].astype(np.float32), nan=0, posinf=0, neginf=0)
            preds, probs = onnx_predict(sess, X)
            del sess
        elif joblib_path and os.path.exists(joblib_path):
            loaded = joblib.load(joblib_path)
            model = loaded if not isinstance(loaded, dict) else loaded.get('model', loaded.get('estimator', loaded))
            n_feat = getattr(model, 'n_features_in_', X_test.shape[1])
            X = np.nan_to_num(X_test[:, :n_feat], nan=0, posinf=0, neginf=0)
            probs = model.predict_proba(X.astype(np.float64))
            preds = np.argmax(probs, axis=1)
            del model
        else:
            return {'status': 'SKIP', 'reason': 'no_model_file'}

        n = len(preds)
        dist = Counter(preds.astype(int))
        short_pct = dist.get(0, 0) / n * 100
        flat_pct = dist.get(1, 0) / n * 100
        long_pct = dist.get(2, 0) / n * 100

        # Shannon entropy of class distribution (max = log2(3) ≈ 1.585)
        p = np.array([short_pct, flat_pct, long_pct]) / 100 + 1e-10
        entropy = -np.sum(p * np.log2(p))

        # FAIL criteria: flat > 70% OR single class > 90% OR entropy < 0.5
        single_dom = max(short_pct, flat_pct, long_pct) > 90
        if flat_pct > 70 or single_dom or entropy < 0.5:
            status = 'FAIL'
        elif flat_pct > 50 or entropy < 1.0:
            status = 'WARN'
        else:
            status = 'PASS'

        return {
            'status': status,
            'short_pct': round(short_pct, 1),
            'flat_pct': round(flat_pct, 1),
            'long_pct': round(long_pct, 1),
            'entropy': round(float(entropy), 3),
            'n_samples': n
        }

    except Exception as e:
        return {'status': 'ERROR', 'reason': str(e)[:80]}


# ──────────────────────────────────────────────────────────────────────
# Layer 4: Trade Viability
# ──────────────────────────────────────────────────────────────────────

def test_trade_viability(brain, model_files, X_val, min_trades=10):
    """Check if model produces enough non-FLAT signals on validation data."""
    onnx_path = model_files.get('onnx')
    joblib_path = model_files.get('joblib')

    try:
        if onnx_path and os.path.exists(onnx_path):
            sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            n_feat = sess.get_inputs()[0].shape[1] if len(sess.get_inputs()[0].shape) > 1 else X_val.shape[1]
            X = np.nan_to_num(X_val[:, :n_feat].astype(np.float32), nan=0, posinf=0, neginf=0)
            preds, _ = onnx_predict(sess, X)
            del sess
        elif joblib_path and os.path.exists(joblib_path):
            loaded = joblib.load(joblib_path)
            model = loaded if not isinstance(loaded, dict) else loaded.get('model', loaded.get('estimator', loaded))
            n_feat = getattr(model, 'n_features_in_', X_val.shape[1])
            X = np.nan_to_num(X_val[:, :n_feat], nan=0, posinf=0, neginf=0)
            preds = model.predict(X.astype(np.float64))
            del model
        else:
            return {'status': 'SKIP', 'reason': 'no_model_file'}

        n = len(preds)
        non_flat = (preds != 1).sum()
        signal_rate = non_flat / max(n, 1) * 100

        # Count "trades" = transitions from FLAT to non-FLAT
        trades = 0
        in_trade = False
        for p in preds:
            if p != 1 and not in_trade:
                trades += 1
                in_trade = True
            elif p == 1:
                in_trade = False

        return {
            'status': 'PASS' if trades >= min_trades else 'FAIL',
            'total_bars': n,
            'non_flat_signals': int(non_flat),
            'signal_rate_pct': round(signal_rate, 1),
            'trade_count': trades
        }

    except Exception as e:
        return {'status': 'ERROR', 'reason': str(e)[:80]}


# ──────────────────────────────────────────────────────────────────────
# Main audit runner
# ──────────────────────────────────────────────────────────────────────

def run_audit(pairs=None, tfs=None):
    pairs = pairs or ALL_PAIRS
    tfs = tfs or PROD_TFS

    quarantined, cond_q = load_quarantine()
    all_results = []
    t0 = time.time()

    for pair in pairs:
        for tf in tfs:
            parquet_path = PROJECT_ROOT / 'features' / f'{pair}_{tf}_features.parquet'
            if not parquet_path.exists():
                print(f'  SKIP {pair}_{tf}: no feature parquet')
                continue

            print(f'\n{"="*80}')
            print(f'  AUDITING {pair}_{tf}')
            print(f'{"="*80}')

            # Load training data once for this block
            train = pd.read_parquet(str(parquet_path))
            feat_cols = get_feature_cols(train)

            # Prepare test sets
            np.random.seed(42)
            n_rows = len(train)
            rand_idx = np.random.choice(range(200, n_rows - 200), min(500, n_rows - 400), replace=False)
            val_start = int(n_rows * 0.7)  # Last 30% as validation
            val_idx = list(range(val_start, n_rows))

            X_rand = train.iloc[rand_idx][feat_cols].values
            X_val = train.iloc[val_idx][feat_cols].values

            # Layer 1: Feature Parity (once per pair+TF)
            print(f'  Layer 1: Feature Parity...', end=' ', flush=True)
            feat_result = test_feature_parity(pair, tf)
            print(feat_result['status'])

            # Find models for this block
            models = find_models(pair, tf)

            for brain, files in sorted(models.items()):
                # Check quarantine
                block_key = f'{pair}_{tf}'
                is_q = brain in quarantined
                if brain in cond_q:
                    policy = cond_q[brain]
                    if policy.get('policy') == 'quarantined_globally_except':
                        is_q = block_key not in policy.get('exceptions', [])

                if is_q:
                    continue  # Skip quarantined brains

                print(f'  {brain:<25}', end=' ', flush=True)

                # Layer 2: Export Parity
                export_result = test_export_parity(brain, files, X_rand[:100])

                # Layer 3: Directional Coverage
                dir_result = test_directional_coverage(brain, files, X_rand)

                # Layer 4: Trade Viability
                trade_result = test_trade_viability(brain, files, X_val)

                # Determine eligibility
                layers = [feat_result, export_result, dir_result, trade_result]
                any_fail = any(r['status'] == 'FAIL' for r in layers)
                any_error = any(r['status'] == 'ERROR' for r in layers)
                eligible = not any_fail and not any_error

                row = {
                    'pair': pair,
                    'tf': tf,
                    'brain': brain,
                    'quarantined': is_q,
                    'feature_parity': feat_result,
                    'export_parity': export_result,
                    'directional_coverage': dir_result,
                    'trade_viability': trade_result,
                    'eligible': eligible,
                }
                all_results.append(row)

                # One-line summary
                fp = feat_result.get('pct_ok', '?')
                ep = export_result.get('class_match_pct', export_result.get('status', '?'))
                flat = dir_result.get('flat_pct', '?')
                lng = dir_result.get('long_pct', '?')
                sht = dir_result.get('short_pct', '?')
                ent = dir_result.get('entropy', '?')
                trades = trade_result.get('trade_count', '?')
                elig = 'YES' if eligible else 'NO'
                print(f'FP={fp}%  EP={ep}  F={flat}% L={lng}% S={sht}%  H={ent}  T={trades}  -> {elig}')

            del train

    elapsed = time.time() - t0

    # Save results
    out_path = PROJECT_ROOT / 'inference' / 'brain_audit_results.json'
    with open(out_path, 'w') as f:
        json.dump({'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                   'elapsed_sec': round(elapsed, 1),
                   'results': all_results}, f, indent=2, default=str)
    print(f'\nResults saved to {out_path}')

    # Print summary table
    print(f'\n{"="*120}')
    print(f'{"Brain":<25} {"Pair_TF":<12} {"FeatPar":>8} {"ExportP":>8} {"Flat%":>6} {"Long%":>6} {"Short%":>6} {"Entropy":>8} {"Trades":>7} {"ELIGIBLE":>9}')
    print(f'{"="*120}')

    eligible_count = 0
    total_count = 0
    for r in all_results:
        total_count += 1
        fp = r['feature_parity'].get('pct_ok', '?')
        ep_stat = r['export_parity'].get('status', '?')
        ep = r['export_parity'].get('class_match_pct', ep_stat)
        dc = r['directional_coverage']
        flat = dc.get('flat_pct', '?')
        lng = dc.get('long_pct', '?')
        sht = dc.get('short_pct', '?')
        ent = dc.get('entropy', '?')
        tv = r['trade_viability']
        trades = tv.get('trade_count', tv.get('status', '?'))
        elig = 'YES' if r['eligible'] else 'NO'
        if r['eligible']:
            eligible_count += 1
        print(f'{r["brain"]:<25} {r["pair"]}_{r["tf"]:<5} {fp:>7}% {ep:>8} {flat:>5}% {lng:>5}% {sht:>5}% {ent:>8} {trades:>7} {elig:>9}')

    print(f'\nEligible: {eligible_count}/{total_count} brains')
    print(f'Elapsed: {elapsed:.0f}s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CHAOS V1.0 Brain Eligibility Audit')
    parser.add_argument('--pairs', nargs='+', default=None)
    parser.add_argument('--tfs', nargs='+', default=None)
    args = parser.parse_args()
    run_audit(pairs=args.pairs, tfs=args.tfs)
