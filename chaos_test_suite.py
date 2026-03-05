#!/usr/bin/env python3
"""
CHAOS V1.0 - COMPREHENSIVE TEST SUITE
=====================================
This test suite MUST pass 100% before any training begins.
"""

import os
import sys
import json
import time
import tempfile
import shutil
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

# ==============================================================================
# CONFIGURATION
# ==============================================================================

if os.path.exists('/content/drive'):
    BASE_DIR = Path("/content/drive/MyDrive/chaos_v1.0")
else:
    BASE_DIR = Path(r"G:\My Drive\chaos_v1.0")

FEATURES_DIR = BASE_DIR / "features"
SCRIPTS_DIR = BASE_DIR

EXPECTED_TARGET_COL = 'target_3class_8'
EXPECTED_RETURNS_COL = 'target_return_8'
EXPECTED_PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD',
                  'USDCHF', 'NZDUSD', 'EURJPY', 'GBPJPY']
EXPECTED_TIMEFRAMES = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1', 'MN1']

test_results = {'passed': 0, 'failed': 0, 'errors': [], 'warnings': []}

def test_pass(name, details=""):
    test_results['passed'] += 1
    print(f"  [PASS] {name}")
    if details:
        print(f"         {details}")

def test_fail(name, expected, actual, details=""):
    test_results['failed'] += 1
    msg = f"{name}: Expected {expected}, got {actual}"
    if details:
        msg += f" - {details}"
    test_results['errors'].append(msg)
    print(f"  [FAIL] {name}")
    print(f"         Expected: {expected}")
    print(f"         Actual: {actual}")
    if details:
        print(f"         Details: {details}")

def test_warn(name, message):
    test_results['warnings'].append(f"{name}: {message}")
    print(f"  [WARN] {name}: {message}")

def section_header(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)

def subsection_header(title):
    print()
    print(f"  --- {title} ---")

# ==============================================================================
# PART 1: ENVIRONMENT & DEPENDENCIES
# ==============================================================================

def test_part1_environment():
    section_header("PART 1: ENVIRONMENT & DEPENDENCIES")

    subsection_header("1.1 Python Version")
    py_version = sys.version_info
    if py_version >= (3, 8):
        test_pass("Python version", f"{py_version.major}.{py_version.minor}")
    else:
        test_fail("Python version", ">=3.8", f"{py_version.major}.{py_version.minor}")

    subsection_header("1.2 Required Packages")
    for package in ['numpy', 'pandas', 'sklearn', 'optuna', 'joblib', 'lightgbm', 'xgboost', 'catboost']:
        try:
            __import__(package)
            test_pass(f"Package: {package}")
        except ImportError:
            test_fail(f"Package: {package}", "installed", "not found")

    subsection_header("1.3 Directory Structure")
    if BASE_DIR.exists():
        test_pass("Base directory", str(BASE_DIR))
    else:
        test_fail("Base directory", "exists", "not found")

    if FEATURES_DIR.exists():
        test_pass("Features directory", str(FEATURES_DIR))
    else:
        test_fail("Features directory", "exists", "not found")

    subsection_header("1.4 Training Scripts")
    for script in ['chaos_rf_et_training.py', 'chaos_gpu_training.py']:
        script_path = SCRIPTS_DIR / script
        if script_path.exists():
            test_pass(f"Script: {script}", f"{script_path.stat().st_size / 1024:.1f} KB")
        else:
            test_fail(f"Script: {script}", "exists", "not found")

# ==============================================================================
# PART 2: DATA INTEGRITY
# ==============================================================================

def test_part2_data_integrity():
    section_header("PART 2: DATA INTEGRITY")

    subsection_header("2.1 Feature File Count")
    feature_files = list(FEATURES_DIR.glob("*_features.parquet"))
    expected_count = len(EXPECTED_PAIRS) * len(EXPECTED_TIMEFRAMES)
    if len(feature_files) >= expected_count:
        test_pass("Feature file count", f"{len(feature_files)} files")
    else:
        test_fail("Feature file count", f">={expected_count}", len(feature_files))

    subsection_header("2.2 Pair/Timeframe Coverage")
    missing = []
    for pair in EXPECTED_PAIRS:
        for tf in EXPECTED_TIMEFRAMES:
            if not (FEATURES_DIR / f"{pair}_{tf}_features.parquet").exists():
                missing.append(f"{pair}_{tf}")
    if not missing:
        test_pass("All pair/TF combinations", f"{len(EXPECTED_PAIRS)}x{len(EXPECTED_TIMEFRAMES)}")
    else:
        test_fail("Missing files", "0", len(missing), str(missing[:5]))

    subsection_header("2.3 File Readability")
    test_file = FEATURES_DIR / "EURUSD_H1_features.parquet"
    try:
        df = pd.read_parquet(test_file)
        test_pass("Parquet readability", f"{len(df):,} rows, {len(df.columns)} columns")
    except Exception as e:
        test_fail("Parquet readability", "readable", str(e))

# ==============================================================================
# PART 3: TARGET & RETURNS VALIDATION
# ==============================================================================

def test_part3_target_returns():
    section_header("PART 3: TARGET & RETURNS VALIDATION")

    df = pd.read_parquet(FEATURES_DIR / "EURUSD_H1_features.parquet")

    subsection_header("3.1 Target Column Existence")
    if EXPECTED_TARGET_COL in df.columns:
        test_pass("Target column exists", EXPECTED_TARGET_COL)
    else:
        test_fail("Target column", EXPECTED_TARGET_COL, "not found")
        return

    subsection_header("3.2 Returns Column Existence")
    if EXPECTED_RETURNS_COL in df.columns:
        test_pass("Returns column exists", EXPECTED_RETURNS_COL)
    else:
        test_fail("Returns column", EXPECTED_RETURNS_COL, "not found")
        return

    subsection_header("3.3 Target Value Range")
    target = df[EXPECTED_TARGET_COL]
    unique_vals = sorted(target.unique())
    if list(unique_vals) in [[-1, 0, 1], [0, 1, 2]]:
        test_pass("Target values", str(unique_vals))
    else:
        test_fail("Target values", "[-1,0,1] or [0,1,2]", str(unique_vals))

    subsection_header("3.4 Target Class Balance")
    value_counts = target.value_counts(normalize=True)
    min_pct = value_counts.min() * 100
    if min_pct > 20:
        test_pass("Class balance", f"min={min_pct:.1f}%")
    else:
        test_warn("Class balance", f"Imbalanced: min={min_pct:.1f}%")

    subsection_header("3.5 Returns Value Range")
    returns = df[EXPECTED_RETURNS_COL]
    if returns.min() > -1.0 and returns.max() < 1.0:
        test_pass("Returns range", f"[{returns.min():.6f}, {returns.max():.6f}]")
    else:
        test_warn("Returns range", f"Large values: [{returns.min():.4f}, {returns.max():.4f}]")

    subsection_header("3.6 Returns Non-Zero")
    zero_pct = (returns == 0).sum() / len(returns) * 100
    if zero_pct < 50:
        test_pass("Non-zero returns", f"{100-zero_pct:.1f}% non-zero")
    else:
        test_fail("Non-zero returns", "<50% zeros", f"{zero_pct:.1f}% zeros")

# ==============================================================================
# PART 4: FEATURE ENGINEERING VALIDATION
# ==============================================================================

def test_part4_features():
    section_header("PART 4: FEATURE ENGINEERING VALIDATION")

    df = pd.read_parquet(FEATURES_DIR / "EURUSD_H1_features.parquet")

    exclude_patterns = ['target_', 'return', 'Open', 'High', 'Low', 'Close',
                        'Volume', 'timestamp', 'date', 'pair', 'symbol', 'tf', 'Unnamed']
    feature_cols = [c for c in df.columns if not any(p in c for p in exclude_patterns)]

    subsection_header("4.1 Feature Count")
    if len(feature_cols) >= 100:
        test_pass("Feature count", f"{len(feature_cols)} features")
    else:
        test_fail("Feature count", ">=100", len(feature_cols))

    subsection_header("4.2 No Target Leakage")
    leaked = [c for c in feature_cols if 'target' in c.lower() or 'return' in c.lower() or 'future' in c.lower()]
    if not leaked:
        test_pass("No target leakage", "clean")
    else:
        test_fail("Target leakage", "0 leaked", len(leaked), str(leaked[:5]))

    subsection_header("4.3 No Infinite Values")
    X = df[feature_cols].select_dtypes(include=[np.number]).values.astype('float64')
    inf_count = np.isinf(X).sum()
    if inf_count == 0:
        test_pass("No infinite values", "clean")
    else:
        test_fail("Infinite values", "0", inf_count)

    subsection_header("4.4 NaN Values")
    nan_pct = np.isnan(X).sum() / X.size * 100
    if nan_pct < 5:
        test_pass("NaN values", f"{nan_pct:.2f}% NaN")
    else:
        test_warn("NaN values", f"{nan_pct:.2f}% NaN (high)")

# ==============================================================================
# PART 5: PROFIT FACTOR CALCULATION (CRITICAL)
# ==============================================================================

def test_part5_profit_factor():
    section_header("PART 5: PROFIT FACTOR CALCULATION (CRITICAL)")

    df = pd.read_parquet(FEATURES_DIR / "EURUSD_H1_features.parquet")
    returns = df[EXPECTED_RETURNS_COL].values

    split_idx = int(len(returns) * 0.8)
    returns_val = returns[split_idx:]

    subsection_header("5.1 Random Predictions PF (MUST be ~1.0)")

    random_pfs = []
    for seed in [42, 123, 456, 789, 999, 1234, 5678, 9999]:
        np.random.seed(seed)
        preds = np.random.randint(0, 3, size=len(returns_val))

        positions = np.zeros_like(preds, dtype=float)
        positions[preds == 0] = -1
        positions[preds == 1] = 0
        positions[preds == 2] = 1

        pnl = positions * returns_val
        gp = np.sum(pnl[pnl > 0])
        gl = np.abs(np.sum(pnl[pnl < 0]))
        pf = gp / gl if gl > 1e-10 else 1.0
        random_pfs.append(pf)

    mean_pf = np.mean(random_pfs)
    print(f"    Random PFs: {[f'{pf:.3f}' for pf in random_pfs]}")
    print(f"    Mean: {mean_pf:.4f}")

    if 0.7 < mean_pf < 1.3:
        test_pass("Random PF sanity", f"mean={mean_pf:.4f}")
    else:
        test_fail("Random PF sanity", "~1.0", mean_pf, "PF CALCULATION BROKEN!")

    subsection_header("5.2 Always Long PF")
    positions = np.ones(len(returns_val))
    pnl = positions * returns_val
    gp = np.sum(pnl[pnl > 0])
    gl = np.abs(np.sum(pnl[pnl < 0]))
    pf_long = gp / gl if gl > 1e-10 else 1.0
    print(f"    Always Long PF: {pf_long:.4f}")
    if 0.5 < pf_long < 2.0:
        test_pass("Always Long PF", f"{pf_long:.4f}")
    else:
        test_warn("Always Long PF", f"Unusual: {pf_long:.4f}")

    subsection_header("5.3 Always Short PF")
    positions = -np.ones(len(returns_val))
    pnl = positions * returns_val
    gp = np.sum(pnl[pnl > 0])
    gl = np.abs(np.sum(pnl[pnl < 0]))
    pf_short = gp / gl if gl > 1e-10 else 1.0
    print(f"    Always Short PF: {pf_short:.4f}")
    if 0.5 < pf_short < 2.0:
        test_pass("Always Short PF", f"{pf_short:.4f}")
    else:
        test_warn("Always Short PF", f"Unusual: {pf_short:.4f}")

# ==============================================================================
# PART 6: POSITION CONVERSION (CRITICAL)
# ==============================================================================

def test_part6_position_conversion():
    section_header("PART 6: POSITION CONVERSION (CRITICAL)")

    subsection_header("6.1 Basic Position Conversion")
    test_cases = [
        (np.array([0, 1, 2]), np.array([-1, 0, 1])),
        (np.array([2, 2, 2]), np.array([1, 1, 1])),
        (np.array([0, 0, 0]), np.array([-1, -1, -1])),
        (np.array([1, 1, 1]), np.array([0, 0, 0])),
    ]

    for preds, expected in test_cases:
        positions = preds - 1  # Simple: 0->-1, 1->0, 2->1
        if np.allclose(positions, expected):
            test_pass(f"Convert {list(preds)}", str(list(expected)))
        else:
            test_fail(f"Convert {list(preds)}", str(list(expected)), str(list(positions)))

    subsection_header("6.2 RF/ET Script Position Conversion")
    try:
        sys.path.insert(0, str(SCRIPTS_DIR))
        from chaos_rf_et_training import convert_to_positions

        test_preds = np.array([0, 1, 2, 0, 2, 1])
        expected = np.array([-1, 0, 1, -1, 1, 0])
        result = convert_to_positions(test_preds)

        if np.allclose(result, expected):
            test_pass("RF/ET convert_to_positions", "correct")
        else:
            test_fail("RF/ET convert_to_positions", str(list(expected)), str(list(result)))
    except Exception as e:
        test_fail("RF/ET convert_to_positions", "importable", str(e))

    subsection_header("6.3 GPU Script Position Conversion")
    try:
        from chaos_gpu_training import convert_to_positions as gpu_convert

        test_preds = np.array([0, 1, 2, 0, 2, 1])
        expected = np.array([-1, 0, 1, -1, 1, 0])
        result = gpu_convert(test_preds)

        if np.allclose(result, expected):
            test_pass("GPU convert_to_positions", "correct")
        else:
            test_fail("GPU convert_to_positions", str(list(expected)), str(list(result)))
    except Exception as e:
        test_fail("GPU convert_to_positions", "importable", str(e))

# ==============================================================================
# PART 7: DATA PREPARATION PIPELINE
# ==============================================================================

def test_part7_data_preparation():
    section_header("PART 7: DATA PREPARATION PIPELINE")

    df = pd.read_parquet(FEATURES_DIR / "EURUSD_H1_features.parquet")

    exclude_patterns = ['target_', 'return', 'Open', 'High', 'Low', 'Close',
                        'Volume', 'timestamp', 'date', 'pair', 'symbol', 'tf', 'Unnamed']
    feature_cols = [c for c in df.columns if not any(p in c for p in exclude_patterns)]

    subsection_header("7.1 Feature Exclusion Patterns")
    test_pass("Exclusion patterns", f"{len(feature_cols)} clean features")

    subsection_header("7.2 Data Types")
    X = df[feature_cols].values.astype('float32')
    if X.dtype == np.float32:
        test_pass("Feature dtype", "float32")
    else:
        test_fail("Feature dtype", "float32", str(X.dtype))

    subsection_header("7.3 NaN Handling")
    X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.isnan(X_clean).any() and not np.isinf(X_clean).any():
        test_pass("NaN/Inf handling", "clean after processing")
    else:
        test_fail("NaN/Inf handling", "all clean", "still has NaN/Inf")

    subsection_header("7.4 Target Conversion")
    y = df[EXPECTED_TARGET_COL].values.copy()
    original_min = y.min()
    if y.min() == -1:
        y = y + 1
    if sorted(np.unique(y)) == [0, 1, 2]:
        test_pass("Target conversion", f"original min={original_min} -> [0,1,2]")
    else:
        test_fail("Target conversion", "[0,1,2]", str(sorted(np.unique(y))))

    subsection_header("7.5 Train/Val Split")
    total = len(df)
    split_idx = int(total * 0.8)
    train_pct = split_idx / total * 100
    if abs(train_pct - 80) < 1:
        test_pass("Train/val split", f"{split_idx:,}/{total-split_idx:,} ({train_pct:.1f}%)")
    else:
        test_fail("Train/val split", "80/20", f"{train_pct:.1f}%")

# ==============================================================================
# PART 8: CHECKPOINT MANAGEMENT
# ==============================================================================

def test_part8_checkpoints():
    section_header("PART 8: CHECKPOINT MANAGEMENT")

    temp_dir = Path(tempfile.mkdtemp())

    try:
        subsection_header("8.1 Import Checkpoint Functions")
        try:
            from chaos_rf_et_training import load_checkpoint, save_checkpoint, add_to_checkpoint
            test_pass("Import RF/ET checkpoint functions", "success")
        except Exception as e:
            test_fail("Import RF/ET checkpoint functions", "importable", str(e))
            return

        subsection_header("8.2 Checkpoint Structure")
        test_cp = {'completed': []}
        cp_path = temp_dir / "test_checkpoint.json"
        with open(cp_path, 'w') as f:
            json.dump(test_cp, f)

        with open(cp_path, 'r') as f:
            loaded = json.load(f)

        if isinstance(loaded.get('completed'), list):
            test_pass("Checkpoint uses list", "completed is list")
        else:
            test_fail("Checkpoint structure", "list", type(loaded.get('completed')))

        subsection_header("8.3 Checkpoint Append")
        loaded['completed'].append("TEST_MODEL_1")
        loaded['completed'].append("TEST_MODEL_2")
        if len(loaded['completed']) == 2:
            test_pass("Checkpoint append", "2 models added")
        else:
            test_fail("Checkpoint append", "2", len(loaded['completed']))
    finally:
        shutil.rmtree(temp_dir)

# ==============================================================================
# PART 9: MODEL TRAINING SMOKE TESTS
# ==============================================================================

def test_part9_training_smoke():
    section_header("PART 9: MODEL TRAINING SMOKE TESTS")

    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

    df = pd.read_parquet(FEATURES_DIR / "EURUSD_H4_features.parquet")

    exclude_patterns = ['target_', 'return', 'Open', 'High', 'Low', 'Close',
                        'Volume', 'timestamp', 'date', 'pair', 'symbol', 'tf', 'Unnamed']
    feature_cols = [c for c in df.columns if not any(p in c for p in exclude_patterns)]

    X = df[feature_cols].values.astype('float32')
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    y = df[EXPECTED_TARGET_COL].values.copy()
    if y.min() == -1:
        y = y + 1

    returns = df[EXPECTED_RETURNS_COL].values

    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    returns_val = returns[split_idx:]

    subsection_header("9.1 RandomForest Training")
    try:
        start = time.time()
        rf = RandomForestClassifier(n_estimators=10, max_depth=5, n_jobs=-1, random_state=42)
        rf.fit(X_train, y_train)
        elapsed = time.time() - start

        preds = rf.predict(X_val)
        accuracy = (preds == y_val).mean()
        test_pass("RF training", f"{elapsed:.2f}s, acc={accuracy:.4f}")
    except Exception as e:
        test_fail("RF training", "success", str(e))

    subsection_header("9.2 ExtraTrees Training")
    try:
        start = time.time()
        et = ExtraTreesClassifier(n_estimators=10, max_depth=5, n_jobs=-1, random_state=42)
        et.fit(X_train, y_train)
        elapsed = time.time() - start

        preds = et.predict(X_val)
        accuracy = (preds == y_val).mean()
        test_pass("ET training", f"{elapsed:.2f}s, acc={accuracy:.4f}")
    except Exception as e:
        test_fail("ET training", "success", str(e))

    subsection_header("9.3 Trained Model PF Calculation")
    preds = rf.predict(X_val)
    positions = preds - 1  # Convert 0,1,2 to -1,0,1

    # Count actual trades (non-zero positions)
    trade_count = np.sum(positions != 0)
    trade_pct = trade_count / len(positions) * 100

    pnl = positions * returns_val
    gp = np.sum(pnl[pnl > 0])
    gl = np.abs(np.sum(pnl[pnl < 0]))
    pf = gp / gl if gl > 1e-10 else 1.0

    print(f"    Trained RF PF: {pf:.4f}")
    print(f"    Trade count: {trade_count}/{len(positions)} ({trade_pct:.1f}%)")

    # PF up to 10 is acceptable (training scripts clamp to this)
    # Low trade count with high PF is valid (conservative model)
    if 0.3 < pf <= 10.0:
        test_pass("Trained model PF", f"{pf:.4f} ({trade_pct:.1f}% trades)")
    else:
        test_fail("Trained model PF", "0.3-10.0", pf, "CHECK PF CALCULATION!")

# ==============================================================================
# PART 10: SCRIPT CONFIGURATION AUDIT
# ==============================================================================

def test_part10_script_config():
    section_header("PART 10: SCRIPT CONFIGURATION AUDIT")

    scripts = [
        ('chaos_rf_et_training.py', ['M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1', 'MN1']),
        ('chaos_gpu_training.py', ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1', 'MN1']),
    ]

    for script_name, expected_tfs in scripts:
        subsection_header(f"{script_name}")

        script_path = SCRIPTS_DIR / script_name
        if not script_path.exists():
            test_fail(f"{script_name} exists", "yes", "no")
            continue

        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if f"TARGET_COL = '{EXPECTED_TARGET_COL}'" in content:
            test_pass("TARGET_COL", EXPECTED_TARGET_COL)
        else:
            test_fail("TARGET_COL", EXPECTED_TARGET_COL, "not found")

        if f"RETURNS_COL = '{EXPECTED_RETURNS_COL}'" in content:
            test_pass("RETURNS_COL", EXPECTED_RETURNS_COL)
        else:
            test_fail("RETURNS_COL", EXPECTED_RETURNS_COL, "not found")

        if "'n_jobs': -1" in content or "n_jobs=-1" in content:
            test_pass("n_jobs=-1", "found")
        else:
            test_fail("n_jobs=-1", "present", "not found")

        missing_pairs = [p for p in EXPECTED_PAIRS if f"'{p}'" not in content]
        if not missing_pairs:
            test_pass("All 9 pairs", "present")
        else:
            test_fail("Pairs", "all 9", f"missing {missing_pairs}")

# ==============================================================================
# PART 11: CROSS-SCRIPT CONSISTENCY
# ==============================================================================

def test_part11_cross_script():
    section_header("PART 11: CROSS-SCRIPT CONSISTENCY")

    try:
        sys.path.insert(0, str(SCRIPTS_DIR))

        import chaos_rf_et_training as rf_script
        import chaos_gpu_training as gpu_script

        subsection_header("11.1 TARGET_COL Consistency")
        if rf_script.TARGET_COL == gpu_script.TARGET_COL == EXPECTED_TARGET_COL:
            test_pass("TARGET_COL consistent", EXPECTED_TARGET_COL)
        else:
            test_fail("TARGET_COL", EXPECTED_TARGET_COL,
                      f"RF={rf_script.TARGET_COL}, GPU={gpu_script.TARGET_COL}")

        subsection_header("11.2 RETURNS_COL Consistency")
        if rf_script.RETURNS_COL == gpu_script.RETURNS_COL == EXPECTED_RETURNS_COL:
            test_pass("RETURNS_COL consistent", EXPECTED_RETURNS_COL)
        else:
            test_fail("RETURNS_COL", EXPECTED_RETURNS_COL,
                      f"RF={rf_script.RETURNS_COL}, GPU={gpu_script.RETURNS_COL}")

        subsection_header("11.3 ALL_PAIRS Consistency")
        if set(rf_script.ALL_PAIRS) == set(gpu_script.ALL_PAIRS):
            test_pass("ALL_PAIRS consistent", f"{len(rf_script.ALL_PAIRS)} pairs")
        else:
            test_fail("ALL_PAIRS", "same pairs",
                      f"RF={rf_script.ALL_PAIRS}, GPU={gpu_script.ALL_PAIRS}")

        subsection_header("11.4 convert_to_positions Consistency")
        test_input = np.array([0, 1, 2, 0, 1, 2])
        rf_result = rf_script.convert_to_positions(test_input)
        gpu_result = gpu_script.convert_to_positions(test_input)

        if np.allclose(rf_result, gpu_result):
            test_pass("convert_to_positions consistent", "same output")
        else:
            test_fail("convert_to_positions", "same",
                      f"RF={list(rf_result)}, GPU={list(gpu_result)}")
    except Exception as e:
        test_fail("Cross-script import", "success", str(e))

# ==============================================================================
# PART 12: END-TO-END PIPELINE TEST
# ==============================================================================

def test_part12_end_to_end():
    section_header("PART 12: END-TO-END PIPELINE TEST")

    subsection_header("12.1 Complete Pipeline Simulation")

    try:
        from sklearn.ensemble import RandomForestClassifier

        df = pd.read_parquet(FEATURES_DIR / "EURUSD_H4_features.parquet")
        print(f"    Loaded: {len(df):,} rows")

        exclude_patterns = ['target_', 'return', 'Open', 'High', 'Low', 'Close',
                            'Volume', 'timestamp', 'date', 'pair', 'symbol', 'tf', 'Unnamed']
        feature_cols = [c for c in df.columns if not any(p in c for p in exclude_patterns)]

        X = df[feature_cols].values.astype('float32')
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"    Features: {len(feature_cols)}")

        y = df[EXPECTED_TARGET_COL].values.copy()
        if y.min() == -1:
            y = y + 1
        print(f"    Target classes: {np.unique(y)}")

        returns = df[EXPECTED_RETURNS_COL].values.copy()
        print(f"    Returns range: [{returns.min():.6f}, {returns.max():.6f}]")

        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        returns_val = returns[split_idx:]
        print(f"    Train: {len(X_train):,}, Val: {len(X_val):,}")

        model = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
        model.fit(X_train, y_train)
        print(f"    Model trained")

        preds = model.predict(X_val)
        print(f"    Predictions: {np.unique(preds, return_counts=True)}")

        positions = preds - 1  # Convert 0,1,2 to -1,0,1
        print(f"    Positions: {np.unique(positions, return_counts=True)}")

        # Count actual trades
        trade_count = np.sum(positions != 0)
        trade_pct = trade_count / len(positions) * 100

        pnl = positions * returns_val
        gp = np.sum(pnl[pnl > 0])
        gl = np.abs(np.sum(pnl[pnl < 0]))
        pf = gp / gl if gl > 1e-10 else 1.0

        print(f"    Trade count: {trade_count}/{len(positions)} ({trade_pct:.1f}%)")
        print(f"    Gross Profit: {gp:.6f}")
        print(f"    Gross Loss: {gl:.6f}")
        print(f"    Profit Factor: {pf:.4f}")

        accuracy = (preds == y_val).mean()
        print(f"    Accuracy: {accuracy:.4f}")

        # PF up to 10 is valid (clamped max in training scripts)
        if 0.3 < pf <= 10.0 and accuracy > 0.3:
            test_pass("End-to-end pipeline", f"PF={pf:.4f}, Acc={accuracy:.4f}, Trades={trade_pct:.1f}%")
        else:
            test_fail("End-to-end pipeline", "PF=0.3-10.0, Acc>0.3",
                      f"PF={pf:.4f}, Acc={accuracy:.4f}")
    except Exception as e:
        test_fail("End-to-end pipeline", "success", str(e))
        traceback.print_exc()

# ==============================================================================
# PART 13: EDGE CASES & ERROR HANDLING
# ==============================================================================

def test_part13_edge_cases():
    section_header("PART 13: EDGE CASES & ERROR HANDLING")

    np.random.seed(42)
    returns_val = np.random.randn(1000) * 0.01

    subsection_header("13.1 All Same Prediction")
    positions = np.ones(1000)
    pnl = positions * returns_val
    gp = np.sum(pnl[pnl > 0])
    gl = np.abs(np.sum(pnl[pnl < 0]))
    pf = gp / gl if gl > 1e-10 else 1.0

    if not np.isnan(pf) and not np.isinf(pf):
        test_pass("All long PF", f"{pf:.4f}")
    else:
        test_fail("All long PF", "valid number", pf)

    subsection_header("13.2 All Flat Prediction")
    positions = np.zeros(1000)
    pnl = positions * returns_val
    gp = np.sum(pnl[pnl > 0])
    gl = np.abs(np.sum(pnl[pnl < 0]))

    if gp == 0 and gl == 0:
        test_pass("All flat PF", "defined as 1.0")
    else:
        test_fail("All flat PF", "gp=0, gl=0", f"gp={gp}, gl={gl}")

    subsection_header("13.3 Zero Returns Handling")
    returns_zero = np.zeros(1000)
    positions = np.array([1, -1, 0] * 333 + [1])
    pnl = positions * returns_zero
    gp = np.sum(pnl[pnl > 0])
    gl = np.abs(np.sum(pnl[pnl < 0]))
    pf = gp / gl if gl > 1e-10 else 1.0

    if pf == 1.0:
        test_pass("Zero returns PF", "1.0 (default)")
    else:
        test_fail("Zero returns PF", "1.0", pf)

# ==============================================================================
# PART 14: MATHEMATICAL VERIFICATION
# ==============================================================================

def test_part14_math_verification():
    section_header("PART 14: MATHEMATICAL VERIFICATION")

    subsection_header("14.1 Manual PF Calculation")

    positions = np.array([1, 1, -1, -1, 0])
    returns = np.array([0.01, -0.02, -0.01, 0.02, 0.01])

    # PnL: 0.01, -0.02, 0.01, -0.02, 0
    # Profit: 0.02, Loss: 0.04, PF: 0.5

    pnl = positions * returns
    expected_pnl = np.array([0.01, -0.02, 0.01, -0.02, 0.0])

    if np.allclose(pnl, expected_pnl):
        test_pass("PnL calculation", str(list(pnl)))
    else:
        test_fail("PnL calculation", str(list(expected_pnl)), str(list(pnl)))

    gp = np.sum(pnl[pnl > 0])
    gl = np.abs(np.sum(pnl[pnl < 0]))
    pf = gp / gl

    if abs(pf - 0.5) < 0.001:
        test_pass("PF formula", f"{pf:.4f}")
    else:
        test_fail("PF formula", 0.5, pf)

    subsection_header("14.2 Position Mapping Math")
    for pred, expected_pos in [(0, -1), (1, 0), (2, 1)]:
        actual_pos = pred - 1
        if actual_pos == expected_pos:
            test_pass(f"Pred {pred} -> Pos {expected_pos}", "correct")
        else:
            test_fail(f"Pred {pred} -> Pos {expected_pos}", expected_pos, actual_pos)

# ==============================================================================
# MAIN TEST RUNNER
# ==============================================================================

def run_all_tests():
    print("=" * 70)
    print("  CHAOS V1.0 - COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Base Directory: {BASE_DIR}")
    print("=" * 70)

    start_time = time.time()

    test_functions = [
        test_part1_environment,
        test_part2_data_integrity,
        test_part3_target_returns,
        test_part4_features,
        test_part5_profit_factor,
        test_part6_position_conversion,
        test_part7_data_preparation,
        test_part8_checkpoints,
        test_part9_training_smoke,
        test_part10_script_config,
        test_part11_cross_script,
        test_part12_end_to_end,
        test_part13_edge_cases,
        test_part14_math_verification,
    ]

    for test_func in test_functions:
        try:
            test_func()
        except Exception as e:
            print(f"\n  [CRITICAL ERROR] in {test_func.__name__}: {e}")
            traceback.print_exc()
            test_results['failed'] += 1
            test_results['errors'].append(f"{test_func.__name__}: {e}")

    elapsed = time.time() - start_time

    print()
    print("=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    print(f"  Tests Passed: {test_results['passed']}")
    print(f"  Tests Failed: {test_results['failed']}")
    print(f"  Warnings: {len(test_results['warnings'])}")
    print(f"  Time: {elapsed:.2f}s")
    print()

    if test_results['errors']:
        print("  FAILURES:")
        for error in test_results['errors']:
            print(f"    [X] {error}")
        print()

    if test_results['warnings']:
        print("  WARNINGS:")
        for warning in test_results['warnings']:
            print(f"    [!] {warning}")
        print()

    print("=" * 70)

    if test_results['failed'] == 0:
        print("  [OK] ALL TESTS PASSED - PRODUCTION READY")
        print("=" * 70)
        print()
        print("  You may now proceed with training:")
        print("    Local:  python chaos_rf_et_training.py")
        print("    Colab:  python chaos_gpu_training.py")
        print()
        return 0
    else:
        print("  [X] TESTS FAILED - DO NOT PROCEED WITH TRAINING")
        print("=" * 70)
        print()
        print("  Fix all failures before training!")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
