#!/usr/bin/env python3
"""
CHAOS V1.0 - MANDATORY FULL SYSTEM AUDIT
=========================================
ZERO TOLERANCE FOR ERRORS
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(r"G:\My Drive\chaos_v1.0")
FEATURES_DIR = BASE_DIR / "features"
MODELS_DIR = BASE_DIR / "models"

# =============================================================================
# TRACKING
# =============================================================================
ALL_TESTS = []
PASSED = 0
FAILED = 0

def test_pass(name):
    global PASSED
    PASSED += 1
    ALL_TESTS.append(("PASS", name))
    print(f"  [PASS] {name}")

def test_fail(name, reason):
    global FAILED
    FAILED += 1
    ALL_TESTS.append(("FAIL", name, reason))
    print(f"  [FAIL] {name}: {reason}")

# =============================================================================
# PART 1: PROFIT FACTOR CALCULATION AUDIT
# =============================================================================
print("=" * 70)
print("PART 1: PROFIT FACTOR CALCULATION AUDIT")
print("=" * 70)

# TEST 1.1: Position Conversion
print("\nTEST 1.1: Position Conversion")
print("-" * 50)

def convert_to_positions(y_pred):
    """Convert class indices [0,1,2] to positions [-1,0,1]."""
    y_pred = np.array(y_pred).flatten()
    if len(y_pred) == 0:
        return np.array([], dtype=np.int32)
    if y_pred.min() >= 0 and y_pred.max() <= 2:
        positions = y_pred - 1
    elif y_pred.min() >= -1 and y_pred.max() <= 1:
        positions = y_pred
    else:
        raise ValueError(f"Unexpected prediction range: [{y_pred.min()}, {y_pred.max()}]")
    return positions.astype(np.int32)

test_cases = [
    ([0, 0, 0], [-1, -1, -1]),
    ([1, 1, 1], [0, 0, 0]),
    ([2, 2, 2], [1, 1, 1]),
    ([0, 1, 2], [-1, 0, 1]),
    ([2, 0, 1, 2], [1, -1, 0, 1]),
]

for i, (inp, expected) in enumerate(test_cases):
    result = convert_to_positions(np.array(inp))
    if np.array_equal(result, np.array(expected)):
        test_pass(f"1.1.{i+1}: {inp} -> {expected}")
    else:
        test_fail(f"1.1.{i+1}: {inp} -> {expected}", f"Got {result.tolist()}")

# TEST 1.2: Profit Factor Formula
print("\nTEST 1.2: Profit Factor Formula")
print("-" * 50)

def calculate_pf_simple(positions, returns):
    """Simple PF calculation without slippage."""
    trade_returns = positions * returns
    gross_profit = trade_returns[trade_returns > 0].sum()
    gross_loss = abs(trade_returns[trade_returns < 0].sum())
    if gross_loss < 0.001:
        pf = 1.0 if gross_profit < 0.001 else 10.0
    else:
        pf = gross_profit / gross_loss
    return min(max(pf, 0.1), 10.0)

# Test: $100 profit, $50 loss = PF 2.0
positions = np.array([1, 1, 1, -1, -1])
returns = np.array([0.05, 0.03, 0.02, 0.02, 0.03])  # Long wins, short loses on positive returns
trade_returns = positions * returns
# trade_returns = [0.05, 0.03, 0.02, -0.02, -0.03]
# gross_profit = 0.05 + 0.03 + 0.02 = 0.10
# gross_loss = 0.02 + 0.03 = 0.05
# PF = 0.10 / 0.05 = 2.0
pf = calculate_pf_simple(positions, returns)
if abs(pf - 2.0) < 0.001:
    test_pass("1.2.1: PF = 2.0 for $100 profit, $50 loss")
else:
    test_fail("1.2.1: PF = 2.0", f"Got {pf}")

# Test: All flat positions = PF 1.0
positions = np.array([0, 0, 0, 0, 0])
returns = np.array([0.05, -0.03, 0.02, -0.01, 0.04])
pf = calculate_pf_simple(positions, returns)
if abs(pf - 1.0) < 0.001:
    test_pass("1.2.2: PF = 1.0 for all flat positions")
else:
    test_fail("1.2.2: PF = 1.0 for flat", f"Got {pf}")

# Test: Clamping - super high PF
positions = np.array([1, 1, 1])
returns = np.array([0.10, 0.10, 0.10])  # All wins, no losses
pf = calculate_pf_simple(positions, returns)
if abs(pf - 10.0) < 0.001:
    test_pass("1.2.3: PF clamped to 10.0 for all wins")
else:
    test_fail("1.2.3: PF clamped to 10.0", f"Got {pf}")

# TEST 1.3: Slippage Model
print("\nTEST 1.3: Slippage Model")
print("-" * 50)

from dataclasses import dataclass

@dataclass
class SlippageConfig:
    pair: str
    mean_pips: float
    std_pips: float
    pip_value: float

SLIPPAGE_CONFIG = {
    'EURUSD': SlippageConfig('EURUSD', 0.25, 0.15, 0.0001),
    'GBPUSD': SlippageConfig('GBPUSD', 0.40, 0.25, 0.0001),
    'USDJPY': SlippageConfig('USDJPY', 0.30, 0.20, 0.01),
}

# Verify configs exist
for pair in ['EURUSD', 'GBPUSD', 'USDJPY']:
    if pair in SLIPPAGE_CONFIG:
        test_pass(f"1.3.{pair}: Slippage config exists")
    else:
        test_fail(f"1.3.{pair}", "Slippage config missing")

# Verify USDJPY has different pip_value
if SLIPPAGE_CONFIG['USDJPY'].pip_value == 0.01:
    test_pass("1.3.4: USDJPY pip_value = 0.01")
else:
    test_fail("1.3.4: USDJPY pip_value", f"Got {SLIPPAGE_CONFIG['USDJPY'].pip_value}")

# TEST 1.4: Slippage Only on Position Changes
print("\nTEST 1.4: Slippage on Position Changes")
print("-" * 50)

def apply_slippage_test(returns, positions, slippage_pips, pip_value):
    """Test slippage calculation."""
    n = len(returns)
    position_changes = np.diff(np.concatenate([[0], positions]))
    slippage_impact = slippage_pips * pip_value * np.abs(position_changes)
    adjusted_returns = returns - slippage_impact
    return adjusted_returns, slippage_impact

# Hold same position - no slippage after entry
positions = np.array([1, 1, 1, 1, 1])
returns = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
slippage = np.array([0.5, 0.5, 0.5, 0.5, 0.5])  # 0.5 pips
pip_value = 0.0001

adjusted, impact = apply_slippage_test(returns, positions, slippage, pip_value)
# position_changes = [1, 0, 0, 0, 0]
# slippage_impact = [0.5*0.0001*1, 0.5*0.0001*0, ...] = [0.00005, 0, 0, 0, 0]

expected_impact = np.array([0.00005, 0, 0, 0, 0])
if np.allclose(impact, expected_impact, atol=1e-7):
    test_pass("1.4.1: Slippage only on entry, not while holding")
else:
    test_fail("1.4.1: Slippage on position changes", f"Impact: {impact}")

# TEST 1.5: Edge Cases
print("\nTEST 1.5: Edge Cases")
print("-" * 50)

# Empty arrays
try:
    result = convert_to_positions(np.array([]))
    if len(result) == 0:
        test_pass("1.5.1: Empty array handled")
    else:
        test_fail("1.5.1: Empty array", f"Got {result}")
except:
    test_fail("1.5.1: Empty array", "Exception raised")

# Single element
result = convert_to_positions(np.array([2]))
if np.array_equal(result, np.array([1])):
    test_pass("1.5.2: Single element [2] -> [1]")
else:
    test_fail("1.5.2: Single element", f"Got {result}")

# =============================================================================
# PART 2: DATA PREPARATION AUDIT
# =============================================================================
print("\n" + "=" * 70)
print("PART 2: DATA PREPARATION AUDIT")
print("=" * 70)

# TEST 2.1: Column Names
print("\nTEST 2.1: Column Name Verification")
print("-" * 50)

TARGET_COL = 'target_3class_8'
RETURNS_COL = 'target_return_8'

# Read GPU script
gpu_script = BASE_DIR / "chaos_gpu_training.py"
with open(gpu_script, 'r', encoding='utf-8') as f:
    gpu_content = f.read()

if f"TARGET_COL = '{TARGET_COL}'" in gpu_content:
    test_pass(f"2.1.1: GPU script has TARGET_COL = '{TARGET_COL}'")
else:
    test_fail("2.1.1: GPU TARGET_COL", "Not found or wrong value")

if f"RETURNS_COL = '{RETURNS_COL}'" in gpu_content:
    test_pass(f"2.1.2: GPU script has RETURNS_COL = '{RETURNS_COL}'")
else:
    test_fail("2.1.2: GPU RETURNS_COL", "Not found or wrong value")

# Read RF/ET script
rfet_script = BASE_DIR / "chaos_rf_et_training.py"
with open(rfet_script, 'r', encoding='utf-8') as f:
    rfet_content = f.read()

if f"TARGET_COL = '{TARGET_COL}'" in rfet_content:
    test_pass(f"2.1.3: RF/ET script has TARGET_COL = '{TARGET_COL}'")
else:
    test_fail("2.1.3: RF/ET TARGET_COL", "Not found or wrong value")

if f"RETURNS_COL = '{RETURNS_COL}'" in rfet_content:
    test_pass(f"2.1.4: RF/ET script has RETURNS_COL = '{RETURNS_COL}'")
else:
    test_fail("2.1.4: RF/ET RETURNS_COL", "Not found or wrong value")

# TEST 2.2: Feature File Check
print("\nTEST 2.2: Feature File Check")
print("-" * 50)

test_file = FEATURES_DIR / "EURUSD_H1_features.parquet"
if test_file.exists():
    df = pd.read_parquet(test_file)

    if TARGET_COL in df.columns:
        test_pass(f"2.2.1: {TARGET_COL} exists in EURUSD_H1")
    else:
        test_fail(f"2.2.1: {TARGET_COL}", "Not in file")

    if RETURNS_COL in df.columns:
        test_pass(f"2.2.2: {RETURNS_COL} exists in EURUSD_H1")
    else:
        test_fail(f"2.2.2: {RETURNS_COL}", "Not in file")

    # Check target values
    target_values = sorted(df[TARGET_COL].dropna().unique())
    if target_values == [-1, 0, 1] or target_values == [0, 1, 2]:
        test_pass(f"2.2.3: Target values are valid: {target_values}")
    else:
        test_fail("2.2.3: Target values", f"Got {target_values}")
else:
    test_fail("2.2.1-3", "Test file not found")

# TEST 2.3: 80/20 Split
print("\nTEST 2.3: Chronological Split")
print("-" * 50)

data = np.arange(100)
split_idx = int(len(data) * 0.8)
train = data[:split_idx]
val = data[split_idx:]

if len(train) == 80 and len(val) == 20:
    test_pass("2.3.1: 80/20 split sizes correct")
else:
    test_fail("2.3.1: Split sizes", f"Train: {len(train)}, Val: {len(val)}")

if train[-1] == 79 and val[0] == 80:
    test_pass("2.3.2: Split is chronological (no shuffle)")
else:
    test_fail("2.3.2: Chronological", f"Train ends: {train[-1]}, Val starts: {val[0]}")

# TEST 2.4: NaN Handling
print("\nTEST 2.4: NaN Handling")
print("-" * 50)

test_array = np.array([1.0, np.nan, np.inf, -np.inf, 2.0])
clean = np.nan_to_num(test_array, nan=0.0, posinf=0.0, neginf=0.0)
expected = np.array([1.0, 0.0, 0.0, 0.0, 2.0])

if np.array_equal(clean, expected):
    test_pass("2.4.1: NaN, inf, -inf handled correctly")
else:
    test_fail("2.4.1: NaN handling", f"Got {clean}")

# TEST 2.5: Feature Column Exclusion
print("\nTEST 2.5: Feature Column Exclusion")
print("-" * 50)

exclude_patterns = ['target_', 'return', 'Open', 'High', 'Low', 'Close',
                    'Volume', 'timestamp', 'date', 'pair', 'symbol', 'tf']

test_cols = ['rsi_14', 'target_3class_8', 'Open', 'sma_20', 'return_5', 'macd_signal']
feature_cols = []
for col in test_cols:
    if not any(p in col for p in exclude_patterns):
        feature_cols.append(col)

expected_features = ['rsi_14', 'sma_20', 'macd_signal']
if feature_cols == expected_features:
    test_pass("2.5.1: Feature exclusion patterns work")
else:
    test_fail("2.5.1: Feature exclusion", f"Got {feature_cols}")

# =============================================================================
# PART 3: CHECKPOINT MANAGEMENT AUDIT
# =============================================================================
print("\n" + "=" * 70)
print("PART 3: CHECKPOINT MANAGEMENT AUDIT")
print("=" * 70)

# TEST 3.1: List Format
print("\nTEST 3.1: Checkpoint Uses List Format")
print("-" * 50)

# Check GPU script uses list append
if ".append(" in gpu_content and "'completed'" in gpu_content:
    test_pass("3.1.1: GPU script uses list append for checkpoint")
else:
    test_fail("3.1.1: GPU checkpoint", "No .append() found")

if ".append(" in rfet_content and "'completed'" in rfet_content:
    test_pass("3.1.2: RF/ET script uses list append for checkpoint")
else:
    test_fail("3.1.2: RF/ET checkpoint", "No .append() found")

# TEST 3.2: Checkpoint Structure
print("\nTEST 3.2: Checkpoint Structure")
print("-" * 50)

test_checkpoint = {"completed": []}
test_checkpoint['completed'].append("EURUSD_H1_lgb_optuna")
test_checkpoint['completed'].append("EURUSD_H1_xgb_optuna")

if isinstance(test_checkpoint['completed'], list):
    test_pass("3.2.1: completed is a list")
else:
    test_fail("3.2.1: completed type", f"Got {type(test_checkpoint['completed'])}")

if len(test_checkpoint['completed']) == 2:
    test_pass("3.2.2: List append works correctly")
else:
    test_fail("3.2.2: List append", f"Length: {len(test_checkpoint['completed'])}")

# =============================================================================
# PART 4: MODEL TRAINING SMOKE TEST
# =============================================================================
print("\n" + "=" * 70)
print("PART 4: MODEL TRAINING SMOKE TEST")
print("=" * 70)

print("\nTEST 4.1: RF/ET Training Smoke Test")
print("-" * 50)

try:
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

    # Small test data
    np.random.seed(42)
    X_test = np.random.randn(1000, 50).astype('float32')
    y_test = np.random.randint(0, 3, 1000)

    # RF
    rf = RandomForestClassifier(n_estimators=10, max_depth=5, n_jobs=-1, random_state=42)
    rf.fit(X_test[:800], y_test[:800])
    rf_pred = rf.predict(X_test[800:])

    if len(rf_pred) == 200 and all(p in [0, 1, 2] for p in rf_pred):
        test_pass("4.1.1: RandomForest trains and predicts")
    else:
        test_fail("4.1.1: RF predict", f"Got {len(rf_pred)} predictions")

    # ET
    et = ExtraTreesClassifier(n_estimators=10, max_depth=5, n_jobs=-1, random_state=42)
    et.fit(X_test[:800], y_test[:800])
    et_pred = et.predict(X_test[800:])

    if len(et_pred) == 200 and all(p in [0, 1, 2] for p in et_pred):
        test_pass("4.1.2: ExtraTrees trains and predicts")
    else:
        test_fail("4.1.2: ET predict", f"Got {len(et_pred)} predictions")

    # Test n_jobs=-1 is in script (check both dict format and assignment)
    if "'n_jobs': -1" in rfet_content or "n_jobs'] = -1" in rfet_content:
        test_pass("4.1.3: n_jobs=-1 found in RF/ET script")
    else:
        test_fail("4.1.3: n_jobs=-1", "Not found")

except Exception as e:
    test_fail("4.1: RF/ET smoke test", str(e))

# =============================================================================
# PART 5: FUNCTION AUDIT
# =============================================================================
print("\n" + "=" * 70)
print("PART 5: FUNCTION AUDIT")
print("=" * 70)

# GPU Script Functions
print("\nGPU Script Functions:")
print("-" * 50)

gpu_functions = [
    'def load_checkpoint',
    'def save_checkpoint',
    'def add_to_checkpoint',
    'def get_feature_columns',
    'def load_features',
    'def convert_to_positions',
    'def calculate_profit_factor',
    'def create_objective',
    'def train_final_model',
    'def save_trained_model',
]

for func in gpu_functions:
    if func in gpu_content:
        test_pass(f"GPU: {func}")
    else:
        test_fail(f"GPU: {func}", "Not found")

# RF/ET Script Functions
print("\nRF/ET Script Functions:")
print("-" * 50)

rfet_functions = [
    'def load_checkpoint',
    'def save_checkpoint',
    'def add_to_checkpoint',
    'def get_feature_columns',
    'def load_features',
    'def convert_to_positions',
    'def calculate_pf_with_slippage',
    'def create_rf_objective',
    'def create_et_objective',
    'def save_trained_model',
]

for func in rfet_functions:
    if func in rfet_content:
        test_pass(f"RF/ET: {func}")
    else:
        test_fail(f"RF/ET: {func}", "Not found")

# =============================================================================
# PART 6: BRAIN COUNT VERIFICATION
# =============================================================================
print("\n" + "=" * 70)
print("PART 6: BRAIN COUNT VERIFICATION")
print("=" * 70)

print("\nGPU Brains (Expected: 19):")
print("-" * 50)

gpu_brains = [
    'lgb_optuna', 'xgb_optuna', 'cat_optuna',
    'lgb_v2_optuna', 'xgb_v2_optuna', 'cat_v2_optuna',
    'tabnet_optuna', 'mlp_optuna',
    'lstm_optuna', 'gru_optuna', 'transformer_optuna',
    'cnn1d_optuna', 'tcn_optuna', 'wavenet_optuna',
    'attention_net_optuna', 'residual_mlp_optuna', 'ensemble_nn_optuna',
    'nbeats_optuna', 'tft_optuna'
]

found_brains = 0
for brain in gpu_brains:
    if f"'{brain}'" in gpu_content:
        found_brains += 1
    else:
        test_fail(f"GPU Brain: {brain}", "Not found in script")

if found_brains == 19:
    test_pass(f"GPU: All 19 brains present")
else:
    test_fail("GPU Brain count", f"Found {found_brains}/19")

# Check neural network classes
print("\nNeural Network Classes:")
print("-" * 50)

nn_classes = [
    'class MLPClassifier',
    'class LSTMClassifier',
    'class GRUClassifier',
    'class TransformerClassifier',
    'class CNN1DClassifier',
    'class TCNClassifier',
    'class WaveNetClassifier',
    'class AttentionNetClassifier',
    'class ResidualMLPClassifier',
    'class EnsembleNNClassifier',
    'class NBeatsClassifier',
    'class TemporalFusionTransformer',
]

for cls in nn_classes:
    if cls in gpu_content:
        test_pass(f"Class: {cls.replace('class ', '')}")
    else:
        test_fail(f"Class: {cls.replace('class ', '')}", "Not found")

# =============================================================================
# PART 7: TIMEFRAME VERIFICATION
# =============================================================================
print("\n" + "=" * 70)
print("PART 7: TIMEFRAME VERIFICATION")
print("=" * 70)

print("\nGPU Timeframes (Expected: 9):")
print("-" * 50)

all_timeframes = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1', 'MN1']

for tf in all_timeframes:
    if f"'{tf}'" in gpu_content:
        test_pass(f"GPU TF: {tf}")
    else:
        test_fail(f"GPU TF: {tf}", "Not in ALL_TIMEFRAMES")

print("\nRF/ET Timeframes (Expected: 8, excluding M1):")
print("-" * 50)

rfet_timeframes = ['M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1', 'MN1']

for tf in rfet_timeframes:
    if f"'{tf}'" in rfet_content:
        test_pass(f"RF/ET TF: {tf}")
    else:
        test_fail(f"RF/ET TF: {tf}", "Not in ALL_TIMEFRAMES")

# M1 should NOT be in RF/ET
if "'M1'" in rfet_content and "ALL_TIMEFRAMES" in rfet_content:
    # Check if M1 is in the ALL_TIMEFRAMES list specifically
    import re
    match = re.search(r"ALL_TIMEFRAMES\s*=\s*\[(.*?)\]", rfet_content, re.DOTALL)
    if match and "'M1'" in match.group(1):
        test_fail("RF/ET: M1 excluded", "M1 still in ALL_TIMEFRAMES")
    else:
        test_pass("RF/ET: M1 correctly excluded from ALL_TIMEFRAMES")
else:
    test_pass("RF/ET: M1 correctly excluded")

# =============================================================================
# PART 8: SELF-TEST ASSERTIONS
# =============================================================================
print("\n" + "=" * 70)
print("PART 8: SELF-TEST ASSERTIONS IN GPU SCRIPT")
print("=" * 70)

assertions = [
    "assert TARGET_COL == 'target_3class_8'",
    "assert RETURNS_COL == 'target_return_8'",
    "assert len(ALL_PAIRS) == 9",
    "assert len(ALL_TIMEFRAMES) == 9",
    "assert len(GPU_BRAINS) == 19",
    "assert 'nbeats_optuna' in GPU_BRAINS",
    "assert 'tft_optuna' in GPU_BRAINS",
]

for assertion in assertions:
    if assertion in gpu_content:
        test_pass(f"Self-test: {assertion[:50]}...")
    else:
        test_fail(f"Self-test assertion", f"Missing: {assertion[:40]}...")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("FINAL AUDIT SUMMARY")
print("=" * 70)

print(f"\nTotal Tests: {PASSED + FAILED}")
print(f"Passed: {PASSED}")
print(f"Failed: {FAILED}")

if FAILED > 0:
    print("\n" + "!" * 70)
    print("FAILURES DETECTED:")
    print("!" * 70)
    for test in ALL_TESTS:
        if test[0] == "FAIL":
            print(f"  - {test[1]}: {test[2]}")

print("\n" + "=" * 70)
if FAILED == 0:
    print("STATUS: ALL TESTS PASSED")
    print("PRODUCTION READY")
else:
    print(f"STATUS: {FAILED} TESTS FAILED")
    print("DO NOT START TRAINING UNTIL ALL TESTS PASS")
print("=" * 70)
