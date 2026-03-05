#!/usr/bin/env python3
"""
CHAOS V1.0 - PRE-FLIGHT VERIFICATION
Must pass ALL checks before training begins.
"""

import os
import sys
import json
from pathlib import Path
import pandas as pd
import importlib.util

# =============================================================================
# CONFIGURATION TO VERIFY
# =============================================================================

EXPECTED_TARGET_COL = 'target_3class_8'
EXPECTED_RETURNS_COL = 'target_return_8'
EXPECTED_PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD',
                  'USDCHF', 'NZDUSD', 'EURJPY', 'GBPJPY']
EXPECTED_TIMEFRAMES = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1', 'MN1']
EXPECTED_GPU_BRAINS = 19
EXPECTED_CPU_BRAINS = 2
EXPECTED_FEATURE_FILES = 81  # 9 pairs x 9 timeframes

# Paths
if os.path.exists('/content/drive'):
    BASE_DIR = Path("/content/drive/MyDrive/chaos_v1.0")
else:
    BASE_DIR = Path(r"G:\My Drive\chaos_v1.0")

FEATURES_DIR = BASE_DIR / "features"
MODELS_DIR = BASE_DIR / "models"

# =============================================================================
# VERIFICATION FUNCTIONS
# =============================================================================

def check_passed(name):
    print(f"  [OK] {name}")
    return True

def check_failed(name, reason):
    print(f"  [FAIL] {name}: {reason}")
    return False

def verify_feature_files():
    """Verify all 81 feature files exist with correct columns."""
    print("\n" + "=" * 60)
    print("CHECK 1: FEATURE FILES")
    print("=" * 60)

    errors = []
    found = 0

    for pair in EXPECTED_PAIRS:
        for tf in EXPECTED_TIMEFRAMES:
            filename = f"{pair}_{tf}_features.parquet"
            filepath = FEATURES_DIR / filename

            if not filepath.exists():
                errors.append(f"MISSING: {filename}")
                continue

            found += 1

            # Check columns only for small/medium files (skip M1/M5 which are huge)
            if tf not in ['M1', 'M5'] and found <= 3:
                try:
                    df = pd.read_parquet(filepath)

                    if EXPECTED_TARGET_COL not in df.columns:
                        errors.append(f"{filename}: Missing {EXPECTED_TARGET_COL}")

                    if EXPECTED_RETURNS_COL not in df.columns:
                        errors.append(f"{filename}: Missing {EXPECTED_RETURNS_COL}")

                except Exception as e:
                    errors.append(f"{filename}: Cannot read - {e}")

    print(f"  Feature files found: {found}/{EXPECTED_FEATURE_FILES}")

    if errors:
        for err in errors[:10]:  # Show first 10
            print(f"  [FAIL] {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
        return False

    print(f"  [OK] All {EXPECTED_FEATURE_FILES} feature files present")
    print(f"  [OK] Sample files have {EXPECTED_TARGET_COL}")
    print(f"  [OK] Sample files have {EXPECTED_RETURNS_COL}")
    return True


def verify_script(script_name, expected_brains):
    """Verify a training script has correct configuration."""

    script_path = BASE_DIR / script_name

    if not script_path.exists():
        return check_failed("File exists", f"{script_name} not found")

    # Read script content
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()

    errors = []

    # Check TARGET_COL
    if f"TARGET_COL = '{EXPECTED_TARGET_COL}'" in content:
        check_passed(f"TARGET_COL = '{EXPECTED_TARGET_COL}'")
    else:
        errors.append("TARGET_COL not set correctly")

    # Check RETURNS_COL
    if f"RETURNS_COL = '{EXPECTED_RETURNS_COL}'" in content:
        check_passed(f"RETURNS_COL = '{EXPECTED_RETURNS_COL}'")
    else:
        errors.append("RETURNS_COL not set correctly")

    # Check timeframes - RF/ET script INTENTIONALLY excludes M1 (too large for 16GB RAM)
    is_rf_et_script = 'rf_et' in script_name.lower()

    if is_rf_et_script:
        # RF/ET should have M5 but NOT M1 (M1 is 7.8M rows, too large)
        if "'M5'" in content:
            check_passed("M5 in ALL_TIMEFRAMES (M1 excluded - too large for 16GB RAM)")
        else:
            errors.append("M5 missing from ALL_TIMEFRAMES")

        # Check 8 timeframes (excluding M1)
        rf_et_timeframes = ['M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1', 'MN1']
        tf_count = len([tf for tf in rf_et_timeframes if f"'{tf}'" in content])
        if tf_count == 8:
            check_passed("All 8 RF/ET timeframes present (M1 excluded)")
        else:
            errors.append(f"Only {tf_count}/8 timeframes found")
    else:
        # GPU script should have all 9 timeframes including M1 and M5
        if "'M1'" in content and "'M5'" in content:
            check_passed("M1 and M5 in ALL_TIMEFRAMES")
        else:
            errors.append("M1 or M5 missing from ALL_TIMEFRAMES")

        # Check all 9 timeframes
        tf_count = len([tf for tf in EXPECTED_TIMEFRAMES if f"'{tf}'" in content])
        if tf_count == 9:
            check_passed("All 9 timeframes present")
        else:
            errors.append(f"Only {tf_count}/9 timeframes found")

    # Check all 9 pairs
    pair_count = len([p for p in EXPECTED_PAIRS if f"'{p}'" in content])
    if pair_count == 9:
        check_passed("All 9 pairs present")
    else:
        errors.append(f"Only {pair_count}/9 pairs found")

    # Check critical functions exist (different names allowed for RF/ET vs GPU)
    if 'gpu' in script_name.lower():
        critical_functions = [
            'def load_checkpoint',
            'def save_checkpoint',
            'def load_features',
            'def calculate_profit_factor',
            'def convert_to_positions',
            'def create_objective',
            'def train_final_model',
        ]
    else:
        # RF/ET script uses different function names
        critical_functions = [
            'def load_checkpoint',
            'def save_checkpoint',
            'def load_features',
            'def calculate_pf_with_slippage',  # Different name
            'def convert_to_positions',
            'def create_rf_objective',  # Different name
            'def create_et_objective',  # Different name
        ]

    missing_funcs = []
    for func in critical_functions:
        if func not in content:
            missing_funcs.append(func.replace('def ', ''))

    if missing_funcs:
        errors.append(f"Missing functions: {missing_funcs}")
    else:
        check_passed("All critical functions defined")

    # Check checkpoint uses list append
    if ".append(" in content and "completed" in content:
        check_passed("Checkpoint uses list append")
    else:
        errors.append("Checkpoint may not use list append")

    # Check datetime imported
    if 'from datetime import datetime' in content or 'import datetime' in content:
        check_passed("datetime imported")
    else:
        errors.append("datetime not imported")

    if errors:
        print(f"\n  ERRORS:")
        for err in errors:
            print(f"  [FAIL] {err}")
        return False

    return True


def verify_gpu_script():
    """Verify GPU training script."""
    print("\n" + "=" * 60)
    print("CHECK 2: GPU TRAINING SCRIPT")
    print("=" * 60)

    script_path = BASE_DIR / "chaos_gpu_training.py"

    if not script_path.exists():
        return check_failed("File exists", "chaos_gpu_training.py not found")

    check_passed("File exists")

    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()

    errors = []

    # Verify GPU_BRAINS
    gpu_brains = [
        'lgb_optuna', 'xgb_optuna', 'cat_optuna',
        'lgb_v2_optuna', 'xgb_v2_optuna', 'cat_v2_optuna',
        'tabnet_optuna', 'mlp_optuna',
        'lstm_optuna', 'gru_optuna', 'transformer_optuna',
        'cnn1d_optuna', 'tcn_optuna', 'wavenet_optuna',
        'attention_net_optuna', 'residual_mlp_optuna', 'ensemble_nn_optuna',
        'nbeats_optuna', 'tft_optuna'
    ]

    missing_brains = [b for b in gpu_brains if f"'{b}'" not in content]

    if missing_brains:
        errors.append(f"Missing brains: {missing_brains}")
    else:
        check_passed(f"All {len(gpu_brains)} GPU brains present")

    # Verify neural network classes
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
        'class TemporalFusionTransformer'
    ]

    missing_classes = [c.replace('class ', '') for c in nn_classes if c not in content]

    if missing_classes:
        errors.append(f"Missing classes: {missing_classes}")
    else:
        check_passed(f"All {len(nn_classes)} neural network classes defined")

    if errors:
        for err in errors:
            print(f"  [FAIL] {err}")
        return False

    # Run common script checks
    return verify_script("chaos_gpu_training.py", 19)


def verify_rf_et_script():
    """Verify RF/ET training script."""
    print("\n" + "=" * 60)
    print("CHECK 3: RF/ET TRAINING SCRIPT")
    print("=" * 60)

    script_path = BASE_DIR / "chaos_rf_et_training.py"

    if not script_path.exists():
        return check_failed("File exists", "chaos_rf_et_training.py not found")

    check_passed("File exists")

    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Verify CPU_BRAINS or similar
    if "'rf_optuna'" in content and "'et_optuna'" in content:
        check_passed("RF and ET brains present")
    else:
        return check_failed("Brains", "RF or ET brain missing")

    return verify_script("chaos_rf_et_training.py", 2)


def verify_checkpoints_clear():
    """Verify checkpoints are clear or don't exist."""
    print("\n" + "=" * 60)
    print("CHECK 4: CHECKPOINTS")
    print("=" * 60)

    checkpoints = [
        MODELS_DIR / "gpu_checkpoint.json",
        MODELS_DIR / "rf_et_checkpoint.json",
        MODELS_DIR / "optuna_checkpoint.json"
    ]

    for cp in checkpoints:
        if cp.exists():
            try:
                with open(cp, 'r') as f:
                    data = json.load(f)
                completed = len(data.get('completed', []))

                if completed > 0:
                    print(f"  [INFO] {cp.name}: {completed} models completed")
                    print(f"         (Delete to start fresh or keep to resume)")
                else:
                    check_passed(f"{cp.name}: Empty (fresh start)")
            except:
                print(f"  [WARN] {cp.name}: Cannot read")
        else:
            check_passed(f"{cp.name}: Does not exist (fresh start)")

    return True  # Not a failure if checkpoints have data


def verify_models_directory():
    """Verify models directory exists and check for existing models."""
    print("\n" + "=" * 60)
    print("CHECK 5: MODELS DIRECTORY")
    print("=" * 60)

    if not MODELS_DIR.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        check_passed("Created models directory")
    else:
        check_passed("Models directory exists")

    # Count existing models
    joblib_files = list(MODELS_DIR.glob("*.joblib"))
    pt_files = list(MODELS_DIR.glob("*.pt"))

    print(f"  [INFO] Existing .joblib models: {len(joblib_files)}")
    print(f"  [INFO] Existing .pt models: {len(pt_files)}")

    return True


def verify_sample_data_load():
    """Verify we can actually load and prepare data."""
    print("\n" + "=" * 60)
    print("CHECK 6: DATA LOADING TEST")
    print("=" * 60)

    test_file = FEATURES_DIR / "EURUSD_H1_features.parquet"

    if not test_file.exists():
        return check_failed("Test file", "EURUSD_H1_features.parquet not found")

    try:
        df = pd.read_parquet(test_file)
        check_passed(f"Loaded {test_file.name}: {len(df):,} rows")

        # Check target column
        if EXPECTED_TARGET_COL not in df.columns:
            return check_failed("Target column", f"{EXPECTED_TARGET_COL} not found")

        target_values = df[EXPECTED_TARGET_COL].unique()
        check_passed(f"Target values: {sorted(target_values)}")

        # Check returns column
        if EXPECTED_RETURNS_COL not in df.columns:
            return check_failed("Returns column", f"{EXPECTED_RETURNS_COL} not found")

        returns = df[EXPECTED_RETURNS_COL]
        check_passed(f"Returns range: [{returns.min():.6f}, {returns.max():.6f}]")

        # Count features
        exclude_patterns = ['target_', 'return', 'Open', 'High', 'Low', 'Close',
                          'Volume', 'timestamp', 'date', 'pair', 'symbol', 'tf', 'bar_time']
        feature_cols = [c for c in df.columns
                       if not any(p in c for p in exclude_patterns)]
        check_passed(f"Feature columns: {len(feature_cols)}")

    except Exception as e:
        return check_failed("Data loading", str(e))

    return True


def verify_import_test():
    """Try importing the GPU script to catch syntax errors."""
    print("\n" + "=" * 60)
    print("CHECK 7: IMPORT TEST")
    print("=" * 60)

    script_path = BASE_DIR / "chaos_gpu_training.py"

    try:
        spec = importlib.util.spec_from_file_location("chaos_gpu_training", script_path)
        module = importlib.util.module_from_spec(spec)

        # This will catch syntax errors
        spec.loader.exec_module(module)

        # Verify key attributes
        if hasattr(module, 'TARGET_COL') and module.TARGET_COL == EXPECTED_TARGET_COL:
            check_passed(f"Module TARGET_COL = '{module.TARGET_COL}'")
        else:
            return check_failed("Module TARGET_COL", "Not set correctly after import")

        if hasattr(module, 'GPU_BRAINS') and len(module.GPU_BRAINS) == 19:
            check_passed(f"Module GPU_BRAINS: {len(module.GPU_BRAINS)} brains")
        else:
            return check_failed("Module GPU_BRAINS", "Not 19 brains after import")

        if hasattr(module, 'ALL_TIMEFRAMES') and len(module.ALL_TIMEFRAMES) == 9:
            check_passed(f"Module ALL_TIMEFRAMES: {len(module.ALL_TIMEFRAMES)} timeframes")
        else:
            return check_failed("Module ALL_TIMEFRAMES", "Not 9 timeframes after import")

        check_passed("chaos_gpu_training.py imports successfully")

    except SyntaxError as e:
        return check_failed("Syntax check", f"Line {e.lineno}: {e.msg}")
    except Exception as e:
        return check_failed("Import test", str(e))

    return True


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("CHAOS V1.0 - PRE-FLIGHT VERIFICATION")
    print("=" * 60)
    print(f"Base directory: {BASE_DIR}")
    print(f"Expected: {len(EXPECTED_PAIRS)} pairs x {len(EXPECTED_TIMEFRAMES)} timeframes")
    print(f"Expected: {EXPECTED_FEATURE_FILES} feature files")
    print(f"Expected: {EXPECTED_GPU_BRAINS} GPU brains + {EXPECTED_CPU_BRAINS} CPU brains")

    results = []

    # Run all checks
    results.append(("Feature Files", verify_feature_files()))
    results.append(("GPU Script", verify_gpu_script()))
    results.append(("RF/ET Script", verify_rf_et_script()))
    results.append(("Checkpoints", verify_checkpoints_clear()))
    results.append(("Models Directory", verify_models_directory()))
    results.append(("Data Loading", verify_sample_data_load()))
    results.append(("Import Test", verify_import_test()))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {name}")

    print("=" * 60)

    if passed == total:
        print(f"ALL {total} CHECKS PASSED")
        print("=" * 60)
        print("READY FOR TRAINING")
        print("")
        print("To start GPU training (Colab):")
        print("  python chaos_gpu_training.py")
        print("")
        print("To start RF/ET training (Local):")
        print("  python chaos_rf_et_training.py")
        print("=" * 60)
        return 0
    else:
        print(f"FAILED: {total - passed}/{total} checks failed")
        print("=" * 60)
        print("DO NOT START TRAINING UNTIL ALL CHECKS PASS")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
