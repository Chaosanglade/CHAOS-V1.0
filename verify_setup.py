"""
CHAOS V1.0 - VERIFICATION SCRIPT
================================
Run on BOTH Local (Windows) and Colab to verify identical configuration.
"""
import pandas as pd
from pathlib import Path

print("=" * 70)
print("CHAOS V1.0 - CONFIGURATION VERIFICATION")
print("=" * 70)

# Detect environment
import sys
if 'google.colab' in sys.modules:
    BASE_DIR = Path("/content/drive/MyDrive/chaos_v1.0")
    ENV = "COLAB"
else:
    BASE_DIR = Path(r"G:\My Drive\chaos_v1.0")
    ENV = "LOCAL (Windows)"

print(f"Environment: {ENV}")
print(f"Base path: {BASE_DIR}")

# Expected configuration
TARGET_COL = 'target_3class_8'
RETURNS_COL = 'target_return_8'
FILE_PATTERN = "{pair}_{tf}_phase2.parquet"

checks = []

# Check 1: Base directory exists
check1 = BASE_DIR.exists()
checks.append(("Base directory exists", check1))
print(f"\n[{'PASS' if check1 else 'FAIL'}] Base directory exists: {BASE_DIR}")

# Check 2: Features directory exists
features_dir = BASE_DIR / "features"
check2 = features_dir.exists()
checks.append(("Features directory exists", check2))
print(f"[{'PASS' if check2 else 'FAIL'}] Features directory exists")

# Check 3: Load test file
test_file = features_dir / "EURUSD_H1_phase2.parquet"
check3 = test_file.exists()
checks.append(("Test file exists", check3))
print(f"[{'PASS' if check3 else 'FAIL'}] Test file exists: {test_file.name}")

if check3:
    df = pd.read_parquet(test_file)
    print(f"\n  File loaded: {len(df):,} rows × {len(df.columns)} columns")

    # Check 4: Target column
    check4 = TARGET_COL in df.columns
    checks.append(("Target column exists", check4))
    if check4:
        values = sorted(df[TARGET_COL].dropna().unique())
        print(f"[PASS] Target column: {TARGET_COL}")
        print(f"       Values: {values}")
    else:
        print(f"[FAIL] Target column: {TARGET_COL} NOT FOUND")
        target_cols = [c for c in df.columns if 'target' in c.lower()]
        print(f"       Available: {target_cols[:10]}")

    # Check 5: Returns column
    check5 = RETURNS_COL in df.columns
    checks.append(("Returns column exists", check5))
    if check5:
        r = df[RETURNS_COL]
        print(f"[PASS] Returns column: {RETURNS_COL}")
        print(f"       Range: [{r.min():.6f}, {r.max():.6f}]")
    else:
        print(f"[FAIL] Returns column: {RETURNS_COL} NOT FOUND")
        return_cols = [c for c in df.columns if 'return' in c.lower()]
        print(f"       Available: {return_cols[:10]}")

# Check 6: Models directory
models_dir = BASE_DIR / "models"
check6 = models_dir.exists()
checks.append(("Models directory exists", check6))
print(f"\n[{'PASS' if check6 else 'FAIL'}] Models directory exists")

# Check 7: Checkpoint file format
checkpoint_path = models_dir / "rf_et_checkpoint.json"
if checkpoint_path.exists():
    import json
    with open(checkpoint_path, 'r') as f:
        cp = json.load(f)
    check7 = isinstance(cp.get('completed'), list)
    checks.append(("Checkpoint uses list format", check7))
    print(f"[{'PASS' if check7 else 'FAIL'}] Checkpoint format: {'list' if check7 else 'dict'}")
    print(f"       Completed: {len(cp.get('completed', []))} models")
else:
    print(f"[INFO] Checkpoint file not found (will be created)")
    checks.append(("Checkpoint uses list format", True))

# Summary
print("\n" + "=" * 70)
passed = sum(1 for _, v in checks if v)
total = len(checks)
print(f"VERIFICATION RESULT: {passed}/{total} CHECKS PASSED")
print("=" * 70)

if passed == total:
    print("\nALL CHECKS PASSED!")
    print(f"\nVerified configuration:")
    print(f"  TARGET_COL = '{TARGET_COL}'")
    print(f"  RETURNS_COL = '{RETURNS_COL}'")
    print(f"  FILE_PATTERN = '{FILE_PATTERN}'")
    print(f"\nReady to run: python chaos_rf_et_training.py")
else:
    print(f"\nWARNING: {total - passed} checks failed!")

print("=" * 70)
