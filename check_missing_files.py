"""
CHAOS V1.0 - CHECK MISSING FILES
================================
Identifies what feature files are missing.
"""
from pathlib import Path

BASE_DIR = Path(r"G:\My Drive\chaos_v1.0")
FEATURES_DIR = BASE_DIR / "features"
OHLCV_DIR = BASE_DIR / "ohlcv_data"

ALL_PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD',
             'USDCHF', 'NZDUSD', 'EURJPY', 'GBPJPY']

# Timeframes with OHLCV data available
AVAILABLE_TFS = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1']

# Additional timeframes (need resampling)
EXTRA_TFS = ['W1', 'MN1']

print("=" * 70)
print("CHAOS V1.0 - FILE INVENTORY")
print("=" * 70)

# Check OHLCV data
print("\n[1] OHLCV DATA STATUS")
print("-" * 40)
ohlcv_missing = []
for pair in ALL_PAIRS:
    for tf in AVAILABLE_TFS:
        ohlcv_file = OHLCV_DIR / pair / f"{pair}_{tf}.parquet"
        if not ohlcv_file.exists():
            ohlcv_missing.append(f"{pair}_{tf}")

if ohlcv_missing:
    print(f"Missing OHLCV: {len(ohlcv_missing)}")
    for m in ohlcv_missing:
        print(f"  - {m}")
else:
    print(f"All {len(ALL_PAIRS) * len(AVAILABLE_TFS)} OHLCV files present")

# Check phase2 feature files
print("\n[2] PHASE2 FEATURE FILES STATUS")
print("-" * 40)
phase2_missing = []
phase2_present = []
for pair in ALL_PAIRS:
    for tf in AVAILABLE_TFS:
        feature_file = FEATURES_DIR / f"{pair}_{tf}_phase2.parquet"
        if not feature_file.exists():
            phase2_missing.append(f"{pair}_{tf}_phase2.parquet")
        else:
            phase2_present.append(f"{pair}_{tf}")

print(f"Present: {len(phase2_present)} files")
if phase2_missing:
    print(f"Missing: {len(phase2_missing)} files")
    for m in phase2_missing:
        print(f"  - {m}")
else:
    print("All phase2 files present!")

# Check W1/MN1 (these don't exist)
print("\n[3] W1/MN1 STATUS (OPTIONAL)")
print("-" * 40)
w1_mn1_missing = []
for pair in ALL_PAIRS:
    for tf in EXTRA_TFS:
        feature_file = FEATURES_DIR / f"{pair}_{tf}_phase2.parquet"
        if not feature_file.exists():
            w1_mn1_missing.append(f"{pair}_{tf}_phase2.parquet")

print(f"W1/MN1 files: {18 - len(w1_mn1_missing)}/18 present")
print(f"NOTE: W1/MN1 require resampling from D1 data")
print(f"      W1 = ~520 rows/10yrs, MN1 = ~120 rows/10yrs (may be too few for ML)")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
CURRENT STATE:
  - Phase2 feature files: {len(phase2_present)}/{len(ALL_PAIRS) * len(AVAILABLE_TFS)}
  - W1/MN1 feature files: 0/18 (not generated)

MISSING (PHASE2):
{chr(10).join(f'  - {m}' for m in phase2_missing) if phase2_missing else '  None - all present!'}

W1/MN1 CONSIDERATION:
  - These timeframes have very few data points
  - W1 (weekly): ~52 bars/year × 10 years = ~520 rows
  - MN1 (monthly): ~12 bars/year × 10 years = ~120 rows
  - May not be suitable for ML training (data leakage risk)

RECOMMENDATION:
  - Focus on M15, M30, H1, H4, D1 (sufficient data)
  - Skip W1/MN1 for ML models (use rule-based for longer timeframes)
""")
print("=" * 70)
