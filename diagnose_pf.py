#!/usr/bin/env python3
"""Data leakage / target-return diagnostic script."""

import pandas as pd
import numpy as np

print("=" * 70)
print("DATA LEAKAGE / TARGET-RETURN DIAGNOSTIC")
print("=" * 70)

# Load data
df = pd.read_parquet("G:/My Drive/chaos_v1.0/features/EURUSD_H1_features.parquet")
print(f"Loaded: {len(df):,} rows")

# ============================================================
# CHECK 1: Target-Return Alignment
# ============================================================
print("\n" + "=" * 70)
print("CHECK 1: TARGET-RETURN ALIGNMENT")
print("=" * 70)

target = df['target_3class_8'].values
returns = df['target_return_8'].values

# The target should PREDICT the return
# If target = LONG (2 or 1), return should be POSITIVE
# If target = SHORT (0 or -1), return should be NEGATIVE

if target.min() == -1:
    long_mask = target == 1
    short_mask = target == -1
else:
    long_mask = target == 2
    short_mask = target == 0

print(f"When target = LONG:")
print(f"  Count: {long_mask.sum():,}")
print(f"  Mean return: {returns[long_mask].mean():.6f}")
print(f"  Positive returns: {(returns[long_mask] > 0).sum()} ({(returns[long_mask] > 0).mean()*100:.1f}%)")

print(f"\nWhen target = SHORT:")
print(f"  Count: {short_mask.sum():,}")
print(f"  Mean return: {returns[short_mask].mean():.6f}")
print(f"  Negative returns: {(returns[short_mask] < 0).sum()} ({(returns[short_mask] < 0).mean()*100:.1f}%)")

# CRITICAL CHECK: What % of LONG signals have positive returns?
long_accuracy = (returns[long_mask] > 0).mean() * 100
short_accuracy = (returns[short_mask] < 0).mean() * 100

print(f"\n*** CRITICAL ***")
print(f"LONG signal accuracy: {long_accuracy:.1f}%")
print(f"SHORT signal accuracy: {short_accuracy:.1f}%")

if long_accuracy > 90 or short_accuracy > 90:
    print("!!! WARNING: >90% accuracy suggests TARGET CONTAINS FUTURE INFO !!!")
elif long_accuracy > 70 or short_accuracy > 70:
    print("!!! WARNING: >70% accuracy is suspicious - check for leakage !!!")
else:
    print("Target accuracy looks reasonable (not leaking future)")

# ============================================================
# CHECK 2: Feature-Target Correlation
# ============================================================
print("\n" + "=" * 70)
print("CHECK 2: FEATURE-TARGET CORRELATION (TOP 20)")
print("=" * 70)

# Exclude non-feature columns
exclude = ['target_', 'return', 'Open', 'High', 'Low', 'Close', 'Volume',
           'timestamp', 'date', 'pair', 'symbol', 'tf', 'Unnamed']
feature_cols = [c for c in df.columns if not any(p in c for p in exclude)]

# Calculate correlation with target
correlations = []
for col in feature_cols:
    try:
        corr = np.corrcoef(df[col].fillna(0).values, target)[0, 1]
        if not np.isnan(corr):
            correlations.append((col, abs(corr), corr))
    except:
        pass

# Sort by absolute correlation
correlations.sort(key=lambda x: x[1], reverse=True)

print("Features most correlated with target:")
for col, abs_corr, corr in correlations[:20]:
    flag = " <- SUSPICIOUS!" if abs_corr > 0.3 else ""
    print(f"  {col}: {corr:.4f}{flag}")

if len(correlations) > 0:
    if correlations[0][1] > 0.5:
        print("\n!!! CRITICAL: Feature with >0.5 correlation - LIKELY LEAKAGE !!!")
    elif correlations[0][1] > 0.3:
        print("\n!!! WARNING: Feature with >0.3 correlation - check for leakage !!!")

# ============================================================
# CHECK 3: Feature-Return Correlation
# ============================================================
print("\n" + "=" * 70)
print("CHECK 3: FEATURE-RETURN CORRELATION (TOP 20)")
print("=" * 70)

correlations_ret = []
for col in feature_cols:
    try:
        corr = np.corrcoef(df[col].fillna(0).values, returns)[0, 1]
        if not np.isnan(corr):
            correlations_ret.append((col, abs(corr), corr))
    except:
        pass

correlations_ret.sort(key=lambda x: x[1], reverse=True)

print("Features most correlated with returns:")
for col, abs_corr, corr in correlations_ret[:20]:
    flag = " <- SUSPICIOUS!" if abs_corr > 0.2 else ""
    print(f"  {col}: {corr:.4f}{flag}")

# ============================================================
# CHECK 4: Look for Obvious Leakage Column Names
# ============================================================
print("\n" + "=" * 70)
print("CHECK 4: SUSPICIOUS COLUMN NAMES")
print("=" * 70)

suspicious_patterns = ['future', 'forward', 'next', 'lead', 'target', 'return', 'label', 'y_']
suspicious_cols = []

for col in df.columns:
    col_lower = col.lower()
    for pattern in suspicious_patterns:
        if pattern in col_lower and col not in ['target_3class_8', 'target_return_8']:
            suspicious_cols.append(col)
            break

if suspicious_cols:
    print("!!! SUSPICIOUS COLUMNS FOUND !!!")
    for col in suspicious_cols:
        print(f"  - {col}")
else:
    print("No obviously suspicious column names found")

# ============================================================
# CHECK 5: Verify Returns Are Forward-Looking
# ============================================================
print("\n" + "=" * 70)
print("CHECK 5: VERIFY RETURNS CALCULATION")
print("=" * 70)

# If we have Close price, verify returns match
if 'Close' in df.columns:
    close = df['Close'].values

    # Calculate what 8-bar forward return SHOULD be
    expected_return_8 = (np.roll(close, -8) - close) / close
    expected_return_8[-8:] = 0  # Last 8 bars have no forward return

    actual_return_8 = returns

    # Compare (excluding last 8 bars)
    diff = np.abs(expected_return_8[:-8] - actual_return_8[:-8])
    match_pct = (diff < 0.0001).mean() * 100

    print(f"Return calculation verification:")
    print(f"  Expected vs Actual match: {match_pct:.1f}%")

    if match_pct < 95:
        print("  !!! WARNING: Returns may not be calculated correctly !!!")
    else:
        print("  Returns appear to be calculated correctly")
else:
    print("Close column not found - cannot verify return calculation")

# ============================================================
# CHECK 6: Perfect Prediction Test
# ============================================================
print("\n" + "=" * 70)
print("CHECK 6: PERFECT PREDICTION SANITY CHECK")
print("=" * 70)

# If we use the target as predictions, what's the PF?
# This tells us the theoretical maximum

if target.min() == -1:
    perfect_positions = target.astype(float)
else:
    perfect_positions = target.astype(float) - 1

pnl = perfect_positions * returns
gp = np.sum(pnl[pnl > 0])
gl = np.abs(np.sum(pnl[pnl < 0]))
perfect_pf = gp / gl if gl > 1e-10 else 99

print(f"Perfect prediction PF: {perfect_pf:.2f}")
print(f"  (This is the max PF possible if model predicted perfectly)")

if perfect_pf > 20:
    print("  !!! WARNING: Perfect PF > 20 suggests target definition issue !!!")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)

issues = []
if long_accuracy > 70 or short_accuracy > 70:
    issues.append("Target may contain future information")
if len(correlations) > 0 and correlations[0][1] > 0.3:
    issues.append(f"High feature-target correlation: {correlations[0][0]} = {correlations[0][2]:.3f}")
if suspicious_cols:
    issues.append(f"Suspicious columns: {suspicious_cols}")
if perfect_pf > 20:
    issues.append(f"Perfect PF = {perfect_pf:.1f} is too high")

if issues:
    print("ISSUES FOUND:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
else:
    print("No obvious issues found - investigate further")
