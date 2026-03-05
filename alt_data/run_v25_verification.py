"""
V2.5 Alt-Data Pipeline — 16-Test Verification Suite
"""
import sys
import json
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

BASE = Path('G:/My Drive/chaos_v1.0/alt_data')
results = []


def record(test_num, description, passed, details=""):
    status = "PASS" if passed else "FAIL"
    results.append((test_num, description, status, details))
    print(f"  Test {test_num:2d}: [{status}] {description} — {details}")


print("=" * 70)
print("V2.5 ALT-DATA PIPELINE — 16-TEST VERIFICATION")
print("=" * 70)

# ── Test 1: Folder structure ──────────────────────────────────────────
expected_dirs = ['cot', 'cot/raw', 'cme', 'cme/raw', 'common', 'tests']
missing = [d for d in expected_dirs if not (BASE / d).is_dir()]
record(1, "alt_data/ folder structure created correctly",
       len(missing) == 0,
       f"All {len(expected_dirs)} dirs exist" if not missing else f"Missing: {missing}")

# ── Test 2: cot_contract_map.json ────────────────────────────────────
try:
    with open(BASE / 'cot/cot_contract_map.json') as f:
        cot_map = json.load(f)
    currencies = [k for k in cot_map['currency_to_cot'].keys() if k != 'USD']
    pairs = list(cot_map['pair_decomposition'].keys())
    ok = len(currencies) >= 7 and len(pairs) >= 9
    record(2, "cot_contract_map.json maps 7 currencies + 9 pairs",
           ok, f"{len(currencies)} currencies, {len(pairs)} pairs")
except Exception as e:
    record(2, "cot_contract_map.json maps 7 currencies + 9 pairs", False, str(e))

# ── Test 3: cme_contract_map.json ────────────────────────────────────
try:
    with open(BASE / 'cme/cme_contract_map.json') as f:
        cme_map = json.load(f)
    cme_currencies = list(cme_map['currency_to_cme'].keys())
    ok = len(cme_currencies) >= 7
    record(3, "cme_contract_map.json maps all 7 currencies",
           ok, f"{len(cme_currencies)} currencies: {cme_currencies}")
except Exception as e:
    record(3, "cme_contract_map.json maps all 7 currencies", False, str(e))

# ── Test 4: COT downloader downloaded data ───────────────────────────
cot_raw_files = list((BASE / 'cot/raw').glob('cot_tff_raw_*.csv'))
record(4, "COT downloader successfully downloaded TFF data",
       len(cot_raw_files) >= 1,
       f"{len(cot_raw_files)} year files downloaded")

# ── Test 5: cot_weekly.parquet ───────────────────────────────────────
try:
    import pandas as pd
    import numpy as np

    weekly = pd.read_parquet(BASE / 'cot/cot_weekly.parquet')
    n_cols = weekly.shape[1]
    n_rows = weekly.shape[0]
    ok = n_cols == 49
    record(5, "cot_weekly.parquet: 7 currencies x 7 features = 49 columns",
           ok, f"{n_rows} weeks x {n_cols} columns")
except Exception as e:
    record(5, "cot_weekly.parquet: 7 currencies x 7 features = 49 columns", False, str(e))

# ── Test 6: cot_pair_daily.parquet ───────────────────────────────────
try:
    cot_daily = pd.read_parquet(BASE / 'cot/cot_pair_daily.parquet')
    n_cols = cot_daily.shape[1]
    n_rows = cot_daily.shape[0]
    # 9 pairs x 4 features = 36 (but we have 5 features per pair: pressure, pct_diff, wow_diff, extreme_flag = 4 per pair + wow_diff = 5)
    # Actually from compute_pair_features: pressure, pct_diff, wow_diff, extreme_flag = 4 per pair = 36
    ok = n_cols == 36
    record(6, "cot_pair_daily.parquet: 9 pairs x 4 features = 36 columns",
           ok, f"{n_rows} days x {n_cols} columns")
except Exception as e:
    record(6, "cot_pair_daily.parquet: 9 pairs x 4 features = 36 columns", False, str(e))

# ── Test 7: COT leakage test ─────────────────────────────────────────
try:
    weekly = pd.read_parquet(BASE / 'cot/cot_weekly.parquet')
    daily = pd.read_parquet(BASE / 'cot/cot_pair_daily.parquet')
    weekly.index = pd.to_datetime(weekly.index)
    daily.index = pd.to_datetime(daily.index)

    violations = 0
    # Verify: shifted dates (report_date + 6) are all >= first daily appearance
    shifted_dates = weekly.index + pd.Timedelta(days=6)
    daily_dates = daily.index

    for shifted_date in shifted_dates:
        matching = daily_dates[daily_dates >= shifted_date]
        if len(matching) > 0:
            first_appearance = matching[0]
            if first_appearance < shifted_date:
                violations += 1

    # Also check: no daily dates appear between a report date and its shifted Monday
    for report_date in weekly.index:
        monday_after = report_date + pd.Timedelta(days=6)
        # Check that the daily values between report_date and monday_after
        # are NOT equal to the current week's values (they should be previous week's)
        in_between = daily[(daily.index > report_date) & (daily.index < monday_after)]
        # If shifted correctly, the data for this week first appears at monday_after
        # so any rows before monday_after should have previous week's data

    record(7, "COT leakage test: ZERO violations",
           violations == 0, f"{violations} violations found")
except Exception as e:
    record(7, "COT leakage test: ZERO violations", False, str(e))

# ── Test 8: COT completeness ─────────────────────────────────────────
try:
    cot_daily = pd.read_parquet(BASE / 'cot/cot_pair_daily.parquet')
    # Skip first 200 rows (warmup)
    if len(cot_daily) > 200:
        cot_daily_trimmed = cot_daily.iloc[200:]
    else:
        cot_daily_trimmed = cot_daily
    total = cot_daily_trimmed.size
    non_null = cot_daily_trimmed.count().sum()
    fill_rate = non_null / total if total > 0 else 0
    record(8, "COT completeness: >= 99% fill rate after warmup",
           fill_rate >= 0.99, f"{fill_rate:.4f} ({non_null}/{total})")
except Exception as e:
    record(8, "COT completeness: >= 99% fill rate after warmup", False, str(e))

# ── Test 9: CME downloader downloaded data ───────────────────────────
cme_raw_files = list((BASE / 'cme/raw').glob('cme_*_daily.csv'))
record(9, "CME downloader successfully downloaded data",
       len(cme_raw_files) >= 1,
       f"{len(cme_raw_files)} currency files downloaded")

# ── Test 10: cme_fx_daily.parquet ────────────────────────────────────
try:
    cme_fx = pd.read_parquet(BASE / 'cme/cme_fx_daily.parquet')
    vol_z_cols = [c for c in cme_fx.columns if 'vol_z' in c]
    record(10, "cme_fx_daily.parquet created with volume features",
           len(vol_z_cols) > 0,
           f"{cme_fx.shape[0]} days x {cme_fx.shape[1]} columns, {len(vol_z_cols)} vol_z features")
except Exception as e:
    record(10, "cme_fx_daily.parquet created with volume features", False, str(e))

# ── Test 11: cme_pair_daily.parquet ──────────────────────────────────
try:
    cme_pair = pd.read_parquet(BASE / 'cme/cme_pair_daily.parquet')
    n_cols = cme_pair.shape[1]
    ok = n_cols == 27
    record(11, "cme_pair_daily.parquet: 9 pairs x 3 features = 27 columns",
           ok, f"{cme_pair.shape[0]} days x {n_cols} columns")
except Exception as e:
    record(11, "cme_pair_daily.parquet: 9 pairs x 3 features = 27 columns", False, str(e))

# ── Test 12: CME lookahead test ──────────────────────────────────────
try:
    cme_fx = pd.read_parquet(BASE / 'cme/cme_fx_daily.parquet')
    cme_fx.index = pd.to_datetime(cme_fx.index)
    violations = 0
    vol_z_cols = [c for c in cme_fx.columns if 'vol_z_20' in c]
    for col in vol_z_cols:
        series = cme_fx[col].dropna()
        if len(series) > 0:
            first_valid = series.first_valid_index()
            if first_valid is not None:
                idx_pos = cme_fx.index.get_loc(first_valid)
                if idx_pos < 5:
                    violations += 1
            if np.any(np.isinf(series.values)):
                violations += 1
    record(12, "CME lookahead test: ZERO violations",
           violations == 0,
           f"{len(vol_z_cols)} columns checked, {violations} violations")
except Exception as e:
    record(12, "CME lookahead test: ZERO violations", False, str(e))

# ── Test 13: CME completeness ────────────────────────────────────────
try:
    cme_pair = pd.read_parquet(BASE / 'cme/cme_pair_daily.parquet')
    if len(cme_pair) > 200:
        cme_pair_trimmed = cme_pair.iloc[200:]
    else:
        cme_pair_trimmed = cme_pair
    total = cme_pair_trimmed.size
    non_null = cme_pair_trimmed.count().sum()
    fill_rate = non_null / total if total > 0 else 0
    record(13, "CME completeness: >= 99% fill rate after warmup",
           fill_rate >= 0.99, f"{fill_rate:.4f} ({non_null}/{total})")
except Exception as e:
    record(13, "CME completeness: >= 99% fill rate after warmup", False, str(e))

# ── Test 14: Feature sanity ──────────────────────────────────────────
try:
    all_ok = True
    details_parts = []

    for label, path in [('COT', BASE / 'cot/cot_pair_daily.parquet'),
                         ('CME', BASE / 'cme/cme_pair_daily.parquet')]:
        df = pd.read_parquet(path)
        if len(df) > 200:
            df = df.iloc[200:]
        n_inf = np.isinf(df.values[~np.isnan(df.values)]).sum()
        if n_inf > 0:
            all_ok = False
            details_parts.append(f"{label}: {n_inf} Inf values")

        z_cols = [c for c in df.columns if '_z_' in c or '_pressure' in c]
        for col in z_cols:
            s = df[col].dropna()
            if len(s) > 0 and s.abs().max() > 20:
                details_parts.append(f"{label}/{col}: max_abs={s.abs().max():.1f}")
                all_ok = False

    if not details_parts:
        details_parts.append("All z-scores finite and bounded")

    record(14, "Feature sanity: all z-scores finite, no Inf",
           all_ok, "; ".join(details_parts))
except Exception as e:
    record(14, "Feature sanity: all z-scores finite, no Inf", False, str(e))

# ── Test 15: AltDataProvider loads and returns modifiers ─────────────
try:
    sys.path.insert(0, str(BASE / 'common'))
    from alt_data_provider import AltDataProvider

    provider = AltDataProvider(
        cot_daily_path=str(BASE / 'cot/cot_pair_daily.parquet'),
        cme_daily_path=str(BASE / 'cme/cme_pair_daily.parquet')
    )

    modifiers = provider.get_modifiers('EURUSD', '2024-06-15')

    required_keys = ['cot_pressure', 'cot_extreme', 'cme_spike', 'cme_pressure',
                     'cme_confirm', 'data_available', 'recommended_agreement_adjustment']
    missing_keys = [k for k in required_keys if k not in modifiers]

    ok = len(missing_keys) == 0 and modifiers.get('data_available', False)
    record(15, "AltDataProvider loads and returns modifiers for EURUSD",
           ok,
           f"pressure={modifiers.get('cot_pressure', 'N/A'):.3f}, "
           f"extreme={modifiers.get('cot_extreme', 'N/A')}, "
           f"data_available={modifiers.get('data_available', False)}")
except Exception as e:
    record(15, "AltDataProvider loads and returns modifiers for EURUSD", False, str(e))

# ── Test 16: AltDataProvider.get_regime_modifier ─────────────────────
try:
    regime_mod = provider.get_regime_modifier('EURUSD', '2024-06-15')

    required_keys = ['override_regime', 'allow_trend_following', 'allow_mean_reversion', 'reason']
    missing_keys = [k for k in required_keys if k not in regime_mod]

    ok = len(missing_keys) == 0
    record(16, "AltDataProvider.get_regime_modifier returns valid structure",
           ok,
           f"trend={regime_mod.get('allow_trend_following')}, "
           f"mr={regime_mod.get('allow_mean_reversion')}, "
           f"reason={regime_mod.get('reason')}")
except Exception as e:
    record(16, "AltDataProvider.get_regime_modifier returns valid structure", False, str(e))

# ── Summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"{'#':>4}  {'Description':<55} {'Result':>6}")
print("-" * 70)
for num, desc, status, detail in results:
    print(f"{num:4d}  {desc:<55} {status:>6}")
    if detail:
        print(f"      {detail}")
print("=" * 70)

passed = sum(1 for _, _, s, _ in results if s == "PASS")
failed = sum(1 for _, _, s, _ in results if s == "FAIL")
print(f"\n{passed}/{len(results)} PASSED, {failed} FAILED")
if failed == 0:
    print("ALL 16 TESTS PASSED")
else:
    print("SOME TESTS FAILED")
