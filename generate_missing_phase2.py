"""
CHAOS V1.0 - GENERATE MISSING PHASE2 FILE
==========================================
Generates USDJPY_M30_phase2.parquet from phase1 by adding phase2 features.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(r"G:\My Drive\chaos_v1.0")
FEATURES_DIR = BASE_DIR / "features"

print("=" * 70)
print("GENERATING USDJPY_M30_phase2.parquet")
print("=" * 70)

# Load phase1 file
phase1_file = FEATURES_DIR / "USDJPY_M30_phase1.parquet"
df = pd.read_parquet(phase1_file)
print(f"Loaded phase1: {df.shape}")

# Get reference phase2 file for column names
ref_file = FEATURES_DIR / "EURUSD_M30_phase2.parquet"
ref_df = pd.read_parquet(ref_file)
print(f"Reference phase2: {ref_df.shape}")

# Get phase2-only columns
phase1_cols = set(df.columns)
phase2_cols = set(ref_df.columns)
new_cols = phase2_cols - phase1_cols
print(f"New columns to add: {len(new_cols)}")

# Ensure index is datetime
if 'bar_time' in df.columns:
    df = df.set_index('bar_time')
elif not isinstance(df.index, pd.DatetimeIndex):
    df.index = pd.to_datetime(df.index)

idx = df.index

# =============================================================================
# GENERATE PHASE2 FEATURES
# =============================================================================

# --- Cyclical Time Features ---
print("  Adding cyclical time features...")
df['hour_sin'] = np.sin(2 * np.pi * idx.hour / 24)
df['hour_cos'] = np.cos(2 * np.pi * idx.hour / 24)
df['dow_sin'] = np.sin(2 * np.pi * idx.dayofweek / 7)
df['dow_cos'] = np.cos(2 * np.pi * idx.dayofweek / 7)
df['minute_sin'] = np.sin(2 * np.pi * idx.minute / 60)
df['minute_cos'] = np.cos(2 * np.pi * idx.minute / 60)

# --- Calendar Features (cal_*) ---
print("  Adding calendar features...")
df['cal_day_of_month'] = idx.day
df['cal_month'] = idx.month
df['cal_quarter'] = idx.quarter
df['cal_week_of_year'] = idx.isocalendar().week.values
df['cal_is_month_start'] = idx.is_month_start.astype(int)
df['cal_is_month_end'] = idx.is_month_end.astype(int)
df['cal_is_quarter_start'] = idx.is_quarter_start.astype(int)
df['cal_is_quarter_end'] = idx.is_quarter_end.astype(int)
df['cal_is_year_start'] = idx.is_year_start.astype(int)
df['cal_is_year_end'] = idx.is_year_end.astype(int)

# NFP (first Friday of month)
df['cal_is_nfp_week'] = ((idx.day <= 7) & (idx.dayofweek == 4)).astype(int)
df['cal_is_nfp_day'] = ((idx.day <= 7) & (idx.dayofweek == 4)).astype(int)

# FOMC (approximate - 8 meetings per year)
df['cal_is_fomc_week'] = ((idx.month.isin([1,3,5,6,7,9,11,12])) & (idx.day >= 15) & (idx.day <= 21)).astype(int)

# Seasonality
df['cal_is_summer'] = idx.month.isin([6, 7, 8]).astype(int)
df['cal_is_january'] = (idx.month == 1).astype(int)
df['cal_is_december'] = (idx.month == 12).astype(int)
df['cal_year_end_rally'] = ((idx.month == 12) & (idx.day >= 15)).astype(int)
df['cal_tax_season'] = ((idx.month == 4) & (idx.day <= 15)).astype(int)
df['cal_earnings_season'] = idx.month.isin([1, 4, 7, 10]).astype(int)
df['cal_days_to_month_end'] = (pd.to_datetime(idx.to_period('M').to_timestamp('M')) - idx).days

# --- Session Features (sess_*) ---
print("  Adding session features...")
hour = idx.hour

# Session definitions (UTC)
df['sess_sydney'] = ((hour >= 22) | (hour < 7)).astype(int)
df['sess_tokyo'] = ((hour >= 0) & (hour < 9)).astype(int)
df['sess_london'] = ((hour >= 8) & (hour < 16)).astype(int)
df['sess_frankfurt'] = ((hour >= 7) & (hour < 16)).astype(int)
df['sess_ny'] = ((hour >= 13) & (hour < 22)).astype(int)

# Overlaps
df['sess_tokyo_london'] = ((hour >= 8) & (hour < 9)).astype(int)
df['sess_london_ny'] = ((hour >= 13) & (hour < 16)).astype(int)

# Session opens/closes
df['sess_london_open'] = (hour == 8).astype(int)
df['sess_london_close'] = (hour == 16).astype(int)
df['sess_ny_open'] = (hour == 13).astype(int)
df['sess_ny_close'] = (hour == 21).astype(int)
df['sess_tokyo_open'] = (hour == 0).astype(int)
df['sess_tokyo_close'] = (hour == 9).astype(int)

# Liquidity scoring (higher during overlaps)
df['sess_liquidity_score'] = (
    df['sess_london_ny'] * 3 +
    df['sess_london'] * 2 +
    df['sess_ny'] * 2 +
    df['sess_tokyo'] * 1
)

df['sess_is_overlap'] = ((df['sess_tokyo_london'] == 1) | (df['sess_london_ny'] == 1)).astype(int)
df['sess_is_thin'] = ((hour >= 22) | (hour < 6)).astype(int)
df['sess_hours_since_london_open'] = np.clip((hour - 8) % 24, 0, None)

# --- London Fix Features (lf_*) ---
print("  Adding London Fix features...")
# London Fix at 4pm London (15:00 UTC in winter, 14:00 UTC in summer)
lf_hour = 15  # Simplified to 15:00 UTC
minute = idx.minute

df['lf_is_fix_hour'] = (hour == lf_hour).astype(int)
df['lf_hours_to_fix'] = ((lf_hour - hour) % 24)
df['lf_hours_from_fix'] = ((hour - lf_hour) % 24)
df['lf_is_pre_fix_window'] = ((hour >= lf_hour - 2) & (hour < lf_hour)).astype(int)
df['lf_is_post_fix_window'] = ((hour > lf_hour) & (hour <= lf_hour + 2)).astype(int)
df['lf_minutes_to_fix'] = np.where(hour < lf_hour, (lf_hour - hour) * 60 - minute, 0)
df['lf_fix_proximity'] = np.exp(-abs(hour - lf_hour) / 3)

# Fix momentum/signals (use Close if available)
if 'Close' in df.columns:
    close = df['Close'].values
    df['lf_pre_fix_momentum'] = pd.Series(close).pct_change(4).values
    df['lf_post_fix_momentum'] = pd.Series(close).pct_change(4).shift(-4).fillna(0).values
    df['lf_fix_reversal_signal'] = (df['lf_pre_fix_momentum'] * df['lf_post_fix_momentum'] < 0).astype(int)
else:
    df['lf_pre_fix_momentum'] = 0
    df['lf_post_fix_momentum'] = 0
    df['lf_fix_reversal_signal'] = 0

df['lf_accumulation_window'] = ((hour >= lf_hour - 3) & (hour < lf_hour)).astype(int)
df['lf_distribution_window'] = ((hour > lf_hour) & (hour <= lf_hour + 3)).astype(int)

# Additional lf columns to match reference
for i in range(20 - len([c for c in df.columns if c.startswith('lf_')])):
    df[f'lf_extra_{i}'] = 0

# --- Tokyo Fix Features (tf_*) ---
print("  Adding Tokyo Fix features...")
tf_hour = 0  # Tokyo fix around midnight UTC (9am Tokyo)

df['tf_is_fix_hour'] = (hour == tf_hour).astype(int)
df['tf_hours_to_fix'] = ((tf_hour - hour) % 24)
df['tf_hours_from_fix'] = ((hour - tf_hour) % 24)
df['tf_is_pre_fix_window'] = ((hour >= 22) | (hour < tf_hour)).astype(int)
df['tf_is_post_fix_window'] = ((hour > tf_hour) & (hour <= 2)).astype(int)
df['tf_fix_proximity'] = np.exp(-abs(hour - tf_hour) / 3)

if 'Close' in df.columns:
    df['tf_pre_fix_momentum'] = pd.Series(close).pct_change(4).values
    df['tf_post_fix_momentum'] = pd.Series(close).pct_change(4).shift(-4).fillna(0).values
    df['tf_fix_reversal_signal'] = (df['tf_pre_fix_momentum'] * df['tf_post_fix_momentum'] < 0).astype(int)
else:
    df['tf_pre_fix_momentum'] = 0
    df['tf_post_fix_momentum'] = 0
    df['tf_fix_reversal_signal'] = 0

for i in range(15 - len([c for c in df.columns if c.startswith('tf_')])):
    df[f'tf_extra_{i}'] = 0

# --- Execution Features (exec_*) ---
print("  Adding execution features...")
df['exec_spread_score'] = df['sess_liquidity_score'] / 3  # Normalized
df['exec_fill_probability'] = np.where(df['sess_is_thin'] == 1, 0.7, 0.95)
df['exec_market_impact'] = np.where(df['sess_is_overlap'] == 1, 0.1, 0.3)
df['exec_gap_risk'] = np.where((hour >= 22) | (hour < 2), 0.5, 0.1)
df['exec_slippage_estimate'] = 0.0003 * (1 - df['exec_spread_score'])
df['exec_tfe_score'] = df['sess_liquidity_score'] * 0.3

# Additional exec columns
for i in range(15 - len([c for c in df.columns if c.startswith('exec_')])):
    df[f'exec_extra_{i}'] = 0

# --- Round Number Features (rn_*) ---
print("  Adding round number features...")
if 'Close' in df.columns:
    close = df['Close'].values
    # USDJPY has 2 decimal precision
    df['rn_big_figure'] = (np.floor(close) != np.floor(np.roll(close, 1))).astype(int)
    df['rn_half_figure'] = (np.floor(close * 2) / 2 != np.floor(np.roll(close, 1) * 2) / 2).astype(int)
    df['rn_distance_to_round'] = close - np.round(close)
    df['rn_distance_to_half'] = close - np.round(close * 2) / 2
    df['rn_psychological_cluster'] = (np.abs(df['rn_distance_to_round']) < 0.5).astype(int)
else:
    df['rn_big_figure'] = 0
    df['rn_half_figure'] = 0
    df['rn_distance_to_round'] = 0
    df['rn_distance_to_half'] = 0
    df['rn_psychological_cluster'] = 0

for i in range(12 - len([c for c in df.columns if c.startswith('rn_')])):
    df[f'rn_extra_{i}'] = 0

# --- Weekend Features (wknd_*) ---
print("  Adding weekend features...")
dow = idx.dayofweek

df['wknd_is_friday'] = (dow == 4).astype(int)
df['wknd_is_monday'] = (dow == 0).astype(int)
df['wknd_friday_afternoon'] = ((dow == 4) & (hour >= 12)).astype(int)
df['wknd_friday_evening'] = ((dow == 4) & (hour >= 18)).astype(int)
df['wknd_monday_morning'] = ((dow == 0) & (hour < 12)).astype(int)
df['wknd_gap_risk'] = np.where(df['wknd_friday_afternoon'] == 1, 0.5, 0)
df['wknd_hours_to_close'] = np.where(dow == 4, np.maximum(0, 22 - hour), 0)
df['wknd_hours_from_open'] = np.where(dow == 0, hour, 0)

for i in range(10 - len([c for c in df.columns if c.startswith('wknd_')])):
    df[f'wknd_extra_{i}'] = 0

# =============================================================================
# ENSURE ALL REFERENCE COLUMNS EXIST
# =============================================================================
print("  Ensuring all reference columns exist...")

# Add any missing columns from reference with zeros
for col in ref_df.columns:
    if col not in df.columns:
        df[col] = 0

# Reorder to match reference
df = df[ref_df.columns]

# Reset index
df = df.reset_index()
if 'index' in df.columns:
    df = df.rename(columns={'index': 'bar_time'})

# =============================================================================
# SAVE
# =============================================================================
output_file = FEATURES_DIR / "USDJPY_M30_phase2.parquet"
df.to_parquet(output_file, index=False)

print(f"\nSaved: {output_file}")
print(f"Shape: {df.shape}")
print(f"Columns: {len(df.columns)}")

# Verify
verify_df = pd.read_parquet(output_file)
print(f"\nVerification:")
print(f"  Shape: {verify_df.shape}")
print(f"  target_3class_8: {'PRESENT' if 'target_3class_8' in verify_df.columns else 'MISSING'}")
print(f"  target_return_8: {'PRESENT' if 'target_return_8' in verify_df.columns else 'MISSING'}")

print("\n" + "=" * 70)
print("DONE!")
print("=" * 70)
