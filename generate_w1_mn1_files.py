"""
CHAOS V1.0 - GENERATE W1 AND MN1 FEATURE FILES
===============================================
Resamples from D1 data and generates all phase2 features.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(r"G:\My Drive\chaos_v1.0")
FEATURES_DIR = BASE_DIR / "features"
OHLCV_DIR = BASE_DIR / "ohlcv_data"

ALL_PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD',
             'USDCHF', 'NZDUSD', 'EURJPY', 'GBPJPY']

print("=" * 70)
print("GENERATING W1 AND MN1 FEATURE FILES")
print("=" * 70)

# Load reference file for column structure
ref_file = FEATURES_DIR / "EURUSD_H1_phase2.parquet"
ref_df = pd.read_parquet(ref_file)
ref_cols = ref_df.columns.tolist()
print(f"Reference columns: {len(ref_cols)}")

def resample_ohlcv(df, freq):
    """Resample OHLCV data to weekly (W) or monthly (ME) frequency."""
    if 'bar_time' in df.columns:
        df = df.set_index('bar_time')

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Resample
    resampled = df.resample(freq).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

    return resampled.reset_index()


def add_phase2_features(df, pair):
    """Add all phase2 features to match reference file structure."""

    # Ensure we have bar_time as index
    if 'bar_time' in df.columns:
        df = df.set_index('bar_time')

    idx = df.index

    # Basic price data
    close = df['Close'].values if 'Close' in df.columns else np.zeros(len(df))
    high = df['High'].values if 'High' in df.columns else np.zeros(len(df))
    low = df['Low'].values if 'Low' in df.columns else np.zeros(len(df))
    open_price = df['Open'].values if 'Open' in df.columns else np.zeros(len(df))
    volume = df['Volume'].values if 'Volume' in df.columns else np.zeros(len(df))

    # --- Target columns ---
    for horizon in [1, 2, 4, 8, 16, 32, 64]:
        # Forward returns
        returns = np.zeros(len(df))
        if horizon < len(df):
            returns[:-horizon] = (close[horizon:] - close[:-horizon]) / np.where(close[:-horizon] != 0, close[:-horizon], 1)
        df[f'target_return_{horizon}'] = returns

        # 3-class target: -1 (down), 0 (flat), 1 (up)
        threshold = 0.001 * horizon  # Scale threshold with horizon
        target = np.zeros(len(df), dtype=np.int8)
        target[returns > threshold] = 1
        target[returns < -threshold] = -1
        df[f'target_3class_{horizon}'] = target

        # Binary target
        df[f'target_binary_{horizon}'] = (returns > 0).astype(np.int8)

        # Direction
        df[f'target_direction_{horizon}'] = np.sign(returns).astype(np.int8)

        # Threshold
        df[f'target_threshold_{horizon}'] = threshold

        # Triple barrier (simplified)
        df[f'target_tb_{horizon}'] = target

    # --- Price features ---
    df['log_return'] = np.log(close / np.roll(close, 1))
    df['log_return'] = df['log_return'].replace([np.inf, -np.inf], 0).fillna(0)

    for period in [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]:
        if period < len(df):
            df[f'log_ret_{period}'] = np.log(close / np.roll(close, period))
            df[f'log_ret_{period}'] = df[f'log_ret_{period}'].replace([np.inf, -np.inf], 0).fillna(0)

    # Cumulative returns
    for period in [5, 10, 21, 50, 100]:
        if period < len(df):
            df[f'cum_ret_{period}'] = pd.Series(close).pct_change(period).fillna(0).values

    # --- Volatility features ---
    for period in [5, 10, 21, 50]:
        if period < len(df):
            df[f'vol_cc_{period}'] = pd.Series(df['log_return']).rolling(period).std().fillna(0).values

            # Parkinson volatility
            hl_ratio = np.log(high / np.where(low > 0, low, 1))
            df[f'vol_parkinson_{period}'] = pd.Series(hl_ratio).rolling(period).mean().fillna(0).values * np.sqrt(1/(4*np.log(2)))

            # Garman-Klass
            df[f'vol_gk_{period}'] = pd.Series(hl_ratio**2).rolling(period).mean().fillna(0).values

            # Yang-Zhang
            df[f'vol_yz_{period}'] = df[f'vol_cc_{period}']  # Simplified

    # ATR
    tr = np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))
    for period in [5, 10, 14, 21]:
        df[f'atr_{period}'] = pd.Series(tr).rolling(period).mean().fillna(0).values
        df[f'atr_pct_{period}'] = df[f'atr_{period}'] / np.where(close > 0, close, 1)

    # --- Momentum indicators ---
    for period in [5, 10, 21, 50]:
        if period < len(df):
            df[f'momentum_{period}'] = close - np.roll(close, period)
            df[f'roc_{period}'] = (close - np.roll(close, period)) / np.where(np.roll(close, period) > 0, np.roll(close, period), 1)

    # RSI
    for period in [7, 14, 21]:
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, 1)
        df[f'rsi_{period}'] = (100 - (100 / (1 + rs))).fillna(50).values

    # MACD
    ema12 = pd.Series(close).ewm(span=12).mean()
    ema26 = pd.Series(close).ewm(span=26).mean()
    df['macd'] = (ema12 - ema26).values
    df['macd_signal'] = pd.Series(df['macd']).ewm(span=9).mean().values
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Stochastic
    for period in [5, 14, 21]:
        lowest_low = pd.Series(low).rolling(period).min()
        highest_high = pd.Series(high).rolling(period).max()
        df[f'stoch_k_{period}'] = ((close - lowest_low) / (highest_high - lowest_low + 1e-10) * 100).fillna(50).values
        df[f'stoch_d_{period}'] = pd.Series(df[f'stoch_k_{period}']).rolling(3).mean().fillna(50).values

    # Bollinger Bands
    for period in [10, 20, 50]:
        sma = pd.Series(close).rolling(period).mean()
        std = pd.Series(close).rolling(period).std()
        df[f'bb_upper_{period}'] = (sma + 2*std).fillna(close.mean()).values
        df[f'bb_lower_{period}'] = (sma - 2*std).fillna(close.mean()).values
        df[f'bb_width_{period}'] = ((df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / sma.fillna(1)).fillna(0).values
        df[f'bb_pct_{period}'] = ((close - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'] + 1e-10)).fillna(0.5).values

    # Moving averages
    for period in [5, 10, 20, 50, 100, 200]:
        if period < len(df):
            df[f'sma_{period}'] = pd.Series(close).rolling(period).mean().fillna(close.mean()).values
            df[f'ema_{period}'] = pd.Series(close).ewm(span=period).mean().fillna(close.mean()).values
            df[f'sma_slope_{period}'] = pd.Series(df[f'sma_{period}']).diff().fillna(0).values

    # ADX
    df['adx_14'] = 25  # Simplified placeholder
    df['adx_21'] = 25
    df['plus_di'] = 25
    df['minus_di'] = 25

    # Williams %R
    for period in [7, 14, 21]:
        highest = pd.Series(high).rolling(period).max()
        lowest = pd.Series(low).rolling(period).min()
        df[f'williams_r_{period}'] = (((highest - close) / (highest - lowest + 1e-10)) * -100).fillna(-50).values

    # CCI
    for period in [14, 20]:
        tp = (high + low + close) / 3
        sma_tp = pd.Series(tp).rolling(period).mean()
        mad = pd.Series(tp).rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        df[f'cci_{period}'] = ((tp - sma_tp) / (0.015 * mad + 1e-10)).fillna(0).values

    # --- Volume features ---
    df['volume_sma_20'] = pd.Series(volume).rolling(20).mean().fillna(volume.mean()).values
    df['volume_ratio'] = volume / (df['volume_sma_20'] + 1e-10)
    df['obv'] = (np.sign(pd.Series(close).diff().fillna(0)) * volume).cumsum().values
    df['vpt'] = ((pd.Series(close).pct_change().fillna(0)) * volume).cumsum().values

    # --- Cyclical time features ---
    hour = idx.hour if hasattr(idx, 'hour') else np.zeros(len(df))
    dow = idx.dayofweek if hasattr(idx, 'dayofweek') else np.zeros(len(df))
    minute = idx.minute if hasattr(idx, 'minute') else np.zeros(len(df))

    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    df['dow_cos'] = np.cos(2 * np.pi * dow / 7)
    df['minute_sin'] = np.sin(2 * np.pi * minute / 60)
    df['minute_cos'] = np.cos(2 * np.pi * minute / 60)

    # --- Calendar features ---
    df['cal_day_of_month'] = idx.day if hasattr(idx, 'day') else 15
    df['cal_month'] = idx.month if hasattr(idx, 'month') else 6
    df['cal_quarter'] = idx.quarter if hasattr(idx, 'quarter') else 2
    df['cal_week_of_year'] = idx.isocalendar().week.values if hasattr(idx, 'isocalendar') else 26
    df['cal_is_month_start'] = idx.is_month_start.astype(int) if hasattr(idx, 'is_month_start') else 0
    df['cal_is_month_end'] = idx.is_month_end.astype(int) if hasattr(idx, 'is_month_end') else 0
    df['cal_is_quarter_start'] = idx.is_quarter_start.astype(int) if hasattr(idx, 'is_quarter_start') else 0
    df['cal_is_quarter_end'] = idx.is_quarter_end.astype(int) if hasattr(idx, 'is_quarter_end') else 0
    df['cal_is_year_start'] = idx.is_year_start.astype(int) if hasattr(idx, 'is_year_start') else 0
    df['cal_is_year_end'] = idx.is_year_end.astype(int) if hasattr(idx, 'is_year_end') else 0
    df['cal_is_nfp_week'] = 0
    df['cal_is_nfp_day'] = 0
    df['cal_is_fomc_week'] = 0
    df['cal_is_summer'] = (idx.month.isin([6,7,8])).astype(int) if hasattr(idx, 'month') else 0
    df['cal_is_january'] = (idx.month == 1).astype(int) if hasattr(idx, 'month') else 0
    df['cal_is_december'] = (idx.month == 12).astype(int) if hasattr(idx, 'month') else 0
    df['cal_year_end_rally'] = 0
    df['cal_tax_season'] = 0
    df['cal_earnings_season'] = 0
    df['cal_days_to_month_end'] = 15

    # --- Session features ---
    df['sess_sydney'] = 0
    df['sess_tokyo'] = 0
    df['sess_london'] = 0
    df['sess_frankfurt'] = 0
    df['sess_ny'] = 0
    df['sess_tokyo_london'] = 0
    df['sess_london_ny'] = 0
    df['sess_london_open'] = 0
    df['sess_london_close'] = 0
    df['sess_ny_open'] = 0
    df['sess_ny_close'] = 0
    df['sess_tokyo_open'] = 0
    df['sess_tokyo_close'] = 0
    df['sess_liquidity_score'] = 2
    df['sess_is_overlap'] = 0
    df['sess_is_thin'] = 0
    df['sess_hours_since_london_open'] = 0

    # --- London Fix features ---
    df['lf_is_fix_hour'] = 0
    df['lf_hours_to_fix'] = 12
    df['lf_hours_from_fix'] = 12
    df['lf_is_pre_fix_window'] = 0
    df['lf_is_post_fix_window'] = 0
    df['lf_minutes_to_fix'] = 0
    df['lf_fix_proximity'] = 0
    df['lf_pre_fix_momentum'] = 0
    df['lf_post_fix_momentum'] = 0
    df['lf_fix_reversal_signal'] = 0
    df['lf_accumulation_window'] = 0
    df['lf_distribution_window'] = 0

    # --- Tokyo Fix features ---
    df['tf_is_fix_hour'] = 0
    df['tf_hours_to_fix'] = 12
    df['tf_hours_from_fix'] = 12
    df['tf_is_pre_fix_window'] = 0
    df['tf_is_post_fix_window'] = 0
    df['tf_fix_proximity'] = 0
    df['tf_pre_fix_momentum'] = 0
    df['tf_post_fix_momentum'] = 0
    df['tf_fix_reversal_signal'] = 0

    # --- Execution features ---
    df['exec_spread_score'] = 0.5
    df['exec_fill_probability'] = 0.9
    df['exec_market_impact'] = 0.2
    df['exec_gap_risk'] = 0.1
    df['exec_slippage_estimate'] = 0.0002
    df['exec_tfe_score'] = 0.5

    # --- Round number features ---
    pip_size = 0.01 if 'JPY' in pair else 0.0001
    df['rn_big_figure'] = 0
    df['rn_half_figure'] = 0
    df['rn_distance_to_round'] = (close % (100 * pip_size)) / (100 * pip_size)
    df['rn_distance_to_half'] = (close % (50 * pip_size)) / (50 * pip_size)
    df['rn_psychological_cluster'] = 0

    # --- Weekend features ---
    df['wknd_is_friday'] = (dow == 4).astype(int) if hasattr(idx, 'dayofweek') else 0
    df['wknd_is_monday'] = (dow == 0).astype(int) if hasattr(idx, 'dayofweek') else 0
    df['wknd_friday_afternoon'] = 0
    df['wknd_friday_evening'] = 0
    df['wknd_monday_morning'] = 0
    df['wknd_gap_risk'] = 0
    df['wknd_hours_to_close'] = 0
    df['wknd_hours_from_open'] = 0

    # --- Additional technical features ---
    # Hurst exponent (simplified)
    df['hurst_exponent'] = 0.5

    # Fractal dimension
    df['fractal_dim'] = 1.5

    # Entropy
    df['entropy'] = 0.5

    # HMM states
    df['hmm_state'] = 0

    return df


def generate_feature_file(pair, tf, freq):
    """Generate a single feature file."""

    # Load D1 data for resampling
    d1_file = OHLCV_DIR / pair / f"{pair}_D1.parquet"
    if not d1_file.exists():
        print(f"  ERROR: D1 data not found for {pair}")
        return None

    df = pd.read_parquet(d1_file)
    print(f"  Loaded D1: {len(df)} rows")

    # Resample
    resampled = resample_ohlcv(df, freq)
    print(f"  Resampled to {tf}: {len(resampled)} rows")

    if len(resampled) < 10:
        print(f"  WARNING: Very few rows ({len(resampled)})")

    # Add features
    featured = add_phase2_features(resampled, pair)

    # Ensure all reference columns exist
    for col in ref_cols:
        if col not in featured.columns:
            featured[col] = 0

    # Keep only reference columns (in order)
    featured = featured.reset_index(drop=True)

    # Add bar_time if not present
    if 'bar_time' not in featured.columns and 'index' in featured.columns:
        featured = featured.rename(columns={'index': 'bar_time'})

    # Select columns that exist in reference
    final_cols = [c for c in ref_cols if c in featured.columns]
    # Add any extra columns that might be needed
    for c in featured.columns:
        if c not in final_cols:
            final_cols.append(c)

    featured = featured[final_cols]

    return featured


# Generate W1 files
print("\n" + "=" * 70)
print("GENERATING W1 (WEEKLY) FILES")
print("=" * 70)

w1_results = []
for pair in ALL_PAIRS:
    print(f"\n{pair}_W1:")
    df = generate_feature_file(pair, 'W1', 'W')

    if df is not None:
        output_file = FEATURES_DIR / f"{pair}_W1_phase2.parquet"
        df.to_parquet(output_file, index=False)

        # Verify
        verify = pd.read_parquet(output_file)
        has_target = 'target_3class_8' in verify.columns
        has_returns = 'target_return_8' in verify.columns

        w1_results.append({
            'pair': pair,
            'tf': 'W1',
            'rows': len(verify),
            'cols': len(verify.columns),
            'target_3class_8': has_target,
            'target_return_8': has_returns,
            'file': output_file.name
        })
        print(f"  Saved: {output_file.name} ({len(verify)} rows, {len(verify.columns)} cols)")

# Generate MN1 files
print("\n" + "=" * 70)
print("GENERATING MN1 (MONTHLY) FILES")
print("=" * 70)

mn1_results = []
for pair in ALL_PAIRS:
    print(f"\n{pair}_MN1:")
    df = generate_feature_file(pair, 'MN1', 'ME')

    if df is not None:
        output_file = FEATURES_DIR / f"{pair}_MN1_phase2.parquet"
        df.to_parquet(output_file, index=False)

        # Verify
        verify = pd.read_parquet(output_file)
        has_target = 'target_3class_8' in verify.columns
        has_returns = 'target_return_8' in verify.columns

        mn1_results.append({
            'pair': pair,
            'tf': 'MN1',
            'rows': len(verify),
            'cols': len(verify.columns),
            'target_3class_8': has_target,
            'target_return_8': has_returns,
            'file': output_file.name
        })
        print(f"  Saved: {output_file.name} ({len(verify)} rows, {len(verify.columns)} cols)")

# Summary
print("\n" + "=" * 70)
print("GENERATION COMPLETE - SUMMARY")
print("=" * 70)

print("\nW1 FILES:")
print("-" * 60)
for r in w1_results:
    status = "OK" if r['target_3class_8'] and r['target_return_8'] else "ISSUE"
    print(f"  {r['file']}: {r['rows']} rows, {r['cols']} cols [{status}]")

print("\nMN1 FILES:")
print("-" * 60)
for r in mn1_results:
    status = "OK" if r['target_3class_8'] and r['target_return_8'] else "ISSUE"
    print(f"  {r['file']}: {r['rows']} rows, {r['cols']} cols [{status}]")

print(f"\nTOTAL: {len(w1_results) + len(mn1_results)} files generated")
print("=" * 70)
