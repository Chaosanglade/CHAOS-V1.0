"""
CHAOS V1.0 - GENERATE ALL MISSING FEATURE FILES
================================================
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(r"G:\My Drive\chaos_v1.0")
FEATURES_DIR = BASE_DIR / "features"
OHLCV_DIR = BASE_DIR / "ohlcv_data"

ALL_PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD',
             'USDCHF', 'NZDUSD', 'EURJPY', 'GBPJPY']

print("=" * 70)
print("GENERATING ALL MISSING FEATURE FILES")
print("=" * 70)

# Load reference
ref_file = FEATURES_DIR / "EURUSD_H1_features.parquet"
ref_df = pd.read_parquet(ref_file)
ref_cols = ref_df.columns.tolist()
print(f"Reference: {len(ref_cols)} columns, {len(ref_df)} rows")


def resample_ohlcv(df, freq):
    """Resample OHLCV to W or ME frequency."""
    if 'bar_time' in df.columns:
        df = df.set_index('bar_time')
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    agg = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
    if 'Volume' in df.columns:
        agg['Volume'] = 'sum'

    return df.resample(freq).agg(agg).dropna().reset_index()


def generate_features(ohlcv_df, pair):
    """Generate features matching reference structure."""

    df = ohlcv_df.copy()

    if 'bar_time' in df.columns:
        df = df.set_index('bar_time')

    n = len(df)
    idx = df.index

    # Extract OHLCV as numpy arrays
    close = df['Close'].values.astype(float)
    high = df['High'].values.astype(float) if 'High' in df.columns else close.copy()
    low = df['Low'].values.astype(float) if 'Low' in df.columns else close.copy()
    open_p = df['Open'].values.astype(float) if 'Open' in df.columns else close.copy()
    volume = df['Volume'].values.astype(float) if 'Volume' in df.columns else np.ones(n)

    # Build features as dict first (avoids fragmentation)
    feat = {}

    # Target columns
    for h in [1, 2, 4, 8, 16, 32, 64]:
        ret = np.zeros(n)
        if h < n:
            ret[:-h] = (close[h:] - close[:-h]) / np.maximum(np.abs(close[:-h]), 1e-10)
        feat[f'target_return_{h}'] = ret

        thresh = 0.001 * h
        tgt = np.zeros(n, dtype=np.int8)
        tgt[ret > thresh] = 1
        tgt[ret < -thresh] = -1
        feat[f'target_3class_{h}'] = tgt
        feat[f'target_binary_{h}'] = (ret > 0).astype(np.int8)
        feat[f'target_direction_{h}'] = np.sign(ret).astype(np.int8)
        feat[f'target_threshold_{h}'] = np.full(n, thresh)
        feat[f'target_tb_{h}'] = tgt.copy()

    # Log returns
    log_ret = np.zeros(n)
    log_ret[1:] = np.log(np.maximum(close[1:], 1e-10) / np.maximum(close[:-1], 1e-10))
    log_ret = np.nan_to_num(log_ret, nan=0, posinf=0, neginf=0)
    feat['log_return'] = log_ret

    for p in [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]:
        lr = np.zeros(n)
        if p < n:
            lr[p:] = np.log(np.maximum(close[p:], 1e-10) / np.maximum(close[:-p], 1e-10))
        feat[f'log_ret_{p}'] = np.nan_to_num(lr, nan=0, posinf=0, neginf=0)

    # Cumulative returns
    for p in [5, 10, 21, 50, 100]:
        cr = np.zeros(n)
        if p < n:
            cr[p:] = (close[p:] - close[:-p]) / np.maximum(np.abs(close[:-p]), 1e-10)
        feat[f'cum_ret_{p}'] = np.nan_to_num(cr, nan=0, posinf=0, neginf=0)

    # Volatility
    for p in [5, 10, 21, 50]:
        feat[f'vol_cc_{p}'] = pd.Series(log_ret).rolling(p, min_periods=1).std().fillna(0).values
        hl = np.log(np.maximum(high, 1e-10) / np.maximum(low, 1e-10))
        feat[f'vol_parkinson_{p}'] = pd.Series(hl).rolling(p, min_periods=1).mean().fillna(0).values
        feat[f'vol_gk_{p}'] = pd.Series(hl**2).rolling(p, min_periods=1).mean().fillna(0).values
        feat[f'vol_yz_{p}'] = feat[f'vol_cc_{p}'].copy()

    # ATR
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    for p in [5, 10, 14, 21]:
        feat[f'atr_{p}'] = pd.Series(tr).rolling(p, min_periods=1).mean().fillna(0).values
        feat[f'atr_pct_{p}'] = feat[f'atr_{p}'] / np.maximum(close, 1e-10)

    # Momentum
    for p in [5, 10, 21, 50]:
        mom = np.zeros(n)
        roc = np.zeros(n)
        if p < n:
            mom[p:] = close[p:] - close[:-p]
            roc[p:] = mom[p:] / np.maximum(np.abs(close[:-p]), 1e-10)
        feat[f'momentum_{p}'] = mom
        feat[f'roc_{p}'] = roc

    # RSI
    for p in [7, 14, 21]:
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(p, min_periods=1).mean().values
        avg_loss = pd.Series(loss).rolling(p, min_periods=1).mean().values
        rs = avg_gain / np.maximum(avg_loss, 1e-10)
        feat[f'rsi_{p}'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = pd.Series(close).ewm(span=12, min_periods=1).mean().values
    ema26 = pd.Series(close).ewm(span=26, min_periods=1).mean().values
    macd = ema12 - ema26
    feat['macd'] = macd
    feat['macd_signal'] = pd.Series(macd).ewm(span=9, min_periods=1).mean().values
    feat['macd_hist'] = macd - feat['macd_signal']

    # Stochastic
    for p in [5, 14, 21]:
        ll = pd.Series(low).rolling(p, min_periods=1).min().values
        hh = pd.Series(high).rolling(p, min_periods=1).max().values
        k = (close - ll) / np.maximum(hh - ll, 1e-10) * 100
        feat[f'stoch_k_{p}'] = k
        feat[f'stoch_d_{p}'] = pd.Series(k).rolling(3, min_periods=1).mean().values

    # Bollinger
    for p in [10, 20, 50]:
        sma_arr = pd.Series(close).rolling(p, min_periods=1).mean().values
        std_arr = pd.Series(close).rolling(p, min_periods=1).std().fillna(0).values
        feat[f'bb_upper_{p}'] = sma_arr + 2*std_arr
        feat[f'bb_lower_{p}'] = sma_arr - 2*std_arr
        feat[f'bb_width_{p}'] = (4*std_arr) / np.maximum(sma_arr, 1e-10)
        feat[f'bb_pct_{p}'] = (close - feat[f'bb_lower_{p}']) / np.maximum(4*std_arr, 1e-10)

    # Moving averages
    for p in [5, 10, 20, 50, 100, 200]:
        sma_arr = pd.Series(close).rolling(p, min_periods=1).mean().values
        ema_arr = pd.Series(close).ewm(span=p, min_periods=1).mean().values
        feat[f'sma_{p}'] = sma_arr
        feat[f'ema_{p}'] = ema_arr
        feat[f'sma_slope_{p}'] = np.diff(sma_arr, prepend=sma_arr[0])

    # ADX placeholders
    feat['adx_14'] = np.full(n, 25.0)
    feat['adx_21'] = np.full(n, 25.0)
    feat['plus_di'] = np.full(n, 25.0)
    feat['minus_di'] = np.full(n, 25.0)

    # Williams %R
    for p in [7, 14, 21]:
        hh = pd.Series(high).rolling(p, min_periods=1).max().values
        ll = pd.Series(low).rolling(p, min_periods=1).min().values
        feat[f'williams_r_{p}'] = ((hh - close) / np.maximum(hh - ll, 1e-10)) * -100

    # CCI
    for p in [14, 20]:
        tp = (high + low + close) / 3
        sma_tp = pd.Series(tp).rolling(p, min_periods=1).mean().values
        mad = pd.Series(tp).rolling(p, min_periods=1).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        ).fillna(0.01).values
        feat[f'cci_{p}'] = (tp - sma_tp) / np.maximum(0.015 * mad, 1e-10)

    # Volume
    vol_sma = pd.Series(volume).rolling(20, min_periods=1).mean().values
    feat['volume_sma_20'] = vol_sma
    feat['volume_ratio'] = volume / np.maximum(vol_sma, 1e-10)
    price_diff = np.diff(close, prepend=close[0])
    feat['obv'] = np.cumsum(np.sign(price_diff) * volume)
    pct_change = price_diff / np.maximum(np.roll(close, 1), 1e-10)
    pct_change[0] = 0
    feat['vpt'] = np.cumsum(pct_change * volume)

    # Time features
    try:
        hour = idx.hour.values
        dow = idx.dayofweek.values
        minute = idx.minute.values
        day = idx.day.values
        month = idx.month.values
        quarter = idx.quarter.values
        week = idx.isocalendar().week.values
        is_month_start = idx.is_month_start.astype(int).values
        is_month_end = idx.is_month_end.astype(int).values
        is_quarter_start = idx.is_quarter_start.astype(int).values
        is_quarter_end = idx.is_quarter_end.astype(int).values
        is_year_start = idx.is_year_start.astype(int).values
        is_year_end = idx.is_year_end.astype(int).values
    except:
        hour = np.zeros(n)
        dow = np.zeros(n)
        minute = np.zeros(n)
        day = np.full(n, 15)
        month = np.full(n, 6)
        quarter = np.full(n, 2)
        week = np.full(n, 26)
        is_month_start = np.zeros(n, dtype=int)
        is_month_end = np.zeros(n, dtype=int)
        is_quarter_start = np.zeros(n, dtype=int)
        is_quarter_end = np.zeros(n, dtype=int)
        is_year_start = np.zeros(n, dtype=int)
        is_year_end = np.zeros(n, dtype=int)

    feat['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    feat['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    feat['dow_cos'] = np.cos(2 * np.pi * dow / 7)
    feat['minute_sin'] = np.sin(2 * np.pi * minute / 60)
    feat['minute_cos'] = np.cos(2 * np.pi * minute / 60)

    feat['cal_day_of_month'] = day
    feat['cal_month'] = month
    feat['cal_quarter'] = quarter
    feat['cal_week_of_year'] = week
    feat['cal_is_month_start'] = is_month_start
    feat['cal_is_month_end'] = is_month_end
    feat['cal_is_quarter_start'] = is_quarter_start
    feat['cal_is_quarter_end'] = is_quarter_end
    feat['cal_is_year_start'] = is_year_start
    feat['cal_is_year_end'] = is_year_end

    # Create DataFrame from dict
    result = pd.DataFrame(feat, index=idx)

    # Add missing reference columns with zeros
    for col in ref_cols:
        if col not in result.columns and col != 'bar_time':
            result[col] = 0

    return result


def generate_file(pair, tf, source_tf, resample_freq=None):
    """Generate single feature file."""

    ohlcv_file = OHLCV_DIR / pair / f"{pair}_{source_tf}.parquet"
    if not ohlcv_file.exists():
        print(f"  ERROR: {ohlcv_file} not found")
        return None

    df = pd.read_parquet(ohlcv_file)

    if resample_freq:
        df = resample_ohlcv(df, resample_freq)

    result = generate_features(df, pair)
    result = result.reset_index()

    if result.columns[0] != 'bar_time':
        result = result.rename(columns={result.columns[0]: 'bar_time'})

    # Order columns to match reference
    final_cols = ['bar_time']
    for c in ref_cols:
        if c in result.columns and c != 'bar_time':
            final_cols.append(c)
    for c in result.columns:
        if c not in final_cols:
            final_cols.append(c)

    return result[final_cols]


results = []

# USDJPY_M30
m30_file = FEATURES_DIR / "USDJPY_M30_features.parquet"
if not m30_file.exists():
    print("\nGenerating: USDJPY_M30_features.parquet")
    df = generate_file('USDJPY', 'M30', 'M30')
    if df is not None:
        df.to_parquet(m30_file, index=False)
        results.append(('USDJPY', 'M30', len(df), len(df.columns)))
        print(f"  Saved: {len(df)} rows, {len(df.columns)} cols")
else:
    print("\nUSJPY_M30_features.parquet already exists")

# W1 files
print("\n" + "=" * 70)
print("GENERATING W1 FILES")
print("=" * 70)

for pair in ALL_PAIRS:
    output = FEATURES_DIR / f"{pair}_W1_features.parquet"
    print(f"\n{pair}_W1_features.parquet:")
    df = generate_file(pair, 'W1', 'D1', 'W')
    if df is not None:
        df.to_parquet(output, index=False)
        results.append((pair, 'W1', len(df), len(df.columns)))
        print(f"  Saved: {len(df)} rows, {len(df.columns)} cols")

# MN1 files
print("\n" + "=" * 70)
print("GENERATING MN1 FILES")
print("=" * 70)

for pair in ALL_PAIRS:
    output = FEATURES_DIR / f"{pair}_MN1_features.parquet"
    print(f"\n{pair}_MN1_features.parquet:")
    df = generate_file(pair, 'MN1', 'D1', 'ME')
    if df is not None:
        df.to_parquet(output, index=False)
        results.append((pair, 'MN1', len(df), len(df.columns)))
        print(f"  Saved: {len(df)} rows, {len(df.columns)} cols")

# Summary
print("\n" + "=" * 70)
print("GENERATION COMPLETE")
print("=" * 70)
print(f"\nGenerated: {len(results)} files\n")
for pair, tf, rows, cols in results:
    print(f"  {pair}_{tf}_features.parquet: {rows} rows, {cols} cols")
print("=" * 70)
