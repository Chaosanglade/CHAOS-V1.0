"""
CHAOS V1.0 — Comprehensive Parity Audit
========================================
Compare training parquet features vs production pipeline output
for the 116 features the EURUSD model actually uses.

Run locally or on Colab:
    python colab/COMPREHENSIVE_PARITY_AUDIT.py

On Colab, mount Drive first and set CHAOS_BASE_DIR.
"""
import os
import sys
import json
import time
import warnings
import gc

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict

# ── Setup paths ──
PROJECT_ROOT = Path(os.environ.get('CHAOS_BASE_DIR', '.'))
if not (PROJECT_ROOT / 'features').exists():
    # Try Google Drive Colab path
    PROJECT_ROOT = Path('/content/drive/MyDrive/chaos_v1.0')
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'inference'))

print(f'Project root: {PROJECT_ROOT}')
print(f'Features dir: {PROJECT_ROOT / "features"}')


# ═══════════════════════════════════════════════════════════════════════
# Cell 1: Load Data
# ═══════════════════════════════════════════════════════════════════════

print('\n=== CELL 1: LOAD DATA ===')

# Feature config: the 116 features the model actually uses
config_path = PROJECT_ROOT / 'ml_signals' / 'production_models' / 'EURUSD_feature_config.json'
with open(config_path) as f:
    feature_config = json.load(f)
MODEL_FEATURES = feature_config['features']
print(f'Model uses {len(MODEL_FEATURES)} features')

# Training parquet
parquet_path = PROJECT_ROOT / 'features' / 'EURUSD_M5_features.parquet'
print(f'Loading training parquet...', end=' ', flush=True)
t0 = time.time()
train_df = pd.read_parquet(str(parquet_path))
print(f'{train_df.shape[0]:,} rows x {train_df.shape[1]} cols ({time.time()-t0:.1f}s)')

# Check which model features exist in parquet
parquet_cols = set(train_df.columns)
missing_in_parquet = [f for f in MODEL_FEATURES if f not in parquet_cols]
present_in_parquet = [f for f in MODEL_FEATURES if f in parquet_cols]
print(f'Model features in parquet: {len(present_in_parquet)}/{len(MODEL_FEATURES)}')
if missing_in_parquet:
    print(f'MISSING from parquet: {missing_in_parquet}')

# Raw OHLCV for production computation
ohlcv_path = PROJECT_ROOT / 'ohlcv_data' / 'EURUSD' / 'EURUSD_M5.parquet'
print(f'Loading raw OHLCV...', end=' ', flush=True)
t0 = time.time()
ohlcv_df = pd.read_parquet(str(ohlcv_path))
print(f'{ohlcv_df.shape[0]:,} rows x {ohlcv_df.shape[1]} cols ({time.time()-t0:.1f}s)')
print(f'OHLCV columns: {list(ohlcv_df.columns[:10])}')


# ═══════════════════════════════════════════════════════════════════════
# Cell 2: Sample Selection
# ═══════════════════════════════════════════════════════════════════════

print('\n=== CELL 2: SAMPLE SELECTION ===')

N_SAMPLES = 100
LOOKBACK = 500

# Need datetime index for year-based sampling
if isinstance(train_df.index, pd.DatetimeIndex):
    years = train_df.index.year
elif 'bar_time' in train_df.columns:
    years = pd.to_datetime(train_df['bar_time']).dt.year
else:
    years = pd.Series(range(len(train_df))) // (len(train_df) // 15)  # Fake years

# Sample across years, skip first 1000 bars
np.random.seed(42)
valid_indices = np.where(np.arange(len(train_df)) > 1000)[0]
unique_years = sorted(set(years.iloc[valid_indices]))
samples_per_year = max(1, N_SAMPLES // len(unique_years))

sample_indices = []
for yr in unique_years:
    yr_mask = (years == yr) & (pd.Series(range(len(train_df))) > 1000)
    yr_indices = np.where(yr_mask)[0]
    if len(yr_indices) > 0:
        chosen = np.random.choice(yr_indices, min(samples_per_year, len(yr_indices)), replace=False)
        sample_indices.extend(chosen)

sample_indices = sorted(sample_indices[:N_SAMPLES])
print(f'Selected {len(sample_indices)} sample bars across {len(unique_years)} years')
print(f'Index range: {sample_indices[0]} to {sample_indices[-1]}')


# ═══════════════════════════════════════════════════════════════════════
# Cell 3: Production Feature Computation
# ═══════════════════════════════════════════════════════════════════════

print('\n=== CELL 3: PRODUCTION FEATURE COMPUTATION ===')

cache_path = PROJECT_ROOT / 'audit_production_features.parquet'
if cache_path.exists():
    print(f'Loading cached production features from {cache_path.name}')
    prod_df = pd.read_parquet(str(cache_path))
    print(f'Cached: {prod_df.shape}')
else:
    from inference.live_feature_adapter import LiveFeatureAdapter

    adapter = LiveFeatureAdapter(
        schema_path=PROJECT_ROOT / 'schema' / 'feature_schema.json',
        column_order_path=PROJECT_ROOT / 'schema' / 'feature_columns_by_pair_tf.json',
    )
    schema_names = adapter.get_feature_names()

    # Also need column order for this pair
    col_order_path = PROJECT_ROOT / 'schema' / 'feature_columns_by_pair_tf.json'
    if col_order_path.exists():
        with open(col_order_path) as f:
            col_orders = json.load(f)
        training_cols = col_orders.get('EURUSD_M5', col_orders.get('EURUSD_H1', schema_names))
    else:
        training_cols = schema_names

    # Build production features for each sample bar
    prod_rows = []
    t0 = time.time()
    for i, idx in enumerate(sample_indices):
        if i % 10 == 0:
            print(f'  Computing bar {i+1}/{len(sample_indices)} (idx={idx})...', flush=True)

        # Extract 500-bar window from OHLCV
        start = max(0, idx - LOOKBACK + 1)
        window = ohlcv_df.iloc[start:idx + 1]

        bars = []
        for _, row in window.iterrows():
            bar = {}
            for src in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if src in row.index:
                    bar[src.lower()] = float(row[src])
            if hasattr(window.index, 'strftime'):
                bar['time'] = row.name.strftime('%Y-%m-%dT%H:%M:%S')
            elif 'bar_time' in row.index:
                bar['time'] = str(row['bar_time'])
            else:
                bar['time'] = '2020-01-01T00:00:00'
            bars.append(bar)

        schema_vec = adapter.compute('EURUSD', 'M5', {'M5': bars})
        if schema_vec is None:
            prod_rows.append({f: np.nan for f in MODEL_FEATURES})
            continue

        # Map schema vector to model features by name
        row_dict = {}
        for fname in MODEL_FEATURES:
            if fname in adapter._name_to_idx:
                row_dict[fname] = float(schema_vec[adapter._name_to_idx[fname]])
            else:
                # Try training column order
                row_dict[fname] = np.nan
        prod_rows.append(row_dict)

    prod_df = pd.DataFrame(prod_rows, index=sample_indices)
    elapsed = time.time() - t0
    print(f'Production features computed: {prod_df.shape} ({elapsed:.0f}s)')

    # Cache
    try:
        prod_df.to_parquet(str(cache_path))
        print(f'Cached to {cache_path.name}')
    except Exception as e:
        print(f'Cache save failed: {e}')


# ═══════════════════════════════════════════════════════════════════════
# Cell 4: Parquet Feature Lookup
# ═══════════════════════════════════════════════════════════════════════

print('\n=== CELL 4: PARQUET FEATURE LOOKUP ===')

parq_rows = []
for idx in sample_indices:
    row_dict = {}
    for fname in MODEL_FEATURES:
        if fname in train_df.columns:
            row_dict[fname] = float(train_df[fname].iloc[idx])
        else:
            row_dict[fname] = np.nan
    parq_rows.append(row_dict)

parq_df = pd.DataFrame(parq_rows, index=sample_indices)
print(f'Parquet features extracted: {parq_df.shape}')


# ═══════════════════════════════════════════════════════════════════════
# Cell 5: Comparison
# ═══════════════════════════════════════════════════════════════════════

print('\n=== CELL 5: FEATURE-BY-FEATURE COMPARISON ===')

comparison = []
for fname in MODEL_FEATURES:
    parq_vals = parq_df[fname].dropna().values
    prod_vals = prod_df[fname].dropna().values if fname in prod_df.columns else np.array([])

    n_common = min(len(parq_vals), len(prod_vals))
    if n_common < 10:
        comparison.append({
            'feature': fname, 'correlation': np.nan, 'mae': np.nan,
            'parq_mean': np.nan, 'prod_mean': np.nan,
            'parq_std': np.nan, 'prod_std': np.nan,
            'parq_min': np.nan, 'parq_max': np.nan,
            'prod_min': np.nan, 'prod_max': np.nan,
            'pct_match_1pct': 0.0, 'n_samples': n_common,
            'classification': 'INSUFFICIENT_DATA',
        })
        continue

    pv = parq_vals[:n_common]
    rv = prod_vals[:n_common]

    # Correlation
    if np.std(pv) > 1e-10 and np.std(rv) > 1e-10:
        corr = np.corrcoef(pv, rv)[0, 1]
    elif np.std(pv) < 1e-10 and np.std(rv) < 1e-10:
        corr = 1.0  # Both constant
    else:
        corr = 0.0

    mae = np.mean(np.abs(pv - rv))

    # % matching within 1% tolerance
    denom = np.maximum(np.abs(pv), 1e-10)
    pct_diff = np.abs(pv - rv) / denom
    pct_match = np.mean(pct_diff < 0.01) * 100

    # Classification
    if np.isnan(corr):
        cls = 'NAN_CORR'
    elif corr > 0.99 and pct_match > 80:
        cls = 'PERFECT'
    elif corr > 0.95:
        cls = 'SCALE_DIFF'
    elif corr > 0.5:
        cls = 'FORMULA_DIFF'
    else:
        cls = 'UNRELATED'

    comparison.append({
        'feature': fname, 'correlation': round(corr, 4), 'mae': round(mae, 6),
        'parq_mean': round(np.mean(pv), 6), 'prod_mean': round(np.mean(rv), 6),
        'parq_std': round(np.std(pv), 6), 'prod_std': round(np.std(rv), 6),
        'parq_min': round(np.min(pv), 6), 'parq_max': round(np.max(pv), 6),
        'prod_min': round(np.min(rv), 6), 'prod_max': round(np.max(rv), 6),
        'pct_match_1pct': round(pct_match, 1), 'n_samples': n_common,
        'classification': cls,
    })

comp_df = pd.DataFrame(comparison)


# ═══════════════════════════════════════════════════════════════════════
# Cell 6: Classification Summary
# ═══════════════════════════════════════════════════════════════════════

print('\n=== CELL 6: CLASSIFICATION SUMMARY ===')

class_counts = comp_df['classification'].value_counts()
print(f'\nClassification distribution:')
for cls, cnt in class_counts.items():
    print(f'  {cls:<20} {cnt:>3} features')

# Print detailed table sorted by classification severity
print(f'\n{"Feature":<35} {"Corr":>6} {"MAE":>10} {"ParqMean":>10} {"ProdMean":>10} {"Match%":>7} {"Class":<15}')
print('=' * 100)

# Sort: UNRELATED first, then FORMULA_DIFF, then SCALE_DIFF, then PERFECT
order = {'UNRELATED': 0, 'FORMULA_DIFF': 1, 'NAN_CORR': 2,
         'INSUFFICIENT_DATA': 3, 'SCALE_DIFF': 4, 'PERFECT': 5}
comp_sorted = comp_df.copy()
comp_sorted['sort_key'] = comp_sorted['classification'].map(order).fillna(3)
comp_sorted = comp_sorted.sort_values('sort_key')

for _, row in comp_sorted.iterrows():
    corr_str = f'{row["correlation"]:.4f}' if not np.isnan(row['correlation']) else 'N/A'
    mae_str = f'{row["mae"]:.6f}' if not np.isnan(row['mae']) else 'N/A'
    pm = f'{row["parq_mean"]:.4f}' if not np.isnan(row['parq_mean']) else 'N/A'
    rm = f'{row["prod_mean"]:.4f}' if not np.isnan(row['prod_mean']) else 'N/A'
    print(f'{row["feature"]:<35} {corr_str:>6} {mae_str:>10} {pm:>10} {rm:>10} '
          f'{row["pct_match_1pct"]:>6.1f}% {row["classification"]:<15}')


# ═══════════════════════════════════════════════════════════════════════
# Cell 7: Impact Assessment
# ═══════════════════════════════════════════════════════════════════════

print('\n=== CELL 7: IMPACT ASSESSMENT ===')

# Try to load feature importance from XGBoost model
importance = {}
try:
    import joblib as jl
    import glob
    model_files = glob.glob(str(PROJECT_ROOT / 'models' / 'EURUSD_M5_xgb*.joblib'))
    if not model_files:
        model_files = glob.glob(str(PROJECT_ROOT / 'models' / 'v2_retrained' / 'EURUSD_*xgb*.joblib'))
    if model_files:
        loaded = jl.load(model_files[0])
        model = loaded if not isinstance(loaded, dict) else loaded.get('model', loaded)
        if hasattr(model, 'feature_importances_'):
            fi = model.feature_importances_
            # Map to feature names if available
            fnames = loaded.get('feature_names', MODEL_FEATURES) if isinstance(loaded, dict) else MODEL_FEATURES
            for i, fname in enumerate(fnames[:len(fi)]):
                if fname in MODEL_FEATURES:
                    importance[fname] = float(fi[i])
            print(f'Loaded feature importance from {model_files[0]}')
except Exception as e:
    print(f'Could not load feature importance: {e}')

# If no importance available, use uniform
if not importance:
    for f in MODEL_FEATURES:
        importance[f] = 1.0 / len(MODEL_FEATURES)
    print('Using uniform importance (no model loaded)')

total_importance = sum(importance.values())
comp_df['importance'] = comp_df['feature'].map(importance).fillna(0)
comp_df['importance_pct'] = comp_df['importance'] / total_importance * 100

# Impact by classification
print(f'\nImportance-weighted impact:')
for cls in ['PERFECT', 'SCALE_DIFF', 'FORMULA_DIFF', 'UNRELATED', 'NAN_CORR', 'INSUFFICIENT_DATA']:
    mask = comp_df['classification'] == cls
    n = mask.sum()
    imp = comp_df.loc[mask, 'importance_pct'].sum()
    print(f'  {cls:<20} {n:>3} features  {imp:>6.1f}% of total importance')

affected = comp_df[comp_df['classification'].isin(['FORMULA_DIFF', 'UNRELATED'])]
affected_pct = affected['importance_pct'].sum()

print(f'\nAFFECTED (FORMULA_DIFF + UNRELATED): {len(affected)} features, {affected_pct:.1f}% importance')
if affected_pct < 5:
    print('VERDICT: Surgical retrain is VIABLE (<5% affected)')
elif affected_pct < 20:
    print('VERDICT: Surgical retrain is RISKY (5-20% affected)')
else:
    print('VERDICT: Full retrain MANDATORY (>20% affected)')

# List affected features
if len(affected) > 0:
    print(f'\nAffected features (sorted by importance):')
    affected_sorted = affected.sort_values('importance_pct', ascending=False)
    for _, row in affected_sorted.iterrows():
        print(f'  {row["feature"]:<35} imp={row["importance_pct"]:.1f}%  '
              f'class={row["classification"]}  corr={row["correlation"]}')


# ═══════════════════════════════════════════════════════════════════════
# Cell 8: Save Reports
# ═══════════════════════════════════════════════════════════════════════

print('\n=== CELL 8: SAVE REPORTS ===')

report_csv = PROJECT_ROOT / 'parity_audit_report.csv'
comp_df.to_csv(str(report_csv), index=False)
print(f'Full report: {report_csv}')

summary = {
    'timestamp': pd.Timestamp.now().isoformat(),
    'n_features': len(MODEL_FEATURES),
    'n_samples': len(sample_indices),
    'classification_counts': class_counts.to_dict(),
    'affected_importance_pct': round(affected_pct, 1),
    'verdict': 'VIABLE' if affected_pct < 5 else ('RISKY' if affected_pct < 20 else 'MANDATORY_RETRAIN'),
    'affected_features': affected.sort_values('importance_pct', ascending=False)[
        ['feature', 'classification', 'correlation', 'importance_pct']
    ].to_dict('records') if len(affected) > 0 else [],
}
summary_path = PROJECT_ROOT / 'parity_audit_summary.json'
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print(f'Summary: {summary_path}')

print('\nDONE.')
