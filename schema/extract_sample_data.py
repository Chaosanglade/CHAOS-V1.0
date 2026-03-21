"""
Extract 10,000 rows from EURUSD_M30_features.parquet using canonical feature schema.
Output is used for deterministic parity testing of ONNX exports.

The execution lag in training is applied to RETURNS (shift -1), not features.
See chaos_gpu_training.py line 708. Features are used as-is after nan_to_num.
"""
import pandas as pd
import numpy as np
import json

# Load schema
with open('G:/My Drive/chaos_v1.0/schema/feature_schema.json') as f:
    schema = json.load(f)

feature_names = [f['name'] for f in schema['features']]

# Load parquet
df = pd.read_parquet('G:/My Drive/chaos_v1.0/features/EURUSD_M30_features.parquet')

# Extract schema features and apply nan_to_num (same as training script line 718)
sample = df[feature_names].copy()
sample = sample.apply(pd.to_numeric, errors='coerce')

# Take last 10,000 rows (falls within validation period 2023-2025)
sample = sample.tail(10000).copy()

# Apply nan_to_num same as training
values = sample.values.astype(np.float64)
values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
sample = pd.DataFrame(values, columns=feature_names)

# Validate before saving
assert sample.shape[1] == schema['feature_count'], \
    f"Column count mismatch: {sample.shape[1]} vs {schema['feature_count']}"
assert sample.isna().sum().sum() == 0, \
    f"NaN detected: {sample.isna().sum().sum()}"
assert not np.any(np.isinf(sample.values)), "Inf detected"

# Save
output_path = 'G:/My Drive/chaos_v1.0/schema/sample_features_10k.parquet'
sample.to_parquet(output_path, index=False)
print(f"Saved {len(sample)} rows x {len(feature_names)} features to {output_path}")
print(f"Dtypes: {sample.dtypes.value_counts().to_dict()}")
print(f"NaN count: {sample.isna().sum().sum()}")
print(f"Inf count: {np.isinf(sample.values).sum()}")
print(f"Value range: [{sample.values.min():.4f}, {sample.values.max():.4f}]")
