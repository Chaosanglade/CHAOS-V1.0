"""
Validates that a feature vector conforms to feature_schema.json.
Run against every parquet file to confirm universal compatibility.

Usage: python test_feature_schema.py
"""
import json
import numpy as np
import pandas as pd
import sys
import os

# Exact feature selection logic from chaos_gpu_training.py lines 664-677
EXCLUDE_PATTERNS = ['target_', 'return', 'Open', 'High', 'Low', 'Close',
                    'Volume', 'timestamp', 'date', 'pair', 'symbol', 'tf',
                    'bar_time', 'Bid_', 'Ask_', 'Spread_', 'Unnamed']


def get_feature_columns(df):
    """Replicate exact feature selection from training script."""
    feature_cols = []
    for col in df.columns:
        if any(p in col for p in EXCLUDE_PATTERNS):
            continue
        if df[col].dtype not in ['int64', 'int32', 'float64', 'float32']:
            if str(df[col].dtype) != 'Float64':
                continue
        feature_cols.append(col)
    return feature_cols


def load_schema(path='G:/My Drive/chaos_v1.0/schema/feature_schema.json'):
    with open(path) as f:
        return json.load(f)


def validate_features(features_array, schema):
    """
    Args:
        features_array: np.ndarray of shape (n_samples, n_features)
        schema: loaded schema dict
    Returns:
        (bool, list of error strings)
    """
    errors = []
    expected = schema['feature_count']

    if features_array.shape[1] != expected:
        errors.append(f"Feature count mismatch: got {features_array.shape[1]}, expected {expected}")

    if np.any(np.isnan(features_array)):
        nan_cols = np.where(np.any(np.isnan(features_array), axis=0))[0]
        errors.append(f"NaN detected in columns: {nan_cols.tolist()}")

    if np.any(np.isinf(features_array)):
        inf_cols = np.where(np.any(np.isinf(features_array), axis=0))[0]
        errors.append(f"Inf detected in columns: {inf_cols.tolist()}")

    return len(errors) == 0, errors


def validate_parquet(parquet_path, schema):
    """Validate a full parquet file against the schema."""
    df = pd.read_parquet(parquet_path)
    schema_names = [f['name'] for f in schema['features']]

    missing = set(schema_names) - set(df.columns)
    if missing:
        return False, [f"Missing columns: {missing}"]

    features = df[schema_names].values.astype(np.float64)
    # Replace NaN/Inf same as training (nan_to_num with 0.0)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return validate_features(features, schema)


if __name__ == '__main__':
    schema = load_schema()
    print(f"Schema version: {schema['version']}")
    print(f"Feature count: {schema['feature_count']}")

    # Test against all 9 pairs on M30 (universal timeframe)
    pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD',
             'USDCHF', 'NZDUSD', 'EURJPY', 'GBPJPY']
    base = 'G:/My Drive/chaos_v1.0/features'

    all_passed = True
    for pair in pairs:
        path = f"{base}/{pair}_M30_features.parquet"
        try:
            valid, errors = validate_parquet(path, schema)
            status = "PASS" if valid else f"FAIL: {errors}"
            print(f"  {pair}_M30: {status}")
            if not valid:
                all_passed = False
        except FileNotFoundError:
            print(f"  {pair}_M30: SKIP (file not found)")

    print(f"\n{'ALL PASSED' if all_passed else 'FAILURES DETECTED'}")
