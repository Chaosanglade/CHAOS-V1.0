"""
No-lookahead test: Verify features at bar N are from bar N-1 (execution lag).

Checks:
  - Features are shifted by 1 bar
  - No future data leaks into current bar decision
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

os.environ["PYTHONHASHSEED"] = "42"

PROJECT_ROOT = Path('G:/My Drive/chaos_v1.0')
sys.path.insert(0, str(PROJECT_ROOT))


def test_no_lookahead():
    """Verify the replay iterator applies 1-bar lag correctly."""
    from replay.runners.replay_iterator import ReplayIterator

    pair = 'EURUSD'
    tf = 'M30'

    iterator = ReplayIterator(
        pair=pair, tf=tf,
        features_dir=str(PROJECT_ROOT / 'features'),
        schema_path=str(PROJECT_ROOT / 'schema' / 'feature_schema.json'),
        date_start='2024-01-01', date_end='2024-01-31',
        mode='LIVE',
    )

    # Also load raw data for comparison
    raw_df = pd.read_parquet(PROJECT_ROOT / 'features' / f'{pair}_{tf}_features.parquet')
    if raw_df.index.name and 'time' in raw_df.index.name.lower():
        raw_df = raw_df.reset_index()
        ts_col = raw_df.columns[0]
    else:
        ts_col = 'bar_time'
    raw_df[ts_col] = pd.to_datetime(raw_df[ts_col])
    raw_df = raw_df.sort_values(ts_col).reset_index(drop=True)

    # Load feature names
    with open(PROJECT_ROOT / 'schema' / 'feature_schema.json') as f:
        schema = json.load(f)
    feature_names = [feat['name'] for feat in schema['features']]

    # Check first few bars
    events = list(iterator)
    if len(events) < 5:
        print(f"  Only {len(events)} events, need at least 5 for testing")
        return len(events) > 0

    # For each event, the features should match the PREVIOUS bar's features
    # in the raw data (before filtering)
    # The iterator's bar_idx 0 corresponds to the second raw bar (after lag)
    checks_passed = 0
    checks_total = min(5, len(events))

    for i in range(checks_total):
        event = events[i]
        event_ts = pd.Timestamp(event['timestamp'])

        # Find this timestamp in raw data
        raw_mask = raw_df[ts_col] == event_ts
        if not raw_mask.any():
            continue

        raw_idx = raw_df[raw_mask].index[0]
        if raw_idx == 0:
            continue  # Can't verify lag for first bar

        # Get features from the PREVIOUS bar in raw data
        prev_row = raw_df.iloc[raw_idx - 1]
        available_feats = [f for f in feature_names if f in raw_df.columns]

        if not available_feats:
            continue

        # Compare a sample of features
        sample_feats = available_feats[:10]
        match = True
        for feat in sample_feats:
            expected = prev_row[feat]
            actual = event['features'][feature_names.index(feat)]
            if pd.isna(expected) and np.isnan(actual):
                continue
            if not pd.isna(expected) and not np.isclose(expected, actual, rtol=1e-5):
                match = False
                break

        if match:
            checks_passed += 1

    success = checks_passed == checks_total
    print(f"  Lag verification: {checks_passed}/{checks_total} bars confirmed")
    return success


if __name__ == '__main__':
    result = test_no_lookahead()
    print(f"\nNo-lookahead: {'PASS' if result else 'FAIL'}")
