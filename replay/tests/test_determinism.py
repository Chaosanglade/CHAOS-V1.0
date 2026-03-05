"""
Determinism test: Run replay twice with identical seed, compare outputs.

Parquet files should be identical (excluding write-time metadata).
"""
import os
import sys
import hashlib
from pathlib import Path

os.environ["PYTHONHASHSEED"] = "42"

PROJECT_ROOT = Path('G:/My Drive/chaos_v1.0')
sys.path.insert(0, str(PROJECT_ROOT))


def hash_parquet_content(path):
    """Hash the logical content of a parquet file (not raw bytes, which include metadata)."""
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    # Convert to pandas for deterministic comparison
    df = table.to_pandas()
    # Sort by all columns for deterministic ordering
    df = df.sort_values(list(df.columns)).reset_index(drop=True)
    return hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()


def test_determinism(max_bars=50):
    """Run replay twice and compare."""
    import numpy as np
    import random

    # Run 1
    os.environ["PYTHONHASHSEED"] = "42"
    np.random.seed(42)
    random.seed(42)

    from replay.runners.run_replay import ReplayRunner
    runner1 = ReplayRunner()
    run_id_1 = runner1.run(pairs=['EURUSD'], tfs=['M30'], max_bars=max_bars)

    # Run 2 (reset state)
    os.environ["PYTHONHASHSEED"] = "42"
    np.random.seed(42)
    random.seed(42)

    runner2 = ReplayRunner()
    run_id_2 = runner2.run(pairs=['EURUSD'], tfs=['M30'], max_bars=max_bars)

    # Compare
    runs_dir = PROJECT_ROOT / 'replay' / 'outputs' / 'runs'
    dir1 = runs_dir / run_id_1
    dir2 = runs_dir / run_id_2

    files_to_compare = ['decision_ledger.parquet', 'trades.parquet', 'positions.parquet']
    all_match = True

    for fname in files_to_compare:
        f1 = dir1 / fname
        f2 = dir2 / fname
        if f1.exists() and f2.exists():
            h1 = hash_parquet_content(f1)
            h2 = hash_parquet_content(f2)
            if h1 == h2:
                print(f"  MATCH  {fname}")
            else:
                print(f"  DIFFER {fname}")
                all_match = False
        elif not f1.exists() and not f2.exists():
            print(f"  MATCH  {fname} (both missing)")
        else:
            print(f"  DIFFER {fname} (one missing)")
            all_match = False

    return all_match


if __name__ == '__main__':
    result = test_determinism()
    print(f"\nDeterminism: {'PASS' if result else 'FAIL'}")
