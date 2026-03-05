"""
Replay alignment test: Verify cross-file joins work correctly.

Checks:
  - Every request_id in trades.parquet exists in decision_ledger.parquet
  - Every position change has a corresponding trade or decision
  - Ledger row count = total bars processed (100% coverage)
"""
import os
import sys
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

os.environ["PYTHONHASHSEED"] = "42"

PROJECT_ROOT = Path('G:/My Drive/chaos_v1.0')
sys.path.insert(0, str(PROJECT_ROOT))


def find_latest_run() -> Path:
    runs_dir = PROJECT_ROOT / 'replay' / 'outputs' / 'runs'
    run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], reverse=True)
    if not run_dirs:
        raise FileNotFoundError("No replay runs found")
    return run_dirs[0]


def test_cross_file_joins(run_dir: Path = None):
    """Verify cross-file consistency."""
    if run_dir is None:
        run_dir = find_latest_run()

    results = {}

    # Load files
    ledger_path = run_dir / 'decision_ledger.parquet'
    trades_path = run_dir / 'trades.parquet'
    positions_path = run_dir / 'positions.parquet'

    if not ledger_path.exists():
        return {'error': 'No decision_ledger.parquet'}

    ledger_df = pd.read_parquet(ledger_path)
    trades_df = pd.read_parquet(trades_path) if trades_path.exists() else pd.DataFrame()
    positions_df = pd.read_parquet(positions_path) if positions_path.exists() else pd.DataFrame()

    # Test 1: Every trade request_id exists in ledger
    if len(trades_df) > 0 and 'request_id' in trades_df.columns:
        trade_request_ids = set(trades_df['request_id'].unique())
        ledger_request_ids = set(ledger_df['request_id'].unique())
        # Exclude END_OF_REPLAY forced closes
        trade_request_ids.discard('END_OF_REPLAY')
        missing = trade_request_ids - ledger_request_ids
        results['trade_ids_in_ledger'] = {
            'total_trade_ids': len(trade_request_ids),
            'found_in_ledger': len(trade_request_ids) - len(missing),
            'missing': len(missing),
            'passed': len(missing) == 0,
        }
    else:
        results['trade_ids_in_ledger'] = {
            'total_trade_ids': 0,
            'passed': True,
            'note': 'No trades to check',
        }

    # Test 2: Position changes have corresponding entries
    if len(positions_df) > 0 and 'request_id' in positions_df.columns:
        pos_request_ids = set(positions_df['request_id'].dropna().unique())
        ledger_request_ids = set(ledger_df['request_id'].unique())
        missing = pos_request_ids - ledger_request_ids
        results['positions_in_ledger'] = {
            'total_position_ids': len(pos_request_ids),
            'missing': len(missing),
            'passed': len(missing) == 0,
        }
    else:
        results['positions_in_ledger'] = {
            'total_position_ids': 0,
            'passed': True,
            'note': 'No positions to check',
        }

    # Test 3: Ledger coverage (checked via manifest)
    import json
    manifest_path = run_dir / 'manifest.json'
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        total_bars = manifest.get('total_bars', 0)
        ledger_rows = len(ledger_df)
        coverage = ledger_rows / total_bars if total_bars > 0 else 0
        results['ledger_coverage'] = {
            'total_bars': total_bars,
            'ledger_rows': ledger_rows,
            'coverage': round(coverage, 4),
            'passed': coverage >= 1.0,
        }

    return results


if __name__ == '__main__':
    results = test_cross_file_joins()
    for name, detail in results.items():
        status = 'PASS' if detail.get('passed', False) else 'FAIL'
        print(f"  {status}  {name}: {detail}")
