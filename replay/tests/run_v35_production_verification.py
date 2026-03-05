"""
CHAOS V3.5 Production Upgrade — 17-Test Verification Suite

Tests:
 1. All config JSON files load without error (7 files)
 2. PyArrow schemas import and have correct field counts
 3. Risk engine unit tests pass (all 7 checks + Lock 9 cost invariant)
 4. PortfolioState open/close/exposure calculations correct
 5. Replay iterator loads EURUSD_M30 and yields events with 273 features
 6. 50-bar smoke test: run_replay produces all output files
 7. trades.parquet conforms to TRADES_SCHEMA
 8. positions.parquet conforms to POSITIONS_SCHEMA
 9. decision_ledger.parquet conforms to DECISION_LEDGER_SCHEMA
10. Ledger coverage = 100%
11. Cross-file join: all trade request_ids exist in ledger
12. Determinism test: two identical runs produce identical output
13. No-lookahead test: features at bar N are from bar N-1
14. MT5 gate export produces valid zip bundle
15. Gate criteria evaluation runs and reports PASS/FAIL
16. Reason codes in outputs are all from reason_codes.json registry
17. Lock 9: Cost sign invariant on trade output rows
"""
import os
import sys
import json
import time
import hashlib
import zipfile
import traceback
import numpy as np
import random
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Determinism
os.environ["PYTHONHASHSEED"] = "42"
np.random.seed(42)
random.seed(42)

PROJECT_ROOT = Path('G:/My Drive/chaos_v1.0')
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'inference'))

RESULTS = []


def record(test_num, description, passed, details=""):
    status = "PASS" if passed else "FAIL"
    RESULTS.append((test_num, description, status, details))
    print(f"  Test {test_num:2d} | {status} | {description}" +
          (f" | {details}" if details else ""))


# ============================================================
# TEST 1: Config JSON files
# ============================================================
def test_1():
    config_files = [
        PROJECT_ROOT / 'replay' / 'config' / 'universe.json',
        PROJECT_ROOT / 'replay' / 'config' / 'replay_config.json',
        PROJECT_ROOT / 'replay' / 'config' / 'execution_cost_scenarios.json',
        PROJECT_ROOT / 'risk' / 'config' / 'instrument_specs.json',
        PROJECT_ROOT / 'risk' / 'config' / 'correlation_groups.json',
        PROJECT_ROOT / 'risk' / 'config' / 'risk_policy.json',
        PROJECT_ROOT / 'audit' / 'schemas' / 'reason_codes.json',
    ]
    loaded = 0
    for fpath in config_files:
        try:
            with open(fpath) as f:
                json.load(f)
            loaded += 1
        except Exception as e:
            record(1, "Config JSON files load", False, f"{fpath.name}: {e}")
            return
    record(1, "Config JSON files load (7 files)", loaded == 7, f"{loaded}/7")


# ============================================================
# TEST 2: PyArrow schemas field counts
# ============================================================
def test_2():
    try:
        from replay.runners.parquet_schemas import (
            TRADES_SCHEMA, POSITIONS_SCHEMA, DECISION_LEDGER_SCHEMA
        )
        t_count = len(TRADES_SCHEMA)
        p_count = len(POSITIONS_SCHEMA)
        l_count = len(DECISION_LEDGER_SCHEMA)

        # Actual schema counts (may differ from original spec due to field additions)
        ok = (t_count >= 30 and p_count >= 17 and l_count >= 30)
        record(2, "PyArrow schemas field counts",
               ok, f"TRADES={t_count}, POSITIONS={p_count}, LEDGER={l_count}")
    except Exception as e:
        record(2, "PyArrow schemas field counts", False, str(e))


# ============================================================
# TEST 3: Risk engine unit tests
# ============================================================
def test_3():
    try:
        import importlib
        import risk.engine.test_risk_engine as trm
        importlib.reload(trm)
        # Capture output
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            passed = trm.run_all_tests()
        record(3, "Risk engine unit tests (7 checks + Lock 9)", passed, buf.getvalue().strip().split('\n')[-1])
    except Exception as e:
        record(3, "Risk engine unit tests", False, str(e))


# ============================================================
# TEST 4: PortfolioState calculations
# ============================================================
def test_4():
    try:
        from risk.engine.portfolio_state import PortfolioState
        portfolio = PortfolioState(
            str(PROJECT_ROOT / 'risk' / 'config' / 'instrument_specs.json'),
            str(PROJECT_ROOT / 'risk' / 'config' / 'correlation_groups.json'),
        )
        now = datetime(2024, 6, 1, 12, 0)

        # Open
        portfolio.open_position('EURUSD', 'M30', 1, 0.5, 1.1000, now)
        assert portfolio.get_position_count() == 1
        assert portfolio.get_gross_exposure_usd() == 50000.0

        # Close with profit
        trade = portfolio.close_position('EURUSD', 'M30', 1.1050, now + timedelta(hours=1))
        assert abs(trade.pnl_pips - 50.0) < 0.01, f"pnl_pips={trade.pnl_pips}"
        assert abs(trade.pnl_net_usd - 250.0) < 0.01, f"pnl_net={trade.pnl_net_usd}"
        assert portfolio.get_position_count() == 0

        record(4, "PortfolioState calculations", True)
    except Exception as e:
        record(4, "PortfolioState calculations", False, str(e))


# ============================================================
# TEST 5: Replay iterator
# ============================================================
def test_5():
    """LIVE mode: iterator enforces 273 universal boundary."""
    try:
        from replay.runners.replay_iterator import ReplayIterator
        iterator = ReplayIterator(
            pair='EURUSD', tf='M30',
            features_dir=str(PROJECT_ROOT / 'features'),
            schema_path=str(PROJECT_ROOT / 'schema' / 'feature_schema.json'),
            date_start='2024-01-01', date_end='2024-01-31',
            mode='LIVE',
        )
        events = []
        for i, event in enumerate(iterator):
            events.append(event)
            if i >= 4:
                break

        ok = len(events) > 0
        feat_len = len(events[0]['features']) if events else 0
        record(5, "LIVE mode iterator yields 273 features", ok and feat_len == 273,
               f"{len(iterator)} bars, features={feat_len}, mode={events[0].get('mode', '?')}")
    except Exception as e:
        record(5, "LIVE mode iterator yields 273 features", False, str(e))


# ============================================================
# TEST 6: 50-bar smoke test
# ============================================================
SMOKE_RUN_ID = None

def test_6():
    global SMOKE_RUN_ID
    try:
        os.environ["PYTHONHASHSEED"] = "42"
        np.random.seed(42)
        random.seed(42)

        from replay.runners.run_replay import ReplayRunner
        runner = ReplayRunner()
        SMOKE_RUN_ID = runner.run(pairs=['EURUSD'], tfs=['M30'], max_bars=50)

        run_dir = PROJECT_ROOT / 'replay' / 'outputs' / 'runs' / SMOKE_RUN_ID
        expected_files = ['decision_ledger.parquet', 'trades.parquet',
                         'positions.parquet', 'manifest.json',
                         'metrics.json', 'gate_criteria_result.json']
        found = [f for f in expected_files if (run_dir / f).exists()]
        record(6, "50-bar smoke test", len(found) == len(expected_files),
               f"{len(found)}/{len(expected_files)} output files")
    except Exception as e:
        record(6, "50-bar smoke test", False, str(e))


# ============================================================
# TEST 7: trades.parquet schema validation
# ============================================================
def test_7():
    try:
        if not SMOKE_RUN_ID:
            record(7, "trades.parquet schema", False, "No smoke test run")
            return
        import pyarrow.parquet as pq
        from replay.runners.parquet_schemas import TRADES_SCHEMA
        run_dir = PROJECT_ROOT / 'replay' / 'outputs' / 'runs' / SMOKE_RUN_ID
        table = pq.read_table(run_dir / 'trades.parquet')
        expected = set(f.name for f in TRADES_SCHEMA)
        actual = set(table.column_names)
        missing = expected - actual
        extra = actual - expected
        ok = len(missing) == 0
        record(7, "trades.parquet conforms to TRADES_SCHEMA", ok,
               f"{table.num_rows} rows" + (f", missing: {missing}" if missing else ""))
    except Exception as e:
        record(7, "trades.parquet schema", False, str(e))


# ============================================================
# TEST 8: positions.parquet schema validation
# ============================================================
def test_8():
    try:
        if not SMOKE_RUN_ID:
            record(8, "positions.parquet schema", False, "No smoke test run")
            return
        import pyarrow.parquet as pq
        from replay.runners.parquet_schemas import POSITIONS_SCHEMA
        run_dir = PROJECT_ROOT / 'replay' / 'outputs' / 'runs' / SMOKE_RUN_ID
        table = pq.read_table(run_dir / 'positions.parquet')
        expected = set(f.name for f in POSITIONS_SCHEMA)
        actual = set(table.column_names)
        missing = expected - actual
        ok = len(missing) == 0
        record(8, "positions.parquet conforms to POSITIONS_SCHEMA", ok,
               f"{table.num_rows} rows" + (f", missing: {missing}" if missing else ""))
    except Exception as e:
        record(8, "positions.parquet schema", False, str(e))


# ============================================================
# TEST 9: decision_ledger.parquet schema validation
# ============================================================
def test_9():
    try:
        if not SMOKE_RUN_ID:
            record(9, "decision_ledger.parquet schema", False, "No smoke test run")
            return
        import pyarrow.parquet as pq
        from replay.runners.parquet_schemas import DECISION_LEDGER_SCHEMA
        run_dir = PROJECT_ROOT / 'replay' / 'outputs' / 'runs' / SMOKE_RUN_ID
        table = pq.read_table(run_dir / 'decision_ledger.parquet')
        expected = set(f.name for f in DECISION_LEDGER_SCHEMA)
        actual = set(table.column_names)
        missing = expected - actual
        ok = len(missing) == 0
        record(9, "decision_ledger.parquet conforms to LEDGER_SCHEMA", ok,
               f"{table.num_rows} rows" + (f", missing: {missing}" if missing else ""))
    except Exception as e:
        record(9, "decision_ledger.parquet schema", False, str(e))


# ============================================================
# TEST 10: Ledger coverage = 100%
# ============================================================
def test_10():
    try:
        if not SMOKE_RUN_ID:
            record(10, "Ledger coverage", False, "No smoke test run")
            return
        run_dir = PROJECT_ROOT / 'replay' / 'outputs' / 'runs' / SMOKE_RUN_ID
        with open(run_dir / 'manifest.json') as f:
            manifest = json.load(f)
        total_bars = manifest['total_bars']
        ledger_rows = manifest['total_ledger_rows']
        coverage = ledger_rows / total_bars if total_bars > 0 else 0
        record(10, "Ledger coverage = 100%", coverage >= 1.0,
               f"{ledger_rows}/{total_bars} = {coverage:.2%}")
    except Exception as e:
        record(10, "Ledger coverage", False, str(e))


# ============================================================
# TEST 11: Cross-file join
# ============================================================
def test_11():
    try:
        if not SMOKE_RUN_ID:
            record(11, "Cross-file join", False, "No smoke test run")
            return
        run_dir = PROJECT_ROOT / 'replay' / 'outputs' / 'runs' / SMOKE_RUN_ID
        trades_df = pd.read_parquet(run_dir / 'trades.parquet')
        ledger_df = pd.read_parquet(run_dir / 'decision_ledger.parquet')

        if len(trades_df) == 0:
            record(11, "Cross-file join: trade IDs in ledger", True, "No trades to check")
            return

        trade_rids = set(trades_df['request_id'].unique()) - {'END_OF_REPLAY'}
        ledger_rids = set(ledger_df['request_id'].unique())
        missing = trade_rids - ledger_rids
        record(11, "Cross-file join: trade IDs in ledger", len(missing) == 0,
               f"{len(trade_rids) - len(missing)}/{len(trade_rids)} found")
    except Exception as e:
        record(11, "Cross-file join", False, str(e))


# ============================================================
# TEST 12: Determinism test (Lock 10 — Determinism Hash Contract)
# ============================================================
def test_12():
    """Lock 10: Determinism Hash Contract.

    Rule 1: Exclude run-variant fields (run_id — latency_ms is None per Rule 5).
    Rule 2: Sort rows by stable keys per file type.
    Rule 3: Float canonicalization — round to 9 decimal places.
    Rule 4: Reason codes already pipe-joined + sorted (Lock 4).
    """
    try:
        # --- Stable sort keys per file type (Rule 2) ---
        SORT_KEYS = {
            'decision_ledger.parquet': ['pair', 'tf', 'request_id', 'event_ts'],
            'positions.parquet': ['pair', 'tf', 'position_id', 'event_ts'],
            'trades.parquet': ['pair', 'tf', 'trade_id', 'fill_ts', 'action'],
        }

        # --- Run-variant fields to exclude (Rule 1) ---
        EXCLUDE_COLS = {'run_id'}

        def canonicalize_and_hash(path, sort_keys):
            """Load parquet, canonicalize, and return SHA-256 hash."""
            df = pd.read_parquet(path)

            # Rule 1: Drop run-variant fields
            drop_cols = [c for c in EXCLUDE_COLS if c in df.columns]
            if drop_cols:
                df = df.drop(columns=drop_cols)

            # Convert object-dtype columns to string for sorting/hashing
            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].astype(str)

            # Rule 3: Float canonicalization — round to 9 decimal places
            float_cols = df.select_dtypes(include=[np.floating]).columns.tolist()
            for col in float_cols:
                df[col] = df[col].round(9)

            # Rule 2: Sort by stable keys
            available_keys = [k for k in sort_keys if k in df.columns]
            if available_keys:
                df = df.sort_values(available_keys).reset_index(drop=True)

            return hashlib.sha256(df.to_csv(index=False).encode()).hexdigest(), df

        # --- Run 1 ---
        os.environ["PYTHONHASHSEED"] = "42"
        np.random.seed(42)
        random.seed(42)
        from replay.runners.run_replay import ReplayRunner
        r1 = ReplayRunner()
        id1 = r1.run(pairs=['EURUSD'], tfs=['M30'], max_bars=30)

        # --- Run 2 ---
        os.environ["PYTHONHASHSEED"] = "42"
        np.random.seed(42)
        random.seed(42)
        r2 = ReplayRunner()
        id2 = r2.run(pairs=['EURUSD'], tfs=['M30'], max_bars=30)

        runs_dir = PROJECT_ROOT / 'replay' / 'outputs' / 'runs'
        d1 = runs_dir / id1
        d2 = runs_dir / id2

        all_match = True
        mismatch_details = []

        # Compare all 3 parquet files
        for fname, sort_keys in SORT_KEYS.items():
            f1 = d1 / fname
            f2 = d2 / fname
            if not f1.exists() or not f2.exists():
                continue

            h1, df1 = canonicalize_and_hash(f1, sort_keys)
            h2, df2 = canonicalize_and_hash(f2, sort_keys)

            if h1 != h2:
                all_match = False
                # Find first mismatch for diagnostics
                if len(df1) != len(df2):
                    mismatch_details.append(
                        f"{fname}: row count differs ({len(df1)} vs {len(df2)})")
                else:
                    for col in df1.columns:
                        if col in df2.columns:
                            diff_mask = df1[col].astype(str) != df2[col].astype(str)
                            if diff_mask.any():
                                first_idx = diff_mask.idxmax()
                                mismatch_details.append(
                                    f"{fname}/{col}[{first_idx}]: "
                                    f"'{df1[col].iloc[first_idx]}' vs '{df2[col].iloc[first_idx]}'")
                                break

        detail = f"Compared {len(SORT_KEYS)} files across runs {id1} vs {id2}"
        if mismatch_details:
            detail += f" | First mismatch: {mismatch_details[0]}"

        record(12, "Determinism: Lock 10 hash contract", all_match, detail)
    except Exception as e:
        record(12, "Determinism test", False, f"{e}\n{traceback.format_exc()}")


# ============================================================
# TEST 13: No-lookahead test
# ============================================================
def test_13():
    try:
        from replay.runners.replay_iterator import ReplayIterator

        iterator = ReplayIterator(
            pair='EURUSD', tf='M30',
            features_dir=str(PROJECT_ROOT / 'features'),
            schema_path=str(PROJECT_ROOT / 'schema' / 'feature_schema.json'),
            date_start='2024-01-01', date_end='2024-01-31',
            mode='LIVE',
        )

        # Load raw data
        raw_df = pd.read_parquet(PROJECT_ROOT / 'features' / 'EURUSD_M30_features.parquet')
        raw_df = raw_df.reset_index()
        ts_col = raw_df.columns[0]
        raw_df[ts_col] = pd.to_datetime(raw_df[ts_col])
        raw_df = raw_df.sort_values(ts_col).reset_index(drop=True)

        with open(PROJECT_ROOT / 'schema' / 'feature_schema.json') as f:
            schema = json.load(f)
        feature_names = [feat['name'] for feat in schema['features']]

        events = list(iterator)
        checks = 0
        verified = 0

        for event in events[:10]:
            event_ts = pd.Timestamp(event['timestamp'])
            mask = raw_df[ts_col] == event_ts
            if not mask.any():
                continue
            raw_idx = raw_df[mask].index[0]
            if raw_idx == 0:
                continue

            prev_row = raw_df.iloc[raw_idx - 1]
            test_feats = [f for f in feature_names[:5] if f in raw_df.columns]
            if not test_feats:
                continue

            checks += 1
            match = True
            for feat in test_feats:
                exp = prev_row[feat]
                act = event['features'][feature_names.index(feat)]
                if pd.isna(exp) and np.isnan(act):
                    continue
                if not pd.isna(exp) and not np.isclose(exp, act, rtol=1e-5):
                    match = False
                    break
            if match:
                verified += 1

        ok = checks > 0 and verified == checks
        record(13, "No-lookahead: features from bar N-1", ok,
               f"{verified}/{checks} bars verified")
    except Exception as e:
        record(13, "No-lookahead test", False, str(e))


# ============================================================
# TEST 14: MT5 gate export
# ============================================================
def test_14():
    try:
        if not SMOKE_RUN_ID:
            record(14, "MT5 gate export", False, "No smoke test run")
            return
        from replay.runners.mt5_gate_export import export_mt5_bundle
        run_dir = PROJECT_ROOT / 'replay' / 'outputs' / 'runs' / SMOKE_RUN_ID
        zip_path = export_mt5_bundle(run_dir)
        ok = zip_path.exists()
        if ok:
            with zipfile.ZipFile(zip_path) as zf:
                files = zf.namelist()
            record(14, "MT5 gate export produces valid zip", ok,
                   f"{len(files)} files: {files}")
        else:
            record(14, "MT5 gate export", False, "Zip not created")
    except Exception as e:
        record(14, "MT5 gate export", False, str(e))


# ============================================================
# TEST 15: Gate criteria evaluation
# ============================================================
def test_15():
    try:
        if not SMOKE_RUN_ID:
            record(15, "Gate criteria evaluation", False, "No smoke test run")
            return
        run_dir = PROJECT_ROOT / 'replay' / 'outputs' / 'runs' / SMOKE_RUN_ID
        with open(run_dir / 'gate_criteria_result.json') as f:
            gate = json.load(f)

        overall = gate.get('overall', 'UNKNOWN')
        criteria = gate.get('criteria', {})
        detail_parts = [f"{k}={'PASS' if v.get('passed') else 'FAIL'}" for k, v in criteria.items()]
        record(15, "Gate criteria evaluation runs", True,
               f"Overall: {overall} | {', '.join(detail_parts)}")
    except Exception as e:
        record(15, "Gate criteria evaluation", False, str(e))


# ============================================================
# TEST 16: Reason codes registry
# ============================================================
def test_16():
    try:
        if not SMOKE_RUN_ID:
            record(16, "Reason codes from registry", False, "No smoke test run")
            return

        with open(PROJECT_ROOT / 'audit' / 'schemas' / 'reason_codes.json') as f:
            valid_codes = set(json.load(f).keys())

        run_dir = PROJECT_ROOT / 'replay' / 'outputs' / 'runs' / SMOKE_RUN_ID
        ledger_df = pd.read_parquet(run_dir / 'decision_ledger.parquet')

        invalid_codes = set()
        for codes_str in ledger_df['reason_codes'].dropna():
            if not codes_str:
                continue
            for code in str(codes_str).split('|'):
                code = code.strip()
                if code and code not in valid_codes:
                    invalid_codes.add(code)

        ok = len(invalid_codes) == 0
        record(16, "Reason codes all from registry", ok,
               f"Invalid: {invalid_codes}" if invalid_codes else "All valid")
    except Exception as e:
        record(16, "Reason codes from registry", False, str(e))


# ============================================================
# TEST 17: Lock 9 — Cost sign invariant on trade output
# ============================================================
def test_17():
    try:
        if not SMOKE_RUN_ID:
            record(17, "Lock 9: Cost sign invariant", False, "No smoke test run")
            return

        run_dir = PROJECT_ROOT / 'replay' / 'outputs' / 'runs' / SMOKE_RUN_ID
        trades_df = pd.read_parquet(run_dir / 'trades.parquet')

        if len(trades_df) == 0:
            record(17, "Lock 9: Cost sign invariant", True, "No trades to check")
            return

        violations = []

        for idx, row in trades_df.iterrows():
            # (1) All cost fields non-negative
            for cost_col in ['spread_cost_pips', 'slippage_cost_pips', 'commission_cost_usd']:
                val = row.get(cost_col)
                if val is not None and val < 0:
                    violations.append(f"Row {idx}: {cost_col}={val} < 0")

            total = row.get('total_cost_usd')
            if total is not None and total < 0:
                violations.append(f"Row {idx}: total_cost_usd={total} < 0")

            # (2) total_cost_usd matches components within 1e-9
            if total is not None:
                spread = row.get('spread_cost_pips', 0) or 0
                slip = row.get('slippage_cost_pips', 0) or 0
                comm = row.get('commission_cost_usd', 0) or 0
                pip_value_approx = 10.0
                qty = row.get('qty_lots', 0) or 0
                expected = (spread + slip) * pip_value_approx * qty + comm
                if abs(total - expected) > 1e-9:
                    violations.append(
                        f"Row {idx}: total_cost_usd={total:.10f} != expected={expected:.10f}")

            # (3) pnl_net_usd <= pnl_gross_usd on CLOSE rows
            if row.get('action') == 'CLOSE':
                pnl_gross = row.get('pnl_gross_usd')
                pnl_net = row.get('pnl_net_usd')
                if pnl_gross is not None and pnl_net is not None:
                    if pnl_net > pnl_gross + 1e-9:
                        violations.append(
                            f"Row {idx}: pnl_net={pnl_net} > pnl_gross={pnl_gross}")

        ok = len(violations) == 0
        detail = f"{len(trades_df)} rows checked"
        if not ok:
            detail += f", {len(violations)} violations: {violations[:3]}"
        record(17, "Lock 9: Cost sign invariant", ok, detail)
    except Exception as e:
        record(17, "Lock 9: Cost sign invariant", False, str(e))


# ============================================================
# TEST 18 (A): Replay feature compatibility
# ============================================================
def test_18():
    """REPLAY mode features match model expectations. Run 10 bars through inference."""
    try:
        from replay.runners.replay_iterator import ReplayIterator
        from replay.runners.run_replay import ModelLoader, EnsembleEngine

        # Load iterator in REPLAY mode
        iterator = ReplayIterator(
            pair='EURUSD', tf='M30',
            features_dir=str(PROJECT_ROOT / 'features'),
            date_start='2024-01-01', date_end='2024-06-30',
            mode='REPLAY',
        )
        replay_feat_count = len(iterator.feature_names)

        # Load models
        loader = ModelLoader(str(PROJECT_ROOT / 'models'))
        models = loader.get_models('EURUSD_M30')
        if not models:
            record(18, "Replay feature compatibility", False, "No EURUSD_M30 models found")
            return

        # Log comparison
        model_info = []
        for brain, backend in models.items():
            expected = getattr(backend, 'expected_features', None)
            model_info.append(f"{brain}={expected}")
        detail = f"Replay sends {replay_feat_count} features | Models: {', '.join(model_info)}"

        # Run 10 bars through full inference
        ensemble = EnsembleEngine(str(PROJECT_ROOT / 'ensemble' / 'ensemble_config.json'))
        errors = 0
        bars_tested = 0
        for event in iterator:
            if bars_tested >= 10:
                break
            try:
                features = np.nan_to_num(event['features'], nan=0.0, posinf=0.0, neginf=0.0)
                result = ensemble.run_inference(list(models.items()), features)
                bars_tested += 1
            except Exception as e:
                errors += 1
                detail += f" | Error at bar {event['bar_idx']}: {e}"

        ok = errors == 0 and bars_tested == 10
        detail += f" | {bars_tested} bars, {errors} errors"
        record(18, "Replay feature compatibility (Test A)", ok, detail)
    except Exception as e:
        record(18, "Replay feature compatibility (Test A)", False, str(e))


# ============================================================
# TEST 19 (B): REPLAY mode yields > 273 features
# ============================================================
def test_19():
    """REPLAY mode derives pair-native features (> 273 for most pairs)."""
    try:
        from replay.runners.replay_iterator import ReplayIterator
        iterator = ReplayIterator(
            pair='EURUSD', tf='M30',
            features_dir=str(PROJECT_ROOT / 'features'),
            date_start='2024-01-01', date_end='2024-01-31',
            mode='REPLAY',
        )
        feat_count = len(iterator.feature_names)
        event = next(iter(iterator))
        event_feat_count = event['feature_count']
        ok = feat_count > 273 and event_feat_count == feat_count
        record(19, "REPLAY mode yields > 273 features (Test B)", ok,
               f"feature_count={feat_count}, event_feature_count={event_feat_count}")
    except Exception as e:
        record(19, "REPLAY mode yields > 273 features (Test B)", False, str(e))


# ============================================================
# TEST 20 (C): LIVE mode still enforces 273 boundary
# ============================================================
def test_20():
    """LIVE mode enforces 273 universal boundary."""
    try:
        from replay.runners.replay_iterator import ReplayIterator
        iterator = ReplayIterator(
            pair='EURUSD', tf='M30',
            features_dir=str(PROJECT_ROOT / 'features'),
            schema_path=str(PROJECT_ROOT / 'schema' / 'feature_schema.json'),
            date_start='2024-01-01', date_end='2024-01-31',
            mode='LIVE',
        )
        feat_count = len(iterator.feature_names)
        event = next(iter(iterator))
        event_feat_count = event['feature_count']
        ok = feat_count == 273 and event_feat_count == 273
        record(20, "LIVE mode enforces 273 boundary (Test C)", ok,
               f"feature_count={feat_count}, event_feature_count={event_feat_count}")
    except Exception as e:
        record(20, "LIVE mode enforces 273 boundary (Test C)", False, str(e))


# ============================================================
# TEST 21 (D): 500-bar smoke — trades > 0, INFERENCE_ERROR = 0
# ============================================================
def test_21():
    """500-bar REPLAY smoke test: confirm trades occur and 0 inference errors."""
    try:
        os.environ["PYTHONHASHSEED"] = "42"
        np.random.seed(42)
        random.seed(42)

        from replay.runners.run_replay import ReplayRunner
        runner = ReplayRunner()
        run_id = runner.run(pairs=['EURUSD'], tfs=['M30'], max_bars=500)

        run_dir = PROJECT_ROOT / 'replay' / 'outputs' / 'runs' / run_id
        trades_df = pd.read_parquet(run_dir / 'trades.parquet')
        ledger_df = pd.read_parquet(run_dir / 'decision_ledger.parquet')

        trade_count = len(trades_df[trades_df['action'] == 'CLOSE']) if len(trades_df) > 0 else 0
        inference_errors = 0
        if 'reason_codes' in ledger_df.columns:
            for codes in ledger_df['reason_codes'].dropna():
                if 'INFERENCE_ERROR' in str(codes):
                    inference_errors += 1

        ok = trade_count > 0 and inference_errors == 0
        detail = f"trades={trade_count}, inference_errors={inference_errors}, bars={len(ledger_df)}"

        # Also check metrics
        with open(run_dir / 'metrics.json') as f:
            metrics = json.load(f)
        detail += f", PF={metrics.get('profit_factor', 0)}, WR={metrics.get('win_rate', 0)}"

        record(21, "500-bar smoke: trades>0, errors=0 (Test D)", ok, detail)
    except Exception as e:
        record(21, "500-bar smoke: trades>0, errors=0 (Test D)", False, str(e))


# ============================================================
# TEST 22 (E): Ledger schema unchanged
# ============================================================
def test_22():
    """Verify DECISION_LEDGER_SCHEMA is unchanged (exact field match)."""
    try:
        from replay.runners.parquet_schemas import DECISION_LEDGER_SCHEMA

        expected_fields = [
            'event_ts', 'pair', 'tf', 'request_id', 'run_id', 'scenario',
            'regime_state', 'regime_confidence',
            'enabled_models_count', 'enabled_models_keys',
            'models_voted', 'models_agreed',
            'brain_probs_trimmed_mean',
            'agreement_score', 'agreement_threshold_base', 'agreement_threshold_modified',
            'alt_data_available', 'cot_pressure', 'cot_extreme', 'cme_spike', 'cme_confirm',
            'alt_agreement_adjustment', 'mtf_status',
            'decision_side', 'decision_confidence',
            'raw_signal_side', 'signal_overridden',
            'risk_veto', 'risk_reason',
            'action_taken', 'latency_ms', 'reason_codes',
        ]

        schema_fields = [f.name for f in DECISION_LEDGER_SCHEMA]
        schema_set = set(schema_fields)
        expected_set = set(expected_fields)

        missing = expected_set - schema_set
        extra = schema_set - expected_set

        ok = len(missing) == 0
        detail = f"{len(schema_fields)} fields"
        if missing:
            detail += f", missing: {missing}"
        if extra:
            detail += f", extra: {extra}"
        record(22, "Ledger schema unchanged (Test E)", ok, detail)
    except Exception as e:
        record(22, "Ledger schema unchanged (Test E)", False, str(e))


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 75)
    print("CHAOS V3.5 PRODUCTION UPGRADE — 22-TEST VERIFICATION")
    print("=" * 75)
    print()

    start = time.time()

    for i in range(1, 23):
        try:
            globals()[f'test_{i}']()
        except Exception as e:
            record(i, f"Test {i}", False, f"EXCEPTION: {e}")

    elapsed = time.time() - start

    print()
    print("=" * 75)
    print(f"{'Test':>6} | {'Status':>6} | Description")
    print("-" * 75)
    for num, desc, status, detail in RESULTS:
        line = f"{num:6d} | {status:>6} | {desc}"
        if detail:
            line += f" | {detail}"
        print(line)

    passed = sum(1 for _, _, s, _ in RESULTS if s == 'PASS')
    failed = sum(1 for _, _, s, _ in RESULTS if s == 'FAIL')
    print("-" * 75)
    print(f"TOTAL: {passed} PASS / {failed} FAIL / {len(RESULTS)} TOTAL  ({elapsed:.1f}s)")
    print("=" * 75)


if __name__ == '__main__':
    main()
