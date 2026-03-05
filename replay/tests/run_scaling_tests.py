"""
CHAOS V3.5 — Scaling Runs

3 progressively larger replay runs x 3 cost scenarios each.
(Run 1 skipped — already complete, baseline frozen.)
Prints PF, trades, win rate, max DD, error rate, risk activations after each.

Usage:
    python -u -W ignore replay/tests/run_scaling_tests.py
"""
import os
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["SKLEARN_WARNINGS"] = "ignore"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["ORT_LOG_SEVERITY_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore")

# Suppress ONNX Runtime warnings before any ORT imports
try:
    import onnxruntime as ort
    ort.set_default_logger_severity(3)  # ERROR only
except ImportError:
    pass

import sys
import json
import time
import traceback
import tracemalloc
import gc
import numpy as np
import random
import pandas as pd
from pathlib import Path

# Determinism
os.environ["PYTHONHASHSEED"] = "42"
np.random.seed(42)
random.seed(42)

PROJECT_ROOT = Path('G:/My Drive/chaos_v1.0')
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'inference'))

SCENARIOS = ['IBKR_BASE', 'BASE_PLUS_25', 'STRESS_PLUS_75', 'VOLATILITY_SPIKE']

ALL_PAIRS = [
    'AUDUSD', 'EURUSD', 'EURJPY', 'GBPJPY', 'GBPUSD',
    'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY',
]

RUNS = [
    # Run 1 skipped — baseline frozen
    # Run 2 skipped — completed (PF 2.55/2.18/1.65 across 3 scenarios, all PASS)
    {
        'name': 'Run 3: EURUSD multi-TF',
        'pairs': ['EURUSD'],
        'tfs': ['M5', 'M15', 'M30', 'H1'],
        'max_bars': 100000,  # capped to avoid M5 OOM (224K bars + 48 models = 6+ GB)
    },
    {
        'name': 'Run 4: All 9 pairs x 4 TFs (full gate bundle)',
        'pairs': ALL_PAIRS,
        'tfs': ['M5', 'M15', 'M30', 'H1'],
        'max_bars': 50000,  # capped — 9 pairs x 4 TFs = 36 combos, memory constrained
    },
]


def print_metrics(run_name, scenario, run_id, wall_time, peak_mem_mb):
    """Print key metrics for a completed run."""
    run_dir = PROJECT_ROOT / 'replay' / 'outputs' / 'runs' / run_id

    with open(run_dir / 'metrics.json') as f:
        metrics = json.load(f)
    with open(run_dir / 'manifest.json') as f:
        manifest = json.load(f)

    # Count risk activations from ledger
    risk_activations = 0
    cooldowns = 0
    correlation_blocks = 0
    ledger_path = run_dir / 'decision_ledger.parquet'
    if ledger_path.exists():
        ledger_df = pd.read_parquet(ledger_path)
        if 'risk_veto' in ledger_df.columns:
            risk_activations = int(ledger_df['risk_veto'].sum())
        if 'reason_codes' in ledger_df.columns:
            for codes in ledger_df['reason_codes'].dropna():
                codes_str = str(codes)
                if 'EXEC_COOLDOWN_ACTIVE' in codes_str:
                    cooldowns += 1
                if 'CORR_BLOCK' in codes_str:
                    correlation_blocks += 1

    pf = metrics.get('profit_factor', 0)
    trades = metrics.get('total_trades', 0)
    wr = metrics.get('win_rate', 0)
    max_dd = metrics.get('max_drawdown_pct', 0)
    total_pnl = metrics.get('total_pnl_net_usd', 0)
    errors = manifest.get('errors_count', 0)
    total_bars = manifest.get('total_bars', 0)
    error_rate = (errors / total_bars * 100) if total_bars > 0 else 0
    bars_per_sec = total_bars / wall_time if wall_time > 0 else 0

    print(f"  {'Scenario':<20s} {scenario}")
    print(f"  {'Run ID':<20s} {run_id}")
    print(f"  {'Bars':<20s} {total_bars:,}")
    print(f"  {'Trades':<20s} {trades}")
    print(f"  {'Profit Factor':<20s} {pf:.4f}")
    print(f"  {'Win Rate':<20s} {wr:.2%}")
    print(f"  {'Total PnL':<20s} ${total_pnl:,.2f}")
    print(f"  {'Max Drawdown':<20s} {max_dd:.2f}%")
    print(f"  {'Errors':<20s} {errors} ({error_rate:.4f}%)")
    print(f"  {'Risk Vetoes':<20s} {risk_activations}")
    print(f"  {'Cooldowns':<20s} {cooldowns}")
    print(f"  {'Corr Blocks':<20s} {correlation_blocks}")
    print(f"  {'Wall Time':<20s} {wall_time:.1f}s")
    print(f"  {'Throughput':<20s} {bars_per_sec:.1f} bars/sec")
    print(f"  {'Peak Memory':<20s} {peak_mem_mb:.1f} MB")

    # Print per-pair-tf breakdown if available
    ptf_path = run_dir / 'metrics_by_pair_tf.json'
    if ptf_path.exists():
        with open(ptf_path) as f:
            ptf = json.load(f)
        if ptf:
            print(f"  --- Per pair/tf breakdown ---")
            print(f"  {'Pair_TF':<16s} {'Trades':>7s} {'PF':>8s} {'WR':>7s} {'PnL':>12s} {'MaxDD':>10s}")
            for key in sorted(ptf.keys()):
                m = ptf[key]
                print(f"  {key:<16s} {m['trades']:>7d} {m['profit_factor']:>8.2f} "
                      f"{m['win_rate']:>6.1%} ${m['total_pnl_usd']:>10,.2f} "
                      f"${m['max_drawdown_usd']:>8,.2f}")

    # Print errors.log if it exists
    errors_path = run_dir / 'errors.log'
    if errors_path.exists():
        with open(errors_path) as f:
            err_lines = f.readlines()
        if err_lines:
            print(f"  --- Errors ({len(err_lines)}) ---")
            for line in err_lines[:5]:  # first 5 errors only
                print(f"    {line.strip()}")
            if len(err_lines) > 5:
                print(f"    ... and {len(err_lines) - 5} more")

    # Print gate result + regression triggers
    gate_path = run_dir / 'gate_criteria_result.json'
    if gate_path.exists():
        with open(gate_path) as f:
            gate = json.load(f)
        overall = gate.get('overall', 'UNKNOWN')
        criteria = gate.get('criteria', {})
        criteria_str = ', '.join(
            f"{k}={'PASS' if v.get('passed') else 'FAIL'}" for k, v in criteria.items())
        print(f"  {'Gate Result':<20s} {overall} ({criteria_str})")
        # Regression triggers
        triggers = gate.get('regression_triggers', {})
        for tname, tdata in triggers.items():
            status = tdata.get('status', '?')
            severity = tdata.get('severity', '?')
            detail = ''
            if tname == 'veto_dominance':
                detail = f" ({tdata.get('dominant_cause', '?')} @ {tdata.get('dominant_pct', 0):.0%})"
            print(f"  {'  Trigger: ' + tname:<20s} {status} [{severity}]{detail}")

    # Coverage / ensemble health
    cov_path = run_dir / 'coverage.json'
    if cov_path.exists():
        with open(cov_path) as f:
            cov = json.load(f)
        eh = cov.get('ensemble_health', {})
        if eh.get('vote_entropy_mean') is not None:
            print(f"  {'Entropy (mean)':<20s} {eh['vote_entropy_mean']:.4f}")
            print(f"  {'Confidence (mean)':<20s} {eh['vote_confidence_mean']:.4f}")
            print(f"  {'Decisiveness':<20s} {eh['decisiveness_ratio']:.4f}")

    # Veto breakdown top 5
    veto_path = run_dir / 'veto_breakdown.json'
    if veto_path.exists():
        with open(veto_path) as f:
            veto = json.load(f)
        top5 = list(veto.get('top_10_reasons', {}).items())[:5]
        if top5:
            print(f"  --- Veto top 5 ---")
            for code, count in top5:
                print(f"    {code}: {count}")

    # Partial fill metrics
    pfm = metrics.get('partial_fill', {})
    if pfm:
        print(f"  {'Partial Fill Rate':<20s} {pfm.get('partial_fill_rate', 0):.4f}")
        print(f"  {'Fill Invariant':<20s} {'PASS' if pfm.get('fill_invariant_pass', True) else 'FAIL'}")

    # Bar cap info
    bar_caps = manifest.get('bar_caps', {})
    if bar_caps.get('cap_active'):
        print(f"  --- Bar Cap Info (max {bar_caps.get('max_bars_per_pair_tf', '?')}) ---")
        per_ptf = bar_caps.get('per_pair_tf', {})
        for key in sorted(per_ptf.keys()):
            cap = per_ptf[key]
            print(f"    {key}: {cap['bars_processed']:,}/{cap['bars_available']:,} "
                  f"({cap['coverage_of_window_pct']}%)")
    if gate_path.exists() and gate.get('window_capped'):
        print(f"  {'Window Capped':<20s} YES")

    # Brain contributions
    brain_path = run_dir / 'brain_contributions.json'
    if brain_path.exists():
        with open(brain_path) as f:
            brain = json.load(f)
        quarantine = brain.get('quarantine_candidates', [])
        n_brains = brain.get('brains_tracked', 0)
        n_bars = brain.get('total_bars_tracked', 0)
        print(f"  {'Brains Tracked':<20s} {n_brains} ({n_bars:,} bars)")
        if quarantine:
            print(f"  {'Quarantine Flag':<20s} {', '.join(quarantine)}")
        else:
            print(f"  {'Quarantine Flag':<20s} NONE (all brains healthy)")

    # VOLATILITY_SPIKE specific
    if scenario == 'VOLATILITY_SPIKE':
        vs_pass = pf >= 1.20
        print(f"  {'VOL_SPIKE Threshold':<20s} PF {pf:.4f} >= 1.20 -> {'PASS' if vs_pass else 'FAIL'}")

    print()
    sys.stdout.flush()


def main():
    print("=" * 80)
    print("CHAOS V3.5 — SCALING RUNS (batch inference, chunked writes)")
    print("=" * 80)
    print()
    sys.stdout.flush()

    from replay.runners.run_replay import ReplayRunner

    total_start = time.perf_counter()
    is_first_run = True

    for run_cfg in RUNS:
        print("-" * 80)
        print(f"  {run_cfg['name']}")
        print(f"  Pairs: {run_cfg['pairs']}")
        print(f"  TFs: {run_cfg['tfs']}")
        print(f"  Max bars: {run_cfg['max_bars'] or 'unlimited'}")
        print("-" * 80)
        sys.stdout.flush()

        # ONE runner per run config — reuse across scenarios
        # Only load models for this run's pairs/tfs (avoids loading all 300+ models)
        runner = ReplayRunner(pairs=run_cfg['pairs'], tfs=run_cfg['tfs'])
        print(f"  Models loaded. Starting scenarios...")
        sys.stdout.flush()

        for scenario in SCENARIOS:
            # Reset determinism per scenario
            os.environ["PYTHONHASHSEED"] = "42"
            np.random.seed(42)
            random.seed(42)

            # Reset runner state (keeps models loaded) + free memory
            runner.reset_state()
            gc.collect()

            tracemalloc.start()
            t0 = time.perf_counter()

            try:
                run_id = runner.run(
                    pairs=run_cfg['pairs'],
                    tfs=run_cfg['tfs'],
                    max_bars=run_cfg['max_bars'],
                    scenario_name=scenario,
                )
                wall_time = time.perf_counter() - t0
                _, peak_mem = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                peak_mem_mb = peak_mem / (1024 * 1024)

                # Print bars/sec gauge after first scenario of Run 2
                if is_first_run and scenario == 'IBKR_BASE':
                    run_dir = PROJECT_ROOT / 'replay' / 'outputs' / 'runs' / run_id
                    with open(run_dir / 'manifest.json') as f:
                        m = json.load(f)
                    bars = m.get('total_bars', 0)
                    rate = bars / wall_time if wall_time > 0 else 0
                    print(f"\n  >>> THROUGHPUT GAUGE: {rate:.1f} bars/sec "
                          f"({bars:,} bars in {wall_time:.1f}s) <<<\n")
                    sys.stdout.flush()
                    is_first_run = False

                print_metrics(run_cfg['name'], scenario, run_id, wall_time, peak_mem_mb)

            except Exception as e:
                wall_time = time.perf_counter() - t0
                try:
                    tracemalloc.stop()
                except RuntimeError:
                    pass
                print(f"  FAILED ({scenario}): {e}")
                print(f"  {traceback.format_exc()}")
                print()
                sys.stdout.flush()

    total_elapsed = time.perf_counter() - total_start
    print("=" * 80)
    print(f"  All scaling runs complete in {total_elapsed:.1f}s")
    print("=" * 80)
    sys.stdout.flush()


if __name__ == '__main__':
    main()
