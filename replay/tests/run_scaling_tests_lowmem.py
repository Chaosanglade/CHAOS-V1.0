"""
CHAOS V3.5 — Scaling Runs (LOW MEMORY version)

Processes Run 4 ONE PAIR AT A TIME to avoid OOM.
Each pair loads ~80 models, runs all 4 scenarios, then frees memory.
Results are aggregated across pairs at the end.

Run 3 is skipped (already completed in previous session).

Usage:
    python -u -W ignore replay/tests/run_scaling_tests_lowmem.py
"""
import os
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["SKLEARN_WARNINGS"] = "ignore"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["ORT_LOG_SEVERITY_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore")

try:
    import onnxruntime as ort
    ort.set_default_logger_severity(3)
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
from datetime import datetime, timezone

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

ALL_TFS = ['M5', 'M15', 'M30', 'H1']


def collect_run_metrics(run_id):
    """Collect metrics from a completed run."""
    run_dir = PROJECT_ROOT / 'replay' / 'outputs' / 'runs' / run_id
    result = {'run_id': run_id}

    metrics_path = run_dir / 'metrics.json'
    if metrics_path.exists():
        with open(metrics_path) as f:
            result['metrics'] = json.load(f)

    manifest_path = run_dir / 'manifest.json'
    if manifest_path.exists():
        with open(manifest_path) as f:
            result['manifest'] = json.load(f)

    ptf_path = run_dir / 'metrics_by_pair_tf.json'
    if ptf_path.exists():
        with open(ptf_path) as f:
            result['metrics_by_pair_tf'] = json.load(f)

    brain_path = run_dir / 'brain_contributions.json'
    if brain_path.exists():
        with open(brain_path) as f:
            result['brain_contributions'] = json.load(f)

    coverage_path = run_dir / 'coverage.json'
    if coverage_path.exists():
        with open(coverage_path) as f:
            result['coverage'] = json.load(f)

    veto_path = run_dir / 'veto_breakdown.json'
    if veto_path.exists():
        with open(veto_path) as f:
            result['veto_breakdown'] = json.load(f)

    gate_path = run_dir / 'gate_criteria_result.json'
    if gate_path.exists():
        with open(gate_path) as f:
            result['gate'] = json.load(f)

    return result


def print_scenario_summary(scenario, pair, run_data, wall_time, peak_mem_mb):
    """Print concise per-pair scenario summary."""
    m = run_data.get('metrics', {})
    pf = m.get('profit_factor', 0)
    trades = m.get('total_trades', 0)
    total_pnl = m.get('total_pnl_net_usd', 0)
    wr = m.get('win_rate', 0)
    max_dd = m.get('max_drawdown_pct', 0)

    gate = run_data.get('gate', {})
    gate_result = gate.get('overall', '?')

    brain = run_data.get('brain_contributions', {})
    quarantine = brain.get('quarantine_candidates', [])

    print(f"    {pair:<10s} {scenario:<18s} PF={pf:.4f}  Trades={trades:<5d} "
          f"PnL=${total_pnl:>10,.2f}  WR={wr:.1%}  DD={max_dd:.2f}%  "
          f"Gate={gate_result}  Q={'|'.join(quarantine) if quarantine else 'NONE'}  "
          f"({wall_time:.0f}s, {peak_mem_mb:.0f}MB)")
    sys.stdout.flush()


def aggregate_results(all_results):
    """
    Aggregate per-pair results into a portfolio-level summary.
    all_results: {scenario: [{pair, run_data, ...}, ...]}
    """
    print("\n" + "=" * 100)
    print("  FULL-COVERAGE RUN 4 — AGGREGATED RESULTS (per-pair sequential, 689 models)")
    print("=" * 100)

    # 1. Scenario Summary Table
    print("\n  === SCENARIO SUMMARY ===")
    print(f"  {'Scenario':<20s} {'PF':>8s} {'Trades':>8s} {'PnL':>12s} {'MaxDD':>8s} {'Gate':>6s}")
    for scenario in SCENARIOS:
        pairs_data = all_results.get(scenario, [])
        total_gross_win = 0
        total_gross_loss = 0
        total_trades = 0
        total_pnl = 0
        max_dd = 0

        for pd_item in pairs_data:
            m = pd_item['run_data'].get('metrics', {})
            total_trades += m.get('total_trades', 0)
            total_pnl += m.get('total_pnl_net_usd', 0)
            max_dd = max(max_dd, m.get('max_drawdown_pct', 0))
            # Reconstruct PF from gross win/loss
            total_gross_win += m.get('gross_profit_usd', 0)
            total_gross_loss += abs(m.get('gross_loss_usd', 0))

        pf = total_gross_win / total_gross_loss if total_gross_loss > 0 else 0
        # Gate: PASS if PF > 1.15 under BASE_PLUS_25, or PF > 1.20 for VOL_SPIKE
        if scenario == 'VOLATILITY_SPIKE':
            gate = 'PASS' if pf >= 1.20 else 'FAIL'
        elif scenario == 'BASE_PLUS_25':
            gate = 'PASS' if pf >= 1.15 else 'FAIL'
        else:
            gate = 'PASS' if pf >= 1.0 else 'FAIL'

        print(f"  {scenario:<20s} {pf:>8.2f} {total_trades:>8d} ${total_pnl:>10,.2f} "
              f"{max_dd:>7.2f}% {gate:>6s}")

    # 2. Brain Coverage Per Pair
    print("\n  === BRAIN COVERAGE PER PAIR ===")
    # Use IBKR_BASE scenario for brain counts
    ibkr_data = all_results.get('IBKR_BASE', [])
    for pd_item in ibkr_data:
        pair = pd_item['pair']
        brain = pd_item['run_data'].get('brain_contributions', {})
        n_brains = brain.get('brains_tracked', 0)
        print(f"  {pair}: {n_brains}/21 brains loaded")

    # 3. Ensemble Dilution Comparison
    print("\n  === ENSEMBLE DILUTION COMPARISON ===")
    print(f"  {'':>22s} {'12-brain (prev)':>16s} {'20-brain (new)':>16s} {'Change':>10s}")

    ibkr_total_pnl = sum(
        d['run_data'].get('metrics', {}).get('total_pnl_net_usd', 0)
        for d in ibkr_data
    )
    ibkr_total_trades = sum(
        d['run_data'].get('metrics', {}).get('total_trades', 0)
        for d in ibkr_data
    )
    ibkr_gross_win = sum(
        d['run_data'].get('metrics', {}).get('gross_profit_usd', 0)
        for d in ibkr_data
    )
    ibkr_gross_loss = sum(
        abs(d['run_data'].get('metrics', {}).get('gross_loss_usd', 0))
        for d in ibkr_data
    )
    ibkr_pf = ibkr_gross_win / ibkr_gross_loss if ibkr_gross_loss > 0 else 0

    # Previous 12-brain Run 4 values from resume package
    prev_pf = 2.46
    prev_trades = 4265
    prev_pnl = 8910

    pf_change = ((ibkr_pf - prev_pf) / prev_pf * 100) if prev_pf > 0 else 0
    trades_change = ((ibkr_total_trades - prev_trades) / prev_trades * 100) if prev_trades > 0 else 0
    pnl_change = ((ibkr_total_pnl - prev_pnl) / prev_pnl * 100) if prev_pnl > 0 else 0

    print(f"  {'PF (IBKR):':<22s} {prev_pf:>16.2f} {ibkr_pf:>16.2f} {pf_change:>+9.1f}%")
    print(f"  {'Trades:':<22s} {prev_trades:>16,d} {ibkr_total_trades:>16,d} {trades_change:>+9.1f}%")
    print(f"  {'PnL:':<22s} ${prev_pnl:>14,.2f} ${ibkr_total_pnl:>14,.2f} {pnl_change:>+9.1f}%")

    if ibkr_total_trades < prev_trades * 0.5 or ibkr_pf < prev_pf * 0.7:
        print(f"\n  VERDICT: *** DILUTION DETECTED ***")
    elif ibkr_total_trades > prev_trades * 0.8 and ibkr_pf > prev_pf * 0.9:
        print(f"\n  VERDICT: ENSEMBLE ROBUST")
    else:
        print(f"\n  VERDICT: MIXED — review per-pair breakdown")

    # 4. Brain Contribution Report
    print("\n  === BRAIN CONTRIBUTION REPORT ===")
    all_quarantine = set()
    all_weak = []
    all_strong = []
    for pd_item in ibkr_data:
        brain = pd_item['run_data'].get('brain_contributions', {})
        for q in brain.get('quarantine_candidates', []):
            all_quarantine.add(q)
        per_brain = brain.get('per_brain', {})
        for bname, bdata in per_brain.items():
            all_weak.append((bname, pd_item['pair'],
                           bdata.get('vote_alignment', 0),
                           bdata.get('aligned_total_pnl', 0)))
            all_strong.append((bname, pd_item['pair'],
                             bdata.get('vote_alignment', 0),
                             bdata.get('aligned_total_pnl', 0)))

    print(f"  Quarantine flags: {', '.join(sorted(all_quarantine)) if all_quarantine else 'NONE'}")

    if all_weak:
        all_weak.sort(key=lambda x: x[2])
        print(f"\n  Bottom 5 brains (by vote alignment):")
        for bname, pair, va, pnl in all_weak[:5]:
            print(f"    {bname:<25s} ({pair}) alignment={va:.3f} pnl=${pnl:.2f}")

    if all_strong:
        all_strong.sort(key=lambda x: x[2], reverse=True)
        print(f"\n  Top 5 brains (by vote alignment):")
        for bname, pair, va, pnl in all_strong[:5]:
            print(f"    {bname:<25s} ({pair}) alignment={va:.3f} pnl=${pnl:.2f}")

    # 5. Pair x TF Matrix (IBKR_BASE, H1+M30)
    print("\n  === PAIR x TF MATRIX (IBKR_BASE, trade-enabled only) ===")
    print(f"  {'Pair':<10s} {'H1 PF':>8s} {'H1 Tr':>7s} {'M30 PF':>8s} {'M30 Tr':>7s}")
    for pd_item in ibkr_data:
        pair = pd_item['pair']
        ptf = pd_item['run_data'].get('metrics_by_pair_tf', {})
        h1 = ptf.get(f'{pair}_H1', {})
        m30 = ptf.get(f'{pair}_M30', {})
        h1_pf = h1.get('profit_factor', 0)
        h1_tr = h1.get('trades', 0)
        m30_pf = m30.get('profit_factor', 0)
        m30_tr = m30.get('trades', 0)
        print(f"  {pair:<10s} {h1_pf:>8.2f} {h1_tr:>7d} {m30_pf:>8.2f} {m30_tr:>7d}")

    # 6. Ensemble Health
    print("\n  === ENSEMBLE HEALTH ===")
    for pd_item in ibkr_data:
        pair = pd_item['pair']
        cov = pd_item['run_data'].get('coverage', {})
        eh = cov.get('ensemble_health', {})
        if eh.get('vote_entropy_mean') is not None:
            print(f"  {pair}: entropy={eh['vote_entropy_mean']:.4f} "
                  f"confidence={eh['vote_confidence_mean']:.4f} "
                  f"decisiveness={eh['decisiveness_ratio']:.4f}")

    # 7. Veto Breakdown Top 5 (aggregated)
    print("\n  === VETO BREAKDOWN (aggregated across pairs) ===")
    veto_totals = {}
    for pd_item in ibkr_data:
        veto = pd_item['run_data'].get('veto_breakdown', {})
        for code, count in veto.get('top_10_reasons', {}).items():
            veto_totals[code] = veto_totals.get(code, 0) + count
    for code, count in sorted(veto_totals.items(), key=lambda x: -x[1])[:5]:
        print(f"    {code}: {count:,}")

    # 8. Per-TF Breakdown
    print("\n  === PER-TF BREAKDOWN (IBKR_BASE) ===")
    tf_totals = {}
    for pd_item in ibkr_data:
        ptf = pd_item['run_data'].get('metrics_by_pair_tf', {})
        for key, m in ptf.items():
            tf = key.split('_')[1]
            if tf not in tf_totals:
                tf_totals[tf] = {'trades': 0, 'gross_win': 0, 'gross_loss': 0, 'pnl': 0}
            tf_totals[tf]['trades'] += m.get('trades', 0)
            tf_totals[tf]['pnl'] += m.get('total_pnl_usd', 0)
            tf_totals[tf]['gross_win'] += m.get('gross_profit_usd', 0)
            tf_totals[tf]['gross_loss'] += abs(m.get('gross_loss_usd', 0))

    for tf in ['H1', 'M30', 'M15', 'M5']:
        t = tf_totals.get(tf, {})
        trades = t.get('trades', 0)
        pf = t['gross_win'] / t['gross_loss'] if t.get('gross_loss', 0) > 0 else 0
        role = 'trade' if tf in ('H1', 'M30') else 'confirm only'
        print(f"  {tf}: {trades} trades, PF {pf:.2f} ({role})")

    print("\n" + "=" * 100)
    sys.stdout.flush()


def main():
    print("=" * 100)
    print("  CHAOS V3.5 — RUN 4: LOW-MEMORY MODE (one pair at a time)")
    print("  9 pairs x 4 TFs x 4 scenarios | 50K bar cap | models loaded per-pair")
    print("=" * 100)
    print()
    sys.stdout.flush()

    from replay.runners.run_replay import ReplayRunner

    total_start = time.perf_counter()
    all_results = {s: [] for s in SCENARIOS}

    for pair_idx, pair in enumerate(ALL_PAIRS):
        pair_start = time.perf_counter()
        print(f"\n{'-' * 80}")
        print(f"  PAIR {pair_idx + 1}/9: {pair}")
        print(f"  Loading models for {pair} x {ALL_TFS}...")
        print(f"{'-' * 80}")
        sys.stdout.flush()

        # Create runner for THIS PAIR ONLY — loads only ~80 models
        runner = ReplayRunner(pairs=[pair], tfs=ALL_TFS)
        print(f"  Models loaded for {pair}. Running 4 scenarios...")
        sys.stdout.flush()

        for scenario in SCENARIOS:
            os.environ["PYTHONHASHSEED"] = "42"
            np.random.seed(42)
            random.seed(42)

            runner.reset_state()
            gc.collect()

            tracemalloc.start()
            t0 = time.perf_counter()

            try:
                run_id = runner.run(
                    pairs=[pair],
                    tfs=ALL_TFS,
                    max_bars=50000,
                    scenario_name=scenario,
                )
                wall_time = time.perf_counter() - t0
                _, peak_mem = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                peak_mem_mb = peak_mem / (1024 * 1024)

                run_data = collect_run_metrics(run_id)
                all_results[scenario].append({
                    'pair': pair,
                    'run_id': run_id,
                    'run_data': run_data,
                    'wall_time': wall_time,
                    'peak_mem_mb': peak_mem_mb,
                })

                print_scenario_summary(scenario, pair, run_data, wall_time, peak_mem_mb)

            except Exception as e:
                wall_time = time.perf_counter() - t0
                try:
                    tracemalloc.stop()
                except RuntimeError:
                    pass
                print(f"    FAILED: {pair} {scenario}: {e}")
                print(f"    {traceback.format_exc()}")
                sys.stdout.flush()

        # Free runner + models before next pair
        del runner
        gc.collect()

        pair_elapsed = time.perf_counter() - pair_start
        print(f"  {pair} complete in {pair_elapsed:.0f}s")
        sys.stdout.flush()

    total_elapsed = time.perf_counter() - total_start
    print(f"\n  All pairs complete in {total_elapsed:.0f}s ({total_elapsed/3600:.1f}h)")

    # Print full aggregated report
    aggregate_results(all_results)

    # Save raw results for later analysis
    results_path = PROJECT_ROOT / 'replay' / 'outputs' / 'run4_lowmem_results.json'
    save_data = {}
    for scenario in SCENARIOS:
        save_data[scenario] = []
        for item in all_results[scenario]:
            save_data[scenario].append({
                'pair': item['pair'],
                'run_id': item['run_id'],
                'wall_time': item['wall_time'],
                'peak_mem_mb': item['peak_mem_mb'],
                'pf': item['run_data'].get('metrics', {}).get('profit_factor', 0),
                'trades': item['run_data'].get('metrics', {}).get('total_trades', 0),
                'pnl': item['run_data'].get('metrics', {}).get('total_pnl_net_usd', 0),
            })
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Raw results saved to: {results_path}")

    print(f"\n{'=' * 100}")
    print(f"  RUN 4 LOW-MEMORY COMPLETE — {total_elapsed:.0f}s total")
    print(f"{'=' * 100}")
    sys.stdout.flush()


if __name__ == '__main__':
    main()
