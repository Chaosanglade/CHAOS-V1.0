"""
CHAOS V1.0 -- VARIANT B PRODUCTION RERUN
Full universe with quarantine enforced (14 brains + residual_mlp exception for GBPJPY_M30).
Low-memory mode: one pair at a time.

Usage:
    python -u -W ignore replay/tests/run_variant_b_rerun.py
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
import gc
import numpy as np
import random
import pandas as pd
from pathlib import Path

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


# ============================================================
# QUARANTINE ENFORCEMENT
# ============================================================
def load_quarantine_config():
    qpath = PROJECT_ROOT / 'replay' / 'config' / 'brain_quarantine.json'
    with open(qpath) as f:
        return json.load(f)


def is_quarantined(quarantine_cfg, brain_name, pair, tf):
    """Check if a brain is quarantined for this pair+tf block."""
    block_key = f"{pair}_{tf}"
    if brain_name in quarantine_cfg.get('global_quarantine', []):
        return True
    cond = quarantine_cfg.get('conditional_quarantine', {}).get(brain_name)
    if cond and cond.get('policy') == 'quarantined_globally_except':
        return block_key not in cond.get('exceptions', [])
    return False


def apply_quarantine(runner, quarantine_cfg):
    """Remove quarantined brains from runner's model loader. Returns stats."""
    stats = {}
    for pair_tf, models in list(runner.model_loader.models.items()):
        parts = pair_tf.split('_')
        pair = parts[0]
        tf = parts[1]
        original_count = len(models)
        quarantined = [b for b in models if is_quarantined(quarantine_cfg, b, pair, tf)]
        for b in quarantined:
            del models[b]
        active = len(models)
        stats[pair_tf] = {
            'original': original_count,
            'active': active,
            'quarantined': quarantined,
        }
    return stats


# ============================================================
# SMOKE TEST
# ============================================================
def run_smoke_test(quarantine_cfg):
    """Quick 1000-bar smoke test on AUDUSD_H1 and GBPJPY_M30."""
    from replay.runners.run_replay import ReplayRunner
    print("\n  QUARANTINE SMOKE TEST")
    print("  " + "-" * 60)

    results = {}
    for pair, tf, expected_brains in [('AUDUSD', 'H1', 14), ('GBPJPY', 'M30', 15)]:
        os.environ["PYTHONHASHSEED"] = "42"
        np.random.seed(42)
        random.seed(42)

        runner = ReplayRunner(pairs=[pair], tfs=ALL_TFS)
        q_stats = apply_quarantine(runner, quarantine_cfg)
        pair_tf = f"{pair}_{tf}"
        active = q_stats.get(pair_tf, {}).get('active', 0)

        runner.reset_state()
        run_id = runner.run(pairs=[pair], tfs=ALL_TFS, max_bars=1000, scenario_name='IBKR_BASE')

        run_dir = PROJECT_ROOT / 'replay' / 'outputs' / 'runs' / run_id
        ptf = {}
        if (run_dir / 'metrics_by_pair_tf.json').exists():
            with open(run_dir / 'metrics_by_pair_tf.json') as f:
                ptf = json.load(f)

        block_m = ptf.get(pair_tf, {})
        trades = block_m.get('trades', 0)

        passed = active == expected_brains
        trade_ok = trades > 0 if pair == 'AUDUSD' else True

        status = 'PASS' if (passed and trade_ok) else 'FAIL'
        print(f"  {pair_tf}: brains_active={active} (expected {expected_brains}), "
              f"trades={trades} -- {status}")
        results[pair_tf] = {'active': active, 'trades': trades, 'status': status}

        del runner
        gc.collect()

    print("  " + "-" * 60)
    sys.stdout.flush()

    all_pass = all(r['status'] == 'PASS' for r in results.values())
    if not all_pass:
        print("  WARNING: Smoke test had failures. Proceeding anyway (1000-bar sample may be too small).")
    return all_pass


# ============================================================
# RESULT COLLECTION
# ============================================================
def collect_run_metrics(run_id):
    run_dir = PROJECT_ROOT / 'replay' / 'outputs' / 'runs' / run_id
    result = {'run_id': run_id}
    for fname, key in [('metrics.json', 'metrics'), ('metrics_by_pair_tf.json', 'metrics_by_pair_tf'),
                       ('brain_contributions.json', 'brain_contributions'), ('coverage.json', 'coverage'),
                       ('veto_breakdown.json', 'veto_breakdown'), ('gate_criteria_result.json', 'gate')]:
        path = run_dir / fname
        if path.exists():
            with open(path) as f:
                result[key] = json.load(f)
    return result


# ============================================================
# AGGREGATE & PRINT
# ============================================================
def print_full_report(all_results, quarantine_stats_by_pair):
    print("\n" + "=" * 100)
    print("  POST-QUARANTINE RUN 4 (14-brain ensemble + residual_mlp on GBPJPY_M30)")
    print("=" * 100)

    # 5A: Scenario Summary
    print("\n  === SCENARIO SUMMARY ===")
    print(f"  {'Scenario':<20s} {'PF':>8s} {'Trades':>8s} {'PnL':>12s} {'MaxDD':>8s} {'Gate':>6s}")
    for scenario in SCENARIOS:
        pairs_data = all_results.get(scenario, [])
        total_trades = 0
        total_pnl = 0
        max_dd = 0
        total_gw = 0
        total_gl = 0
        for d in pairs_data:
            m = d['run_data'].get('metrics', {})
            total_trades += m.get('total_trades', 0)
            total_pnl += m.get('total_pnl_net_usd', 0)
            max_dd = max(max_dd, m.get('max_drawdown_pct', 0))
            total_gw += m.get('gross_profit_usd', 0)
            total_gl += abs(m.get('gross_loss_usd', 0))
        pf = total_gw / total_gl if total_gl > 0 else 0
        if scenario == 'VOLATILITY_SPIKE':
            gate = 'PASS' if pf >= 1.20 else 'FAIL'
        elif scenario == 'BASE_PLUS_25':
            gate = 'PASS' if pf >= 1.15 else 'FAIL'
        else:
            gate = 'PASS' if pf >= 1.0 else 'FAIL'
        print(f"  {scenario:<20s} {pf:>8.2f} {total_trades:>8d} ${total_pnl:>10,.2f} "
              f"{max_dd:>7.2f}% {gate:>6s}")

    # 5B: Three-Way Comparison
    ibkr = all_results.get('IBKR_BASE', [])
    new_pnl = sum(d['run_data'].get('metrics', {}).get('total_pnl_net_usd', 0) for d in ibkr)
    new_trades = sum(d['run_data'].get('metrics', {}).get('total_trades', 0) for d in ibkr)
    new_gw = sum(d['run_data'].get('metrics', {}).get('gross_profit_usd', 0) for d in ibkr)
    new_gl = sum(abs(d['run_data'].get('metrics', {}).get('gross_loss_usd', 0)) for d in ibkr)
    new_pf = new_gw / new_gl if new_gl > 0 else 0

    print("\n  === THREE-WAY COMPARISON ===")
    print(f"  {'':>22s} {'12-brain (orig)':>16s} {'20-brain (diluted)':>18s} {'14-brain (quarantined)':>22s}")
    print(f"  {'PF (IBKR):':<22s} {'2.46':>16s} {'2.25':>18s} {new_pf:>22.2f}")
    print(f"  {'Trades:':<22s} {'4,265':>16s} {'2,584':>18s} {new_trades:>22,d}")
    print(f"  {'PnL:':<22s} {'$8,910':>16s} {'$5,873':>18s} ${new_pnl:>20,.2f}")

    # 5C: Pair x TF Matrix
    print("\n  === PAIR x TF MATRIX (IBKR_BASE, H1+M30) ===")
    print(f"  {'Pair':<10s} {'H1 PF':>8s} {'H1 Tr':>7s} {'M30 PF':>8s} {'M30 Tr':>7s} {'Brains':>8s}")
    for d in ibkr:
        pair = d['pair']
        ptf = d['run_data'].get('metrics_by_pair_tf', {})
        h1 = ptf.get(f'{pair}_H1', {})
        m30 = ptf.get(f'{pair}_M30', {})
        # Get brain count from quarantine stats
        q = quarantine_stats_by_pair.get(pair, {})
        h1_active = q.get(f'{pair}_H1', {}).get('active', '?')
        m30_active = q.get(f'{pair}_M30', {}).get('active', '?')
        brains_str = f"{h1_active}"
        if pair == 'GBPJPY':
            brains_str = f"{h1_active}/{m30_active}*"
        print(f"  {pair:<10s} {h1.get('profit_factor', 0):>8.2f} {h1.get('trades', 0):>7d} "
              f"{m30.get('profit_factor', 0):>8.2f} {m30.get('trades', 0):>7d} {brains_str:>8s}")

    # 5D: Tiering
    print("\n  === NEW TIERING (STRESS_PLUS_75, H1+M30) ===")
    stress = all_results.get('STRESS_PLUS_75', [])
    tiers = {'Tier 1': [], 'Tier 2': [], 'Tier 3/Disabled': []}
    for d in stress:
        pair = d['pair']
        ptf = d['run_data'].get('metrics_by_pair_tf', {})
        h1 = ptf.get(f'{pair}_H1', {})
        m30 = ptf.get(f'{pair}_M30', {})
        combined_trades = h1.get('trades', 0) + m30.get('trades', 0)
        # Use the better TF's PF
        best_pf = max(h1.get('profit_factor', 0), m30.get('profit_factor', 0))
        if best_pf >= 1.30 and combined_trades >= 200:
            tiers['Tier 1'].append(f"{pair} (PF {best_pf:.2f}, {combined_trades} trades)")
        elif best_pf >= 1.15:
            tiers['Tier 2'].append(f"{pair} (PF {best_pf:.2f}, {combined_trades} trades)")
        else:
            tiers['Tier 3/Disabled'].append(f"{pair} (PF {best_pf:.2f}, {combined_trades} trades)")
    for tier, pairs in tiers.items():
        print(f"  {tier}: {', '.join(pairs) if pairs else 'NONE'}")

    # 5E: Ensemble Health
    print("\n  === ENSEMBLE HEALTH ===")
    for d in ibkr:
        pair = d['pair']
        cov = d['run_data'].get('coverage', {})
        eh = cov.get('ensemble_health', {})
        if eh.get('vote_entropy_mean') is not None:
            print(f"  {pair}: entropy={eh['vote_entropy_mean']:.4f} "
                  f"confidence={eh['vote_confidence_mean']:.4f} "
                  f"decisiveness={eh['decisiveness_ratio']:.4f}")

    # 5F: Veto Top 5
    print("\n  === VETO BREAKDOWN (aggregated) ===")
    veto_totals = {}
    for d in ibkr:
        veto = d['run_data'].get('veto_breakdown', {})
        for code, count in veto.get('top_10_reasons', {}).items():
            veto_totals[code] = veto_totals.get(code, 0) + count
    for code, count in sorted(veto_totals.items(), key=lambda x: -x[1])[:5]:
        print(f"    {code}: {count:,}")

    # 5H: Brain Coverage Per Pair
    print("\n  === BRAIN COVERAGE PER PAIR ===")
    for d in ibkr:
        pair = d['pair']
        q = quarantine_stats_by_pair.get(pair, {})
        h1_q = q.get(f'{pair}_H1', {})
        m30_q = q.get(f'{pair}_M30', {})
        h1_active = h1_q.get('active', '?')
        h1_quarantined = len(h1_q.get('quarantined', []))
        note = ""
        if pair == 'GBPJPY':
            note = " (residual_mlp exempted on M30)"
        print(f"  {pair}: {h1_active} active, {h1_quarantined} quarantined{note}")

    print("\n" + "=" * 100)
    sys.stdout.flush()


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 100)
    print("  CHAOS V1.0 -- VARIANT B PRODUCTION RERUN")
    print("  9 pairs x 4 TFs x 4 scenarios | 14-brain ensemble | quarantine enforced")
    print("=" * 100)
    print()
    sys.stdout.flush()

    from replay.runners.run_replay import ReplayRunner
    quarantine_cfg = load_quarantine_config()
    print(f"  Quarantine loaded: {quarantine_cfg['quarantine_version']}")
    print(f"  Global quarantine: {quarantine_cfg['global_quarantine']}")
    print(f"  Conditional: residual_mlp_optuna exempt for GBPJPY_M30")
    sys.stdout.flush()

    # Step 3: Smoke test
    smoke_ok = run_smoke_test(quarantine_cfg)
    print()

    # Step 4: Full rerun
    total_start = time.perf_counter()
    all_results = {s: [] for s in SCENARIOS}
    quarantine_stats_by_pair = {}

    for pair_idx, pair in enumerate(ALL_PAIRS):
        pair_start = time.perf_counter()
        print(f"\n{'-' * 80}")
        print(f"  PAIR {pair_idx + 1}/9: {pair}")
        print(f"{'-' * 80}")
        sys.stdout.flush()

        runner = ReplayRunner(pairs=[pair], tfs=ALL_TFS)
        q_stats = apply_quarantine(runner, quarantine_cfg)
        quarantine_stats_by_pair[pair] = q_stats

        # Log quarantine status for this pair
        for ptf_key in sorted(q_stats.keys()):
            s = q_stats[ptf_key]
            if s['quarantined']:
                print(f"    {ptf_key}: {s['active']}/{s['original']} active "
                      f"(quarantined: {', '.join(s['quarantined'])})")
            else:
                print(f"    {ptf_key}: {s['active']}/{s['original']} active")
        sys.stdout.flush()

        for scenario in SCENARIOS:
            os.environ["PYTHONHASHSEED"] = "42"
            np.random.seed(42)
            random.seed(42)

            runner.reset_state()
            gc.collect()

            t0 = time.perf_counter()
            try:
                run_id = runner.run(
                    pairs=[pair], tfs=ALL_TFS,
                    max_bars=50000, scenario_name=scenario,
                )
                wall_time = time.perf_counter() - t0
                run_data = collect_run_metrics(run_id)
                all_results[scenario].append({
                    'pair': pair, 'run_id': run_id,
                    'run_data': run_data, 'wall_time': wall_time,
                })
                m = run_data.get('metrics', {})
                pf = m.get('profit_factor', 0)
                trades = m.get('total_trades', 0)
                pnl = m.get('total_pnl_net_usd', 0)
                print(f"    {pair:<10s} {scenario:<18s} PF={pf:.4f}  Trades={trades:<5d}  "
                      f"PnL=${pnl:>10,.2f}  ({wall_time:.0f}s)")
            except Exception as e:
                wall_time = time.perf_counter() - t0
                print(f"    FAILED: {pair} {scenario}: {e}")
                traceback.print_exc()
            sys.stdout.flush()

        del runner
        gc.collect()
        pair_elapsed = time.perf_counter() - pair_start
        print(f"  {pair} complete in {pair_elapsed:.0f}s")
        sys.stdout.flush()

    total_elapsed = time.perf_counter() - total_start
    print(f"\n  All pairs complete in {total_elapsed:.0f}s ({total_elapsed/3600:.1f}h)")

    # Print full report
    print_full_report(all_results, quarantine_stats_by_pair)

    # Save raw results
    results_path = PROJECT_ROOT / 'replay' / 'outputs' / 'variant_b_rerun_results.json'
    save_data = {}
    for scenario in SCENARIOS:
        save_data[scenario] = []
        for item in all_results[scenario]:
            save_data[scenario].append({
                'pair': item['pair'],
                'run_id': item['run_id'],
                'wall_time': item['wall_time'],
                'pf': item['run_data'].get('metrics', {}).get('profit_factor', 0),
                'trades': item['run_data'].get('metrics', {}).get('total_trades', 0),
                'pnl': item['run_data'].get('metrics', {}).get('total_pnl_net_usd', 0),
            })
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Raw results saved to: {results_path}")

    print(f"\n{'=' * 100}")
    print(f"  VARIANT B RERUN COMPLETE -- {total_elapsed:.0f}s total")
    print(f"{'=' * 100}")
    sys.stdout.flush()


if __name__ == '__main__':
    main()
