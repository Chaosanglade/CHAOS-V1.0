"""
Walk-Forward Validation for CHAOS V1.0 Post-Quarantine Ensemble

Tests temporal stability by running the frozen ensemble across
rolling 3-month test windows. Models are NOT retrained.

20 windows, 3 scenarios each (IBKR_BASE, STRESS_PLUS_75, VOLATILITY_SPIKE).
Low-memory mode: one pair at a time per window.

Usage:
    python -u -W ignore replay/tests/run_walk_forward.py
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
from pathlib import Path
from datetime import datetime

os.environ["PYTHONHASHSEED"] = "42"
np.random.seed(42)
random.seed(42)

PROJECT_ROOT = Path('G:/My Drive/chaos_v1.0')
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'inference'))

ALL_PAIRS = [
    'AUDUSD', 'EURUSD', 'EURJPY', 'GBPJPY', 'GBPUSD',
    'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY',
]
ALL_TFS = ['M5', 'M15', 'M30', 'H1']
WFO_SCENARIOS = ['IBKR_BASE', 'STRESS_PLUS_75', 'VOLATILITY_SPIKE']

# 20 rolling windows: 12-month train (unused — models frozen), 3-month test
WFO_WINDOWS = []
for i in range(20):
    # Test start: 2024-01 + i months
    test_start_year = 2024 + (i // 12)
    test_start_month = 1 + (i % 12)
    # Test end: test_start + 2 months (3-month window)
    test_end_month = test_start_month + 2
    test_end_year = test_start_year
    if test_end_month > 12:
        test_end_month -= 12
        test_end_year += 1

    # Format dates
    ts = f"{test_start_year}-{test_start_month:02d}-01"
    # End date: last day of end month (use 28 for simplicity, replay handles it)
    if test_end_month in (1, 3, 5, 7, 8, 10, 12):
        end_day = 31
    elif test_end_month == 2:
        end_day = 28
    else:
        end_day = 30
    te = f"{test_end_year}-{test_end_month:02d}-{end_day}"

    WFO_WINDOWS.append({
        'name': f"W{i+1}",
        'test_start': ts,
        'test_end': te,
        'label': f"{test_start_year}-{test_start_month:02d} to {test_end_month:02d}",
    })


# ============================================================
# QUARANTINE
# ============================================================
def load_quarantine_config():
    with open(PROJECT_ROOT / 'replay' / 'config' / 'brain_quarantine.json') as f:
        return json.load(f)


def is_quarantined(quarantine_cfg, brain_name, pair, tf):
    block_key = f"{pair}_{tf}"
    if brain_name in quarantine_cfg.get('global_quarantine', []):
        return True
    cond = quarantine_cfg.get('conditional_quarantine', {}).get(brain_name)
    if cond and cond.get('policy') == 'quarantined_globally_except':
        return block_key not in cond.get('exceptions', [])
    return False


def apply_quarantine(runner, quarantine_cfg):
    for pair_tf, models in list(runner.model_loader.models.items()):
        parts = pair_tf.split('_')
        pair, tf = parts[0], parts[1]
        quarantined = [b for b in models if is_quarantined(quarantine_cfg, b, pair, tf)]
        for b in quarantined:
            del models[b]


# ============================================================
# SINGLE WINDOW RUNNER
# ============================================================
def run_window(window, quarantine_cfg):
    """Run all 9 pairs x 3 scenarios for one WFO window. Returns per-scenario aggregated metrics."""
    from replay.runners.run_replay import ReplayRunner

    window_results = {}

    for scenario in WFO_SCENARIOS:
        total_trades = 0
        total_pnl = 0
        total_gw = 0
        total_gl = 0
        max_dd = 0

        for pair in ALL_PAIRS:
            os.environ["PYTHONHASHSEED"] = "42"
            np.random.seed(42)
            random.seed(42)

            runner = ReplayRunner(pairs=[pair], tfs=ALL_TFS)
            apply_quarantine(runner, quarantine_cfg)
            runner.reset_state()
            gc.collect()

            # Override universe dates for this window
            runner.universe['date_start'] = window['test_start']
            runner.universe['date_end'] = window['test_end']

            try:
                run_id = runner.run(
                    pairs=[pair], tfs=ALL_TFS,
                    max_bars=0,  # No cap — 3-month window is small
                    scenario_name=scenario,
                )
                run_dir = PROJECT_ROOT / 'replay' / 'outputs' / 'runs' / run_id
                metrics = {}
                if (run_dir / 'metrics.json').exists():
                    with open(run_dir / 'metrics.json') as f:
                        metrics = json.load(f)

                total_trades += metrics.get('total_trades', 0)
                total_pnl += metrics.get('total_pnl_net_usd', 0)
                total_gw += metrics.get('gross_profit_usd', 0)
                total_gl += abs(metrics.get('gross_loss_usd', 0))
                max_dd = max(max_dd, metrics.get('max_drawdown_pct', 0))

            except Exception as e:
                print(f"      ERROR: {pair} {scenario}: {e}")
                sys.stdout.flush()

            del runner
            gc.collect()

        pf = total_gw / total_gl if total_gl > 0 else 0
        # Sharpe proxy: PnL / window_months / sqrt(max_dd+0.01)
        sharpe_proxy = total_pnl / 3.0 / max(np.sqrt(max_dd + 0.01), 0.1) if total_pnl > 0 else 0

        window_results[scenario] = {
            'pf': round(pf, 4),
            'trades': total_trades,
            'pnl': round(total_pnl, 2),
            'max_dd': round(max_dd, 4),
            'sharpe_proxy': round(sharpe_proxy, 2),
            'gross_win': round(total_gw, 2),
            'gross_loss': round(total_gl, 2),
        }

    # Gate evaluation
    ibkr = window_results.get('IBKR_BASE', {})
    stress = window_results.get('STRESS_PLUS_75', {})
    vol = window_results.get('VOLATILITY_SPIKE', {})

    gate_pass = (
        ibkr.get('pf', 0) >= 1.0 and
        stress.get('pf', 0) >= 1.0 and
        ibkr.get('max_dd', 100) <= 5.0
    )

    window_results['gate'] = 'PASS' if gate_pass else 'FAIL'

    return window_results


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 100)
    print("  WALK-FORWARD VALIDATION -- Post-Quarantine Ensemble v1.0")
    print("  20 windows x 9 pairs x 3 scenarios | 14-brain ensemble | frozen models")
    print("=" * 100)
    print()
    sys.stdout.flush()

    quarantine_cfg = load_quarantine_config()
    print(f"  Quarantine: {quarantine_cfg['quarantine_version']}")
    print(f"  Windows: {len(WFO_WINDOWS)} (3-month test, 1-month roll)")
    print()
    sys.stdout.flush()

    total_start = time.perf_counter()
    all_window_results = {}

    for wi, window in enumerate(WFO_WINDOWS):
        w_start = time.perf_counter()
        print(f"  [{wi+1}/20] {window['name']}: {window['test_start']} to {window['test_end']}")
        sys.stdout.flush()

        try:
            w_results = run_window(window, quarantine_cfg)
            all_window_results[window['name']] = {
                'window': window,
                'results': w_results,
            }

            ibkr = w_results.get('IBKR_BASE', {})
            stress = w_results.get('STRESS_PLUS_75', {})
            vol = w_results.get('VOLATILITY_SPIKE', {})
            gate = w_results.get('gate', '?')

            w_elapsed = time.perf_counter() - w_start
            print(f"    PF(IBKR)={ibkr.get('pf', 0):.2f}  PF(STRESS)={stress.get('pf', 0):.2f}  "
                  f"PF(VOL)={vol.get('pf', 0):.2f}  Trades={ibkr.get('trades', 0)}  "
                  f"MaxDD={ibkr.get('max_dd', 0):.2f}%  Gate={gate}  ({w_elapsed:.0f}s)")

        except Exception as e:
            w_elapsed = time.perf_counter() - w_start
            print(f"    FAILED: {e} ({w_elapsed:.0f}s)")
            traceback.print_exc()
            all_window_results[window['name']] = {
                'window': window,
                'results': {'gate': 'FAIL', 'error': str(e)},
            }

        sys.stdout.flush()

    total_elapsed = time.perf_counter() - total_start

    # ============================================================
    # PRINT RESULTS TABLE
    # ============================================================
    print("\n\n" + "=" * 105)
    print("  WALK-FORWARD VALIDATION -- Post-Quarantine Ensemble v1.0")
    print("=" * 105)
    print(f"  {'Window':<8s} {'Test Period':<18s} {'PF(IBKR)':>9s} {'PF(STRESS)':>11s} "
          f"{'PF(VOL)':>8s} {'Trades':>7s} {'Sharpe':>7s} {'MaxDD':>7s} {'Gate':>5s}")
    print("-" * 105)

    pf_ibkr_list = []
    pf_stress_list = []
    pf_vol_list = []
    pass_count = 0
    pf_above_2 = 0
    worst_window = None
    worst_pf = 999
    best_window = None
    best_pf = 0

    for wi, window in enumerate(WFO_WINDOWS):
        wname = window['name']
        wr = all_window_results.get(wname, {}).get('results', {})
        ibkr = wr.get('IBKR_BASE', {})
        stress = wr.get('STRESS_PLUS_75', {})
        vol = wr.get('VOLATILITY_SPIKE', {})
        gate = wr.get('gate', '?')

        pf_i = ibkr.get('pf', 0)
        pf_s = stress.get('pf', 0)
        pf_v = vol.get('pf', 0)
        trades = ibkr.get('trades', 0)
        sharpe = ibkr.get('sharpe_proxy', 0)
        mdd = ibkr.get('max_dd', 0)

        print(f"  {wname:<8s} {window['test_start']} to {window['test_end'][-5:]}"
              f" {pf_i:>9.2f} {pf_s:>11.2f} {pf_v:>8.2f} {trades:>7d} "
              f"{sharpe:>7.1f} {mdd:>6.2f}% {gate:>5s}")

        pf_ibkr_list.append(pf_i)
        pf_stress_list.append(pf_s)
        pf_vol_list.append(pf_v)

        if gate == 'PASS':
            pass_count += 1
        if pf_i >= 2.0:
            pf_above_2 += 1
        if pf_i < worst_pf:
            worst_pf = pf_i
            worst_window = wname
        if pf_i > best_pf:
            best_pf = pf_i
            best_window = wname

    print("=" * 105)

    # Aggregates
    mean_pf_ibkr = np.mean(pf_ibkr_list) if pf_ibkr_list else 0
    mean_pf_stress = np.mean(pf_stress_list) if pf_stress_list else 0
    mean_pf_vol = np.mean(pf_vol_list) if pf_vol_list else 0
    consistency = (pf_above_2 / len(WFO_WINDOWS) * 100) if WFO_WINDOWS else 0

    print(f"\n  AGGREGATE:")
    print(f"    Mean PF (IBKR):     {mean_pf_ibkr:.2f}")
    print(f"    Mean PF (STRESS):   {mean_pf_stress:.2f}")
    print(f"    Mean PF (VOL):      {mean_pf_vol:.2f}")
    print(f"    Worst Window:       {worst_window} -- PF {worst_pf:.2f}")
    print(f"    Best Window:        {best_window} -- PF {best_pf:.2f}")
    print(f"    Windows PASS:       {pass_count}/20")
    print(f"    Consistency Score:  {consistency:.0f}% (windows where PF > 2.0)")

    overall_gate = 'PASS' if pass_count >= 16 and mean_pf_ibkr >= 1.5 else 'FAIL'
    print(f"\n  GATE VERDICT: {overall_gate}")
    print("=" * 105)
    sys.stdout.flush()

    # Save results
    save_path = PROJECT_ROOT / 'replay' / 'outputs' / 'walk_forward_results.json'
    save_data = {
        'version': quarantine_cfg['quarantine_version'],
        'windows': {},
        'aggregate': {
            'mean_pf_ibkr': round(mean_pf_ibkr, 4),
            'mean_pf_stress': round(mean_pf_stress, 4),
            'mean_pf_vol': round(mean_pf_vol, 4),
            'worst_window': worst_window,
            'worst_pf': round(worst_pf, 4),
            'best_window': best_window,
            'best_pf': round(best_pf, 4),
            'windows_pass': pass_count,
            'windows_total': 20,
            'consistency_pct': round(consistency, 1),
            'gate_verdict': overall_gate,
        },
        'elapsed_seconds': round(total_elapsed, 0),
    }
    for wname, wdata in all_window_results.items():
        save_data['windows'][wname] = wdata

    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Results saved to: {save_path}")

    print(f"\n  Total elapsed: {total_elapsed:.0f}s ({total_elapsed/3600:.1f}h)")
    print(f"\n{'=' * 105}")
    print(f"  WFO COMPLETE")
    print(f"{'=' * 105}")
    sys.stdout.flush()


if __name__ == '__main__':
    main()
