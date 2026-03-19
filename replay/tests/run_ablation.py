"""
CHAOS V1.0 -- TARGETED ABLATION REPLAY
Per-block brain quarantine diagnostic.

3 variants x 7 blocks = 21 replay passes, IBKR_BASE only.
Low-memory mode (one pair at a time).

Usage:
    python -u -W ignore replay/tests/run_ablation.py
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

os.environ["PYTHONHASHSEED"] = "42"
np.random.seed(42)
random.seed(42)

PROJECT_ROOT = Path('G:/My Drive/chaos_v1.0')
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'inference'))

# ============================================================
# TARGET BLOCKS
# ============================================================
TARGET_BLOCKS = [
    ('AUDUSD', 'H1'),
    ('AUDUSD', 'M30'),
    ('EURUSD', 'H1'),
    ('USDCAD', 'H1'),
    ('USDCHF', 'H1'),
    ('GBPJPY', 'M30'),
    ('NZDUSD', 'M30'),
]

# ============================================================
# QUARANTINE MAPS
# ============================================================
FLAGGED_BRAINS = {
    'cnn1d_optuna', 'gru_optuna', 'lstm_optuna',
    'nbeats_optuna', 'residual_mlp_optuna', 'tft_optuna',
}

# Variant A: no exclusions
VARIANT_A = {}

# Variant B: remove all 6 flagged from ALL blocks
VARIANT_B = {f"{p}_{t}": FLAGGED_BRAINS for p, t in TARGET_BLOCKS}

# Variant C: per-block quarantine (only where flagged in Run 4)
VARIANT_C = {
    'AUDUSD_H1':  {'gru_optuna', 'nbeats_optuna', 'tft_optuna'},
    'AUDUSD_M30': {'gru_optuna', 'nbeats_optuna', 'tft_optuna'},
    'GBPJPY_M30': {'residual_mlp_optuna'},
    'NZDUSD_M30': {'cnn1d_optuna'},
    'USDCHF_H1':  {'nbeats_optuna'},
    # EURUSD_H1 and USDCAD_H1: no quarantine flags
}

VARIANTS = [
    ('A_baseline', VARIANT_A),
    ('B_minus_6', VARIANT_B),
    ('C_per_block', VARIANT_C),
]


def run_with_runner(runner, pair, tf, exclude_brains, original_models_cache, bar_cap=50000):
    """
    Run replay for a single pair+TF block with brain exclusion.
    Reuses an existing runner (avoids temp dir issues).
    Returns dict with PF, trades, PnL, brains_used, etc.
    """
    pair_tf = f"{pair}_{tf}"
    all_tfs = ['M5', 'M15', 'M30', 'H1']

    # Reset determinism
    os.environ["PYTHONHASHSEED"] = "42"
    np.random.seed(42)
    random.seed(42)

    # Restore original models for this block first, then apply exclusion
    if pair_tf in original_models_cache:
        runner.model_loader.models[pair_tf] = dict(original_models_cache[pair_tf])

    # Apply brain exclusion
    if exclude_brains and pair_tf in runner.model_loader.models:
        original_models = runner.model_loader.models[pair_tf]
        filtered = {k: v for k, v in original_models.items() if k not in exclude_brains}
        excluded_names = set(original_models.keys()) - set(filtered.keys())
        runner.model_loader.models[pair_tf] = filtered
        print(f"      Excluded {len(excluded_names)} brains from {pair_tf}: {sorted(excluded_names)}")
        print(f"      {len(filtered)} brains active for {pair_tf}")
    else:
        total_for_block = len(runner.model_loader.models.get(pair_tf, {}))
        print(f"      All {total_for_block} brains active for {pair_tf}")

    sys.stdout.flush()

    runner.reset_state()
    gc.collect()

    t0 = time.perf_counter()
    try:
        run_id = runner.run(
            pairs=[pair],
            tfs=all_tfs,
            max_bars=bar_cap,
            scenario_name='IBKR_BASE',
        )
        wall_time = time.perf_counter() - t0

        # Collect results
        run_dir = PROJECT_ROOT / 'replay' / 'outputs' / 'runs' / run_id
        metrics = {}
        if (run_dir / 'metrics.json').exists():
            with open(run_dir / 'metrics.json') as f:
                metrics = json.load(f)

        ptf_metrics = {}
        if (run_dir / 'metrics_by_pair_tf.json').exists():
            with open(run_dir / 'metrics_by_pair_tf.json') as f:
                ptf_metrics = json.load(f)

        brain_data = {}
        if (run_dir / 'brain_contributions.json').exists():
            with open(run_dir / 'brain_contributions.json') as f:
                brain_data = json.load(f)

        block_metrics = ptf_metrics.get(pair_tf, {})

        result = {
            'run_id': run_id,
            'pair': pair,
            'tf': tf,
            'pair_tf': pair_tf,
            'wall_time': wall_time,
            'pf': block_metrics.get('profit_factor', 0),
            'trades': block_metrics.get('trades', 0),
            'pnl': block_metrics.get('total_pnl_usd', 0),
            'win_rate': block_metrics.get('win_rate', 0),
            'brains_used': len(runner.model_loader.models.get(pair_tf, {})),
            'brains_excluded': sorted(exclude_brains) if exclude_brains else [],
            'quarantine_candidates': brain_data.get('quarantine_candidates', []),
            'portfolio_pf': metrics.get('profit_factor', 0),
            'portfolio_trades': metrics.get('total_trades', 0),
            'portfolio_pnl': metrics.get('total_pnl_net_usd', 0),
        }

    except Exception as e:
        wall_time = time.perf_counter() - t0
        print(f"      FAILED: {e}")
        traceback.print_exc()
        result = {
            'pair_tf': pair_tf, 'pf': 0, 'trades': 0, 'pnl': 0,
            'error': str(e), 'wall_time': wall_time,
            'brains_excluded': sorted(exclude_brains) if exclude_brains else [],
        }

    # Restore original models after run
    if pair_tf in original_models_cache:
        runner.model_loader.models[pair_tf] = dict(original_models_cache[pair_tf])

    return result


def print_ablation_table(results):
    """Print the ablation comparison table."""
    print("\n")
    print("ABLATION RESULTS -- IBKR_BASE")
    print("=" * 95)
    print(f"{'Block':<16s} | {'Variant A (all)':<18s} | {'Variant B (-6)':<18s} | {'Variant C (per-block)':<22s}")
    print(f"{'':16s} | {'PF / Trades':<18s} | {'PF / Trades':<18s} | {'PF / Trades':<22s}")
    print("-" * 95)

    for pair, tf in TARGET_BLOCKS:
        pair_tf = f"{pair}_{tf}"
        a = results.get(('A_baseline', pair_tf), {})
        b = results.get(('B_minus_6', pair_tf), {})
        c = results.get(('C_per_block', pair_tf), {})

        a_str = f"{a.get('pf', 0):.2f} / {a.get('trades', 0)}"
        b_str = f"{b.get('pf', 0):.2f} / {b.get('trades', 0)}"
        c_str = f"{c.get('pf', 0):.2f} / {c.get('trades', 0)}"

        print(f"{pair_tf:<16s} | {a_str:<18s} | {b_str:<18s} | {c_str:<22s}")

    print("=" * 95)


def print_marginal_contributions(results):
    """Print per-brain marginal contribution analysis."""
    print("\nPER-BRAIN MARGINAL CONTRIBUTION (per block)")
    print("=" * 80)
    print(f"{'Brain':<22s} | {'Block':<14s} | {'dPF':>8s} | {'dTrades':>8s} | {'Recommend':<12s}")
    print("-" * 80)

    recommendations = {}

    for pair, tf in TARGET_BLOCKS:
        pair_tf = f"{pair}_{tf}"
        a = results.get(('A_baseline', pair_tf), {})
        c = results.get(('C_per_block', pair_tf), {})

        a_pf = a.get('pf', 0)
        a_trades = a.get('trades', 0)
        c_pf = c.get('pf', 0)
        c_trades = c.get('trades', 0)

        excluded_in_c = VARIANT_C.get(pair_tf, set())
        if not excluded_in_c:
            continue

        # Delta from removing the flagged brains
        d_pf = c_pf - a_pf
        d_trades = c_trades - a_trades

        # Recommendation logic
        trade_recovery = d_trades > 0 and (a_trades == 0 or d_trades / max(a_trades, 1) > 0.20)
        pf_ok = a_pf == 0 or c_pf >= a_pf * 0.95 or c_pf > 1.0

        if trade_recovery and pf_ok:
            rec = 'QUARANTINE'
        elif d_pf < -0.5 or (d_trades < -a_trades * 0.1 and a_trades > 0):
            rec = 'KEEP'
        else:
            rec = 'REVIEW'

        for brain in sorted(excluded_in_c):
            print(f"{brain:<22s} | {pair_tf:<14s} | {d_pf:>+7.2f} | {d_trades:>+7d} | {rec:<12s}")
            if pair_tf not in recommendations:
                recommendations[pair_tf] = {'brains': set(), 'action': rec}
            if rec == 'QUARANTINE':
                recommendations[pair_tf]['brains'].add(brain)
                recommendations[pair_tf]['action'] = 'QUARANTINE'

    print("=" * 80)

    # Recommended quarantine map
    print("\nRECOMMENDED PER-BLOCK QUARANTINE MAP")
    print("=" * 50)
    for pair, tf in TARGET_BLOCKS:
        pair_tf = f"{pair}_{tf}"
        excluded = VARIANT_C.get(pair_tf, set())
        rec = recommendations.get(pair_tf, {})
        if rec.get('action') == 'QUARANTINE' and rec.get('brains'):
            brains_str = ', '.join(sorted(rec['brains']))
            print(f"  {pair_tf:<14s}: quarantine {{{brains_str}}}")
        elif not excluded:
            print(f"  {pair_tf:<14s}: no quarantine needed")
        else:
            print(f"  {pair_tf:<14s}: keep all (review)")
    print("=" * 50)

    return recommendations


def main():
    print("=" * 95)
    print("  CHAOS V1.0 -- TARGETED ABLATION REPLAY")
    print("  7 blocks x 3 variants = 21 passes | IBKR_BASE | 50K bar cap")
    print("=" * 95)
    print()
    sys.stdout.flush()

    from replay.runners.run_replay import ReplayRunner
    from collections import defaultdict

    total_start = time.perf_counter()
    results = {}  # (variant_name, pair_tf) -> result dict

    # Group blocks by pair to minimize model loading
    pair_blocks = defaultdict(list)
    for pair, tf in TARGET_BLOCKS:
        pair_blocks[pair].append(tf)

    pass_num = 0
    total_passes = len(TARGET_BLOCKS) * len(VARIANTS)
    all_tfs = ['M5', 'M15', 'M30', 'H1']

    # Process one pair at a time, run all variants x blocks for that pair
    for pair_idx, (pair, tfs_for_pair) in enumerate(pair_blocks.items()):
        print(f"\n{'=' * 80}")
        print(f"  LOADING PAIR {pair_idx+1}/{len(pair_blocks)}: {pair}")
        print(f"  Blocks: {[f'{pair}_{t}' for t in tfs_for_pair]}")
        print(f"{'=' * 80}")
        sys.stdout.flush()

        # Create ONE runner for this pair
        runner = ReplayRunner(pairs=[pair], tfs=all_tfs)

        # Cache original models so we can restore after exclusion
        original_models_cache = {}
        for tf in tfs_for_pair:
            pair_tf = f"{pair}_{tf}"
            if pair_tf in runner.model_loader.models:
                original_models_cache[pair_tf] = dict(runner.model_loader.models[pair_tf])

        print(f"  Models loaded. Running variants...")
        sys.stdout.flush()

        # Run all variants x blocks for this pair
        for tf in tfs_for_pair:
            pair_tf = f"{pair}_{tf}"
            for variant_name, quarantine_map in VARIANTS:
                pass_num += 1
                exclude = quarantine_map.get(pair_tf, set())

                print(f"\n  [{pass_num}/{total_passes}] {pair_tf} -- {variant_name} "
                      f"(exclude: {sorted(exclude) if exclude else 'NONE'})")
                sys.stdout.flush()

                result = run_with_runner(runner, pair, tf, exclude, original_models_cache)
                results[(variant_name, pair_tf)] = result

                pf = result.get('pf', 0)
                trades = result.get('trades', 0)
                pnl = result.get('pnl', 0)
                wt = result.get('wall_time', 0)
                print(f"    -> PF={pf:.4f}  Trades={trades}  PnL=${pnl:.2f}  ({wt:.0f}s)")
                sys.stdout.flush()

        # Free runner before next pair
        del runner
        gc.collect()
        print(f"\n  {pair} complete, runner freed.")

    total_elapsed = time.perf_counter() - total_start
    print(f"\n\n  All {total_passes} passes complete in {total_elapsed:.0f}s ({total_elapsed/3600:.1f}h)")

    # Print results
    print_ablation_table(results)
    recommendations = print_marginal_contributions(results)

    # Save results
    save_path = PROJECT_ROOT / 'replay' / 'outputs' / 'ablation_results.json'
    save_data = {
        'variants': {},
        'quarantine_maps': {
            'A_baseline': {},
            'B_minus_6': {k: sorted(v) for k, v in VARIANT_B.items()},
            'C_per_block': {k: sorted(v) for k, v in VARIANT_C.items()},
        },
        'recommendations': {
            k: {'brains': sorted(v.get('brains', set())), 'action': v.get('action', '')}
            for k, v in (recommendations or {}).items()
        },
    }
    for (variant, pair_tf), r in results.items():
        if variant not in save_data['variants']:
            save_data['variants'][variant] = {}
        # Filter to serializable data
        save_data['variants'][variant][pair_tf] = {
            k: v for k, v in r.items()
            if isinstance(v, (str, int, float, list, bool, type(None)))
        }

    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to: {save_path}")

    print(f"\n{'=' * 95}")
    print(f"  ABLATION COMPLETE -- {total_elapsed:.0f}s total")
    print(f"{'=' * 95}")
    sys.stdout.flush()


if __name__ == '__main__':
    main()
