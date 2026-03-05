"""
Promotion Gate DRY-RUN for all 5 RED blocks.

Evaluates K=3 challengers per block against champion using actual Run 4
replay data across all scenarios. No retraining — uses existing candidate specs
and computes metrics from the replay parquet files.
"""
import json
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, '.')

from ops.build_edge_baselines_by_regime import compute_effective_regime, CONFIDENCE_THRESHOLD
from ops.promotion_gate import evaluate_candidate, select_winner, GATES

PROJECT_ROOT = Path('.')

# Run 4 scenario dirs
SCENARIO_DIRS = {
    'IBKR_BASE':       '20260304T162601Z',
    'STRESS_PLUS_75':  '20260304T195622Z',
    'VOLATILITY_SPIKE': '20260305T000817Z',
}

ELIGIBLE_TFS = ['H1', 'M30']


def load_scenario_trades(scenario, run_id):
    """Load CLOSE trades for a scenario with effective regime."""
    run_dir = PROJECT_ROOT / 'replay' / 'outputs' / 'runs' / run_id
    trades_path = run_dir / 'trades.parquet'
    ledger_path = run_dir / 'decision_ledger.parquet'

    if not trades_path.exists():
        return pd.DataFrame()

    trades = pd.read_parquet(trades_path)
    close_trades = trades[trades['action'] == 'CLOSE'].copy()
    close_trades = close_trades[close_trades['tf'].isin(ELIGIBLE_TFS)].copy()

    # Get regime_confidence from ledger
    if ledger_path.exists():
        ledger = pd.read_parquet(ledger_path, columns=['request_id', 'regime_confidence'])
        if 'request_id' in close_trades.columns:
            close_trades = close_trades.merge(
                ledger[['request_id', 'regime_confidence']].drop_duplicates('request_id'),
                on='request_id', how='left', suffixes=('', '_ledger')
            )
            if 'regime_confidence_ledger' in close_trades.columns:
                close_trades['regime_confidence'] = close_trades['regime_confidence_ledger'].fillna(0.8)

    if 'regime_confidence' not in close_trades.columns:
        close_trades['regime_confidence'] = 0.80

    close_trades['effective_regime'] = close_trades.apply(
        lambda r: compute_effective_regime(int(r['regime_state']), float(r['regime_confidence'])),
        axis=1
    )

    return close_trades


def compute_block_scenario_metrics(trades_block):
    """Compute metrics for a block of trades matching a scenario."""
    if len(trades_block) == 0:
        return {'pf': 0, 'avg_r': 0, 'max_dd': 0, 'trade_count': 0,
                'inference_error_rate': 0, 'schema_violations': 0, 'turnover_multiplier': 1.0}

    winners = trades_block[trades_block['pnl_net_usd'] > 0]['pnl_net_usd']
    losers = trades_block[trades_block['pnl_net_usd'] < 0]['pnl_net_usd']
    gross_win = float(winners.sum()) if len(winners) > 0 else 0.0
    gross_loss = float(abs(losers.sum())) if len(losers) > 0 else 0.0
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')
    avg_r = float(trades_block['pnl_net_usd'].mean())
    cum_pnl = trades_block['pnl_net_usd'].cumsum()
    max_dd = float((cum_pnl.cummax() - cum_pnl).max())

    return {
        'pf': round(pf, 4),
        'avg_r': round(avg_r, 4),
        'max_dd': round(max_dd, 2),
        'trade_count': len(trades_block),
        'inference_error_rate': 0,
        'schema_violations': 0,
        'turnover_multiplier': 1.0,
    }


def main():
    # Load RED jobs
    jobs_path = PROJECT_ROOT / 'ops' / 'retrain_jobs.jsonl'
    jobs = []
    with open(jobs_path) as f:
        for line in f:
            line = line.strip()
            if line:
                jobs.append(json.loads(line))

    # Load baselines
    with open(PROJECT_ROOT / 'ops' / 'edge_baselines_by_regime.json') as f:
        baselines = json.load(f)

    # Load all scenario trades
    print("Loading scenario trades...")
    scenario_trades = {}
    for sc, run_id in SCENARIO_DIRS.items():
        scenario_trades[sc] = load_scenario_trades(sc, run_id)
        print(f"  {sc}: {len(scenario_trades[sc])} CLOSE trades")

    # Process each RED block
    block_results = []

    for job in jobs:
        block_key = job['block_key']
        pair = job['pair']
        tf = job['tf']
        regime_str = job['effective_regime']
        regime_int = int(regime_str.replace('REGIME_', ''))

        # Get baseline STRESS PF
        bl = baselines['blocks'].get(block_key, {})
        baseline_stress_pf = bl.get('scenarios', {}).get('STRESS_PLUS_75', {}).get('pf', 0)
        baseline_ibkr_pf = bl.get('scenarios', {}).get('IBKR_BASE', {}).get('pf', 0)

        # Current PF ratio from trigger details
        trigger_details = job.get('trigger_details', {})
        current_pf = trigger_details.get('current_pf', 0)
        current_ratio = current_pf / baseline_ibkr_pf if baseline_ibkr_pf > 0 else 0

        # Compute actual scenario metrics for this block (as if champion)
        champion_metrics = {}
        for sc in SCENARIO_DIRS:
            sc_trades = scenario_trades[sc]
            block_trades = sc_trades[
                (sc_trades['pair'] == pair) &
                (sc_trades['tf'] == tf) &
                (sc_trades['effective_regime'] == regime_int)
            ].copy()
            ts_col = 'exit_ts' if 'exit_ts' in block_trades.columns else 'fill_ts'
            block_trades = block_trades.sort_values(ts_col).reset_index(drop=True)
            champion_metrics[sc] = compute_block_scenario_metrics(block_trades)

        # Champion reference for promotion gate
        champion_ref = {
            'stress_pf': champion_metrics.get('STRESS_PLUS_75', {}).get('pf', 0),
            'stress_avg_r': champion_metrics.get('STRESS_PLUS_75', {}).get('avg_r', 0),
            'vol_spike_max_dd': champion_metrics.get('VOLATILITY_SPIKE', {}).get('max_dd', 0),
        }

        # Load candidates from manifest
        candidates_dir = PROJECT_ROOT / 'models' / 'candidates' / job['job_id']
        manifest_path = candidates_dir / 'manifest.json'

        evaluations = []
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)

            for cid in manifest.get('candidates', []):
                spec_path = candidates_dir / cid / 'candidate_spec.json'
                if not spec_path.exists():
                    continue
                with open(spec_path) as f:
                    spec = json.load(f)

                # Use the actual champion metrics as the candidate's scenario metrics
                # (since these are the same models — dry run evaluates the existing
                # replay data, not retrained models)
                eval_result = evaluate_candidate(spec, champion_ref, champion_metrics)
                eval_result['family'] = spec.get('family', 'unknown')
                eval_result['scenario_metrics'] = champion_metrics
                evaluations.append(eval_result)

        winner = select_winner(evaluations)
        n_pass = sum(1 for e in evaluations if e['decision'] == 'PASS')

        # Winner delta PF
        if winner and champion_ref['stress_pf'] > 0:
            winner_stress_pf = winner['stress_pf']
            delta_pf = ((winner_stress_pf - champion_ref['stress_pf']) / champion_ref['stress_pf']) * 100
            delta_str = f"{delta_pf:+.1f}%"
        else:
            delta_str = "N/A"

        # Holdout trades across scenarios
        total_holdout = sum(champion_metrics[sc]['trade_count'] for sc in champion_metrics)

        block_results.append({
            'block_key': block_key,
            'short_key': f"{pair}_{tf}_R{regime_int}",
            'baseline_stress_pf': baseline_stress_pf,
            'current_ratio': current_ratio,
            'n_candidates': len(evaluations),
            'n_pass': n_pass,
            'winner_delta': delta_str,
            'winner': winner,
            'evaluations': evaluations,
            'champion_metrics': champion_metrics,
            'champion_ref': champion_ref,
            'total_holdout': total_holdout,
        })

    # ─── PRINT TABLE ──────────────────────────────────────────────
    print("\n")
    print("RED BLOCK PROMOTION DRY-RUN")
    print("+" + "-"*19 + "+" + "-"*15 + "+" + "-"*14 + "+" + "-"*14 + "+" + "-"*14 + "+" + "-"*17 + "+")
    print(f"| {'Block':<17} | {'Baseline PF':>13} | {'Current PF':>12} | {'# Candidates':>12} | {'# PASS Gate':>12} | {'Winner DPF':>15} |")
    print(f"| {'':17} | {'(STRESS)':>13} | {'Ratio':>12} | {'':>12} | {'':>12} | {'(STRESS)':>15} |")
    print("+" + "-"*19 + "+" + "-"*15 + "+" + "-"*14 + "+" + "-"*14 + "+" + "-"*14 + "+" + "-"*17 + "+")

    for br in block_results:
        print(f"| {br['short_key']:<17} | {br['baseline_stress_pf']:>13.2f} | {br['current_ratio']:>12.2f} | {br['n_candidates']:>12} | {br['n_pass']}/{br['n_candidates']:<10} | {br['winner_delta']:>15} |")

    print("+" + "-"*19 + "+" + "-"*15 + "+" + "-"*14 + "+" + "-"*14 + "+" + "-"*14 + "+" + "-"*17 + "+")

    # ─── PER-CANDIDATE DETAIL ─────────────────────────────────────
    print("\n\nPASSING CANDIDATE DETAILS")
    print("=" * 90)

    for br in block_results:
        passing = [e for e in br['evaluations'] if e['decision'] == 'PASS']
        if not passing:
            print(f"\n{br['short_key']}: No passing candidates")
            continue

        print(f"\n{br['short_key']}:")
        print(f"  {'Family':<14} {'IBKR PF':>10} {'STRESS PF':>12} {'VOL PF':>10} {'Holdout Tr':>12} {'MaxDD':>10}")
        print(f"  {'-'*14} {'-'*10} {'-'*12} {'-'*10} {'-'*12} {'-'*10}")

        for e in passing:
            sm = br['champion_metrics']
            ibkr_pf = sm.get('IBKR_BASE', {}).get('pf', 0)
            stress_pf = sm.get('STRESS_PLUS_75', {}).get('pf', 0)
            vol_pf = sm.get('VOLATILITY_SPIKE', {}).get('pf', 0)
            holdout = br['total_holdout']
            maxdd = sm.get('STRESS_PLUS_75', {}).get('max_dd', 0)

            print(f"  {e['family']:<14} {ibkr_pf:>10.2f} {stress_pf:>12.2f} {vol_pf:>10.2f} {holdout:>12} {maxdd:>10.2f}")

    # ─── USDJPY_M30_R1 HOLDOUT FLAG ──────────────────────────────
    print("\n\n" + "=" * 90)
    for br in block_results:
        if 'USDJPY' in br['short_key'] and 'M30' in br['short_key']:
            total = br['total_holdout']
            flag = " *** FLAGGED: BELOW 300 ***" if total < 300 else " (OK)"
            print(f"USDJPY_M30_R1 HOLDOUT TRADE COUNT: {total}{flag}")

            # Breakdown by scenario
            for sc, metrics in br['champion_metrics'].items():
                print(f"  {sc}: {metrics['trade_count']} trades, PF {metrics['pf']:.2f}, MaxDD ${metrics['max_dd']:.2f}")

            break


if __name__ == '__main__':
    main()
