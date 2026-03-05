"""
Retrain Trigger Engine

Monitors live/replay decision ledger + trades against edge baselines.
Produces retrain_jobs.jsonl with ORANGE (watch) and RED (retrain) alerts.

Rules:
  ORANGE: pf < 0.75*baseline_pf OR avg_r < 0.70*baseline_avg_r OR spread_ratio_mean > 2.0*baseline
  RED: (pf < 0.60*baseline_pf OR avg_r < 0.55*baseline_avg_r) AND persists >=5 days OR >=150 trades
  Sample floor: >=80 trades in slow window for RED

Only for TF=H1/M30 and regimes 0/1/2. Skip REGIME_3.
"""
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Optional
import hashlib

logger = logging.getLogger('retrain_trigger')

CONFIDENCE_THRESHOLD = 0.60
ELIGIBLE_TFS = ['H1', 'M30']
ELIGIBLE_REGIMES = [0, 1, 2]

# Trigger thresholds
ORANGE_PF_RATIO = 0.75
ORANGE_AVG_R_RATIO = 0.70
ORANGE_SPREAD_RATIO = 2.0
RED_PF_RATIO = 0.60
RED_AVG_R_RATIO = 0.55
RED_PERSIST_DAYS = 5
RED_PERSIST_TRADES = 150
RED_SAMPLE_FLOOR = 80

# Windows
W_FAST = 50    # Fast window (recent trades)
W_SLOW = 150   # Slow window (persistence check)


def compute_effective_regime(regime_state: int, regime_confidence: float) -> int:
    if regime_confidence < CONFIDENCE_THRESHOLD:
        return 2
    return int(regime_state)


def evaluate_block(
    trades: pd.DataFrame,
    baseline: dict,
    scenario: str = 'IBKR_BASE'
) -> dict:
    """
    Evaluate a (pair, tf, effective_regime) block against its baseline.

    Returns trigger result with level (GREEN/ORANGE/RED) and details.
    """
    bl = baseline.get('scenarios', {}).get(scenario)
    if bl is None:
        return {'level': 'GREEN', 'reason': 'no_baseline', 'details': {}}

    baseline_pf = bl.get('pf', 0)
    baseline_avg_r = bl.get('avg_r', 0)
    baseline_spread = bl.get('spread_mean', 0)

    if len(trades) == 0:
        return {'level': 'GREEN', 'reason': 'no_trades', 'details': {}}

    # Sort by exit time
    ts_col = 'exit_ts' if 'exit_ts' in trades.columns else 'fill_ts'
    trades = trades.sort_values(ts_col).reset_index(drop=True)

    # Fast window metrics
    fast_trades = trades.tail(min(W_FAST, len(trades)))
    winners = fast_trades[fast_trades['pnl_net_usd'] > 0]['pnl_net_usd']
    losers = fast_trades[fast_trades['pnl_net_usd'] < 0]['pnl_net_usd']
    gross_win = float(winners.sum()) if len(winners) > 0 else 0.0
    gross_loss = float(abs(losers.sum())) if len(losers) > 0 else 0.0
    current_pf = gross_win / gross_loss if gross_loss > 0 else float('inf')
    current_avg_r = float(fast_trades['pnl_net_usd'].mean())
    current_spread = float(fast_trades['spread_cost_pips'].mean()) if 'spread_cost_pips' in fast_trades.columns else 0.0

    details = {
        'current_pf': round(current_pf, 4),
        'baseline_pf': baseline_pf,
        'current_avg_r': round(current_avg_r, 4),
        'baseline_avg_r': baseline_avg_r,
        'current_spread': round(current_spread, 4),
        'baseline_spread': baseline_spread,
        'fast_window_trades': len(fast_trades),
        'total_trades': len(trades),
    }

    # Check ORANGE conditions
    orange_reasons = []
    if baseline_pf > 0 and current_pf < ORANGE_PF_RATIO * baseline_pf:
        orange_reasons.append(f'pf_decay: {current_pf:.2f} < {ORANGE_PF_RATIO}*{baseline_pf:.2f}')
    if baseline_avg_r != 0 and current_avg_r < ORANGE_AVG_R_RATIO * baseline_avg_r:
        orange_reasons.append(f'avg_r_decay: {current_avg_r:.4f} < {ORANGE_AVG_R_RATIO}*{baseline_avg_r:.4f}')
    if baseline_spread > 0 and current_spread > ORANGE_SPREAD_RATIO * baseline_spread:
        orange_reasons.append(f'spread_spike: {current_spread:.4f} > {ORANGE_SPREAD_RATIO}*{baseline_spread:.4f}')

    if not orange_reasons:
        return {'level': 'GREEN', 'reason': 'within_bands', 'details': details}

    # Check RED conditions (need slow window + persistence)
    slow_trades = trades.tail(min(W_SLOW, len(trades)))
    slow_winners = slow_trades[slow_trades['pnl_net_usd'] > 0]['pnl_net_usd']
    slow_losers = slow_trades[slow_trades['pnl_net_usd'] < 0]['pnl_net_usd']
    slow_gross_win = float(slow_winners.sum()) if len(slow_winners) > 0 else 0.0
    slow_gross_loss = float(abs(slow_losers.sum())) if len(slow_losers) > 0 else 0.0
    slow_pf = slow_gross_win / slow_gross_loss if slow_gross_loss > 0 else float('inf')
    slow_avg_r = float(slow_trades['pnl_net_usd'].mean())

    red_conditions_met = False
    red_reasons = []

    if baseline_pf > 0 and slow_pf < RED_PF_RATIO * baseline_pf:
        red_reasons.append(f'slow_pf_critical: {slow_pf:.2f} < {RED_PF_RATIO}*{baseline_pf:.2f}')
        red_conditions_met = True
    if baseline_avg_r != 0 and slow_avg_r < RED_AVG_R_RATIO * baseline_avg_r:
        red_reasons.append(f'slow_avg_r_critical: {slow_avg_r:.4f} < {RED_AVG_R_RATIO}*{baseline_avg_r:.4f}')
        red_conditions_met = True

    # Persistence check
    if red_conditions_met and len(slow_trades) >= RED_SAMPLE_FLOOR:
        ts_col_val = 'exit_ts' if 'exit_ts' in slow_trades.columns else 'fill_ts'
        if len(slow_trades) >= RED_PERSIST_TRADES:
            details['persistence'] = f'>={RED_PERSIST_TRADES} trades'
            details['slow_pf'] = round(slow_pf, 4)
            details['slow_avg_r'] = round(slow_avg_r, 4)
            return {
                'level': 'RED',
                'reason': '|'.join(red_reasons),
                'details': details
            }

        # Check time persistence
        first_ts = pd.to_datetime(slow_trades[ts_col_val].iloc[0])
        last_ts = pd.to_datetime(slow_trades[ts_col_val].iloc[-1])
        duration_days = (last_ts - first_ts).total_seconds() / 86400
        if duration_days >= RED_PERSIST_DAYS:
            details['persistence'] = f'{duration_days:.1f} days >= {RED_PERSIST_DAYS}'
            details['slow_pf'] = round(slow_pf, 4)
            details['slow_avg_r'] = round(slow_avg_r, 4)
            return {
                'level': 'RED',
                'reason': '|'.join(red_reasons),
                'details': details
            }

    # Only ORANGE
    return {
        'level': 'ORANGE',
        'reason': '|'.join(orange_reasons),
        'details': details
    }


def run_triggers(
    trades_path: Path,
    ledger_path: Path,
    baselines_path: Path,
    output_path: Path,
    scenario: str = 'IBKR_BASE'
) -> List[dict]:
    """
    Run trigger evaluation across all blocks.

    Returns list of trigger results and writes RED jobs to retrain_jobs.jsonl.
    """
    with open(baselines_path) as f:
        baselines = json.load(f)

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
                close_trades['regime_confidence'] = close_trades['regime_confidence_ledger'].fillna(
                    close_trades.get('regime_confidence', 0.8)
                )

    if 'regime_confidence' not in close_trades.columns:
        close_trades['regime_confidence'] = 0.80

    close_trades['effective_regime'] = close_trades.apply(
        lambda r: compute_effective_regime(int(r['regime_state']), float(r['regime_confidence'])),
        axis=1
    )

    results = []
    jobs = []

    for block_key, baseline in baselines.get('blocks', {}).items():
        pair = baseline['pair']
        tf = baseline['tf']
        eff_regime = baseline['effective_regime']
        regime_int = int(eff_regime.replace('REGIME_', ''))

        block_trades = close_trades[
            (close_trades['pair'] == pair) &
            (close_trades['tf'] == tf) &
            (close_trades['effective_regime'] == regime_int)
        ].copy()

        result = evaluate_block(block_trades, baseline, scenario)
        result['block_key'] = block_key
        result['pair'] = pair
        result['tf'] = tf
        result['effective_regime'] = eff_regime
        results.append(result)

        if result['level'] == 'RED':
            job_id = hashlib.sha256(
                f"{block_key}|{datetime.now(timezone.utc).isoformat()}".encode()
            ).hexdigest()[:12]

            job = {
                'job_id': job_id,
                'status': 'QUEUED',
                'created_at': datetime.now(timezone.utc).isoformat(),
                'block_key': block_key,
                'pair': pair,
                'tf': tf,
                'effective_regime': eff_regime,
                'trigger_level': 'RED',
                'trigger_reason': result['reason'],
                'trigger_details': result['details'],
                'baseline_scenario': scenario,
            }
            jobs.append(job)

    # Write jobs
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for job in jobs:
            f.write(json.dumps(job, default=str) + '\n')

    return results


def main():
    project_root = Path('.')
    runs_dir = project_root / 'replay' / 'outputs' / 'runs'
    baselines_path = project_root / 'ops' / 'edge_baselines_by_regime.json'
    output_path = project_root / 'ops' / 'retrain_jobs.jsonl'

    # Use IBKR_BASE run
    run_dir = runs_dir / '20260304T162601Z'
    trades_path = run_dir / 'trades.parquet'
    ledger_path = run_dir / 'decision_ledger.parquet'

    results = run_triggers(trades_path, ledger_path, baselines_path, output_path)

    # Summary
    counts = {'GREEN': 0, 'ORANGE': 0, 'RED': 0}
    for r in results:
        counts[r['level']] = counts.get(r['level'], 0) + 1

    print(f"\nRETRAIN TRIGGER RESULTS")
    print(f"  GREEN:  {counts['GREEN']}")
    print(f"  ORANGE: {counts['ORANGE']}")
    print(f"  RED:    {counts['RED']}")
    print(f"\nSaved: {output_path}")

    return results


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
