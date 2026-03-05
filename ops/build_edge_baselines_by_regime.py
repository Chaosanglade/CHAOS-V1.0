"""
Edge Baselines by Regime Builder

Reads Run 4 replay outputs and produces edge_baselines_by_regime.json
with baselines per (pair, TF, effective_regime) for IBKR_BASE and STRESS_PLUS_75.

Effective regime rule:
  - If regime_confidence < 0.60 => effective_regime := REGIME_2
  - REGIME_3 => skip (forces FLAT, no trades to baseline)
"""
import hashlib
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger('build_edge_baselines')

# Run 4 directories
RUN4_DIRS = {
    'IBKR_BASE':       '20260304T162601Z',
    'STRESS_PLUS_75':  '20260304T195622Z',
}

CONFIDENCE_THRESHOLD = 0.60
ELIGIBLE_TFS = ['H1', 'M30']
ELIGIBLE_REGIMES = [0, 1, 2]  # Skip REGIME_3
BASELINE_VERSION = '1.0.0'


def compute_file_hash(path: Path) -> str:
    """SHA-256 hash of a file for reproducibility tracking."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()[:16]


def compute_effective_regime(regime_state: int, regime_confidence: float) -> int:
    """Apply effective regime rule: low confidence degrades to REGIME_2."""
    if regime_confidence < CONFIDENCE_THRESHOLD:
        return 2
    return int(regime_state)


def compute_block_metrics(trades_block: pd.DataFrame) -> dict:
    """Compute PF, avg_r, win_rate, max_dd, trade_count for a block of CLOSE trades."""
    if len(trades_block) == 0:
        return None

    winners = trades_block[trades_block['pnl_net_usd'] > 0]['pnl_net_usd']
    losers = trades_block[trades_block['pnl_net_usd'] < 0]['pnl_net_usd']

    gross_win = float(winners.sum()) if len(winners) > 0 else 0.0
    gross_loss = float(abs(losers.sum())) if len(losers) > 0 else 0.0
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')

    avg_r = float(trades_block['pnl_net_usd'].mean())
    win_rate = len(winners) / len(trades_block)

    cum_pnl = trades_block['pnl_net_usd'].cumsum()
    max_dd = float((cum_pnl.cummax() - cum_pnl).max())

    avg_winner = float(winners.mean()) if len(winners) > 0 else 0.0
    avg_loser = float(abs(losers.mean())) if len(losers) > 0 else 0.0
    wl_ratio = avg_winner / avg_loser if avg_loser > 0 else float('inf')

    spread_mean = float(trades_block['spread_cost_pips'].mean()) if 'spread_cost_pips' in trades_block.columns else 0.0

    return {
        'pf': round(pf, 4),
        'avg_r': round(avg_r, 4),
        'win_rate': round(win_rate, 4),
        'max_dd': round(max_dd, 2),
        'wl_ratio': round(wl_ratio, 4),
        'avg_winner': round(avg_winner, 4),
        'avg_loser': round(avg_loser, 4),
        'spread_mean': round(spread_mean, 4),
        'trade_count': len(trades_block),
        'gross_win': round(gross_win, 2),
        'gross_loss': round(gross_loss, 2),
    }


def build_baselines(project_root: Path = None) -> dict:
    """
    Build edge baselines per (pair, TF, effective_regime) from Run 4 data.

    Returns the full baselines dict ready for JSON serialization.
    """
    if project_root is None:
        project_root = Path('.')

    runs_dir = project_root / 'replay' / 'outputs' / 'runs'
    regime_policy_path = project_root / 'regime' / 'regime_policy.json'

    # Compute hashes for reproducibility
    regime_hash = compute_file_hash(regime_policy_path) if regime_policy_path.exists() else 'N/A'

    baselines = {
        'metadata': {
            'baseline_version': BASELINE_VERSION,
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'source_runs': dict(RUN4_DIRS),
            'confidence_threshold': CONFIDENCE_THRESHOLD,
            'eligible_tfs': ELIGIBLE_TFS,
            'eligible_regimes': ELIGIBLE_REGIMES,
            'regime_policy_hash': regime_hash,
        },
        'blocks': {}
    }

    for scenario, run_id in sorted(RUN4_DIRS.items()):
        run_dir = runs_dir / run_id
        trades_path = run_dir / 'trades.parquet'
        ledger_path = run_dir / 'decision_ledger.parquet'

        if not trades_path.exists():
            logger.warning(f"Missing trades.parquet for {scenario} at {run_dir}")
            continue

        trades = pd.read_parquet(trades_path)
        close_trades = trades[trades['action'] == 'CLOSE'].copy()
        close_trades = close_trades[close_trades['tf'].isin(ELIGIBLE_TFS)].copy()

        # Get regime_confidence from ledger if available
        if ledger_path.exists():
            ledger = pd.read_parquet(ledger_path, columns=['request_id', 'regime_confidence'])
            if 'request_id' in close_trades.columns and 'request_id' in ledger.columns:
                close_trades = close_trades.merge(
                    ledger[['request_id', 'regime_confidence']].drop_duplicates('request_id'),
                    on='request_id', how='left', suffixes=('', '_ledger')
                )
                # Use ledger confidence if available
                if 'regime_confidence_ledger' in close_trades.columns:
                    close_trades['regime_confidence'] = close_trades['regime_confidence_ledger'].fillna(
                        close_trades.get('regime_confidence', 0.8)
                    )

        # Ensure regime_confidence column exists
        if 'regime_confidence' not in close_trades.columns:
            close_trades['regime_confidence'] = 0.80  # Default if not available

        # Compute effective regime
        close_trades['effective_regime'] = close_trades.apply(
            lambda r: compute_effective_regime(int(r['regime_state']), float(r['regime_confidence'])),
            axis=1
        )

        # Filter to eligible regimes
        close_trades = close_trades[close_trades['effective_regime'].isin(ELIGIBLE_REGIMES)]

        # Sort deterministically
        ts_col = 'exit_ts' if 'exit_ts' in close_trades.columns else 'fill_ts'
        close_trades = close_trades.sort_values([ts_col, 'pair', 'tf']).reset_index(drop=True)

        pairs = sorted(close_trades['pair'].unique())
        logger.info(f"{scenario}: {len(close_trades)} eligible CLOSE trades, {len(pairs)} pairs")

        for pair in pairs:
            for tf in ELIGIBLE_TFS:
                for regime in ELIGIBLE_REGIMES:
                    block_key = f"{pair}|{tf}|REGIME_{regime}"

                    block_trades = close_trades[
                        (close_trades['pair'] == pair) &
                        (close_trades['tf'] == tf) &
                        (close_trades['effective_regime'] == regime)
                    ].copy()

                    if len(block_trades) == 0:
                        continue

                    metrics = compute_block_metrics(block_trades)
                    if metrics is None:
                        continue

                    if block_key not in baselines['blocks']:
                        baselines['blocks'][block_key] = {
                            'pair': pair,
                            'tf': tf,
                            'effective_regime': f'REGIME_{regime}',
                            'scenarios': {}
                        }

                    baselines['blocks'][block_key]['scenarios'][scenario] = metrics

    # Compute schema hash for the baselines structure
    schema_fields = sorted(baselines['blocks'].keys()) if baselines['blocks'] else []
    schema_str = json.dumps(schema_fields, sort_keys=True)
    baselines['metadata']['schema_hash'] = hashlib.sha256(schema_str.encode()).hexdigest()[:16]
    baselines['metadata']['total_blocks'] = len(baselines['blocks'])

    return baselines


def main():
    project_root = Path('.')
    baselines = build_baselines(project_root)

    output_path = project_root / 'ops' / 'edge_baselines_by_regime.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(baselines, f, indent=2, sort_keys=False, default=str)

    print(f"Saved: {output_path}")
    print(f"Total blocks: {baselines['metadata']['total_blocks']}")

    # Print summary
    scenario_counts = {}
    for block_key, block in baselines['blocks'].items():
        for sc in block['scenarios']:
            scenario_counts[sc] = scenario_counts.get(sc, 0) + 1

    for sc, count in sorted(scenario_counts.items()):
        print(f"  {sc}: {count} blocks")

    return baselines


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
