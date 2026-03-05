"""
Extract execution stats for 3 retrain targets: USDCAD_H1_R1, USDCAD_M30_R1, AUDUSD_M30_R1
Filtered to REGIME_1 (effective_regime), across all 4 Run 4 scenarios.
"""
import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path

sys.path.insert(0, '.')
from ops.build_edge_baselines_by_regime import compute_effective_regime

SCENARIO_DIRS = {
    'IBKR_BASE':        '20260304T162601Z',
    'BASE_PLUS_25':     '20260304T180836Z',
    'STRESS_PLUS_75':   '20260304T195622Z',
    'VOLATILITY_SPIKE': '20260305T000817Z',
}

TARGETS = [
    ('USDCAD', 'H1',  1),
    ('USDCAD', 'M30', 1),
    ('AUDUSD', 'M30', 1),
]

RUNS_DIR = Path('replay/outputs/runs')


def load_all_trades(run_id):
    """Load trades + merge regime_confidence from ledger."""
    run_dir = RUNS_DIR / run_id
    trades = pd.read_parquet(run_dir / 'trades.parquet')
    close_trades = trades[trades['action'] == 'CLOSE'].copy()

    ledger_path = run_dir / 'decision_ledger.parquet'
    if ledger_path.exists():
        ledger = pd.read_parquet(ledger_path, columns=['request_id', 'regime_confidence'])
        if 'request_id' in close_trades.columns:
            close_trades = close_trades.merge(
                ledger[['request_id', 'regime_confidence']].drop_duplicates('request_id'),
                on='request_id', how='left', suffixes=('', '_led')
            )
            if 'regime_confidence_led' in close_trades.columns:
                close_trades['regime_confidence'] = close_trades['regime_confidence_led'].fillna(0.8)

    if 'regime_confidence' not in close_trades.columns:
        close_trades['regime_confidence'] = 0.80

    close_trades['effective_regime'] = close_trades.apply(
        lambda r: compute_effective_regime(int(r['regime_state']), float(r['regime_confidence'])), axis=1
    )
    return close_trades


# Collect stats
results = []

for pair, tf, regime in TARGETS:
    block_key = f"{pair}_{tf}_R{regime}"

    # Aggregate across all scenarios for spread/slippage/fill stats
    # but also count per-scenario trades
    all_block_trades = []
    per_scenario_counts = {}

    for sc_name, run_id in SCENARIO_DIRS.items():
        trades = load_all_trades(run_id)
        block = trades[
            (trades['pair'] == pair) &
            (trades['tf'] == tf) &
            (trades['effective_regime'] == regime)
        ]
        per_scenario_counts[sc_name] = len(block)
        all_block_trades.append(block)

    combined = pd.concat(all_block_trades, ignore_index=True)

    if len(combined) == 0:
        results.append({
            'block': block_key,
            'spread_ratio': 0, 'slippage': 0, 'partial_fill_rate': 0,
            'counts': per_scenario_counts,
        })
        continue

    # Spread ratio: spread_cost_pips relative to pair average
    spread_mean = float(combined['spread_cost_pips'].mean())

    # Slippage mean
    slippage_mean = float(combined['slippage_cost_pips'].mean()) if 'slippage_cost_pips' in combined.columns else 0.0

    # Partial fill rate
    if 'fill_status' in combined.columns:
        partial_count = (combined['fill_status'] == 'PARTIAL').sum()
        partial_rate = partial_count / len(combined) * 100
    elif 'qty_lots' in combined.columns and 'qty_lots_filled' in combined.columns:
        partial_count = (combined['qty_lots_filled'] < combined['qty_lots']).sum()
        partial_rate = partial_count / len(combined) * 100
    else:
        partial_rate = 0.0

    results.append({
        'block': block_key,
        'spread_ratio': spread_mean,
        'slippage': slippage_mean,
        'partial_fill_rate': partial_rate,
        'counts': per_scenario_counts,
    })

# Print table
print()
print("RETRAIN TARGET EXECUTION STATS (REGIME_1 only)")
h1 = "+" + "-"*19 + "+" + "-"*14 + "+" + "-"*14 + "+" + "-"*14 + "+" + "-"*31 + "+"
h2 = f"| {'Block':<17} | {'Spread Ratio':>12} | {'Slippage':>12} | {'Partial Fill':>12} | {'Trades by Scenario':<29} |"
h3 = f"| {'':17} | {'(mean)':>12} | {'(mean pips)':>12} | {'Rate':>12} | {'IBKR/BASE25/STRESS/VOLSPIKE':<29} |"
print(h1)
print(h2)
print(h3)
print(h1)

for r in results:
    c = r['counts']
    counts_str = f"{c.get('IBKR_BASE',0)} / {c.get('BASE_PLUS_25',0)} / {c.get('STRESS_PLUS_75',0)} / {c.get('VOLATILITY_SPIKE',0)}"
    print(f"| {r['block']:<17} | {r['spread_ratio']:>12.2f} | {r['slippage']:>12.2f} | {r['partial_fill_rate']:>11.2f}% | {counts_str:<29} |")

print(h1)
