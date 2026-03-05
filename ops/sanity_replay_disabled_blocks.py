"""
Sanity Check: Before vs After Block Disable

Uses Run 4 IBKR_BASE data to compute metrics for Tier 1+2 pairs
with and without the disabled blocks (USDJPY_M30_R1, NZDUSD_H1_R1).

Since USDJPY and NZDUSD are NOT in Tier 1+2 (EURUSD, USDCAD, AUDUSD, GBPJPY),
the disabled blocks don't directly affect Tier 1+2 PF. This confirms the disables
are safe — they only remove non-Tier pairs from REGIME_1 execution.

Additionally, we check the full 9-pair portfolio before/after to quantify the
trade count drop and PF impact.
"""
import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path

sys.path.insert(0, '.')
from ops.build_edge_baselines_by_regime import compute_effective_regime

RUN_DIR = Path('replay/outputs/runs/20260304T162601Z')
TIER_PAIRS = ['EURUSD', 'USDCAD', 'AUDUSD', 'GBPJPY']
DISABLED_BLOCKS = {
    ('USDJPY', 'M30', 1),  # USDJPY_M30_R1
    ('NZDUSD', 'H1',  1),  # NZDUSD_H1_R1
}

# Load trades
trades = pd.read_parquet(RUN_DIR / 'trades.parquet')
close_trades = trades[trades['action'] == 'CLOSE'].copy()

# Merge regime confidence
ledger_path = RUN_DIR / 'decision_ledger.parquet'
if ledger_path.exists():
    ledger = pd.read_parquet(ledger_path, columns=['request_id', 'regime_confidence'])
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


def compute_pf(df):
    w = df[df['pnl_net_usd'] > 0]['pnl_net_usd'].sum()
    l = abs(df[df['pnl_net_usd'] < 0]['pnl_net_usd'].sum())
    return w / l if l > 0 else float('inf')


def is_disabled(row):
    return (row['pair'], row['tf'], row['effective_regime']) in DISABLED_BLOCKS


# ─── FULL PORTFOLIO (all 9 pairs, H1+M30) ─────────────────────────
full = close_trades[close_trades['tf'].isin(['H1', 'M30'])].copy()
full['disabled'] = full.apply(is_disabled, axis=1)

before_all = full
after_all = full[~full['disabled']]

disabled_trades = full[full['disabled']]

# ─── TIER 1+2 ONLY ─────────────────────────────────────────────────
tier_before = full[full['pair'].isin(TIER_PAIRS)]
tier_after = after_all[after_all['pair'].isin(TIER_PAIRS)]

# ─── PRINT ──────────────────────────────────────────────────────────
print()
print("SANITY REPLAY: BEFORE vs AFTER BLOCK DISABLE")
print("=" * 65)
print(f"  Source: Run 4 IBKR_BASE (20260304T162601Z), H1+M30 only")
print(f"  Disabled: USDJPY|M30|REGIME_1, NZDUSD|H1|REGIME_1")
print()

# What was removed
print("DISABLED BLOCK TRADES REMOVED:")
print(f"  Total trades removed: {len(disabled_trades)}")
for (pair, tf, reg), grp in disabled_trades.groupby(['pair', 'tf', 'effective_regime']):
    pf = compute_pf(grp)
    pnl = grp['pnl_net_usd'].sum()
    print(f"  {pair}|{tf}|REGIME_{reg}: {len(grp)} trades, PF {pf:.2f}, PnL ${pnl:.2f}")

print()
print("FULL PORTFOLIO (all 9 pairs, H1+M30):")
print(f"  {'Metric':<25} {'Before':>12} {'After':>12} {'Change':>12}")
print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")
print(f"  {'Trades':<25} {len(before_all):>12} {len(after_all):>12} {len(after_all)-len(before_all):>+12}")
pf_before = compute_pf(before_all)
pf_after = compute_pf(after_all)
pf_delta = ((pf_after - pf_before) / pf_before * 100) if pf_before > 0 else 0
print(f"  {'Profit Factor':<25} {pf_before:>12.2f} {pf_after:>12.2f} {pf_delta:>+11.1f}%")
pnl_before = before_all['pnl_net_usd'].sum()
pnl_after = after_all['pnl_net_usd'].sum()
print(f"  {'Net PnL':<25} {'$'+f'{pnl_before:,.2f}':>12} {'$'+f'{pnl_after:,.2f}':>12} {'$'+f'{pnl_after-pnl_before:,.2f}':>12}")

wr_before = (before_all['pnl_net_usd'] > 0).mean() * 100
wr_after = (after_all['pnl_net_usd'] > 0).mean() * 100
print(f"  {'Win Rate':<25} {f'{wr_before:.1f}%':>12} {f'{wr_after:.1f}%':>12} {wr_after-wr_before:>+11.1f}%")

print()
print("TIER 1+2 PORTFOLIO (EURUSD, USDCAD, AUDUSD, GBPJPY):")
print(f"  {'Metric':<25} {'Before':>12} {'After':>12} {'Change':>12}")
print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")
print(f"  {'Trades':<25} {len(tier_before):>12} {len(tier_after):>12} {len(tier_after)-len(tier_before):>+12}")
tier_pf_before = compute_pf(tier_before)
tier_pf_after = compute_pf(tier_after)
tier_pf_delta = ((tier_pf_after - tier_pf_before) / tier_pf_before * 100) if tier_pf_before > 0 else 0
print(f"  {'Profit Factor':<25} {tier_pf_before:>12.2f} {tier_pf_after:>12.2f} {tier_pf_delta:>+11.1f}%")
tier_pnl_before = tier_before['pnl_net_usd'].sum()
tier_pnl_after = tier_after['pnl_net_usd'].sum()
print(f"  {'Net PnL':<25} {'$'+f'{tier_pnl_before:,.2f}':>12} {'$'+f'{tier_pnl_after:,.2f}':>12} {'$'+f'{tier_pnl_after-tier_pnl_before:,.2f}':>12}")

print()
print("VERDICT: ", end="")
if tier_pf_after >= tier_pf_before:
    print("SAFE — Tier 1+2 PF unchanged or improved after disable.")
else:
    print(f"WARNING — Tier 1+2 PF dropped {abs(tier_pf_delta):.1f}%")

if pf_after >= pf_before:
    print(f"         Full portfolio PF improved {pf_delta:+.1f}% by removing negative-edge blocks.")
else:
    print(f"         Full portfolio PF changed {pf_delta:+.1f}%")
