"""
CHAOS V1.0 — Volatility Targeting Backtest + Edge Decay Baselines
Tasks 3 & 4 from the prompt.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys

sys.path.insert(0, '.')

from risk.engine.volatility_targeter import VolatilityTargeter
from risk.engine.edge_decay_monitor import EdgeDecayMonitor

# ─── CONFIG ───────────────────────────────────────────────────────
RUN_DIR = Path('replay/outputs/runs/20260304T162601Z')
OUTPUT_DIR = Path('replay/outputs')
EQUITY_CURVES_PATH = OUTPUT_DIR / 'equity_curves.json'

TIER_PAIRS = ['EURUSD', 'USDCAD', 'AUDUSD', 'GBPJPY']
BASE_WEIGHTS = {'EURUSD': 0.35, 'USDCAD': 0.40, 'AUDUSD': 0.20, 'GBPJPY': 0.05}
ENABLED_TFS = ['H1', 'M30']

# ─── LOAD TRADES ──────────────────────────────────────────────────
trades = pd.read_parquet(RUN_DIR / 'trades.parquet')
close_trades = trades[trades['action'] == 'CLOSE'].copy()
close_trades = close_trades[close_trades['tf'].isin(ENABLED_TFS)].copy()
close_trades = close_trades[close_trades['pair'].isin(TIER_PAIRS)].copy()

ts_col = 'exit_ts' if 'exit_ts' in close_trades.columns else 'fill_ts'
close_trades[ts_col] = pd.to_datetime(close_trades[ts_col])
close_trades = close_trades.sort_values(ts_col)

print(f"Loaded {len(close_trades)} trades for backtest")

# ─── BUILD DAILY RETURNS PER PAIR ────────────────────────────────
daily_returns_by_pair = {}
for pair in TIER_PAIRS:
    pt = close_trades[close_trades['pair'] == pair].copy()
    pt = pt.set_index(ts_col)
    daily = pt['pnl_net_usd'].resample('D').sum()
    daily_returns_by_pair[pair] = daily

# Align to common date index
all_dates = sorted(set().union(*[set(d.index) for d in daily_returns_by_pair.values()]))
returns_df = pd.DataFrame(index=all_dates)
for pair in TIER_PAIRS:
    returns_df[pair] = daily_returns_by_pair[pair]
returns_df = returns_df.fillna(0)

# ─── PORTFOLIO A: FIXED WEIGHTS ──────────────────────────────────
total_base = sum(BASE_WEIGHTS.values())
fixed_weights_norm = {p: w / total_base for p, w in BASE_WEIGHTS.items()}

fixed_daily = sum(returns_df[p] * fixed_weights_norm[p] for p in TIER_PAIRS)
fixed_cum = fixed_daily.cumsum()
fixed_sharpe = fixed_daily.mean() / fixed_daily.std() * np.sqrt(252) if fixed_daily.std() > 0 else 0
fixed_maxdd = float((fixed_cum.cummax() - fixed_cum).max())
fixed_total = float(fixed_cum.iloc[-1])
fixed_win_days = int((fixed_daily > 0).sum())
fixed_loss_days = int((fixed_daily < 0).sum())
fixed_win_ratio = fixed_win_days / (fixed_win_days + fixed_loss_days) * 100

# Monthly PnL for worst month
fixed_monthly = fixed_daily.resample('ME').sum()
fixed_worst_month = float(fixed_monthly.min())

# ─── PORTFOLIO B: VOL-TARGETED ───────────────────────────────────
targeter = VolatilityTargeter(
    pairs=TIER_PAIRS,
    base_weights=BASE_WEIGHTS,
    target_annual_vol=0.08,
    vol_lookback=20
)

vol_daily_pnl = []
vol_weights_history = []

for date in returns_df.index:
    day_returns = {p: float(returns_df.loc[date, p]) for p in TIER_PAIRS}

    # Get current risk weights BEFORE updating with today's data
    risk_weights = targeter.get_risk_weights()
    total_rw = sum(risk_weights.values())
    risk_weights_norm = {p: w / total_rw for p, w in risk_weights.items()} if total_rw > 0 else fixed_weights_norm

    # Compute current drawdown for exposure scaling
    cum_so_far = sum(vol_daily_pnl)
    peak = max([0] + [sum(vol_daily_pnl[:i+1]) for i in range(len(vol_daily_pnl))]) if vol_daily_pnl else 0
    dd_pct = ((peak - cum_so_far) / max(peak, 1)) * 100 if peak > 0 else 0
    exposure_scalar = targeter.get_exposure_scalar(dd_pct)

    # Daily portfolio PnL with vol-targeted weights + exposure scaling
    day_pnl = sum(day_returns[p] * risk_weights_norm[p] * exposure_scalar for p in TIER_PAIRS)
    vol_daily_pnl.append(day_pnl)

    vol_weights_history.append({
        'date': str(date.date()),
        'weights': dict(risk_weights_norm),
        'exposure_scalar': exposure_scalar
    })

    # Update targeter with today's returns
    for pair in TIER_PAIRS:
        targeter.update(pair, day_returns[pair])

vol_series = pd.Series(vol_daily_pnl, index=returns_df.index)
vol_cum = vol_series.cumsum()
vol_sharpe = vol_series.mean() / vol_series.std() * np.sqrt(252) if vol_series.std() > 0 else 0
vol_maxdd = float((vol_cum.cummax() - vol_cum).max())
vol_total = float(vol_cum.iloc[-1])
vol_win_days = int((vol_series > 0).sum())
vol_loss_days = int((vol_series < 0).sum())
vol_win_ratio = vol_win_days / (vol_win_days + vol_loss_days) * 100

vol_monthly = vol_series.resample('ME').sum()
vol_worst_month = float(vol_monthly.min())

# Final snapshot weights
final_weights = targeter.get_risk_weights()
final_total = sum(final_weights.values())
final_weights_norm = {p: round(w / final_total, 4) for p, w in final_weights.items()}

# ─── IMPROVEMENTS ─────────────────────────────────────────────────
sharpe_imp = ((vol_sharpe - fixed_sharpe) / fixed_sharpe * 100) if fixed_sharpe != 0 else 0
dd_imp = ((vol_maxdd - fixed_maxdd) / fixed_maxdd * 100) if fixed_maxdd != 0 else 0
pnl_imp = ((vol_total - fixed_total) / abs(fixed_total) * 100) if fixed_total != 0 else 0
wr_imp = vol_win_ratio - fixed_win_ratio
wm_imp = ((vol_worst_month - fixed_worst_month) / abs(fixed_worst_month) * 100) if fixed_worst_month != 0 else 0

# ─── SAVE BACKTEST JSON ──────────────────────────────────────────
backtest_result = {
    'source': 'Run 4, IBKR_BASE, H1+M30',
    'fixed_weights': {
        'weights': fixed_weights_norm,
        'sharpe': round(fixed_sharpe, 2),
        'max_dd': round(fixed_maxdd, 2),
        'total_pnl': round(fixed_total, 2),
        'win_day_ratio': round(fixed_win_ratio, 1),
        'worst_month': round(fixed_worst_month, 2)
    },
    'vol_targeted': {
        'final_weights': final_weights_norm,
        'sharpe': round(vol_sharpe, 2),
        'max_dd': round(vol_maxdd, 2),
        'total_pnl': round(vol_total, 2),
        'win_day_ratio': round(vol_win_ratio, 1),
        'worst_month': round(vol_worst_month, 2)
    },
    'improvements': {
        'sharpe_pct': round(sharpe_imp, 1),
        'max_dd_pct': round(dd_imp, 1),
        'pnl_pct': round(pnl_imp, 1),
        'win_ratio_delta': round(wr_imp, 1),
        'worst_month_pct': round(wm_imp, 1)
    },
    'vol_status': targeter.get_status()
}

bt_path = OUTPUT_DIR / 'vol_targeting_backtest.json'
with open(bt_path, 'w') as f:
    json.dump(backtest_result, f, indent=2, default=str)
print(f"\nSaved: {bt_path}")

# ─── PRINT BACKTEST COMPARISON ────────────────────────────────────
print("\n")
print("VOLATILITY TARGETING BACKTEST — Run 4 Data")
print("=" * 55)
print(f"{'':20s} {'Fixed Weights':>14s}  {'Vol-Targeted':>14s}  {'Change':>10s}")
print("-" * 62)
print(f"{'Sharpe:':20s} {fixed_sharpe:>14.2f}  {vol_sharpe:>14.2f}  {sharpe_imp:>+9.1f}%")
print(f"{'Max Drawdown:':20s} {'$'+f'{fixed_maxdd:.2f}':>14s}  {'$'+f'{vol_maxdd:.2f}':>14s}  {dd_imp:>+9.1f}%")
print(f"{'Total PnL:':20s} {'$'+f'{fixed_total:,.2f}':>14s}  {'$'+f'{vol_total:,.2f}':>14s}  {pnl_imp:>+9.1f}%")
print(f"{'Win Day Ratio:':20s} {f'{fixed_win_ratio:.1f}%':>14s}  {f'{vol_win_ratio:.1f}%':>14s}  {wr_imp:>+9.1f}%")
print(f"{'Worst Month:':20s} {'$'+f'{fixed_worst_month:.2f}':>14s}  {'$'+f'{vol_worst_month:.2f}':>14s}  {wm_imp:>+9.1f}%")
print(f"\nVol-targeted weights (final snapshot):")
for pair in TIER_PAIRS:
    fw = final_weights_norm.get(pair, 0)
    bw = BASE_WEIGHTS[pair]
    print(f"  {pair}: {fw:.4f} (base: {bw:.2f})")


# ═══════════════════════════════════════════════════════════════════
# TASK 4: EDGE DECAY BASELINES
# ═══════════════════════════════════════════════════════════════════
print("\n\n")
print("EDGE DECAY MONITOR — Baselines Set")
print("=" * 55)

edge_baselines = {}
for pair in TIER_PAIRS:
    pt = close_trades[close_trades['pair'] == pair].copy()
    if len(pt) == 0:
        continue

    # Compute baseline metrics from Run 4
    wins = pt[pt['pnl_net_usd'] > 0]
    losses = pt[pt['pnl_net_usd'] < 0]

    baseline_pf = float(wins['pnl_net_usd'].sum() / abs(losses['pnl_net_usd'].sum())) if len(losses) > 0 else float('inf')
    baseline_wr = len(wins) / len(pt)
    avg_win = float(wins['pnl_net_usd'].mean()) if len(wins) > 0 else 0
    avg_loss = float(abs(losses['pnl_net_usd'].mean())) if len(losses) > 0 else 1
    baseline_wl_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')

    # Create monitor and feed all trades to get current status
    monitor = EdgeDecayMonitor(
        pair=pair,
        tf='H1_M30',
        baseline_pf=baseline_pf,
        baseline_wr=baseline_wr,
        baseline_avg_winner_loser_ratio=baseline_wl_ratio,
        rolling_window=20,
        alert_lookback=50
    )

    # Feed all trades
    for _, trade in pt.iterrows():
        ts = trade[ts_col] if ts_col in trade.index else None
        monitor.record_trade(
            pnl=float(trade['pnl_net_usd']),
            timestamp=ts
        )

    status = monitor.get_status()

    edge_baselines[pair] = {
        'baseline_pf': round(baseline_pf, 2),
        'baseline_wr': round(baseline_wr, 3),
        'baseline_wl_ratio': round(baseline_wl_ratio, 2),
        'avg_winner': round(avg_win, 2),
        'avg_loser': round(avg_loss, 2),
        'trade_count': len(pt),
        'current_status': status['status'],
        'alerts': status['alerts'],
        'rolling_metrics': {
            'rolling_pf': round(status['metrics']['rolling_pf'], 2) if status['metrics']['rolling_pf'] else None,
            'rolling_wr': round(status['metrics']['rolling_wr'], 3) if status['metrics']['rolling_wr'] else None,
            'wl_ratio': round(status['metrics']['wl_ratio'], 2) if status['metrics']['wl_ratio'] else None
        }
    }

    print(f"  {pair}: {status['status']} — PF {baseline_pf:.2f}, WR {baseline_wr*100:.1f}%, W/L {baseline_wl_ratio:.2f}, baseline locked")

# Save edge decay baselines
edge_path = OUTPUT_DIR / 'edge_decay_baselines.json'
with open(edge_path, 'w') as f:
    json.dump(edge_baselines, f, indent=2, default=str)
print(f"\nSaved: {edge_path}")

print("\nAll monitors initialized. Will detect edge decay during paper/live trading.")

# ─── FINAL FILE LIST ──────────────────────────────────────────────
print("\n\nFiles created/saved:")
print(f"  risk/engine/volatility_targeter.py")
print(f"  risk/engine/edge_decay_monitor.py")
print(f"  docs/RL_META_CONTROLLER_ARCHITECTURE.md")
print(f"  {bt_path}")
print(f"  {edge_path}")
