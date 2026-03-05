"""
CHAOS V1.0 — Equity Curve Analysis
Run 4, IBKR_BASE scenario, H1+M30 only (TRADE_ENABLED TFs)
Generates: equity_curves.json + equity_curves.html
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────
RUN_DIR = Path('replay/outputs/runs/20260304T162601Z')
OUTPUT_DIR = Path('replay/outputs')
TIER_PAIRS = ['EURUSD', 'USDCAD', 'AUDUSD', 'GBPJPY']
ALLOCATIONS = {'EURUSD': 0.40, 'USDCAD': 0.35, 'AUDUSD': 0.25, 'GBPJPY': 0.10}
ENABLED_TFS = ['H1', 'M30']

# ─── TASK 1: LOAD & FILTER TRADES ────────────────────────────────
trades_path = RUN_DIR / 'trades.parquet'
if not trades_path.exists():
    raise FileNotFoundError(f"trades.parquet not found at {trades_path}")

trades = pd.read_parquet(trades_path)
print(f"Loaded {len(trades)} total rows from {trades_path}")

# Filter to CLOSE rows only
close_trades = trades[trades['action'] == 'CLOSE'].copy()
print(f"CLOSE rows: {len(close_trades)}")

# Filter to TRADE_ENABLED TFs
close_trades = close_trades[close_trades['tf'].isin(ENABLED_TFS)].copy()
print(f"H1+M30 CLOSE rows: {len(close_trades)}")

# Determine timestamp column
ts_col = 'exit_ts' if 'exit_ts' in close_trades.columns else 'fill_ts'
close_trades = close_trades.sort_values(ts_col)

# ─── TASK 1: PER-PAIR EQUITY CURVES ──────────────────────────────
equity_curves = {}
for pair in TIER_PAIRS:
    pair_trades = close_trades[close_trades['pair'] == pair].copy()
    if len(pair_trades) == 0:
        print(f"  {pair}: no trades")
        continue

    pair_trades = pair_trades.sort_values(ts_col)
    pair_trades['cum_pnl'] = pair_trades['pnl_net_usd'].cumsum()

    losers_sum = abs(pair_trades[pair_trades['pnl_net_usd'] < 0]['pnl_net_usd'].sum())
    winners_sum = pair_trades[pair_trades['pnl_net_usd'] > 0]['pnl_net_usd'].sum()
    pf = float(winners_sum / losers_sum) if losers_sum > 0 else float('inf')

    equity_curves[pair] = {
        'timestamps': [str(t) for t in pair_trades[ts_col].tolist()],
        'cum_pnl': pair_trades['cum_pnl'].tolist(),
        'trade_count': len(pair_trades),
        'final_pnl': float(pair_trades['cum_pnl'].iloc[-1]),
        'max_drawdown': float((pair_trades['cum_pnl'].cummax() - pair_trades['cum_pnl']).max()),
        'pf': pf
    }
    print(f"  {pair}: {len(pair_trades)} trades, PF {pf:.2f}, PnL ${pair_trades['cum_pnl'].iloc[-1]:.2f}, MaxDD ${equity_curves[pair]['max_drawdown']:.2f}")

# ─── TASK 2: PORTFOLIO EQUITY CURVE ──────────────────────────────
total_alloc = sum(ALLOCATIONS.values())
all_events = []
for pair, curve in equity_curves.items():
    weight = ALLOCATIONS.get(pair, 0) / total_alloc
    pnl_changes = [curve['cum_pnl'][0]] + [curve['cum_pnl'][i] - curve['cum_pnl'][i-1] for i in range(1, len(curve['cum_pnl']))]
    for ts, pnl_change in zip(curve['timestamps'], pnl_changes):
        all_events.append({
            'timestamp': ts,
            'pair': pair,
            'pnl_change': pnl_change * weight
        })

portfolio_df = pd.DataFrame(all_events).sort_values('timestamp')
portfolio_df['portfolio_cum_pnl'] = portfolio_df['pnl_change'].cumsum()
print(f"\nPortfolio: {len(portfolio_df)} events, final PnL ${portfolio_df['portfolio_cum_pnl'].iloc[-1]:.2f}")

# ─── TASK 3: CORRELATION MATRIX ──────────────────────────────────
daily_returns = {}
for pair, curve in equity_curves.items():
    pair_df = pd.DataFrame({
        'timestamp': curve['timestamps'],
        'pnl': [curve['cum_pnl'][0]] + [curve['cum_pnl'][i] - curve['cum_pnl'][i-1] for i in range(1, len(curve['cum_pnl']))]
    })
    pair_df['timestamp'] = pd.to_datetime(pair_df['timestamp'])
    pair_df = pair_df.set_index('timestamp')
    daily = pair_df['pnl'].resample('D').sum()
    daily_returns[pair] = daily

returns_df = pd.DataFrame(daily_returns)
corr_matrix = returns_df.corr()

print("\nPAIR CORRELATION MATRIX (daily returns):")
print(corr_matrix.round(3).to_string())

# ─── TASK 4: PORTFOLIO STATISTICS ─────────────────────────────────
weights_series = pd.Series({p: ALLOCATIONS.get(p, 0) / total_alloc for p in returns_df.columns})
portfolio_returns = returns_df.fillna(0).dot(weights_series)

sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252) if portfolio_returns.std() > 0 else 0
max_dd_portfolio = float((portfolio_returns.cumsum().cummax() - portfolio_returns.cumsum()).max())
total_pnl_portfolio = float(portfolio_returns.sum())
win_days = int((portfolio_returns > 0).sum())
loss_days = int((portfolio_returns < 0).sum())
flat_days = int((portfolio_returns == 0).sum())
win_ratio = win_days / (win_days + loss_days) * 100 if (win_days + loss_days) > 0 else 0

print(f"\nPORTFOLIO STATISTICS:")
print(f"  Sharpe Ratio (annualized): {sharpe:.2f}")
print(f"  Total PnL: ${total_pnl_portfolio:.2f}")
print(f"  Max Drawdown: ${max_dd_portfolio:.2f}")
print(f"  Win Days: {win_days}, Loss Days: {loss_days}, Flat Days: {flat_days}")
print(f"  Win Day Ratio: {win_ratio:.1f}%")

# ─── TASK 5A: SAVE JSON ──────────────────────────────────────────
output_json = {
    'source': 'Run 4, IBKR_BASE, H1+M30 only',
    'run_dir': str(RUN_DIR),
    'allocations': ALLOCATIONS,
    'per_pair': {},
    'correlation_matrix': {
        'pairs': list(corr_matrix.columns),
        'matrix': corr_matrix.values.tolist()
    },
    'portfolio': {
        'sharpe': round(sharpe, 2),
        'total_pnl': round(total_pnl_portfolio, 2),
        'max_dd': round(max_dd_portfolio, 2),
        'win_days': win_days,
        'loss_days': loss_days,
        'flat_days': flat_days,
        'win_day_ratio': round(win_ratio, 1)
    }
}

for pair, curve in equity_curves.items():
    output_json['per_pair'][pair] = {
        'trades': curve['trade_count'],
        'final_pnl': round(curve['final_pnl'], 2),
        'max_dd': round(curve['max_drawdown'], 2),
        'pf': round(curve['pf'], 2)
    }

json_path = OUTPUT_DIR / 'equity_curves.json'
with open(json_path, 'w') as f:
    json.dump(output_json, f, indent=2)
print(f"\nSaved: {json_path}")

# ─── TASK 5B: SAVE HTML VISUALIZATION ────────────────────────────
# Prepare data for Chart.js
chart_data = {
    'per_pair': {},
    'portfolio_cum_pnl': portfolio_df['portfolio_cum_pnl'].tolist(),
    'labels': portfolio_df['timestamp'].tolist()
}
for pair, curve in equity_curves.items():
    chart_data['per_pair'][pair] = {
        'timestamps': curve['timestamps'],
        'cum_pnl': curve['cum_pnl']
    }

corr_data = {
    'pairs': list(corr_matrix.columns),
    'matrix': [[round(v, 3) for v in row] for row in corr_matrix.values.tolist()]
}

stats_data = {
    'sharpe': round(sharpe, 2),
    'total_pnl': round(total_pnl_portfolio, 2),
    'max_dd': round(max_dd_portfolio, 2),
    'win_day_ratio': f"{win_ratio:.1f}%"
}

html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>CHAOS V1.0 — Equity Curves</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
    <style>
        body {{ font-family: monospace; background: #1a1a2e; color: #e0e0e0; padding: 20px; }}
        .chart-container {{ width: 90%; max-width: 1200px; margin: 20px auto; background: #16213e; padding: 20px; border-radius: 8px; }}
        h1 {{ text-align: center; color: #00d4ff; }}
        h2 {{ color: #00d4ff; }}
        table {{ border-collapse: collapse; margin: 10px auto; }}
        td, th {{ border: 1px solid #333; padding: 8px; text-align: center; }}
        th {{ background: #0f3460; }}
        .positive {{ color: #00ff88; }}
        .negative {{ color: #ff4444; }}
        .stats-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; max-width: 600px; margin: 0 auto; }}
        .stat-box {{ background: #0f3460; padding: 15px; border-radius: 6px; text-align: center; }}
        .stat-value {{ font-size: 1.8em; font-weight: bold; }}
        .stat-label {{ font-size: 0.9em; color: #888; margin-top: 5px; }}
        .pair-table {{ width: 100%; margin: 15px 0; }}
        .pair-table td, .pair-table th {{ padding: 10px 15px; }}
    </style>
</head>
<body>
    <h1>CHAOS V1.0 — Portfolio Equity Curves</h1>
    <p style="text-align:center; color:#888;">Run 4 | IBKR_BASE | H1+M30 Only | Tier 1+2 Pairs</p>

    <div class="chart-container">
        <canvas id="equityChart"></canvas>
    </div>

    <div class="chart-container">
        <h2>Per-Pair Summary</h2>
        <table class="pair-table">
            <tr><th>Pair</th><th>Trades</th><th>PF</th><th>Final PnL</th><th>Max DD</th><th>Weight</th></tr>
"""

for pair in TIER_PAIRS:
    if pair in equity_curves:
        c = equity_curves[pair]
        pnl_class = "positive" if c['final_pnl'] >= 0 else "negative"
        html_content += f'            <tr><td>{pair}</td><td>{c["trade_count"]}</td><td>{c["pf"]:.2f}</td><td class="{pnl_class}">${c["final_pnl"]:,.2f}</td><td class="negative">${c["max_drawdown"]:,.2f}</td><td>{ALLOCATIONS[pair]*100:.0f}%</td></tr>\n'

html_content += f"""        </table>
    </div>

    <div class="chart-container">
        <h2>Correlation Matrix (Daily Returns)</h2>
        <div id="corrMatrix"></div>
    </div>

    <div class="chart-container">
        <h2>Portfolio Statistics (Weighted)</h2>
        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-value positive">{sharpe:.2f}</div>
                <div class="stat-label">Sharpe Ratio (annualized)</div>
            </div>
            <div class="stat-box">
                <div class="stat-value positive">${total_pnl_portfolio:,.2f}</div>
                <div class="stat-label">Total PnL</div>
            </div>
            <div class="stat-box">
                <div class="stat-value negative">${max_dd_portfolio:,.2f}</div>
                <div class="stat-label">Max Drawdown</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{win_ratio:.1f}%</div>
                <div class="stat-label">Win Day Ratio ({win_days}W / {loss_days}L / {flat_days}F)</div>
            </div>
        </div>
    </div>

    <script>
        const curveData = {json.dumps(chart_data, default=str)};
        const corrData = {json.dumps(corr_data)};

        const colors = {{
            'EURUSD': '#00d4ff',
            'USDCAD': '#00ff88',
            'AUDUSD': '#ff8800',
            'GBPJPY': '#ff44aa',
            'PORTFOLIO': '#ffffff'
        }};

        // Build datasets - align all curves to portfolio timeline
        const datasets = [];
        for (const [pair, data] of Object.entries(curveData.per_pair)) {{
            // Create sparse dataset aligned to portfolio labels
            const pairData = [];
            let pairIdx = 0;
            for (let i = 0; i < curveData.labels.length; i++) {{
                if (pairIdx < data.timestamps.length && curveData.labels[i] === data.timestamps[pairIdx]) {{
                    pairData.push(data.cum_pnl[pairIdx]);
                    pairIdx++;
                }} else {{
                    pairData.push(pairIdx > 0 ? data.cum_pnl[pairIdx - 1] : null);
                }}
            }}
            datasets.push({{
                label: pair,
                data: pairData,
                borderColor: colors[pair] || '#888',
                borderWidth: 1.5,
                fill: false,
                pointRadius: 0,
                spanGaps: true
            }});
        }}

        // Portfolio curve
        datasets.push({{
            label: 'PORTFOLIO (weighted)',
            data: curveData.portfolio_cum_pnl,
            borderColor: colors['PORTFOLIO'],
            borderWidth: 3,
            fill: false,
            pointRadius: 0,
            borderDash: [5, 3]
        }});

        // Simplify labels for display
        const displayLabels = curveData.labels.map((l, i) => {{
            if (i % Math.max(1, Math.floor(curveData.labels.length / 20)) === 0) {{
                return l.substring(0, 10);
            }}
            return '';
        }});

        new Chart(document.getElementById('equityChart'), {{
            type: 'line',
            data: {{ labels: displayLabels, datasets: datasets }},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{ display: true, text: 'Cumulative PnL — Tier 1+2 Pairs (IBKR_BASE, H1+M30)', color: '#e0e0e0', font: {{ size: 14 }} }},
                    legend: {{ labels: {{ color: '#e0e0e0' }} }}
                }},
                scales: {{
                    x: {{ ticks: {{ color: '#888', maxRotation: 45 }}, grid: {{ color: '#333' }} }},
                    y: {{ ticks: {{ color: '#888' }}, title: {{ display: true, text: 'PnL ($)', color: '#888' }}, grid: {{ color: '#333' }} }}
                }}
            }}
        }});

        // Correlation matrix table
        let html = '<table><tr><th></th>';
        for (const p of corrData.pairs) html += '<th>' + p + '</th>';
        html += '</tr>';
        for (let i = 0; i < corrData.pairs.length; i++) {{
            html += '<tr><th>' + corrData.pairs[i] + '</th>';
            for (let j = 0; j < corrData.pairs.length; j++) {{
                const val = corrData.matrix[i][j];
                const cls = (i === j) ? '' : (Math.abs(val) > 0.5 ? 'negative' : 'positive');
                html += '<td class="' + cls + '">' + val.toFixed(3) + '</td>';
            }}
            html += '</tr>';
        }}
        html += '</table>';
        document.getElementById('corrMatrix').innerHTML = html;
    </script>
</body>
</html>"""

html_path = OUTPUT_DIR / 'equity_curves.html'
with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html_content)
print(f"Saved: {html_path}")

# ─── TASK 6: PRINT SUMMARY ───────────────────────────────────────
print("\n")
print("EQUITY CURVE ANALYSIS — Run 4 (IBKR_BASE, H1+M30)")
print("=" * 55)
print("\nPer-Pair Results:")
for pair in TIER_PAIRS:
    if pair in equity_curves:
        c = equity_curves[pair]
        print(f"  {pair}: {c['trade_count']} trades, PF {c['pf']:.2f}, PnL ${c['final_pnl']:,.2f}, MaxDD ${c['max_drawdown']:,.2f}")

print(f"\nCorrelation Matrix:")
print(f"         {'  '.join(f'{p:>8}' for p in corr_matrix.columns)}")
for pair in corr_matrix.index:
    vals = '  '.join(f'{corr_matrix.loc[pair, p]:8.3f}' for p in corr_matrix.columns)
    print(f"{pair:>8} {vals}")

print(f"\nPortfolio (weighted):")
print(f"  Sharpe Ratio: {sharpe:.2f}")
print(f"  Total PnL: ${total_pnl_portfolio:,.2f}")
print(f"  Max Drawdown: ${max_dd_portfolio:,.2f}")
print(f"  Win Day Ratio: {win_ratio:.1f}%")

print(f"\nFiles saved:")
print(f"  {json_path}")
print(f"  {html_path}")
