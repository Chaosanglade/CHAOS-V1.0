"""
Quick verification: Cooldown scoping fix (pair -> pair_tf).
Runs EURUSD x M5/M15/M30/H1, 20K bar cap, IBKR_BASE only.
Expected: trades on multiple TFs, cooldown vetoes << 99%.
"""
import os
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["SKLEARN_WARNINGS"] = "ignore"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["ORT_LOG_SEVERITY_LEVEL"] = "3"
os.environ["PYTHONHASHSEED"] = "42"
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
import gc
import numpy as np
import random
import pandas as pd
from pathlib import Path

np.random.seed(42)
random.seed(42)

PROJECT_ROOT = Path('G:/My Drive/chaos_v1.0')
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'inference'))

from replay.runners.run_replay import ReplayRunner

print("=" * 60)
print("QUICK VERIFY: Cooldown scoping fix (pair -> pair_tf)")
print("EURUSD x M5/M15/M30/H1, 20K bar cap, IBKR_BASE")
print("=" * 60)
sys.stdout.flush()

pairs = ['EURUSD']
tfs = ['M5', 'M15', 'M30', 'H1']

runner = ReplayRunner(pairs=pairs, tfs=tfs)
runner.reset_state()
gc.collect()

t0 = time.perf_counter()
run_id = runner.run(
    pairs=pairs,
    tfs=tfs,
    max_bars=20000,
    scenario_name='IBKR_BASE',
)
elapsed = time.perf_counter() - t0

run_dir = PROJECT_ROOT / 'replay' / 'outputs' / 'runs' / run_id

# Load outputs
with open(run_dir / 'metrics.json') as f:
    metrics = json.load(f)
with open(run_dir / 'veto_breakdown.json') as f:
    veto = json.load(f)

# Trades per TF
ptf_path = run_dir / 'metrics_by_pair_tf.json'
if ptf_path.exists():
    with open(ptf_path) as f:
        ptf = json.load(f)
else:
    ptf = {}

# Also get trades from parquet for TF breakdown
trades_df = pd.read_parquet(run_dir / 'trades.parquet')
tf_counts = trades_df.groupby('tf').size().to_dict() if 'tf' in trades_df.columns else {}

print()
print("=" * 60)
print("RESULTS")
print("=" * 60)

# 1. Trades per TF
print(f"\n1. Trades per TF:")
for tf in tfs:
    key = f"EURUSD_{tf}"
    if key in ptf:
        m = ptf[key]
        print(f"   {tf}: {m['trades']} trades, PF {m['profit_factor']:.2f}")
    else:
        count = tf_counts.get(tf, 0)
        print(f"   {tf}: {count} trades (from trades.parquet)")

# 2. Veto breakdown top 3
print(f"\n2. Veto breakdown top 3:")
top_reasons = list(veto.get('top_10_reasons', {}).items())[:3]
for reason, count in top_reasons:
    print(f"   {reason}: {count}")

# 3. PF
pf = metrics.get('profit_factor', 0)
trades_total = metrics.get('total_trades', 0)
wr = metrics.get('win_rate', 0)
print(f"\n3. PF (IBKR_BASE): {pf:.4f} ({trades_total} trades, WR {wr:.1%})")

# 4. Cooldown vetoes %
total_vetoes = veto.get('risk_vetoes_total', 0)
cooldown_vetoes = veto.get('overall_reason_counts', {}).get('RISK_BLOCKED_COOLDOWN', 0)
total_events = veto.get('total_events', 1)
all_reason_counts = veto.get('overall_reason_counts', {})
total_reasons = sum(all_reason_counts.values())
cd_pct = (cooldown_vetoes / total_reasons * 100) if total_reasons > 0 else 0
print(f"\n4. Cooldown vetoes as % of total vetoes: {cd_pct:.1f}%")
print(f"   (RISK_BLOCKED_COOLDOWN: {cooldown_vetoes}, total vetoes: {total_vetoes})")

print(f"\nWall time: {elapsed:.1f}s")
print(f"Run ID: {run_id}")

# Pass/fail summary
has_multitf_trades = sum(1 for tf in tfs if tf_counts.get(tf, 0) > 0) >= 2
print(f"\n{'PASS' if has_multitf_trades else 'FAIL'}: Multi-TF trades present "
      f"({sum(1 for tf in tfs if tf_counts.get(tf, 0) > 0)}/4 TFs traded)")
print(f"{'PASS' if cd_pct < 50 else 'FAIL'}: Cooldown vetoes < 50% of total vetoes ({cd_pct:.1f}%)")
print(f"{'PASS' if pf > 0.5 else 'FAIL'}: PF > 0.50 ({pf:.4f})")

sys.stdout.flush()
