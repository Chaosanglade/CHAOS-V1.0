"""Tests for ops/retrain_trigger.py"""
import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ops.retrain_trigger import (
    compute_effective_regime,
    evaluate_block,
    ORANGE_PF_RATIO, ORANGE_AVG_R_RATIO,
    RED_PF_RATIO, RED_AVG_R_RATIO,
    RED_PERSIST_TRADES, RED_SAMPLE_FLOOR,
    W_FAST, W_SLOW,
)


def _make_trades(pnls, spread=0.5, start_date='2025-01-01'):
    """Helper to create a trades DataFrame."""
    start = pd.Timestamp(start_date, tz='UTC')
    return pd.DataFrame({
        'pnl_net_usd': pnls,
        'spread_cost_pips': [spread] * len(pnls),
        'exit_ts': [start + timedelta(hours=i) for i in range(len(pnls))],
    })


def test_green_within_bands():
    """Trades matching baseline should return GREEN."""
    # Baseline PF = 2.0 (60 win, 30 loss -> PF 2.0)
    baseline = {'scenarios': {'IBKR_BASE': {'pf': 2.0, 'avg_r': 3.0, 'spread_mean': 0.5}}}
    pnls = [10, -5] * 30  # PF = 2.0, avg_r = 2.5
    trades = _make_trades(pnls)

    result = evaluate_block(trades, baseline, 'IBKR_BASE')
    assert result['level'] == 'GREEN'


def test_orange_pf_decay():
    """PF below 75% of baseline triggers ORANGE."""
    baseline = {'scenarios': {'IBKR_BASE': {'pf': 4.0, 'avg_r': 5.0, 'spread_mean': 0.5}}}
    # Current PF ~ 1.0 (well below 0.75*4.0=3.0)
    pnls = [5, -5] * 30
    trades = _make_trades(pnls)

    result = evaluate_block(trades, baseline, 'IBKR_BASE')
    assert result['level'] in ['ORANGE', 'RED']  # At least ORANGE


def test_red_requires_persistence():
    """RED requires sample floor AND persistence (trades or days)."""
    baseline = {'scenarios': {'IBKR_BASE': {'pf': 4.0, 'avg_r': 5.0, 'spread_mean': 0.5}}}

    # Few trades — should be ORANGE not RED (below sample floor)
    pnls = [1, -5] * 20  # 40 trades, PF ~0.2 (severe), but too few for RED
    trades = _make_trades(pnls)

    result = evaluate_block(trades, baseline, 'IBKR_BASE')
    # With only 40 trades, below RED_SAMPLE_FLOOR of 80
    assert result['level'] == 'ORANGE'


def test_red_with_sufficient_trades():
    """RED fires when persistence threshold met with enough trades."""
    baseline = {'scenarios': {'IBKR_BASE': {'pf': 4.0, 'avg_r': 5.0, 'spread_mean': 0.5}}}

    # 200 trades (>150 persist + >80 sample floor), PF very low
    pnls = [1, -5] * 100  # 200 trades, PF ~0.2
    trades = _make_trades(pnls)

    result = evaluate_block(trades, baseline, 'IBKR_BASE')
    assert result['level'] == 'RED'


def test_effective_regime_mapping():
    """Regime mapping for trigger consistency."""
    assert compute_effective_regime(0, 0.80) == 0
    assert compute_effective_regime(1, 0.50) == 2  # Low confidence -> REGIME_2
    assert compute_effective_regime(3, 0.90) == 3  # Hostile stays 3 (skipped by trigger)


def test_no_baseline_returns_green():
    """Missing baseline scenario returns GREEN (no trigger possible)."""
    baseline = {'scenarios': {}}
    trades = _make_trades([10, -5] * 30)

    result = evaluate_block(trades, baseline, 'IBKR_BASE')
    assert result['level'] == 'GREEN'


def test_only_h1_m30_eligible():
    """Trigger should only process eligible TFs from baselines."""
    # This is enforced at the run_triggers level; unit test confirms the constants
    from ops.retrain_trigger import ELIGIBLE_TFS
    assert set(ELIGIBLE_TFS) == {'H1', 'M30'}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
