"""Tests for ops/build_edge_baselines_by_regime.py"""
import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ops.build_edge_baselines_by_regime import (
    compute_effective_regime,
    compute_block_metrics,
    build_baselines,
    CONFIDENCE_THRESHOLD,
    ELIGIBLE_TFS,
    ELIGIBLE_REGIMES,
)


def test_effective_regime_mapping():
    """Effective regime degrades to REGIME_2 when confidence < 0.60."""
    # High confidence -> keep original
    assert compute_effective_regime(0, 0.80) == 0
    assert compute_effective_regime(1, 0.75) == 1
    assert compute_effective_regime(2, 0.60) == 2

    # Low confidence -> degrade to 2
    assert compute_effective_regime(0, 0.59) == 2
    assert compute_effective_regime(1, 0.30) == 2
    assert compute_effective_regime(0, 0.00) == 2

    # Already regime 2 with low confidence -> stays 2
    assert compute_effective_regime(2, 0.50) == 2

    # Regime 3 with high confidence -> stays 3 (filtered out later)
    assert compute_effective_regime(3, 0.90) == 3


def test_block_metrics_computation():
    """Block metrics correctly compute PF, avg_r, win_rate, max_dd."""
    trades = pd.DataFrame({
        'pnl_net_usd': [10, -5, 8, -3, 12, -4, 6, -2, 15, -6],
        'spread_cost_pips': [0.5] * 10,
    })

    metrics = compute_block_metrics(trades)

    assert metrics is not None
    assert metrics['trade_count'] == 10
    assert metrics['win_rate'] == 0.6  # 6 winners / 10 trades

    # PF = sum(winners) / sum(|losers|) = 51/20 = 2.55
    assert abs(metrics['pf'] - 2.55) < 0.01
    assert metrics['avg_r'] == round(trades['pnl_net_usd'].mean(), 4)


def test_block_metrics_empty():
    """Empty trades returns None."""
    trades = pd.DataFrame({'pnl_net_usd': [], 'spread_cost_pips': []})
    assert compute_block_metrics(trades) is None


def test_baselines_structure():
    """Built baselines have correct structure with metadata and blocks."""
    project_root = Path(__file__).resolve().parents[2]
    baselines = build_baselines(project_root)

    assert 'metadata' in baselines
    assert 'blocks' in baselines
    assert baselines['metadata']['baseline_version'] == '1.0.0'
    assert baselines['metadata']['confidence_threshold'] == CONFIDENCE_THRESHOLD
    assert baselines['metadata']['eligible_tfs'] == ELIGIBLE_TFS
    assert baselines['metadata']['eligible_regimes'] == ELIGIBLE_REGIMES
    assert 'schema_hash' in baselines['metadata']
    assert 'total_blocks' in baselines['metadata']

    # Each block should have pair, tf, effective_regime, scenarios
    for key, block in baselines['blocks'].items():
        assert 'pair' in block
        assert 'tf' in block
        assert block['tf'] in ELIGIBLE_TFS
        assert 'effective_regime' in block
        assert block['effective_regime'] in ['REGIME_0', 'REGIME_1', 'REGIME_2']
        assert 'scenarios' in block

        for scenario_name, metrics in block['scenarios'].items():
            assert 'pf' in metrics
            assert 'avg_r' in metrics
            assert 'win_rate' in metrics
            assert 'max_dd' in metrics
            assert 'trade_count' in metrics
            assert metrics['trade_count'] > 0


def test_baselines_created_for_all_scenarios():
    """Baselines should have entries for both IBKR_BASE and STRESS_PLUS_75."""
    project_root = Path(__file__).resolve().parents[2]
    baselines = build_baselines(project_root)

    scenarios_seen = set()
    for block in baselines['blocks'].values():
        for sc in block['scenarios']:
            scenarios_seen.add(sc)

    assert 'IBKR_BASE' in scenarios_seen
    assert 'STRESS_PLUS_75' in scenarios_seen


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
