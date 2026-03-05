"""Tests for ops/promotion_gate.py"""
import json
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ops.promotion_gate import evaluate_candidate, select_winner, GATES


def _make_scenario_metrics(ibkr_pf=2.0, stress_pf=1.5, vol_pf=1.1,
                            trades=300, err_rate=0, violations=0, turnover=1.0):
    """Helper to create scenario metrics."""
    return {
        'IBKR_BASE': {
            'pf': ibkr_pf, 'avg_r': 3.0, 'max_dd': 50.0, 'trade_count': trades,
            'inference_error_rate': err_rate, 'schema_violations': violations,
            'turnover_multiplier': turnover
        },
        'STRESS_PLUS_75': {
            'pf': stress_pf, 'avg_r': 2.0, 'max_dd': 70.0, 'trade_count': trades,
            'inference_error_rate': err_rate, 'schema_violations': violations,
            'turnover_multiplier': turnover
        },
        'VOLATILITY_SPIKE': {
            'pf': vol_pf, 'avg_r': 1.0, 'max_dd': 90.0, 'trade_count': trades,
            'inference_error_rate': err_rate, 'schema_violations': violations,
            'turnover_multiplier': turnover
        },
    }


def test_passing_candidate():
    """Candidate meeting all gates should PASS."""
    candidate = {'candidate_id': 'test_001'}
    champion = {'stress_pf': 1.0, 'stress_avg_r': 1.0, 'vol_spike_max_dd': 100.0}
    metrics = _make_scenario_metrics(ibkr_pf=2.0, stress_pf=1.5, vol_pf=1.1)

    result = evaluate_candidate(candidate, champion, metrics)
    assert result['decision'] == 'PASS'
    assert len(result['gates_failed']) == 0


def test_failing_ibkr_pf():
    """Candidate with IBKR_BASE PF < 1.30 should FAIL."""
    candidate = {'candidate_id': 'test_002'}
    metrics = _make_scenario_metrics(ibkr_pf=1.10)

    result = evaluate_candidate(candidate, None, metrics)
    assert result['decision'] == 'FAIL'
    assert any('IBKR_BASE PF' in g for g in result['gates_failed'])


def test_failing_stress_pf():
    """Candidate with STRESS_PLUS_75 PF < 1.15 should FAIL."""
    candidate = {'candidate_id': 'test_003'}
    metrics = _make_scenario_metrics(stress_pf=1.00)

    result = evaluate_candidate(candidate, None, metrics)
    assert result['decision'] == 'FAIL'
    assert any('STRESS_PLUS_75 PF' in g for g in result['gates_failed'])


def test_must_beat_champion():
    """Candidate must beat champion STRESS PF by >= 5%."""
    candidate = {'candidate_id': 'test_004'}
    champion = {'stress_pf': 1.50, 'stress_avg_r': 2.50, 'vol_spike_max_dd': 100.0}
    # stress_pf 1.55 < 1.50 * 1.05 = 1.575 (does NOT beat by 5%)
    metrics = _make_scenario_metrics(stress_pf=1.55, ibkr_pf=2.0, vol_pf=1.1)
    # Also need avg_r to fail: 2.0 < 2.50*1.05=2.625
    result = evaluate_candidate(candidate, champion, metrics)
    assert result['decision'] == 'FAIL'
    assert any('Does not beat champion' in g for g in result['gates_failed'])


def test_min_trades_gate():
    """Candidate with too few trades should FAIL."""
    candidate = {'candidate_id': 'test_005'}
    metrics = _make_scenario_metrics(trades=50)  # 50*3=150 < 200

    result = evaluate_candidate(candidate, None, metrics)
    assert result['decision'] == 'FAIL'
    assert any('Holdout trades' in g for g in result['gates_failed'])


def test_execution_sanity_schema_violations():
    """Schema violations should cause FAIL."""
    candidate = {'candidate_id': 'test_006'}
    metrics = _make_scenario_metrics(violations=1)

    result = evaluate_candidate(candidate, None, metrics)
    assert result['decision'] == 'FAIL'
    assert any('schema_violations' in g for g in result['gates_failed'])


def test_deterministic_winner_selection():
    """Winner selection is deterministic: max STRESS PF, tie-break by lower DD."""
    evals = [
        {'decision': 'PASS', 'candidate_id': 'a', 'stress_pf': 1.50, 'stress_max_dd': 80, 'stress_avg_r': 2.0, 'total_holdout_trades': 300},
        {'decision': 'PASS', 'candidate_id': 'b', 'stress_pf': 1.80, 'stress_max_dd': 60, 'stress_avg_r': 2.5, 'total_holdout_trades': 300},
        {'decision': 'FAIL', 'candidate_id': 'c', 'stress_pf': 2.00, 'stress_max_dd': 50, 'stress_avg_r': 3.0, 'total_holdout_trades': 300},
    ]

    winner = select_winner(evals)
    assert winner is not None
    assert winner['candidate_id'] == 'b'  # Highest PF among PASS

    # Run again -> same result (deterministic)
    winner2 = select_winner(evals)
    assert winner2['candidate_id'] == winner['candidate_id']


def test_no_winner_when_all_fail():
    """No winner when all candidates fail."""
    evals = [
        {'decision': 'FAIL', 'candidate_id': 'x', 'stress_pf': 0.5, 'stress_max_dd': 100, 'stress_avg_r': 0, 'total_holdout_trades': 100},
    ]
    assert select_winner(evals) is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
