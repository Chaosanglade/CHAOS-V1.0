"""Tests for ops/canary_deployer.py"""
import json
import pytest
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ops.canary_deployer import (
    CanaryDeployer,
    DEPLOYMENT_STAGES,
    ROLLBACK_PF_RATIO,
    ROLLBACK_DD_RATIO,
    ROLLBACK_MIN_TRADES,
)


@pytest.fixture
def deployer(tmp_path):
    """Create a deployer with temp paths."""
    registry_path = tmp_path / 'model_registry.json'
    audit_dir = tmp_path / 'deploy_audit'
    return CanaryDeployer(registry_path, audit_dir)


def test_shadow_deployment(deployer):
    """Deploy to SHADOW stage."""
    result = deployer.deploy_shadow(
        block_key='EURUSD|H1|REGIME_0',
        candidate_id='cand_001',
        candidate_spec={'family': 'lgb', 'artifact_dir': '/tmp/test'},
        champion_metrics={'pf': 5.0, 'max_dd': 50.0}
    )

    assert result['stage'] == 'SHADOW'
    assert result['candidate_id'] == 'cand_001'
    assert result['rollback'] is False


def test_stage_advancement(deployer):
    """Advance through all deployment stages."""
    deployer.deploy_shadow('EURUSD|H1|REGIME_0', 'cand_001', {'family': 'lgb'})

    expected_stages = ['CANARY_10', 'CANARY_50', 'PROMOTE_100']
    for expected in expected_stages:
        result = deployer.advance_stage('EURUSD|H1|REGIME_0')
        assert result['stage'] == expected

    # Already at PROMOTE_100, should stay
    result = deployer.advance_stage('EURUSD|H1|REGIME_0')
    assert result['stage'] == 'PROMOTE_100'


def test_rollback_on_pf_decay(deployer):
    """Rollback when PF < 0.8x champion after 50 trades."""
    deployer.deploy_shadow(
        'EURUSD|H1|REGIME_0', 'cand_001', {'family': 'lgb'},
        champion_metrics={'pf': 5.0, 'max_dd': 50.0}
    )
    deployer.advance_stage('EURUSD|H1|REGIME_0')  # -> CANARY_10

    # PF 3.5 < 0.8*5.0=4.0 => rollback
    result = deployer.check_rollback('EURUSD|H1|REGIME_0',
                                     current_pf=3.5, current_dd=40.0,
                                     trade_count=60)
    assert result['rollback'] is True
    assert 'PF' in result['reason']


def test_rollback_on_dd_spike(deployer):
    """Rollback when DD > 1.2x champion after 50 trades."""
    deployer.deploy_shadow(
        'EURUSD|H1|REGIME_0', 'cand_001', {'family': 'lgb'},
        champion_metrics={'pf': 5.0, 'max_dd': 50.0}
    )

    # DD 65 > 1.2*50=60 => rollback
    result = deployer.check_rollback('EURUSD|H1|REGIME_0',
                                     current_pf=5.0, current_dd=65.0,
                                     trade_count=60)
    assert result['rollback'] is True
    assert 'DD' in result['reason']


def test_no_rollback_before_min_trades(deployer):
    """No rollback before minimum trade count."""
    deployer.deploy_shadow(
        'EURUSD|H1|REGIME_0', 'cand_001', {'family': 'lgb'},
        champion_metrics={'pf': 5.0, 'max_dd': 50.0}
    )

    # Terrible metrics but only 30 trades (< 50 minimum)
    result = deployer.check_rollback('EURUSD|H1|REGIME_0',
                                     current_pf=0.5, current_dd=200.0,
                                     trade_count=30)
    assert result['rollback'] is False


def test_promote_to_champion(deployer):
    """Full lifecycle: shadow -> canary -> promote."""
    deployer.deploy_shadow('EURUSD|H1|REGIME_0', 'cand_001',
                          {'family': 'lgb', 'artifact_dir': '/tmp/test'})
    deployer.advance_stage('EURUSD|H1|REGIME_0')  # CANARY_10
    deployer.advance_stage('EURUSD|H1|REGIME_0')  # CANARY_50
    deployer.advance_stage('EURUSD|H1|REGIME_0')  # PROMOTE_100

    champion = deployer.promote_to_champion('EURUSD|H1|REGIME_0')
    assert champion['champion_id'] == 'cand_001'
    assert champion['family'] == 'lgb'

    # Deployment should be cleaned up
    assert 'EURUSD|H1|REGIME_0' not in deployer.registry['deployments']


def test_audit_records_created(deployer):
    """Audit records written for each action."""
    deployer.deploy_shadow('EURUSD|H1|REGIME_0', 'cand_001', {'family': 'lgb'})

    audit_files = list(deployer.audit_dir.glob('*.json'))
    assert len(audit_files) >= 1

    with open(audit_files[0]) as f:
        audit = json.load(f)
    assert audit['action'] == 'DEPLOY_SHADOW'
    assert audit['block_key'] == 'EURUSD|H1|REGIME_0'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
