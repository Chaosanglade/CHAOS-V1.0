"""
Unit tests for TF Role Gating (CONFIRM_ONLY / EXECUTE).
Tests can_execute_trade, get_confirmation_signal, and apply_confirmation_to_threshold.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path('G:/My Drive/chaos_v1.0')
sys.path.insert(0, str(PROJECT_ROOT))

from replay.runners.run_replay import ReplayRunner


def test_tf_role_enforcement():
    """
    Verify that CONFIRM_ONLY TFs cannot open positions,
    and EXECUTE TFs can.
    """
    # Create a minimal mock runner with tf_roles set directly
    class MockRunner:
        tf_roles = {
            'H1':  {'role': 'EXECUTE', 'can_open_positions': True},
            'M30': {'role': 'EXECUTE', 'can_open_positions': True},
            'M15': {'role': 'CONFIRM_ONLY', 'can_open_positions': False},
            'M5':  {'role': 'CONFIRM_ONLY', 'can_open_positions': False},
        }
        can_execute_trade = ReplayRunner.can_execute_trade

    runner = MockRunner()

    # EXECUTE TFs can trade
    assert runner.can_execute_trade('EURUSD', 'H1', 1) == True
    assert runner.can_execute_trade('EURUSD', 'M30', -1) == True

    # CONFIRM_ONLY TFs cannot trade
    assert runner.can_execute_trade('EURUSD', 'M15', 1) == False
    assert runner.can_execute_trade('EURUSD', 'M5', -1) == False

    # Unknown TF defaults to EXECUTE
    assert runner.can_execute_trade('EURUSD', 'H4', 1) == True

    print("PASS: TF role enforcement verified")


def test_confirmation_logic():
    """
    Verify confirmation signals from CONFIRM_ONLY TFs
    properly adjust agreement thresholds.
    """
    apply = ReplayRunner.apply_confirmation_to_threshold

    # Confirmers align with trade direction -> loosen
    adjusted = apply(0.60, 1, 1, 0.8)
    assert adjusted < 0.60, f"Should loosen: {adjusted}"
    assert abs(adjusted - 0.55) < 1e-9, f"Expected ~0.55, got {adjusted}"

    # Confirmers conflict -> tighten
    adjusted = apply(0.60, 1, -1, 0.8)
    assert adjusted > 0.60, f"Should tighten: {adjusted}"
    assert abs(adjusted - 0.70) < 1e-9, f"Expected ~0.70, got {adjusted}"

    # No confirmation -> no change
    adjusted = apply(0.60, 1, 0, 0.0)
    assert adjusted == 0.60, f"Should not change: {adjusted}"

    # Low strength -> no change
    adjusted = apply(0.60, 1, 1, 0.3)
    assert adjusted == 0.60, f"Low strength should not change: {adjusted}"

    # Edge: loosen can't go below 0.40
    adjusted = apply(0.42, 1, 1, 1.0)
    assert adjusted == 0.40, f"Should clamp at 0.40: {adjusted}"

    # Edge: tighten can't go above 0.90
    adjusted = apply(0.85, 1, -1, 1.0)
    assert adjusted == 0.90, f"Should clamp at 0.90: {adjusted}"

    print("PASS: Confirmation logic verified")


def test_get_confirmation_signal():
    """
    Verify get_confirmation_signal reads from _confirm_signals correctly.
    """
    class MockRunner:
        tf_roles = {
            'H1':  {'role': 'EXECUTE', 'can_open_positions': True},
            'M30': {'role': 'EXECUTE', 'can_open_positions': True},
            'M15': {'role': 'CONFIRM_ONLY', 'can_open_positions': False},
            'M5':  {'role': 'CONFIRM_ONLY', 'can_open_positions': False},
        }
        _confirm_signals = {
            'EURUSD': {'M15': 1, 'M5': 1},  # Both confirm long
        }
        get_confirmation_signal = ReplayRunner.get_confirmation_signal

    runner = MockRunner()

    # Both confirmers agree on LONG
    direction, strength, used, details = runner.get_confirmation_signal('EURUSD')
    assert direction == 1, f"Expected +1, got {direction}"
    assert strength == 1.0, f"Expected 1.0, got {strength}"
    assert sorted(used) == ['M15', 'M5'], f"Expected M15|M5, got {used}"

    # No signals for GBPUSD
    direction, strength, used, details = runner.get_confirmation_signal('GBPUSD')
    assert direction == 0
    assert strength == 0.0
    assert used == []

    # Conflicting signals
    runner._confirm_signals = {'EURUSD': {'M15': 1, 'M5': -1}}
    direction, strength, used, details = runner.get_confirmation_signal('EURUSD')
    assert direction == 0, f"Conflict should give 0, got {direction}"

    print("PASS: get_confirmation_signal verified")


if __name__ == '__main__':
    test_tf_role_enforcement()
    test_confirmation_logic()
    test_get_confirmation_signal()
    print("\nAll TF role gating tests PASSED")
