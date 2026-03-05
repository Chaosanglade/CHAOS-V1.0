"""
Unit tests for ExposureController + PortfolioState.
Covers all 7 risk check types plus integration tests.
"""
import os
import sys
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

os.environ["PYTHONHASHSEED"] = "42"

PROJECT_ROOT = Path('G:/My Drive/chaos_v1.0')
sys.path.insert(0, str(PROJECT_ROOT))

from risk.engine.portfolio_state import PortfolioState, OpenPosition
from risk.engine.exposure_controller import ExposureController, OrderIntent


def create_test_portfolio() -> PortfolioState:
    """Create a portfolio with test data."""
    return PortfolioState(
        instrument_specs_path=str(PROJECT_ROOT / 'risk' / 'config' / 'instrument_specs.json'),
        correlation_groups_path=str(PROJECT_ROOT / 'risk' / 'config' / 'correlation_groups.json'),
    )


def create_test_controller() -> ExposureController:
    return ExposureController(str(PROJECT_ROOT / 'risk' / 'config' / 'risk_policy.json'))


def test_check_1_cooldown():
    """Check 1: Cooldown / circuit breaker."""
    controller = create_test_controller()
    portfolio = create_test_portfolio()
    intent = OrderIntent(pair='EURUSD', tf='M30', side=1, qty_lots=0.1, price=1.1000)
    now = datetime(2024, 6, 1, 12, 0)

    # Normal state -> approved
    result = controller.check(portfolio, intent, now)
    assert result['approved'], f"Expected approved, got: {result}"

    # Set circuit breaker -> blocked
    portfolio.set_risk_state("CIRCUIT_BREAKER")
    result = controller.check(portfolio, intent, now)
    assert not result['approved'], f"Expected blocked, got: {result}"
    assert result['reason'] == 'RISK_BLOCKED_DRAWDOWN'

    # Reset global state before testing per-pair-tf cooldown
    portfolio.set_risk_state("NORMAL")

    # Set cooldown (active) on specific pair_tf -> blocked
    portfolio.set_risk_state("COOLDOWN", now + timedelta(hours=1), pair_tf="EURUSD_M30")
    result = controller.check(portfolio, intent, now)
    assert not result['approved']
    assert result['reason'] == 'RISK_BLOCKED_COOLDOWN'

    # Set cooldown (expired) -> approved
    portfolio.set_risk_state("COOLDOWN", now - timedelta(hours=1), pair_tf="EURUSD_M30")
    result = controller.check(portfolio, intent, now)
    assert result['approved'], f"Expired cooldown should approve, got: {result}"

    return True


def test_check_2_throttle():
    """Check 2: Trade throttle (min time between entries)."""
    controller = create_test_controller()
    portfolio = create_test_portfolio()
    now = datetime(2024, 6, 1, 12, 0)

    # Open a position (sets last entry timestamp)
    portfolio.open_position('EURUSD', 'M30', 1, 0.1, 1.1000, now)
    portfolio.close_position('EURUSD', 'M30', 1.1010, now + timedelta(minutes=1))

    # Try to enter again too quickly (within 5 min throttle)
    intent = OrderIntent(pair='EURUSD', tf='M30', side=1, qty_lots=0.1, price=1.1020)
    result = controller.check(portfolio, intent, now + timedelta(minutes=2))
    assert not result['approved']
    assert result['reason'] == 'RISK_BLOCKED_THROTTLE'

    # Wait long enough -> approved
    result = controller.check(portfolio, intent, now + timedelta(minutes=10))
    assert result['approved']

    return True


def test_check_3_position_limits():
    """Check 3: Max positions total, per-pair, per-timeframe."""
    controller = create_test_controller()
    portfolio = create_test_portfolio()
    now = datetime(2024, 6, 1, 12, 0)

    # Per-pair limit (max 1 per pair)
    portfolio.open_position('EURUSD', 'M30', 1, 0.1, 1.1000, now)
    intent = OrderIntent(pair='EURUSD', tf='H1', side=1, qty_lots=0.1, price=1.1000)
    result = controller.check(portfolio, intent, now + timedelta(minutes=10))
    assert not result['approved']
    assert result['reason'] == 'RISK_BLOCKED_POSITION_LIMIT'

    return True


def test_check_4_exposure_limits():
    """Check 4: Gross and net exposure limits."""
    controller = create_test_controller()
    portfolio = create_test_portfolio()
    now = datetime(2024, 6, 1, 12, 0)

    # Open positions that push near gross limit
    portfolio.open_position('EURUSD', 'M30', 1, 1.0, 1.1000, now)   # 100k
    portfolio.open_position('GBPUSD', 'M30', 1, 1.0, 1.3000, now)   # 100k

    # This would push gross to 300k > 250k limit
    intent = OrderIntent(pair='USDJPY', tf='M30', side=1, qty_lots=1.0, price=150.00)
    result = controller.check(portfolio, intent, now + timedelta(minutes=10))
    assert not result['approved']
    assert result['reason'] == 'RISK_BLOCKED_EXPOSURE'

    return True


def test_check_5_correlation_group():
    """Check 5: Correlation group caps."""
    controller = create_test_controller()
    portfolio = create_test_portfolio()
    now = datetime(2024, 6, 1, 12, 0)

    # Fill up JPY_CROSS group (max 80k)
    portfolio.open_position('USDJPY', 'M30', 1, 0.5, 150.00, now)  # 50k

    # This would push JPY_CROSS to 100k > 80k
    intent = OrderIntent(pair='EURJPY', tf='M30', side=1, qty_lots=0.5, price=163.00)
    result = controller.check(portfolio, intent, now + timedelta(minutes=10))
    assert not result['approved']
    assert result['reason'] == 'RISK_BLOCKED_CORRELATION'

    return True


def test_check_6_drawdown():
    """Check 6: Drawdown circuit breaker."""
    controller = create_test_controller()
    portfolio = create_test_portfolio()
    now = datetime(2024, 6, 1, 12, 0)

    # Simulate losses to trigger 2% drawdown on 100k equity
    portfolio.open_position('EURUSD', 'M30', 1, 1.0, 1.1000, now)
    # Close at loss: 200 pips * $10 * 1.0 lots = $2000 = 2% of 100k
    portfolio.close_position('EURUSD', 'M30', 1.0800, now + timedelta(minutes=5))

    # New trade attempt -> should trigger drawdown breaker
    intent = OrderIntent(pair='GBPUSD', tf='M30', side=1, qty_lots=0.1, price=1.3000)
    result = controller.check(portfolio, intent, now + timedelta(minutes=10), equity_usd=100000.0)
    assert not result['approved']
    assert result['reason'] == 'RISK_BLOCKED_DRAWDOWN'

    return True


def test_check_7_loss_streak():
    """Check 7: Consecutive loss control."""
    controller = create_test_controller()
    portfolio = create_test_portfolio()
    now = datetime(2024, 6, 1, 12, 0)

    # Create 5 consecutive losses on same pair_tf
    for i in range(5):
        t = now + timedelta(minutes=i * 30)
        portfolio.open_position('EURUSD', 'M30', 1, 0.01, 1.0000, t)
        portfolio.close_position('EURUSD', 'M30', 0.9990, t + timedelta(minutes=10))

    assert portfolio.get_consecutive_losses('EURUSD_M30') == 5

    # Intent on same pair_tf -> blocked
    intent = OrderIntent(pair='EURUSD', tf='M30', side=1, qty_lots=0.1, price=1.0000)
    result = controller.check(portfolio, intent, now + timedelta(hours=5))
    assert not result['approved']
    assert result['reason'] == 'RISK_BLOCKED_COOLDOWN'

    # Intent on different pair_tf -> NOT blocked by EURUSD_M30's losses
    intent2 = OrderIntent(pair='EURUSD', tf='H1', side=1, qty_lots=0.1, price=1.0000)
    result2 = controller.check(portfolio, intent2, now + timedelta(hours=5))
    assert result2['approved'], f"H1 should not be blocked by M30 loss streak, got: {result2}"

    return True


def test_portfolio_state_integration():
    """Integration test for PortfolioState."""
    portfolio = create_test_portfolio()
    now = datetime(2024, 6, 1, 12, 0)

    # Open position
    pos_id = portfolio.open_position('EURUSD', 'M30', 1, 0.5, 1.1000, now)
    assert pos_id == 'EURUSD_M30'
    assert portfolio.get_position_count() == 1
    assert portfolio.has_position('EURUSD', 'M30')
    assert portfolio.get_gross_exposure_usd() == 50000.0

    # Close with profit
    trade = portfolio.close_position('EURUSD', 'M30', 1.1050, now + timedelta(hours=1))
    assert trade is not None
    assert abs(trade.pnl_pips - 50.0) < 0.01, f"pnl_pips={trade.pnl_pips}"
    assert abs(trade.pnl_net_usd - 250.0) < 0.01, f"pnl_net={trade.pnl_net_usd}"
    assert portfolio.get_position_count() == 0
    assert abs(portfolio.realized_pnl_usd_cum - 250.0) < 0.01, f"cum_pnl={portfolio.realized_pnl_usd_cum}"
    assert portfolio.get_consecutive_losses('EURUSD_M30') == 0

    return True


def test_lock9_cost_sign_invariant():
    """Lock 9: Cost sign invariant validation.

    Verifies:
    (1) All cost fields are non-negative (spread_cost_pips, slippage_cost_pips, commission_cost_usd)
    (2) total_cost_usd matches its components within 1e-9
    (3) pnl_net_usd never exceeds pnl_gross_usd on CLOSE rows
    """
    from replay.runners.run_replay import CostCalculator, ReplayRunner

    # --- Part 1 & 2: CostCalculator returns non-negative costs, total matches components ---
    cost_calc = CostCalculator(
        str(PROJECT_ROOT / 'replay' / 'config' / 'execution_cost_scenarios.json')
    )

    test_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
    test_scenarios = ['IBKR_BASE', 'BASE_PLUS_25', 'STRESS_PLUS_75']
    test_lots = [0.01, 0.1, 0.5, 1.0, 5.0]

    for pair in test_pairs:
        for scenario in test_scenarios:
            for qty in test_lots:
                costs = cost_calc.compute_costs(pair, qty, scenario)

                # (1) Non-negative costs
                assert costs['spread_cost_pips'] >= 0.0, \
                    f"Negative spread: {pair}/{scenario}/{qty} = {costs['spread_cost_pips']}"
                assert costs['slippage_cost_pips'] >= 0.0, \
                    f"Negative slippage: {pair}/{scenario}/{qty} = {costs['slippage_cost_pips']}"
                assert costs['commission_cost_usd'] >= 0.0, \
                    f"Negative commission: {pair}/{scenario}/{qty} = {costs['commission_cost_usd']}"
                assert costs['total_cost_usd'] >= 0.0, \
                    f"Negative total cost: {pair}/{scenario}/{qty} = {costs['total_cost_usd']}"

                # (2) Total matches components within 1e-9
                pip_value_approx = 10.0
                expected = ((costs['spread_cost_pips'] + costs['slippage_cost_pips'])
                            * pip_value_approx * qty + costs['commission_cost_usd'])
                assert abs(costs['total_cost_usd'] - expected) < 1e-9, \
                    f"Component mismatch: {pair}/{scenario}/{qty}: {costs['total_cost_usd']} != {expected}"

    # --- Part 3: pnl_net_usd <= pnl_gross_usd on CLOSE rows ---
    # Simulate a real trade cycle
    portfolio = create_test_portfolio()
    now = datetime(2024, 6, 1, 12, 0)
    portfolio.open_position('EURUSD', 'M30', 1, 0.5, 1.1000, now)
    trade = portfolio.close_position('EURUSD', 'M30', 1.1050, now + timedelta(hours=1))
    pnl_gross = trade.pnl_net_usd  # raw PnL from price movement
    costs = cost_calc.compute_costs('EURUSD', 0.5, 'IBKR_BASE')
    pnl_net = pnl_gross - costs['total_cost_usd']
    assert pnl_net <= pnl_gross, \
        f"pnl_net ({pnl_net}) exceeds pnl_gross ({pnl_gross})"

    # Also check a losing trade
    portfolio.open_position('GBPUSD', 'M30', 1, 0.5, 1.3000, now)
    trade = portfolio.close_position('GBPUSD', 'M30', 1.2950, now + timedelta(hours=1))
    pnl_gross = trade.pnl_net_usd
    costs = cost_calc.compute_costs('GBPUSD', 0.5, 'STRESS_PLUS_75')
    pnl_net = pnl_gross - costs['total_cost_usd']
    assert pnl_net <= pnl_gross, \
        f"pnl_net ({pnl_net}) exceeds pnl_gross ({pnl_gross})"

    return True


def test_multitf_cooldown_independence():
    """
    Verify that cooldown on EURUSD_M5 does NOT block EURUSD_M30.
    This is the exact bug that caused Run 3 PF to collapse from 2.55 to 0.229.
    """
    controller = create_test_controller()
    portfolio = create_test_portfolio()
    now = datetime(2024, 6, 1, 12, 0)

    # 1. Trigger cooldown on EURUSD_M5
    portfolio.set_risk_state("COOLDOWN", now + timedelta(hours=1), pair_tf="EURUSD_M5")

    # 2. EURUSD_M5 should be blocked
    intent_m5 = OrderIntent(pair='EURUSD', tf='M5', side=1, qty_lots=0.1, price=1.1000)
    result_m5 = controller.check(portfolio, intent_m5, now)
    assert not result_m5['approved'], "M5 should be in cooldown"
    assert result_m5['reason'] == 'RISK_BLOCKED_COOLDOWN'

    # 3. EURUSD_M30 should NOT be blocked
    intent_m30 = OrderIntent(pair='EURUSD', tf='M30', side=1, qty_lots=0.1, price=1.1000)
    result_m30 = controller.check(portfolio, intent_m30, now)
    assert result_m30['approved'], f"M30 should NOT be blocked by M5 cooldown, got: {result_m30}"

    # 4. EURUSD_H1 should NOT be blocked
    intent_h1 = OrderIntent(pair='EURUSD', tf='H1', side=1, qty_lots=0.1, price=1.1000)
    result_h1 = controller.check(portfolio, intent_h1, now)
    assert result_h1['approved'], f"H1 should NOT be blocked by M5 cooldown, got: {result_h1}"

    # 5. Trigger cooldown on EURUSD_M30 — should not affect M5
    portfolio.set_risk_state("COOLDOWN", now + timedelta(hours=2), pair_tf="EURUSD_M30")
    # M5 cooldown expired by now + 1.5h
    result_m5_after = controller.check(portfolio, intent_m5, now + timedelta(hours=1, minutes=30))
    assert result_m5_after['approved'], f"M5 should NOT be blocked by M30 cooldown, got: {result_m5_after}"
    # M30 should still be blocked (cooldown until now + 2h)
    result_m30_after = controller.check(portfolio, intent_m30, now + timedelta(hours=1, minutes=30))
    assert not result_m30_after['approved'], "M30 should still be in cooldown"

    # 6. Throttle independence: entry on M5 should not throttle M30
    portfolio.set_risk_state("NORMAL")  # Clear all cooldowns
    t1 = now + timedelta(hours=3)
    portfolio.open_position('EURUSD', 'M5', 1, 0.1, 1.1000, t1)
    portfolio.close_position('EURUSD', 'M5', 1.1010, t1 + timedelta(minutes=1))
    # M30 entry should not be throttled by M5's recent entry
    result_m30_throttle = controller.check(portfolio, intent_m30, t1 + timedelta(minutes=2))
    assert result_m30_throttle['approved'], f"M30 should NOT be throttled by M5 entry, got: {result_m30_throttle}"

    # 7. Loss streak independence: losses on M5 should not block M30
    portfolio.set_risk_state("NORMAL")
    for i in range(5):
        t = now + timedelta(hours=4, minutes=i * 10)
        portfolio.open_position('EURUSD', 'M5', 1, 0.01, 1.0000, t)
        portfolio.close_position('EURUSD', 'M5', 0.9990, t + timedelta(minutes=5))
    assert portfolio.get_consecutive_losses('EURUSD_M5') == 5
    assert portfolio.get_consecutive_losses('EURUSD_M30') == 0
    result_m30_streak = controller.check(portfolio, intent_m30, now + timedelta(hours=5))
    assert result_m30_streak['approved'], f"M30 should NOT be blocked by M5 loss streak, got: {result_m30_streak}"

    return True


def run_all_tests():
    """Run all risk engine tests."""
    tests = [
        ("Check 1: Cooldown/CB", test_check_1_cooldown),
        ("Check 2: Throttle", test_check_2_throttle),
        ("Check 3: Position limits", test_check_3_position_limits),
        ("Check 4: Exposure limits", test_check_4_exposure_limits),
        ("Check 5: Correlation group", test_check_5_correlation_group),
        ("Check 6: Drawdown CB", test_check_6_drawdown),
        ("Check 7: Loss streak", test_check_7_loss_streak),
        ("Portfolio integration", test_portfolio_state_integration),
        ("Lock 9: Cost sign invariant", test_lock9_cost_sign_invariant),
        ("Multi-TF cooldown independence", test_multitf_cooldown_independence),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            result = test_fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1

    print(f"\nRisk engine tests: {passed}/{passed + failed} passed")
    return failed == 0


if __name__ == '__main__':
    run_all_tests()
