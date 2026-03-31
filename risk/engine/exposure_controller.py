"""
CHAOS V1.0 Exposure Controller (Risk Engine v2)

Deterministic risk layer matching the Colab team spec.
Checks in order (cheapest first):
1. Cooldown / circuit breaker active?
2. Trade throttle (min time between entries per pair)?
3. Max positions (total, per-pair, per-timeframe)?
4. Gross/net exposure limits?
5. Correlation group caps?
6. Drawdown circuit breaker?
7. Consecutive loss control?

Hard rule: Risk engine can only block or reduce size. It CANNOT create trades.
"""
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger('exposure_controller')


@dataclass
class OrderIntent:
    """Proposed trade for risk check."""
    pair: str
    tf: str
    side: int  # -1 SHORT, +1 LONG
    qty_lots: float
    price: float


class ExposureController:
    """
    Deterministic risk engine. All checks return {approved: bool, reason: str}.

    Usage:
        controller = ExposureController(policy_path)
        result = controller.check(portfolio_state, order_intent, event_ts, regime_state)
        if result['approved']:
            execute()
        else:
            log(result['reason'])
    """

    def __init__(self, policy_path='G:/My Drive/chaos_v1.0/risk/config/risk_policy.json'):
        import time as _time
        for attempt in range(3):
            try:
                with open(policy_path) as f:
                    self.policy = json.load(f)
                break
            except (FileNotFoundError, OSError, PermissionError) as e:
                if attempt < 2:
                    logger.warning(f"Retry {attempt+1}/3 reading {policy_path}: {e}")
                    _time.sleep(2.0 * (attempt + 1))
                else:
                    raise
        logger.info(f"Exposure controller loaded: v{self.policy.get('version', '?')}")

    def check(self, portfolio, intent: OrderIntent, event_ts: datetime,
              regime_state: int = 1, equity_usd: float = 100000.0) -> Dict:
        """
        Run all 7 risk checks for a proposed trade.

        Returns first failing check, or 'approved' if all pass.
        """
        if intent.side == 0:
            return {'approved': True, 'reason': 'FLAT_no_trade', 'adjusted_size': None}

        # Check 1: Cooldown / circuit breaker
        result = self._check_cooldown(portfolio, intent, event_ts)
        if not result['approved']:
            return result

        # Check 2: Trade throttle
        result = self._check_throttle(portfolio, intent, event_ts)
        if not result['approved']:
            return result

        # Check 3: Position limits
        result = self._check_position_limits(portfolio, intent)
        if not result['approved']:
            return result

        # Check 4: Gross/net exposure
        result = self._check_exposure_limits(portfolio, intent, equity_usd)
        if not result['approved']:
            return result

        # Check 5: Correlation group caps
        result = self._check_group_caps(portfolio, intent, equity_usd)
        if not result['approved']:
            return result

        # Check 6: Drawdown circuit breaker
        result = self._check_drawdown(portfolio, intent, event_ts, equity_usd)
        if not result['approved']:
            return result

        # Check 7: Consecutive loss control
        result = self._check_loss_streak(portfolio, intent)
        if not result['approved']:
            return result

        return {'approved': True, 'reason': 'approved', 'adjusted_size': None}

    def _check_cooldown(self, portfolio, intent, event_ts) -> Dict:
        """Check 1: Active cooldown or circuit breaker (per pair_tf)."""
        # Global circuit breaker check
        if portfolio.risk_state == "CIRCUIT_BREAKER":
            return {
                'approved': False,
                'reason': 'RISK_BLOCKED_DRAWDOWN',
                'adjusted_size': None,
                'detail': 'Circuit breaker active. Manual review required.'
            }

        # Per (pair, tf) cooldown check
        pair_tf = f"{intent.pair}_{intent.tf}"
        cooldown_until = portfolio.cooldown_until_ts.get(pair_tf)
        if cooldown_until and event_ts < cooldown_until:
            return {
                'approved': False,
                'reason': 'RISK_BLOCKED_COOLDOWN',
                'adjusted_size': None,
                'detail': f'Cooldown until {cooldown_until} for {pair_tf}'
            }
        elif cooldown_until:
            # Cooldown expired — clear it
            portfolio.set_risk_state("NORMAL", pair_tf=pair_tf)

        return {'approved': True, 'reason': 'no_cooldown', 'adjusted_size': None}

    def _check_throttle(self, portfolio, intent, event_ts) -> Dict:
        """Check 2: Minimum time between entries per (pair, tf)."""
        throttle = self.policy.get('trade_throttle', {})
        min_minutes = throttle.get('min_minutes_between_entries_per_pair', 0)

        if min_minutes > 0:
            pair_tf = f"{intent.pair}_{intent.tf}"
            last_entry = portfolio.get_last_entry_ts(pair_tf)
            if last_entry:
                elapsed = (event_ts - last_entry).total_seconds() / 60
                if elapsed < min_minutes:
                    return {
                        'approved': False,
                        'reason': 'RISK_BLOCKED_THROTTLE',
                        'adjusted_size': None,
                        'detail': f'{pair_tf} last entry {elapsed:.1f}m ago (min {min_minutes}m)'
                    }

        return {'approved': True, 'reason': 'throttle_ok', 'adjusted_size': None}

    def _check_position_limits(self, portfolio, intent) -> Dict:
        """Check 3: Max positions total, per-pair, per-timeframe."""
        limits = self.policy['portfolio']

        # Total positions
        if portfolio.get_position_count() >= limits['max_total_open_positions']:
            return {
                'approved': False,
                'reason': 'RISK_BLOCKED_POSITION_LIMIT',
                'adjusted_size': None,
                'detail': f'{portfolio.get_position_count()} positions open (max {limits["max_total_open_positions"]})'
            }

        # Per-pair
        pair_positions = len(portfolio.get_positions_for_pair(intent.pair))
        if pair_positions >= limits['max_positions_per_pair']:
            return {
                'approved': False,
                'reason': 'RISK_BLOCKED_POSITION_LIMIT',
                'adjusted_size': None,
                'detail': f'{intent.pair} has {pair_positions} positions (max {limits["max_positions_per_pair"]})'
            }

        # Per-timeframe
        tf_positions = len(portfolio.get_positions_for_tf(intent.tf))
        if tf_positions >= limits['max_positions_per_tf']:
            return {
                'approved': False,
                'reason': 'RISK_BLOCKED_POSITION_LIMIT',
                'adjusted_size': None,
                'detail': f'{intent.tf} has {tf_positions} positions (max {limits["max_positions_per_tf"]})'
            }

        return {'approved': True, 'reason': 'position_limits_ok', 'adjusted_size': None}

    def _resolve_limit(self, policy_section, key_pct, key_usd, equity_usd) -> float:
        """Resolve a limit: use percentage of equity if set, else fixed USD, else inf."""
        pct = policy_section.get(key_pct)
        usd = policy_section.get(key_usd)
        if pct is not None and pct > 0:
            return equity_usd * pct / 100.0
        if usd is not None and usd > 0:
            return float(usd)
        return float('inf')

    def _check_exposure_limits(self, portfolio, intent, equity_usd=100000.0) -> Dict:
        """Check 4: Gross and net exposure limits (percentage-based or fixed USD)."""
        limits = self.policy['portfolio']
        proposed_notional = intent.qty_lots * 100000

        max_gross = self._resolve_limit(limits, 'max_gross_exposure_pct', 'max_gross_exposure_usd', equity_usd)
        max_net = self._resolve_limit(limits, 'max_net_exposure_pct', 'max_net_exposure_usd', equity_usd)

        # Gross exposure
        current_gross = portfolio.get_gross_exposure_usd()
        if current_gross + proposed_notional > max_gross:
            return {
                'approved': False,
                'reason': 'RISK_BLOCKED_EXPOSURE',
                'adjusted_size': None,
                'detail': f'Gross exposure {current_gross + proposed_notional:.0f} > {max_gross:.0f} ({limits.get("max_gross_exposure_pct", "?")}% of ${equity_usd:.0f})'
            }

        # Net exposure
        current_net = portfolio.get_net_exposure_usd()
        proposed_net = current_net + (intent.side * proposed_notional)
        if abs(proposed_net) > max_net:
            return {
                'approved': False,
                'reason': 'RISK_BLOCKED_EXPOSURE',
                'adjusted_size': None,
                'detail': f'Net exposure |{proposed_net:.0f}| > {max_net:.0f} ({limits.get("max_net_exposure_pct", "?")}% of ${equity_usd:.0f})'
            }

        return {'approved': True, 'reason': 'exposure_ok', 'adjusted_size': None}

    def _check_group_caps(self, portfolio, intent, equity_usd=100000.0) -> Dict:
        """Check 5: Correlation group caps (percentage-based or fixed USD)."""
        group_caps = self.policy.get('group_caps', [])
        proposed_notional = intent.qty_lots * 100000

        for cap in group_caps:
            group_name = cap['group']
            max_exposure = self._resolve_limit(cap, 'max_gross_exposure_pct', 'max_gross_exposure_usd', equity_usd)

            group_pairs = portfolio.correlation_groups.get(group_name, [])
            if intent.pair in group_pairs:
                current = portfolio.get_group_exposure(group_name)
                if current + proposed_notional > max_exposure:
                    return {
                        'approved': False,
                        'reason': 'RISK_BLOCKED_CORRELATION',
                        'adjusted_size': None,
                        'detail': f'{group_name} exposure {current + proposed_notional:.0f} > {max_exposure:.0f}'
                    }

        return {'approved': True, 'reason': 'group_caps_ok', 'adjusted_size': None}

    def _check_drawdown(self, portfolio, intent, event_ts, equity_usd) -> Dict:
        """Check 6: Drawdown circuit breaker (cooldown scoped per pair_tf)."""
        dd_config = self.policy.get('drawdown_circuit_breaker', {})
        if not dd_config.get('enabled', False):
            return {'approved': True, 'reason': 'dd_check_disabled', 'adjusted_size': None}

        max_dd_pct = dd_config['max_intraday_dd_pct']
        cooldown_minutes = dd_config['cooldown_minutes']

        today_pnl = portfolio.get_today_pnl(event_ts)
        dd_pct = abs(today_pnl) / equity_usd * 100 if equity_usd > 0 and today_pnl < 0 else 0

        if dd_pct >= max_dd_pct:
            cooldown_until = event_ts + timedelta(minutes=cooldown_minutes)
            pair_tf = f"{intent.pair}_{intent.tf}"
            portfolio.set_risk_state("COOLDOWN", cooldown_until, pair_tf=pair_tf)
            return {
                'approved': False,
                'reason': 'RISK_BLOCKED_DRAWDOWN',
                'adjusted_size': None,
                'detail': f'Intraday DD {dd_pct:.2f}% >= {max_dd_pct}%. Cooldown {cooldown_minutes}m for {pair_tf}.'
            }

        return {'approved': True, 'reason': 'drawdown_ok', 'adjusted_size': None}

    def _check_loss_streak(self, portfolio, intent) -> Dict:
        """Check 7: Consecutive loss control (per pair_tf)."""
        config = self.policy.get('loss_streak_control', {})
        if not config.get('enabled', False):
            return {'approved': True, 'reason': 'loss_streak_disabled', 'adjusted_size': None}

        max_losses = config['max_consecutive_losses']
        pair_tf = f"{intent.pair}_{intent.tf}"
        consecutive = portfolio.get_consecutive_losses(pair_tf)

        if consecutive >= max_losses:
            return {
                'approved': False,
                'reason': 'RISK_BLOCKED_COOLDOWN',
                'adjusted_size': None,
                'detail': f'{pair_tf}: {consecutive} consecutive losses (max {max_losses}). Cooldown {config["cooldown_bars"]} bars.'
            }

        return {'approved': True, 'reason': 'loss_streak_ok', 'adjusted_size': None}
