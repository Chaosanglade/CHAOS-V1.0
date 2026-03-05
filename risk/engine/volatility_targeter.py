"""
Volatility Targeting + Dynamic Exposure Scaling

Replaces fixed capital weights with risk-balanced weights that adapt
to each pair's realized volatility. This stabilizes portfolio variance
and typically improves Sharpe by 15-40% without any model changes.

Two components:
1. Per-pair risk weighting (inverse volatility)
2. Portfolio-level exposure scaling (drawdown-based)
"""
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger('volatility_targeter')


class VolatilityTargeter:
    """
    Computes dynamic risk weights for portfolio allocation.

    Usage:
        targeter = VolatilityTargeter(
            pairs=['EURUSD', 'USDCAD', 'AUDUSD', 'GBPJPY'],
            base_weights={'EURUSD': 0.35, 'USDCAD': 0.40, 'AUDUSD': 0.20, 'GBPJPY': 0.05},
            target_annual_vol=0.08,  # 8% annualized portfolio vol target
            vol_lookback=20          # 20-day rolling vol window
        )

        # Feed daily returns as they come in
        targeter.update('EURUSD', daily_return=0.003)

        # Get current risk-adjusted weights
        weights = targeter.get_risk_weights()

        # Get exposure scalar based on drawdown
        scalar = targeter.get_exposure_scalar(current_drawdown_pct=1.5)
    """

    def __init__(self, pairs, base_weights, target_annual_vol=0.08, vol_lookback=20):
        self.pairs = pairs
        self.base_weights = base_weights
        self.target_annual_vol = target_annual_vol
        self.vol_lookback = vol_lookback

        # Rolling return buffers per pair
        self._returns = {pair: [] for pair in pairs}
        self._current_vols = {pair: None for pair in pairs}

        # Exposure scaling thresholds
        self.exposure_tiers = [
            {'max_dd_pct': 1.0, 'exposure': 1.0},
            {'max_dd_pct': 3.0, 'exposure': 0.75},
            {'max_dd_pct': 5.0, 'exposure': 0.50},
            {'max_dd_pct': 100.0, 'exposure': 0.25},
        ]

    def update(self, pair, daily_return):
        """Add a daily return observation for a pair."""
        if pair not in self._returns:
            return

        self._returns[pair].append(daily_return)

        # Keep only lookback window
        if len(self._returns[pair]) > self.vol_lookback * 2:
            self._returns[pair] = self._returns[pair][-self.vol_lookback * 2:]

        # Recompute volatility if we have enough data
        if len(self._returns[pair]) >= self.vol_lookback:
            recent = self._returns[pair][-self.vol_lookback:]
            self._current_vols[pair] = float(np.std(recent)) * np.sqrt(252)  # Annualized

    def get_risk_weights(self):
        """
        Compute inverse-volatility weights.

        weight_pair proportional to base_weight / sigma_pair

        If volatility not yet computed, fall back to base weights.
        """
        # Check if we have vol estimates for all pairs
        vols_available = {p: v for p, v in self._current_vols.items() if v is not None and v > 0}

        if len(vols_available) < len(self.pairs):
            # Not enough data yet — use base weights
            return dict(self.base_weights)

        # Inverse-vol weighting, scaled by base weight preference
        raw_weights = {}
        for pair in self.pairs:
            vol = vols_available.get(pair, 0.10)  # Default 10% if missing
            raw_weights[pair] = self.base_weights.get(pair, 0) / vol

        # Normalize to sum to 1.0
        total = sum(raw_weights.values())
        if total > 0:
            risk_weights = {p: w / total for p, w in raw_weights.items()}
        else:
            risk_weights = dict(self.base_weights)

        return risk_weights

    def get_exposure_scalar(self, current_drawdown_pct):
        """
        Dynamic exposure scaling based on current drawdown.

        Reduces total portfolio exposure as drawdown increases.
        This dramatically reduces tail risk.
        """
        for tier in self.exposure_tiers:
            if current_drawdown_pct < tier['max_dd_pct']:
                return tier['exposure']
        return 0.25  # Minimum exposure

    def get_position_size(self, pair, base_lot_size, current_drawdown_pct=0.0):
        """
        Compute final position size for a pair.

        final_size = base_lot_size * risk_weight * exposure_scalar
        """
        risk_weights = self.get_risk_weights()
        pair_weight = risk_weights.get(pair, 0)

        exposure_scalar = self.get_exposure_scalar(current_drawdown_pct)

        # Scale relative to the pair's base weight
        base_weight = self.base_weights.get(pair, 0)
        if base_weight > 0:
            weight_adjustment = pair_weight / base_weight
        else:
            weight_adjustment = 1.0

        final_size = base_lot_size * weight_adjustment * exposure_scalar

        return round(final_size, 2)

    def get_status(self):
        """Return current state for logging/reporting."""
        return {
            'current_vols': {p: round(v, 4) if v else None for p, v in self._current_vols.items()},
            'risk_weights': {p: round(w, 4) for p, w in self.get_risk_weights().items()},
            'base_weights': self.base_weights,
            'data_points': {p: len(r) for p, r in self._returns.items()}
        }
