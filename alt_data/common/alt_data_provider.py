"""
Alt-data provider for the CHAOS V1.0 inference server.

This module integrates COT and CME features into the inference pipeline
WITHOUT modifying the 273-feature boundary contract.

Integration points:
1. Regime gating modifiers -- adjust enabled model sets based on positioning extremes
2. Agreement threshold modifiers -- tighten/loosen ensemble consensus requirements
3. Confidence dampening -- reduce position size signals during data gaps

Usage in inference_server.py:
    provider = AltDataProvider(cot_path, cme_path)
    modifiers = provider.get_modifiers(symbol='EURUSD', timestamp=current_time)
    # Apply modifiers to regime policy and ensemble config
"""
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger('alt_data_provider')


class AltDataProvider:
    """
    Provides alt-data signals for real-time inference modifications.

    Designed for the pre-signal gating architecture:
    - COT extremes adjust which models are enabled
    - CME spikes confirm or dampen signals
    - Data gaps trigger conservative fallback
    """

    def __init__(self,
                 cot_daily_path='G:/My Drive/chaos_v1.0/alt_data/cot/cot_pair_daily.parquet',
                 cme_daily_path='G:/My Drive/chaos_v1.0/alt_data/cme/cme_pair_daily.parquet',
                 cache_stale_hours=24):
        """
        Args:
            cot_daily_path: path to cot_pair_daily.parquet
            cme_daily_path: path to cme_pair_daily.parquet
            cache_stale_hours: hours before cached data is considered stale
        """
        self.cot_daily_path = cot_daily_path
        self.cme_daily_path = cme_daily_path
        self.cache_stale_hours = cache_stale_hours

        self.cot_data = None
        self.cme_data = None
        self._last_load = None

        self._load_data()

    def _load_data(self):
        """Load alt-data parquet files into memory."""
        try:
            self.cot_data = pd.read_parquet(self.cot_daily_path)
            self.cot_data.index = pd.to_datetime(self.cot_data.index)
            logger.info(f"COT data loaded: {self.cot_data.shape[0]} days, "
                       f"{self.cot_data.index.min().date()} to {self.cot_data.index.max().date()}")
        except FileNotFoundError:
            logger.warning(f"COT data not found at {self.cot_daily_path}. COT features disabled.")
            self.cot_data = None

        try:
            self.cme_data = pd.read_parquet(self.cme_daily_path)
            self.cme_data.index = pd.to_datetime(self.cme_data.index)
            logger.info(f"CME data loaded: {self.cme_data.shape[0]} days, "
                       f"{self.cme_data.index.min().date()} to {self.cme_data.index.max().date()}")
        except FileNotFoundError:
            logger.warning(f"CME data not found at {self.cme_daily_path}. CME features disabled.")
            self.cme_data = None

        self._last_load = datetime.now()

    def _check_staleness(self):
        """Reload data if cache is stale."""
        if self._last_load is None:
            self._load_data()
            return

        elapsed = (datetime.now() - self._last_load).total_seconds() / 3600
        if elapsed > self.cache_stale_hours:
            logger.info(f"Alt-data cache stale ({elapsed:.1f}h). Reloading...")
            self._load_data()

    def _get_latest_row(self, data, timestamp, max_age_days=7):
        """
        Get the most recent data row on or before the given timestamp.

        Args:
            data: pd.DataFrame with DatetimeIndex
            timestamp: lookup timestamp
            max_age_days: reject data older than this (stale/missing)

        Returns:
            pd.Series or None if no valid data
        """
        if data is None or data.empty:
            return None

        ts = pd.Timestamp(timestamp)

        # Get most recent row on or before timestamp
        mask = data.index <= ts
        if not mask.any():
            return None

        latest = data.loc[mask].iloc[-1]
        latest_date = data.loc[mask].index[-1]

        # Check staleness
        age_days = (ts - latest_date).days
        if age_days > max_age_days:
            logger.warning(f"Alt-data stale: latest is {age_days} days old (max {max_age_days})")
            return None

        return latest

    def get_modifiers(self, symbol, timestamp):
        """
        Get alt-data modifiers for a given symbol and timestamp.

        Returns dict with:
            cot_pressure: float, COT positioning pressure for this pair
            cot_extreme: bool, whether positioning is at 3-year extreme
            cot_wow_direction: float, week-over-week positioning change direction
            cme_spike: bool, whether CME volume spike is active
            cme_pressure: float, CME volume-weighted directional pressure
            cme_confirm: bool, whether CME confirms COT direction
            data_available: bool, whether alt-data is available (False -> use conservative fallback)
            recommended_agreement_adjustment: float, modifier to ensemble agreement threshold
                0.0 = no change
                +0.1 = tighten (require more consensus, e.g., during data gaps)
                -0.05 = loosen (allow slightly less consensus when data strongly confirms)
        """
        self._check_staleness()

        result = {
            'cot_pressure': 0.0,
            'cot_extreme': False,
            'cot_wow_direction': 0.0,
            'cme_spike': False,
            'cme_pressure': 0.0,
            'cme_confirm': False,
            'data_available': False,
            'recommended_agreement_adjustment': 0.1  # Conservative default: tighten if no data
        }

        # Get COT data
        cot_row = self._get_latest_row(self.cot_data, timestamp)
        cme_row = self._get_latest_row(self.cme_data, timestamp)

        has_cot = cot_row is not None
        has_cme = cme_row is not None

        if not has_cot and not has_cme:
            # No alt-data available -- return conservative defaults
            return result

        result['data_available'] = True

        # Extract COT features
        if has_cot:
            pressure_col = f"cot_{symbol}_pressure"
            extreme_col = f"cot_{symbol}_extreme_flag"
            wow_col = f"cot_{symbol}_wow_diff"

            result['cot_pressure'] = float(cot_row.get(pressure_col, 0))
            result['cot_extreme'] = bool(cot_row.get(extreme_col, 0))
            result['cot_wow_direction'] = float(cot_row.get(wow_col, 0))

        # Extract CME features
        if has_cme:
            cme_pressure_col = f"cme_{symbol}_pressure"
            spike_col = f"cme_{symbol}_spike_flag"

            result['cme_pressure'] = float(cme_row.get(cme_pressure_col, 0))
            result['cme_spike'] = bool(cme_row.get(spike_col, 0))

        # Determine confirmation
        if has_cot and has_cme:
            cot_direction = np.sign(result['cot_pressure'])
            cme_direction = np.sign(result['cme_pressure'])
            result['cme_confirm'] = (cot_direction == cme_direction) and (cot_direction != 0)

        # Compute agreement threshold adjustment
        # Logic:
        #   COT extreme + CME confirms -> loosen slightly (strong institutional signal)
        #   COT extreme + CME contradicts -> tighten (conflicting signals)
        #   No extreme -> no adjustment
        #   Missing data -> tighten (conservative)
        if result['cot_extreme'] and result['cme_confirm'] and result['cme_spike']:
            result['recommended_agreement_adjustment'] = -0.05  # Loosen: strong confirmation
        elif result['cot_extreme'] and not result['cme_confirm']:
            result['recommended_agreement_adjustment'] = 0.1   # Tighten: conflicting signals
        elif not has_cme:
            result['recommended_agreement_adjustment'] = 0.05  # Slightly tighten: partial data
        else:
            result['recommended_agreement_adjustment'] = 0.0   # Neutral

        return result

    def get_regime_modifier(self, symbol, timestamp):
        """
        Determine if alt-data should modify regime gating.

        Returns:
            dict with:
                override_regime: int or None (None = no override)
                allow_trend_following: bool
                allow_mean_reversion: bool
                reason: str
        """
        modifiers = self.get_modifiers(symbol, timestamp)

        result = {
            'override_regime': None,
            'allow_trend_following': True,
            'allow_mean_reversion': True,
            'reason': 'neutral'
        }

        if not modifiers['data_available']:
            result['reason'] = 'no_alt_data_available'
            return result

        # COT extreme + CME participation spike + alignment -> allow trend models
        if modifiers['cot_extreme'] and modifiers['cme_spike'] and modifiers['cme_confirm']:
            result['allow_trend_following'] = True
            result['allow_mean_reversion'] = False
            result['reason'] = 'cot_extreme_cme_confirms_trend'

        # COT extreme + CME contradicts -> allow mean-reversion, disable trend
        elif modifiers['cot_extreme'] and not modifiers['cme_confirm']:
            result['allow_trend_following'] = False
            result['allow_mean_reversion'] = True
            result['reason'] = 'cot_extreme_cme_contradicts_reversal_likely'

        return result
