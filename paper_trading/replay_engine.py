"""
CHAOS V1.0 Paper Trading Replay Engine

Replays historical bar data through the full production inference pipeline
and simulates trade execution with configurable cost models.

Usage:
    python replay_engine.py --pair EURUSD --start 2023-01-01 --end 2025-12-31 --scenario base

Outputs:
    paper_trading/results/{pair}_{scenario}/
        paper_run_events.parquet    — every inference decision with full context
        paper_run_trades.parquet    — executed trades with entry/exit/PnL
        paper_run_metrics.json      — summary statistics
"""
import pandas as pd
import numpy as np
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('replay_engine')

# Import existing infrastructure (direct Python import, no ZeroMQ)
import sys
sys.path.insert(0, 'G:/My Drive/chaos_v1.0')
sys.path.insert(0, 'G:/My Drive/chaos_v1.0/inference')
sys.path.insert(0, 'G:/My Drive/chaos_v1.0/alt_data/common')


class CostScenario:
    """
    Execution cost model with configurable scenarios.

    Three scenarios per Colab team spec:
      Best case: IBKR modeled costs (from V2 evaluation)
      Base case: +25% slippage/spread over IBKR
      Stress:    +75% slippage/spread + intermittent partial fills
    """

    # IBKR baseline costs (round-trip pips) from V2 evaluation
    IBKR_COSTS = {
        'EURUSD': 0.52, 'GBPUSD': 0.72, 'USDJPY': 0.58,
        'AUDUSD': 0.60, 'USDCAD': 0.78, 'USDCHF': 0.73,
        'NZDUSD': 0.88, 'EURJPY': 0.83, 'GBPJPY': 1.05
    }

    SCENARIOS = {
        'best': {
            'spread_multiplier': 1.0,
            'slippage_multiplier': 1.0,
            'fill_rate': 1.0,  # 100% fills
            'description': 'IBKR modeled costs (Equinix NY4, ECN)'
        },
        'base': {
            'spread_multiplier': 1.25,
            'slippage_multiplier': 1.25,
            'fill_rate': 1.0,
            'description': '+25% slippage/spread over IBKR baseline'
        },
        'stress': {
            'spread_multiplier': 1.75,
            'slippage_multiplier': 1.75,
            'fill_rate': 0.85,  # 85% fill rate (15% missed trades)
            'description': '+75% slippage/spread + 15% intermittent missed fills'
        }
    }

    def __init__(self, scenario='base'):
        if scenario not in self.SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}. Options: {list(self.SCENARIOS.keys())}")
        self.scenario = scenario
        self.config = self.SCENARIOS[scenario]

    def get_cost_rt_pips(self, pair):
        """Get round-trip cost in pips for a pair under this scenario."""
        base_cost = self.IBKR_COSTS.get(pair, 1.0)
        return base_cost * self.config['spread_multiplier']

    def apply_fill_filter(self, rng):
        """
        Returns True if this trade fills, False if missed.
        Uses the scenario's fill rate to randomly reject some trades.
        """
        return rng.random() < self.config['fill_rate']


class Position:
    """Tracks a single open position."""

    def __init__(self, pair, direction, entry_price, entry_time, entry_bar_idx, size=1.0):
        self.pair = pair
        self.direction = direction  # 'LONG' or 'SHORT'
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.entry_bar_idx = entry_bar_idx
        self.size = size
        self.exit_price = None
        self.exit_time = None
        self.exit_bar_idx = None
        self.exit_reason = None
        self.pnl_pips = None

    def close(self, exit_price, exit_time, exit_bar_idx, reason, cost_rt_pips):
        """Close position and compute PnL."""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_bar_idx = exit_bar_idx
        self.exit_reason = reason

        # PnL in pips (before costs)
        if self.direction == 'LONG':
            raw_pnl = exit_price - self.entry_price
        else:
            raw_pnl = self.entry_price - exit_price

        # Convert to pips (multiply by pip factor)
        pip_factor = 100 if 'JPY' in self.pair else 10000
        raw_pnl_pips = raw_pnl * pip_factor

        # Subtract round-trip cost
        self.pnl_pips = raw_pnl_pips - cost_rt_pips

        return self.pnl_pips

    def to_dict(self):
        return {
            'pair': self.pair,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time,
            'entry_bar_idx': self.entry_bar_idx,
            'exit_price': self.exit_price,
            'exit_time': self.exit_time,
            'exit_bar_idx': self.exit_bar_idx,
            'exit_reason': self.exit_reason,
            'pnl_pips': self.pnl_pips,
            'size': self.size
        }


class ReplayEngine:
    """
    Deterministic replay engine for paper trading simulation.

    Feeds historical bar data through the production inference pipeline:
    1. Load feature parquet for pair+timeframe
    2. For each bar in the validation window (2023-2025):
       a. Extract 273-feature vector
       b. Determine HMM regime state (from pre-computed or simulated)
       c. Query alt-data provider for COT/CME modifiers
       d. Call ensemble inference (direct Python, not ZeroMQ)
       e. Apply risk engine checks
       f. Simulate trade execution with cost model
       g. Log everything to audit ledger
    3. Compute and output summary metrics
    """

    def __init__(self, pair, timeframe, scenario='base',
                 start_date='2023-01-01', end_date='2025-12-31',
                 base_dir='G:/My Drive/chaos_v1.0',
                 random_seed=42):
        """
        Args:
            pair: e.g., 'EURUSD'
            timeframe: e.g., 'M30'
            scenario: 'best', 'base', or 'stress'
            start_date: replay start (should be within validation window)
            end_date: replay end
            base_dir: project root
            random_seed: for reproducible fill simulation in stress scenario
        """
        self.pair = pair
        self.timeframe = timeframe
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.base_dir = Path(base_dir)
        self.cost_model = CostScenario(scenario)
        self.rng = np.random.RandomState(random_seed)

        # State
        self.current_position = None
        self.events = []  # All decisions
        self.trades = []  # Completed trades

        # Load components
        self._load_schema()
        self._load_feature_data()
        self._load_models()
        self._load_alt_data()
        self._load_risk_engine()

    def _load_schema(self):
        """Load the 273-feature schema."""
        schema_path = self.base_dir / 'schema' / 'feature_schema.json'
        with open(schema_path) as f:
            self.schema = json.load(f)
        self.feature_names = [f['name'] for f in self.schema['features']]
        logger.info(f"Schema loaded: {len(self.feature_names)} features")

    def _load_feature_data(self):
        """Load the feature parquet for this pair+timeframe."""
        # Feature files are in the features/ subdirectory
        parquet_path = self.base_dir / 'features' / f"{self.pair}_{self.timeframe}_features.parquet"
        if not parquet_path.exists():
            # Fallback: try root directory
            parquet_path = self.base_dir / f"{self.pair}_{self.timeframe}_features.parquet"

        df = pd.read_parquet(parquet_path)

        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
            if date_cols:
                df.index = pd.to_datetime(df[date_cols[0]])
            else:
                raise ValueError("Cannot determine timestamps for replay")

        # Need close price for trade simulation
        close_col = None
        for candidate in ['Close', 'close', 'close_price']:
            if candidate in df.columns:
                close_col = candidate
                break

        if close_col is None:
            price_cols = [c for c in df.columns if 'close' in c.lower() and 'pct' not in c.lower() and 'change' not in c.lower()]
            close_col = price_cols[0] if price_cols else None

        if close_col is None:
            raise ValueError(f"Cannot find close price column in {parquet_path}")

        # Filter to validation window
        mask = (df.index >= self.start_date) & (df.index <= self.end_date)
        df = df.loc[mask].copy()

        # Check which schema features exist in the parquet
        available_features = [f for f in self.feature_names if f in df.columns]
        missing_features = [f for f in self.feature_names if f not in df.columns]
        if missing_features:
            logger.warning(f"{len(missing_features)} schema features missing from parquet. Using {len(available_features)}/{len(self.feature_names)}")

        self.active_feature_names = available_features

        # Drop rows with NaN in features (warmup period)
        df = df.dropna(subset=self.active_feature_names)

        self.bar_data = df
        self.close_col = close_col
        logger.info(f"Loaded {len(df)} bars for {self.pair}_{self.timeframe} "
                    f"({df.index.min().date()} to {df.index.max().date()})")

    def _load_models(self):
        """
        Load all models for this pair+timeframe.
        Uses the same dual-backend approach as inference_server.py.
        Returns dict of {brain_name: backend_instance}
        """
        self.models = {}
        models_dir = self.base_dir / 'models'

        if not models_dir.exists():
            logger.warning(f"Models directory not found: {models_dir}")
            return

        pair_tf = f"{self.pair}_{self.timeframe}"

        try:
            from onnx_export import OnnxBackend, SklearnBackend
        except ImportError as e:
            logger.warning(f"Cannot import backends: {e}. No models loaded.")
            return

        # Load ONNX models
        for onnx_file in models_dir.glob(f"{pair_tf}_*.onnx"):
            brain = onnx_file.stem[len(pair_tf) + 1:]  # Remove pair_tf_ prefix
            try:
                self.models[brain] = OnnxBackend(str(onnx_file))
                logger.info(f"  Loaded ONNX: {brain}")
            except Exception as e:
                logger.warning(f"  Failed to load ONNX {brain}: {e}")

        # Load RF/ET via SklearnBackend (joblib files that are RF or ET)
        rf_et_brains = {'rf_optuna', 'et_optuna'}
        for joblib_file in models_dir.glob(f"{pair_tf}_*.joblib"):
            brain = joblib_file.stem[len(pair_tf) + 1:]
            if brain in rf_et_brains:
                try:
                    self.models[brain] = SklearnBackend(str(joblib_file))
                    logger.info(f"  Loaded sklearn: {brain}")
                except Exception as e:
                    logger.warning(f"  Failed to load sklearn {brain}: {e}")

        logger.info(f"Loaded {len(self.models)} models for {pair_tf}")
        if len(self.models) == 0:
            logger.warning(f"No models loaded for {pair_tf}. Replay will produce FLAT signals only.")

    def _load_alt_data(self):
        """Load alt-data provider."""
        try:
            from alt_data_provider import AltDataProvider
            self.alt_data = AltDataProvider(
                cot_daily_path=str(self.base_dir / 'alt_data' / 'cot' / 'cot_pair_daily.parquet'),
                cme_daily_path=str(self.base_dir / 'alt_data' / 'cme' / 'cme_pair_daily.parquet')
            )
            logger.info("Alt-data provider loaded")
        except Exception as e:
            logger.warning(f"Alt-data provider unavailable: {e}. Running without alt-data.")
            self.alt_data = None

    def _load_risk_engine(self):
        """Load risk engine."""
        try:
            sys.path.insert(0, str(self.base_dir / 'risk'))
            from risk_engine import RiskEngine
            risk_policy_path = str(self.base_dir / 'risk' / 'risk_policy.json')
            self.risk_engine = RiskEngine(risk_policy_path)
            logger.info("Risk engine loaded")
        except Exception as e:
            logger.warning(f"Risk engine unavailable: {e}. Running without risk constraints.")
            self.risk_engine = None

    def _simulate_regime(self, bar_idx):
        """
        Simulate HMM regime state for replay.

        In production, this comes from the live HMM model.
        For replay, we use a simple volatility-based proxy:
          - Compute 20-bar realized volatility
          - Low vol -> REGIME_0 (range-bound)
          - Medium vol -> REGIME_1 (trending)
          - High vol -> REGIME_2 (volatile)
          - Extreme vol -> REGIME_3 (crisis)

        Thresholds calibrated against historical FX vol distributions:
          - < 25th percentile -> REGIME_0
          - 25th-75th -> REGIME_1
          - 75th-95th -> REGIME_2
          - > 95th -> REGIME_3
        """
        if bar_idx < 20:
            return 1, 0.5  # Default to trending with low confidence during warmup

        # Get recent close prices
        close_prices = self.bar_data[self.close_col].iloc[max(0, bar_idx-20):bar_idx+1]
        returns = close_prices.pct_change().dropna()

        if len(returns) < 10:
            return 1, 0.5

        realized_vol = returns.std() * np.sqrt(252)  # Annualized

        # Use full history to compute percentile
        all_close = self.bar_data[self.close_col].iloc[:bar_idx+1]
        all_returns = all_close.pct_change().dropna()

        if len(all_returns) < 100:
            # Not enough history — use fixed thresholds
            if realized_vol < 0.05:
                return 0, 0.7
            elif realized_vol < 0.12:
                return 1, 0.8
            elif realized_vol < 0.20:
                return 2, 0.7
            else:
                return 3, 0.9

        # Rolling 20-bar vol for all history
        rolling_vol = all_returns.rolling(20).std() * np.sqrt(252)
        rolling_vol = rolling_vol.dropna()

        pct = (rolling_vol < realized_vol).mean()

        if pct < 0.25:
            regime, confidence = 0, 0.7 + 0.2 * (0.25 - pct) / 0.25
        elif pct < 0.75:
            regime, confidence = 1, 0.6 + 0.3 * (1 - abs(pct - 0.5) / 0.25)
        elif pct < 0.95:
            regime, confidence = 2, 0.7 + 0.2 * (pct - 0.75) / 0.20
        else:
            regime, confidence = 3, 0.85 + 0.1 * min((pct - 0.95) / 0.05, 1.0)

        return regime, min(confidence, 1.0)

    def _run_ensemble(self, features, regime_state, regime_confidence, timestamp):
        """
        Run the full ensemble inference pipeline.

        Replicates inference_server.py logic but called directly (no ZeroMQ).
        """
        # Load regime policy
        with open(self.base_dir / 'regime' / 'regime_policy.json') as f:
            regime_policy = json.load(f)
        with open(self.base_dir / 'ensemble' / 'ensemble_config.json') as f:
            ensemble_config = json.load(f)

        # Regime gating
        regime_key = f"REGIME_{regime_state}"
        regime_states = regime_policy.get('regime_states', {})
        policy = regime_states.get(regime_key, {})

        conf_threshold = regime_policy.get('confidence_threshold', 0.6)
        if regime_confidence < conf_threshold:
            policy = regime_states.get('REGIME_2', policy)

        allowed_tfs = policy.get('allow', [])
        denied_tfs = policy.get('deny', [])

        if self.timeframe in denied_tfs or (allowed_tfs and self.timeframe not in allowed_tfs) or not allowed_tfs:
            return {
                'signal': 'FLAT', 'class_probs': [0.0, 1.0, 0.0], 'confidence': 1.0,
                'models_voted': 0, 'models_agreed': 0, 'agreement_ratio': 0.0,
                'gated_out': True, 'gate_reason': f'regime_{regime_state}_denies_{self.timeframe}'
            }

        # Run all models
        feature_array = features.reshape(1, -1)
        predictions = []
        brain_names = []

        for brain_name, backend in self.models.items():
            try:
                probs = backend.predict_proba(feature_array)
                predictions.append(probs)
                brain_names.append(brain_name)
            except Exception as e:
                logger.debug(f"Model {brain_name} failed: {e}")
                continue

        if not predictions:
            return {
                'signal': 'FLAT', 'class_probs': [0.0, 1.0, 0.0], 'confidence': 0.0,
                'models_voted': 0, 'models_agreed': 0, 'agreement_ratio': 0.0,
                'gated_out': False, 'gate_reason': 'no_models_produced_output'
            }

        # Trimmed mean aggregation
        probs = np.vstack(predictions)
        n_models = probs.shape[0]

        trim_frac = ensemble_config.get('layer_1_brain_voting', {}).get('trim_fraction', 0.1)
        trim_count = max(1, int(n_models * trim_frac))

        aggregated = np.zeros(3)
        for cls in range(3):
            sorted_probs = np.sort(probs[:, cls])
            if n_models > 2 * trim_count:
                trimmed = sorted_probs[trim_count:-trim_count]
            else:
                trimmed = sorted_probs
            aggregated[cls] = trimmed.mean()

        total = aggregated.sum()
        if total > 0:
            aggregated /= total
        else:
            aggregated = np.array([0.0, 1.0, 0.0])

        ensemble_signal = int(np.argmax(aggregated))
        individual_signals = np.argmax(probs, axis=1)
        models_agreed = int(np.sum(individual_signals == ensemble_signal))

        # Agreement check with alt-data modification
        min_agreement = ensemble_config.get('layer_1_brain_voting', {}).get('minimum_agreement_ratio', 0.6)

        if self.alt_data:
            modifiers = self.alt_data.get_modifiers(self.pair, timestamp)
            min_agreement += modifiers.get('recommended_agreement_adjustment', 0.0)
            min_agreement = max(0.3, min(0.9, min_agreement))  # Clamp to reasonable range
        else:
            modifiers = {'data_available': False, 'recommended_agreement_adjustment': 0.0}

        agreement_ratio = models_agreed / n_models if n_models > 0 else 0

        if agreement_ratio < min_agreement:
            signal = 'FLAT'
            aggregated = np.array([0.0, 1.0, 0.0])
        else:
            signal_map = {0: 'SHORT', 1: 'FLAT', 2: 'LONG'}
            signal = signal_map[ensemble_signal]

        return {
            'signal': signal,
            'class_probs': aggregated.tolist(),
            'confidence': float(np.max(aggregated)),
            'models_voted': n_models,
            'models_agreed': models_agreed,
            'agreement_ratio': round(agreement_ratio, 4),
            'agreement_threshold_used': round(min_agreement, 4),
            'gated_out': False,
            'gate_reason': 'none',
            'alt_data_modifiers': modifiers
        }

    def run(self):
        """
        Execute the full replay.

        For each bar in the validation window:
        1. Extract features
        2. Determine regime
        3. Run ensemble
        4. Apply risk checks
        5. Execute trade logic
        6. Log event
        """
        logger.info("=" * 70)
        logger.info(f"REPLAY START: {self.pair}_{self.timeframe} | "
                    f"Scenario: {self.cost_model.scenario} | "
                    f"Bars: {len(self.bar_data)}")
        logger.info("=" * 70)

        start_time = time.perf_counter()

        for bar_idx in range(len(self.bar_data)):
            row = self.bar_data.iloc[bar_idx]
            timestamp = self.bar_data.index[bar_idx]
            close_price = row[self.close_col]

            # Extract features
            features = row[self.active_feature_names].values.astype(np.float64)

            # Skip if features contain NaN/Inf
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                self._log_event(bar_idx, timestamp, close_price, None, 'skip_invalid_features')
                continue

            # Regime detection
            regime_state, regime_confidence = self._simulate_regime(bar_idx)

            # Ensemble inference
            result = self._run_ensemble(features, regime_state, regime_confidence, timestamp)
            signal = result['signal']

            # Risk engine check
            risk_approved = True
            risk_reason = 'approved'
            if self.risk_engine:
                risk_check = self.risk_engine.check_trade(
                    pair=self.pair,
                    direction=signal,
                    current_positions=[self.current_position] if self.current_position else [],
                    equity_curve_pnl=sum(t.pnl_pips for t in self.trades if t.pnl_pips is not None),
                    recent_trades=self.trades[-10:] if self.trades else [],
                    current_time=timestamp.to_pydatetime() if hasattr(timestamp, 'to_pydatetime') else timestamp
                )
                risk_approved = risk_check.get('approved', True)
                risk_reason = risk_check.get('reason', 'approved')

            if not risk_approved:
                signal = 'FLAT'

            # Trade execution logic
            self._execute_trade_logic(bar_idx, timestamp, close_price, signal, result, risk_reason)

            # Log progress every 5000 bars
            if (bar_idx + 1) % 5000 == 0:
                elapsed = time.perf_counter() - start_time
                logger.info(f"  Bar {bar_idx+1}/{len(self.bar_data)} | "
                           f"Trades: {len(self.trades)} | "
                           f"Elapsed: {elapsed:.1f}s")

        # Close any open position at end
        if self.current_position:
            final_price = self.bar_data[self.close_col].iloc[-1]
            final_time = self.bar_data.index[-1]
            cost = self.cost_model.get_cost_rt_pips(self.pair)
            self.current_position.close(final_price, final_time, len(self.bar_data)-1, 'replay_end', cost)
            self.trades.append(self.current_position)
            self.current_position = None

        elapsed = time.perf_counter() - start_time
        logger.info(f"REPLAY COMPLETE: {len(self.trades)} trades in {elapsed:.1f}s")

        return self._compute_metrics()

    def _execute_trade_logic(self, bar_idx, timestamp, close_price, signal, inference_result, risk_reason):
        """
        Execute trade logic based on signal.

        Rules:
        - FLAT signal: close any open position
        - LONG signal: close SHORT if open, open LONG if not in LONG
        - SHORT signal: close LONG if open, open SHORT if not in SHORT
        - Fill rate check applied for stress scenario
        """
        cost = self.cost_model.get_cost_rt_pips(self.pair)

        # Close existing position if signal changes
        if self.current_position:
            should_close = (
                signal == 'FLAT' or
                (signal == 'LONG' and self.current_position.direction == 'SHORT') or
                (signal == 'SHORT' and self.current_position.direction == 'LONG')
            )

            if should_close:
                self.current_position.close(close_price, timestamp, bar_idx, f'signal_{signal}', cost)
                self.trades.append(self.current_position)
                self.current_position = None

        # Open new position if signal is directional and no position open
        if signal in ('LONG', 'SHORT') and self.current_position is None:
            # Apply fill rate check
            if not self.cost_model.apply_fill_filter(self.rng):
                self._log_event(bar_idx, timestamp, close_price, inference_result, f'missed_fill_{signal}')
                return

            self.current_position = Position(
                pair=self.pair,
                direction=signal,
                entry_price=close_price,
                entry_time=timestamp,
                entry_bar_idx=bar_idx
            )

        # Log event
        self._log_event(bar_idx, timestamp, close_price, inference_result, risk_reason)

    def _log_event(self, bar_idx, timestamp, close_price, inference_result, action_reason):
        """Log a decision event for the audit trail."""
        event = {
            'bar_idx': bar_idx,
            'timestamp': timestamp,
            'close_price': close_price,
            'action_reason': action_reason,
            'position_open': self.current_position is not None,
            'position_direction': self.current_position.direction if self.current_position else None,
            'cumulative_trades': len(self.trades),
            'cumulative_pnl_pips': sum(t.pnl_pips for t in self.trades if t.pnl_pips is not None)
        }

        if inference_result:
            event.update({
                'signal': inference_result.get('signal'),
                'confidence': inference_result.get('confidence'),
                'models_voted': inference_result.get('models_voted'),
                'models_agreed': inference_result.get('models_agreed'),
                'agreement_ratio': inference_result.get('agreement_ratio'),
                'agreement_threshold': inference_result.get('agreement_threshold_used'),
                'gated_out': inference_result.get('gated_out'),
                'gate_reason': inference_result.get('gate_reason'),
            })

        self.events.append(event)

    def _compute_metrics(self):
        """Compute summary metrics from completed trades."""
        if not self.trades:
            return {'total_trades': 0, 'error': 'no_trades_generated'}

        pnls = [t.pnl_pips for t in self.trades if t.pnl_pips is not None]

        if not pnls:
            return {'total_trades': len(self.trades), 'error': 'no_closed_trades'}

        pnl_array = np.array(pnls)
        winners = pnl_array[pnl_array > 0]
        losers = pnl_array[pnl_array < 0]

        # Profit Factor
        gross_profit = winners.sum() if len(winners) > 0 else 0
        gross_loss = abs(losers.sum()) if len(losers) > 0 else 0.001  # Avoid division by zero
        profit_factor = gross_profit / gross_loss

        # Max drawdown (in pips)
        cumulative = np.cumsum(pnl_array)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        max_drawdown = drawdowns.max()

        # Sharpe proxy (annualized, assuming daily-ish trades)
        # Using sqrt(252) as annualization factor
        if pnl_array.std() > 0:
            sharpe_proxy = (pnl_array.mean() / pnl_array.std()) * np.sqrt(252)
        else:
            sharpe_proxy = 0

        # Decision rate (fraction of bars where a non-FLAT signal was generated)
        total_signals = sum(1 for e in self.events if e.get('signal') in ('LONG', 'SHORT'))
        decision_rate = total_signals / len(self.events) if self.events else 0

        # Win rate
        win_rate = len(winners) / len(pnl_array) if len(pnl_array) > 0 else 0

        metrics = {
            'pair': self.pair,
            'timeframe': self.timeframe,
            'scenario': self.cost_model.scenario,
            'total_bars': len(self.bar_data),
            'total_events': len(self.events),
            'total_trades': len(pnl_array),
            'win_rate': round(win_rate, 4),
            'profit_factor': round(profit_factor, 4),
            'sharpe_proxy': round(sharpe_proxy, 4),
            'total_pnl_pips': round(float(pnl_array.sum()), 2),
            'avg_pnl_pips': round(float(pnl_array.mean()), 2),
            'max_drawdown_pips': round(float(max_drawdown), 2),
            'avg_winner_pips': round(float(winners.mean()), 2) if len(winners) > 0 else 0,
            'avg_loser_pips': round(float(losers.mean()), 2) if len(losers) > 0 else 0,
            'decision_rate': round(decision_rate, 4),
            'longest_drawdown_bars': int(self._longest_drawdown_streak(pnl_array)),
            'cost_scenario': self.cost_model.config['description'],
            'cost_rt_pips': self.cost_model.get_cost_rt_pips(self.pair),
            'replay_date': datetime.now().isoformat()
        }

        return metrics

    def _longest_drawdown_streak(self, pnl_array):
        """Count the longest consecutive losing streak."""
        max_streak = 0
        current_streak = 0
        for pnl in pnl_array:
            if pnl < 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        return max_streak

    def save_results(self, output_dir=None):
        """Save all outputs to disk."""
        if output_dir is None:
            output_dir = self.base_dir / 'paper_trading' / 'results' / f"{self.pair}_{self.timeframe}_{self.cost_model.scenario}"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Events
        events_df = pd.DataFrame(self.events)
        events_path = output_dir / 'paper_run_events.parquet'
        events_df.to_parquet(events_path, index=False)
        logger.info(f"Saved: {events_path} ({len(events_df)} events)")

        # Trades (ensure schema even when empty)
        trades_data = [t.to_dict() for t in self.trades if t.pnl_pips is not None]
        trade_columns = ['pair', 'direction', 'entry_price', 'entry_time', 'entry_bar_idx',
                         'exit_price', 'exit_time', 'exit_bar_idx', 'exit_reason', 'pnl_pips', 'size']
        if trades_data:
            trades_df = pd.DataFrame(trades_data)
        else:
            trades_df = pd.DataFrame(columns=trade_columns)
        trades_path = output_dir / 'paper_run_trades.parquet'
        trades_df.to_parquet(trades_path, index=False)
        logger.info(f"Saved: {trades_path} ({len(trades_df)} trades)")

        # Metrics
        metrics = self._compute_metrics()
        metrics_path = output_dir / 'paper_run_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info(f"Saved: {metrics_path}")

        return metrics


def run_full_replay(pairs=None, timeframes=None, scenarios=None):
    """
    Run replay across multiple pairs, timeframes, and cost scenarios.

    Default: all production pairs on M30 across all 3 scenarios.
    """
    if pairs is None:
        pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY', 'GBPJPY']
    if timeframes is None:
        timeframes = ['M30']  # Start with M30 (highest pass rate)
    if scenarios is None:
        scenarios = ['best', 'base', 'stress']

    all_metrics = []

    for pair in pairs:
        for tf in timeframes:
            for scenario in scenarios:
                logger.info(f"\n{'='*70}")
                logger.info(f"RUNNING: {pair}_{tf} | Scenario: {scenario}")
                logger.info(f"{'='*70}")

                try:
                    engine = ReplayEngine(pair=pair, timeframe=tf, scenario=scenario)
                    metrics = engine.run()
                    engine.save_results()
                    all_metrics.append(metrics)

                    logger.info(f"  PF: {metrics.get('profit_factor', 'N/A')} | "
                               f"Trades: {metrics.get('total_trades', 0)} | "
                               f"Sharpe: {metrics.get('sharpe_proxy', 'N/A')}")
                except Exception as e:
                    logger.error(f"  FAILED: {e}")
                    all_metrics.append({
                        'pair': pair, 'timeframe': tf, 'scenario': scenario,
                        'error': str(e)
                    })

    # Save combined metrics
    summary_path = Path('G:/My Drive/chaos_v1.0/paper_trading/results/replay_summary.json')
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)

    logger.info(f"\nSummary saved: {summary_path}")
    return all_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CHAOS V1.0 Paper Trading Replay')
    parser.add_argument('--pair', default='EURUSD')
    parser.add_argument('--timeframe', default='M30')
    parser.add_argument('--scenario', default='base', choices=['best', 'base', 'stress'])
    parser.add_argument('--start', default='2023-01-01')
    parser.add_argument('--end', default='2025-12-31')
    parser.add_argument('--full', action='store_true', help='Run all pairs x all scenarios on M30')
    args = parser.parse_args()

    if args.full:
        run_full_replay()
    else:
        engine = ReplayEngine(
            pair=args.pair, timeframe=args.timeframe, scenario=args.scenario,
            start_date=args.start, end_date=args.end
        )
        metrics = engine.run()
        engine.save_results()

        print("\n" + "=" * 50)
        print("REPLAY METRICS")
        print("=" * 50)
        for k, v in metrics.items():
            print(f"  {k}: {v}")
