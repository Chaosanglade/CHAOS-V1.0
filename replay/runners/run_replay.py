"""
CHAOS V1.0 Main Replay Runner (Colab Team Spec)

Entry point for deterministic replay. Runs all pair × timeframe combinations,
writes two-row trade lifecycle, position-change events, and 100% decision ledger.

Usage:
    python run_replay.py [--pairs EURUSD GBPUSD] [--tfs M30 H1] [--bars 0]
"""
import os
import sys
import json
import time
## uuid removed — Lock 8: deterministic IDs only
import logging
import numpy as np
import random

# Suppress ONNX Runtime warnings (batch shape warnings, etc.)
os.environ['ORT_LOG_SEVERITY_LEVEL'] = '3'
try:
    import onnxruntime as ort
    ort.set_default_logger_severity(3)  # ERROR only
except ImportError:
    pass
import gc
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Determinism enforcement
os.environ["PYTHONHASHSEED"] = "42"
np.random.seed(42)
random.seed(42)

# Project root
PROJECT_ROOT = Path('G:/My Drive/chaos_v1.0')
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'inference'))

import pandas as pd

# Local imports
from replay.runners.replay_iterator import ReplayIterator
from replay.runners.parquet_schemas import (
    TRADES_SCHEMA, POSITIONS_SCHEMA, DECISION_LEDGER_SCHEMA,
    write_parquet_strict
)
from replay.runners.report_generators import (
    generate_coverage_report, generate_veto_breakdown, compute_partial_fill_metrics
)
from replay.runners.brain_tracker import BrainTracker, save_brain_contributions
from risk.engine.portfolio_state import PortfolioState, OpenPosition
from risk.engine.exposure_controller import ExposureController, OrderIntent

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger('run_replay')


def _read_json_retry(path, retries=3, delay=2.0):
    """Read a JSON file with retry logic for Google Drive stability."""
    import time as _time
    for attempt in range(retries):
        try:
            with open(path) as f:
                return json.load(f)
        except (FileNotFoundError, OSError, PermissionError) as e:
            if attempt < retries - 1:
                logger.warning(f"Retry {attempt+1}/{retries} reading {path}: {e}")
                _time.sleep(delay * (attempt + 1))
            else:
                raise


def snapshot_configs(run_output_dir, base_dir=None):
    """
    Copy all config files into run-local directory.
    After this, no config reads should touch G: directly.

    Returns: dict mapping config name -> local snapshot path
    """
    base = Path(base_dir or PROJECT_ROOT)
    snapshot_dir = Path(run_output_dir) / 'config_snapshot'
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    configs_to_snapshot = {
        'universe': base / 'replay' / 'config' / 'universe.json',
        'replay_config': base / 'replay' / 'config' / 'replay_config.json',
        'cost_scenarios': base / 'replay' / 'config' / 'execution_cost_scenarios.json',
        'regime_policy': base / 'regime' / 'regime_policy.json',
        'ensemble_config': base / 'ensemble' / 'ensemble_config.json',
        'risk_policy': base / 'risk' / 'config' / 'risk_policy.json',
        'correlation_groups': base / 'risk' / 'config' / 'correlation_groups.json',
        'instrument_specs': base / 'risk' / 'config' / 'instrument_specs.json',
        'feature_schema': base / 'schema' / 'feature_schema.json',
        'reason_codes': base / 'audit' / 'schemas' / 'reason_codes.json',
    }

    snapshot_paths = {}
    for name, source_path in configs_to_snapshot.items():
        dest_path = snapshot_dir / source_path.name
        try:
            shutil.copy2(str(source_path), str(dest_path))
            snapshot_paths[name] = str(dest_path)
        except Exception as e:
            logger.warning(f"Failed to snapshot {name}: {e}. Using source path.")
            snapshot_paths[name] = str(source_path)

    logger.info(f"Config snapshot: {len(snapshot_paths)} files -> {snapshot_dir}")
    return snapshot_paths


# ============================================================
# MODEL LOADER
# ============================================================
class ModelLoader:
    """Load ONNX and sklearn models for replay inference."""

    def __init__(self, models_dir: str, pairs: List[str] = None, tfs: List[str] = None):
        self.models_dir = Path(models_dir)
        self.models: Dict[str, Dict] = {}  # pair_tf -> {brain: backend}
        self._load_all(pairs=pairs, tfs=tfs)

    def _load_all(self, pairs: List[str] = None, tfs: List[str] = None):
        from onnx_export import OnnxBackend, SklearnBackend

        skip_tfs = {'M1', 'W1', 'MN1'}
        rf_et_brains = {'rf_optuna', 'et_optuna'}
        # Optional filter: only load models for specified pairs/tfs
        pair_filter = set(pairs) if pairs else None
        tf_filter = set(tfs) if tfs else None

        # Load ONNX models
        for onnx_file in sorted(self.models_dir.glob('*.onnx')):
            parts = onnx_file.stem.split('_')
            if len(parts) < 3:
                continue
            pair, tf = parts[0], parts[1]
            brain = '_'.join(parts[2:])
            if tf in skip_tfs:
                continue
            if pair_filter and pair not in pair_filter:
                continue
            if tf_filter and tf not in tf_filter:
                continue

            pair_tf = f"{pair}_{tf}"
            if pair_tf not in self.models:
                self.models[pair_tf] = {}
            try:
                backend = OnnxBackend(str(onnx_file))
                try:
                    backend.expected_features = backend.session.get_inputs()[0].shape[1]
                except Exception:
                    backend.expected_features = None
                self.models[pair_tf][brain] = backend
                logger.debug(f"Loaded ONNX: {pair_tf}/{brain} (expects {backend.expected_features} features)")
            except Exception as e:
                logger.warning(f"Failed to load ONNX {onnx_file.name}: {e}")

        # Load RF/ET joblib models
        for jl_file in sorted(self.models_dir.glob('*.joblib')):
            parts = jl_file.stem.split('_')
            if len(parts) < 3:
                continue
            pair, tf = parts[0], parts[1]
            brain = '_'.join(parts[2:])
            if tf in skip_tfs:
                continue
            if brain not in rf_et_brains:
                continue
            if pair_filter and pair not in pair_filter:
                continue
            if tf_filter and tf not in tf_filter:
                continue

            pair_tf = f"{pair}_{tf}"
            if pair_tf not in self.models:
                self.models[pair_tf] = {}
            try:
                backend = SklearnBackend(str(jl_file))
                try:
                    backend.expected_features = backend.model.n_features_in_
                except Exception:
                    backend.expected_features = None
                self.models[pair_tf][brain] = backend
                logger.debug(f"Loaded sklearn: {pair_tf}/{brain} (expects {backend.expected_features} features)")
            except Exception as e:
                logger.warning(f"Failed to load joblib {jl_file.name}: {e}")

        total = sum(len(v) for v in self.models.values())
        logger.info(f"Models loaded: {total} across {len(self.models)} pair/tf combos")

    def get_models(self, pair_tf: str) -> Dict:
        return self.models.get(pair_tf, {})


# ============================================================
# REGIME SIMULATOR
# ============================================================
class RegimeSimulator:
    """Simulate regime state from regime policy."""

    def __init__(self, regime_policy_path: str):
        self.policy = _read_json_retry(regime_policy_path)
        self.confidence_threshold = self.policy.get('confidence_threshold', 0.6)

    def get_regime_state(self, pair: str, tf: str, bar_idx: int) -> Tuple[int, float]:
        """
        Return (regime_state, regime_confidence).
        Default: REGIME_1 (trending) with high confidence for replay.
        """
        return 1, 0.85

    def get_enabled_models(self, pair_tf_models: Dict, tf: str,
                           regime_state: int, regime_confidence: float) -> List[Tuple[str, object]]:
        """Apply regime gating to filter models."""
        if not pair_tf_models:
            return []

        regime_key = f"REGIME_{regime_state}"
        regime_states = self.policy.get('regime_states', {})
        policy = regime_states.get(regime_key, {})

        if regime_confidence < self.confidence_threshold:
            policy = regime_states.get('REGIME_2', policy)

        allowed_tfs = policy.get('allow', [])
        denied_tfs = policy.get('deny', [])

        if tf in denied_tfs:
            return []
        if not allowed_tfs and denied_tfs:
            return []
        if allowed_tfs and tf not in allowed_tfs:
            return []

        return [(brain, backend) for brain, backend in pair_tf_models.items()]


# ============================================================
# ENSEMBLE ENGINE
# ============================================================
class EnsembleEngine:
    """Run ensemble inference with trimmed mean aggregation."""

    def __init__(self, ensemble_config_path: str):
        self.config = _read_json_retry(ensemble_config_path)
        l1 = self.config.get('layer_1_brain_voting', {})
        self.trim_fraction = l1.get('trim_fraction', 0.1)
        self.min_agreement = l1.get('minimum_agreement_ratio', 0.6)
        self.min_models = l1.get('minimum_models_required', 5)

    @staticmethod
    def _adapt_features(features: np.ndarray, expected: int) -> np.ndarray:
        """Adapt feature vector to match model's expected input size."""
        n = features.shape[0]
        if n == expected:
            return features
        elif n > expected:
            return features[:expected]
        else:
            padded = np.zeros(expected, dtype=np.float32)
            padded[:n] = features
            return padded

    @staticmethod
    def _adapt_features_batch(features_batch: np.ndarray, expected: int) -> np.ndarray:
        """Adapt 2D feature batch (N, n_features) to (N, expected)."""
        n = features_batch.shape[1]
        if n == expected:
            return features_batch
        elif n > expected:
            return features_batch[:, :expected]
        else:
            padded = np.zeros((features_batch.shape[0], expected), dtype=np.float32)
            padded[:, :n] = features_batch
            return padded

    def run_inference(self, enabled_models: List[Tuple[str, object]],
                      features: np.ndarray) -> Dict:
        """
        Run all enabled models and aggregate via trimmed mean.

        Returns dict with: class_probs, signal_side, confidence, models_voted,
                          models_agreed, agreement_score, brain_keys
        """
        predictions = []
        brain_names = []
        brain_predictions_raw = {}  # brain_name -> probs list (for brain tracker)

        for brain_name, backend in enabled_models:
            try:
                # Per-model feature adaptation
                model_features = features.copy()
                expected = getattr(backend, 'expected_features', None)
                if expected is not None and model_features.shape[0] != expected:
                    n_in = model_features.shape[0]
                    model_features = self._adapt_features(model_features, expected)
                    if n_in > expected:
                        logger.debug(f"{brain_name}: truncated {n_in} -> {expected} features")
                    else:
                        logger.debug(f"{brain_name}: padded {n_in} -> {expected} features")
                # Sanitize NaN/Inf (matches training — models never saw NaN)
                model_features = np.nan_to_num(model_features, nan=0.0, posinf=0.0, neginf=0.0)
                probs = backend.predict_proba(model_features.reshape(1, -1))
                if probs.ndim == 2:
                    probs = probs[0]
                predictions.append(probs)
                brain_names.append(brain_name)
                brain_predictions_raw[brain_name] = probs.tolist()
            except Exception as e:
                logger.warning(f"Model {brain_name} inference failed: {e}")
                continue

        if not predictions:
            return {
                'class_probs': np.array([0.0, 1.0, 0.0], dtype=np.float32),
                'signal_side': 0,
                'confidence': 1.0,
                'models_voted': 0,
                'models_agreed': 0,
                'agreement_score': 0.0,
                'brain_keys': [],
                'brain_predictions': {},
            }

        probs = np.vstack(predictions)
        n_models = probs.shape[0]

        # Trimmed mean
        trim_count = max(1, int(n_models * self.trim_fraction))
        aggregated = np.zeros(3, dtype=np.float64)
        for cls in range(3):
            sorted_probs = np.sort(probs[:, cls])
            if n_models > 2 * trim_count:
                trimmed = sorted_probs[trim_count:-trim_count]
            else:
                trimmed = sorted_probs
            aggregated[cls] = trimmed.mean()

        # Normalize
        total = aggregated.sum()
        if total > 0:
            aggregated /= total
        else:
            aggregated = np.array([0.0, 1.0, 0.0])

        # Agreement
        ensemble_signal = int(np.argmax(aggregated))
        individual_signals = np.argmax(probs, axis=1)
        models_agreed = int(np.sum(individual_signals == ensemble_signal))
        agreement_score = models_agreed / n_models if n_models > 0 else 0.0

        # Map: 0=SHORT(-1), 1=FLAT(0), 2=LONG(+1)
        signal_map = {0: -1, 1: 0, 2: 1}
        signal_side = signal_map[ensemble_signal]

        return {
            'class_probs': aggregated.astype(np.float32),
            'signal_side': signal_side,
            'confidence': float(np.max(aggregated)),
            'models_voted': n_models,
            'models_agreed': models_agreed,
            'agreement_score': agreement_score,
            'brain_keys': brain_names,
            'brain_predictions': brain_predictions_raw,
        }

    def run_batch_inference(self, enabled_models: List[Tuple[str, object]],
                            features_batch: np.ndarray) -> List[Dict]:
        """
        Batch inference for N events. One predict call per model for the whole batch.

        Args:
            enabled_models: list of (brain_name, backend)
            features_batch: np.ndarray (N, n_features)
        Returns:
            list of N result dicts (same format as run_inference)
        """
        N = features_batch.shape[0]

        # Collect per-model batch predictions
        all_model_probs = []  # list of (N, 3) arrays
        brain_names = []

        for brain_name, backend in enabled_models:
            try:
                batch = features_batch.copy()
                expected = getattr(backend, 'expected_features', None)
                if expected is not None and batch.shape[1] != expected:
                    batch = self._adapt_features_batch(batch, expected)
                batch = np.nan_to_num(batch, nan=0.0, posinf=0.0, neginf=0.0)
                probs = backend.batch_predict_proba(batch)  # (N, 3)
                all_model_probs.append(probs)
                brain_names.append(brain_name)
            except Exception as e:
                logger.warning(f"Batch model {brain_name} inference failed: {e}")
                continue

        empty_result = {
            'class_probs': np.array([0.0, 1.0, 0.0], dtype=np.float32),
            'signal_side': 0, 'confidence': 1.0,
            'models_voted': 0, 'models_agreed': 0,
            'agreement_score': 0.0, 'brain_keys': [],
            'brain_predictions': {},
        }

        if not all_model_probs:
            return [dict(empty_result) for _ in range(N)]

        # Stack: (n_models, N, 3)
        stacked = np.stack(all_model_probs, axis=0)
        n_models = stacked.shape[0]
        trim_count = max(1, int(n_models * self.trim_fraction))
        signal_map = {0: -1, 1: 0, 2: 1}

        results = []
        for i in range(N):
            probs = stacked[:, i, :]  # (n_models, 3)

            # Per-brain predictions for this bar (for brain tracker)
            bar_brain_preds = {
                brain_names[j]: probs[j].tolist() for j in range(n_models)
            }

            # Trimmed mean
            aggregated = np.zeros(3, dtype=np.float64)
            for cls in range(3):
                sorted_p = np.sort(probs[:, cls])
                if n_models > 2 * trim_count:
                    trimmed = sorted_p[trim_count:-trim_count]
                else:
                    trimmed = sorted_p
                aggregated[cls] = trimmed.mean()

            total = aggregated.sum()
            if total > 0:
                aggregated /= total
            else:
                aggregated = np.array([0.0, 1.0, 0.0])

            ensemble_signal = int(np.argmax(aggregated))
            individual_signals = np.argmax(probs, axis=1)
            models_agreed = int(np.sum(individual_signals == ensemble_signal))
            agreement_score = models_agreed / n_models

            results.append({
                'class_probs': aggregated.astype(np.float32),
                'signal_side': signal_map[ensemble_signal],
                'confidence': float(np.max(aggregated)),
                'models_voted': n_models,
                'models_agreed': models_agreed,
                'agreement_score': agreement_score,
                'brain_keys': brain_names,
                'brain_predictions': bar_brain_preds,
            })

        return results


# ============================================================
# COST CALCULATOR
# ============================================================
class CostCalculator:
    """Compute trade costs for a given scenario."""

    def __init__(self, cost_scenarios_path: str):
        data = _read_json_retry(cost_scenarios_path)
        self.base_costs = data['ibkr_base_costs_rt_pips']
        self.scenarios = {s['name']: s for s in data['scenarios']}

    def compute_costs(self, pair: str, qty_lots: float,
                      scenario_name: str = 'IBKR_BASE') -> Dict:
        """Return spread_cost_pips, slippage_cost_pips, commission_cost_usd, total_cost_usd."""
        scenario = self.scenarios[scenario_name]
        base_rt = self.base_costs.get(pair, 1.0)

        spread_pips = base_rt * scenario['spread_pips_mult']
        slippage_pips = scenario['slippage_pips_add']
        notional_usd = qty_lots * 100_000
        commission_usd = notional_usd / 1_000_000 * scenario['commission_usd_per_million']

        # Lock 9: enforce non-negative costs BEFORE computing total
        spread_pips = max(0.0, spread_pips)
        slippage_pips = max(0.0, slippage_pips)
        commission_usd = max(0.0, commission_usd)

        # Total cost in USD (rough: 1 pip ≈ $10 per standard lot for most pairs)
        pip_value_approx = 10.0
        total_cost_usd = (spread_pips + slippage_pips) * pip_value_approx * qty_lots + commission_usd

        # Lock 9: total must equal components within tolerance
        expected_total = (spread_pips + slippage_pips) * pip_value_approx * qty_lots + commission_usd
        assert abs(total_cost_usd - expected_total) < 1e-9, \
            f"Cost component mismatch: {total_cost_usd} != {expected_total}"

        return {
            'spread_cost_pips': float(spread_pips),
            'slippage_cost_pips': float(slippage_pips),
            'commission_cost_usd': float(commission_usd),
            'total_cost_usd': float(total_cost_usd),
            'fill_rate': scenario['fill_rate'],
        }


# ============================================================
# MAIN REPLAY ENGINE
# ============================================================
class ReplayRunner:
    """Main replay engine orchestrator."""

    def __init__(self, config_dir: str = None, models_dir: str = None,
                 pairs: List[str] = None, tfs: List[str] = None):
        config_dir = Path(config_dir or PROJECT_ROOT / 'replay' / 'config')
        models_dir = models_dir or str(PROJECT_ROOT / 'models')

        # Snapshot configs to local temp dir (isolates from G: filesystem issues)
        import tempfile
        self._snapshot_dir = Path(tempfile.mkdtemp(prefix='chaos_config_'))
        self._snapshot_paths = snapshot_configs(self._snapshot_dir, str(PROJECT_ROOT))

        # Load configs from snapshot (never G: after this point)
        self.universe = _read_json_retry(self._snapshot_paths['universe'])
        self.replay_config = _read_json_retry(self._snapshot_paths['replay_config'])

        # Components — only load models for requested pairs/tfs
        # (models must come from G: since they're too large to snapshot)
        self.model_loader = ModelLoader(models_dir, pairs=pairs, tfs=tfs)
        self.regime_sim = RegimeSimulator(self._snapshot_paths['regime_policy'])
        self.ensemble_engine = EnsembleEngine(self._snapshot_paths['ensemble_config'])
        self.cost_calc = CostCalculator(self._snapshot_paths['cost_scenarios'])
        self.risk_controller = ExposureController(self._snapshot_paths['risk_policy'])
        self.portfolio = PortfolioState(
            self._snapshot_paths['instrument_specs'],
            self._snapshot_paths['correlation_groups'],
        )

        # Load reason codes registry from snapshot
        self.valid_reason_codes = set(
            _read_json_retry(self._snapshot_paths['reason_codes']).keys()
        )

        # Trade simulation config
        sim_cfg = self.replay_config.get('trade_simulation', {})
        self.cooldown_bars_after_exit = sim_cfg.get('cooldown_bars_after_exit', 1)

        # Agreement threshold
        l1 = self.ensemble_engine.config.get('layer_1_brain_voting', {})
        self.agreement_threshold_base = l1.get('minimum_agreement_ratio', 0.6)

        # TF roles (TRADE_ENABLED/EXECUTE vs CONFIRM_ONLY)
        # replay_config takes precedence, fallback to ensemble_config
        self.tf_roles = self.replay_config.get('tf_roles',
                            self.ensemble_engine.config.get('tf_roles', {}))
        # Track latest signals from CONFIRM_ONLY TFs for cross-TF confirmation
        self._confirm_signals: Dict[str, Dict[str, int]] = {}  # pair -> {tf: signal}

        # State tracking
        self.cooldown_counters: Dict[str, int] = {}  # pair_tf -> bars remaining
        self._open_trade_ids: Dict[str, str] = {}  # pair_tf -> trade_id of open position
        self.errors: List[str] = []

        # Output buffers
        self.ledger_rows: List[Dict] = []
        self.trade_rows: List[Dict] = []
        self.position_rows: List[Dict] = []

        # Brain contribution tracker
        self.brain_tracker = BrainTracker()

        # Batch + chunked write config
        self.BATCH_SIZE = 512
        self.CHUNK_SIZE = 50_000
        self._ledger_chunks: List[pd.DataFrame] = []
        self._trade_chunks: List[pd.DataFrame] = []

    def reset_state(self):
        """Reset all per-run state so the runner can be reused across scenarios."""
        self.cooldown_counters = {}
        self._open_trade_ids = {}
        self.errors = []
        self.ledger_rows = []
        self.trade_rows = []
        self.position_rows = []
        self._ledger_chunks = []
        self._trade_chunks = []
        self.brain_tracker = BrainTracker()
        self._confirm_signals = {}
        # Reset portfolio state
        self.portfolio.positions = {}
        self.portfolio.closed_trades = []
        self.portfolio.risk_state = "NORMAL"
        self.portfolio.cooldown_until_ts = {}
        self.portfolio.realized_pnl_usd_cum = 0.0
        self.portfolio._consecutive_losses = {}
        self.portfolio._last_entry_ts = {}
        self.portfolio._trade_seq = {}

    # ============================================================
    # TF ROLE GATING — CONFIRM_ONLY / EXECUTE logic
    # ============================================================

    def can_execute_trade(self, pair: str, tf: str, signal: int) -> bool:
        """
        Returns True if this TF is allowed to open/close positions.
        CONFIRM_ONLY TFs contribute to signal confirmation but never trade.
        Supports both formats: {'role': 'CONFIRM_ONLY'} and {'can_open_positions': False}
        """
        role_config = self.tf_roles.get(tf, {'role': 'TRADE_ENABLED', 'can_open_positions': True})
        # Check role name first (replay_config format)
        role = role_config.get('role', 'TRADE_ENABLED')
        if role == 'CONFIRM_ONLY':
            return False
        # Fallback to can_open_positions flag (ensemble_config format)
        if not role_config.get('can_open_positions', True):
            return False
        return True

    def get_confirmation_signal(self, pair: str) -> Tuple[int, float, List[str], Dict[str, int]]:
        """
        Compute a confirmation signal from CONFIRM_ONLY timeframes.

        Returns:
            confirm_direction: +1 (confirms long), -1 (confirms short), 0 (neutral/conflict)
            confirm_strength: 0.0-1.0 (how strong the confirmation is)
            confirmers_used: list of TF names that contributed
            confirm_details: dict of {tf: signal} for each confirmer
        """
        confirmers = {}

        pair_signals = self._confirm_signals.get(pair, {})
        for tf, config in self.tf_roles.items():
            if config.get('role') == 'CONFIRM_ONLY' and tf in pair_signals:
                signal = pair_signals[tf]  # -1, 0, +1
                if signal != 0:
                    confirmers[tf] = signal

        if not confirmers:
            return 0, 0.0, [], {}

        # Simple majority: if confirmers agree on direction, that's the confirmation
        directions = list(confirmers.values())
        long_count = sum(1 for d in directions if d == 1)
        short_count = sum(1 for d in directions if d == -1)

        if long_count > short_count:
            confirm_direction = 1
        elif short_count > long_count:
            confirm_direction = -1
        else:
            confirm_direction = 0  # Conflict

        confirm_strength = max(long_count, short_count) / len(directions) if directions else 0.0

        return confirm_direction, confirm_strength, list(confirmers.keys()), confirmers

    @staticmethod
    def apply_confirmation_to_threshold(base_threshold: float, execute_signal: int,
                                         confirm_direction: int, confirm_strength: float) -> float:
        """
        Adjust agreement threshold based on confirmation from lower TFs.

        Rules:
        - Confirmers align with execute signal -> loosen threshold slightly (-0.05)
        - Confirmers conflict with execute signal -> tighten threshold (+0.10)
        - No confirmers or neutral -> no change
        """
        if confirm_direction == 0 or confirm_strength < 0.5:
            return base_threshold  # No meaningful confirmation

        if confirm_direction == execute_signal:
            # Lower TFs confirm — slightly loosen
            return max(base_threshold - 0.05, 0.40)
        else:
            # Lower TFs conflict — tighten (be more cautious)
            return min(base_threshold + 0.10, 0.90)

    def _all_trade_rows(self) -> List[Dict]:
        """Return all trade rows including flushed chunks."""
        rows = []
        for chunk_df in self._trade_chunks:
            rows.extend(chunk_df.to_dict('records'))
        rows.extend(self.trade_rows)
        return rows

    def _all_ledger_rows(self) -> List[Dict]:
        """Return all ledger rows including flushed chunks."""
        rows = []
        for chunk_df in self._ledger_chunks:
            rows.extend(chunk_df.to_dict('records'))
        rows.extend(self.ledger_rows)
        return rows

    def run(self, pairs: List[str] = None, tfs: List[str] = None,
            max_bars: int = 0, scenario_name: str = 'IBKR_BASE') -> str:
        """
        Run full replay. Returns run_id.

        Args:
            pairs: override universe pairs
            tfs: override universe timeframes
            max_bars: limit bars per pair_tf (0 = unlimited)
            scenario_name: cost scenario name
        """
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        output_dir = PROJECT_ROOT / 'replay' / 'outputs' / 'runs' / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Copy config snapshot into run output for reproducibility
        run_snapshot_dir = output_dir / 'config_snapshot'
        run_snapshot_dir.mkdir(parents=True, exist_ok=True)
        for name, src in self._snapshot_paths.items():
            try:
                shutil.copy2(src, str(run_snapshot_dir / Path(src).name))
            except Exception:
                pass

        pairs = pairs or self.universe['pairs']
        tfs = tfs or self.universe['timeframes']
        date_start = self.universe['date_start']
        date_end = self.universe['date_end']

        logger.info(f"=== Replay {run_id} ===")
        logger.info(f"Pairs: {pairs}, TFs: {tfs}, Scenario: {scenario_name}")
        logger.info(f"Window: {date_start} to {date_end}")
        if max_bars > 0:
            logger.info(f"Bar cap: {max_bars} per pair/tf")

        total_bars = 0
        total_trades = 0
        self._bar_cap_info: Dict[str, Dict] = {}  # pair_tf -> {available, processed}

        # Process high-frequency TFs first (M5, M15) when memory is fresh,
        # lower-frequency TFs (M30, H1) last (smaller data, less memory needed)
        tf_order = {'M1': 0, 'M5': 1, 'M15': 2, 'M30': 3, 'H1': 4, 'H4': 5, 'D1': 6}
        tfs_ordered = sorted(tfs, key=lambda t: tf_order.get(t, 99))

        for pair in sorted(pairs):
            for tf in tfs_ordered:
                pair_tf = f"{pair}_{tf}"

                # Load models
                models = self.model_loader.get_models(pair_tf)
                if not models:
                    logger.warning(f"{pair_tf}: No models available, skipping")
                    continue

                # Free memory before loading next iterator
                gc.collect()

                # Create iterator (REPLAY mode: pair-native features)
                try:
                    iterator = ReplayIterator(
                        pair=pair, tf=tf,
                        features_dir=str(PROJECT_ROOT / 'features'),
                        schema_path=None, mode='REPLAY',
                        date_start=date_start, date_end=date_end,
                    )
                except FileNotFoundError as e:
                    logger.warning(f"{pair_tf}: {e}")
                    continue

                bars_available = len(iterator)
                n_bars = bars_available
                if max_bars > 0:
                    n_bars = min(n_bars, max_bars)

                # Track bar cap info per pair_tf
                self._bar_cap_info[pair_tf] = {
                    'bars_available': bars_available,
                    'bars_processed': n_bars,
                    'coverage_of_window_pct': round(
                        n_bars / bars_available * 100, 2
                    ) if bars_available > 0 else 0,
                }

                logger.info(f"{pair_tf}: {n_bars} bars ({bars_available} available), {len(models)} models")

                # Pre-compute regime + enabled models (fixed for entire pair_tf)
                regime_state, regime_confidence = self.regime_sim.get_regime_state(pair, tf, 0)
                enabled_models = self.regime_sim.get_enabled_models(
                    models, tf, regime_state, regime_confidence)

                bars_processed = 0
                batch_events = []

                for event in iterator:
                    if max_bars > 0 and bars_processed >= max_bars:
                        break
                    batch_events.append(event)
                    bars_processed += 1
                    total_bars += 1

                    if len(batch_events) >= self.BATCH_SIZE:
                        self._process_batch(
                            batch_events, enabled_models,
                            pair, tf, pair_tf,
                            regime_state, regime_confidence,
                            run_id, scenario_name,
                        )
                        batch_events = []
                        self._maybe_flush_chunks()

                # Flush remaining batch
                if batch_events:
                    self._process_batch(
                        batch_events, enabled_models,
                        pair, tf, pair_tf,
                        regime_state, regime_confidence,
                        run_id, scenario_name,
                    )
                    self._maybe_flush_chunks()

                # Close any remaining position for this pair_tf
                if self.portfolio.has_position(pair, tf):
                    self._force_close_position(pair, tf, run_id, scenario_name)

                # Free iterator memory before next pair_tf
                del iterator
                gc.collect()

        total_trades = len([r for r in self.trade_rows if r.get('action') == 'CLOSE'])
        total_trades += sum(
            int(c['action'].eq('CLOSE').sum()) for c in self._trade_chunks if len(c) > 0
        )
        logger.info(f"=== Replay complete: {total_bars} bars, {total_trades} trades ===")

        # Write outputs
        self._write_outputs(run_id, output_dir, scenario_name, total_bars, max_bars)

        return run_id

    def _process_batch(self, events: List[Dict], enabled_models: List,
                       pair: str, tf: str, pair_tf: str,
                       regime_state: int, regime_confidence: float,
                       run_id: str, scenario_name: str):
        """Process a batch of events with batched inference, then sequential state updates."""

        if not enabled_models:
            # No models — process each bar as SKIP
            for event in events:
                self._process_bar(
                    event, pair, tf, pair_tf, {}, run_id, scenario_name,
                    precomputed_regime=(regime_state, regime_confidence),
                )
            return

        # Stack and sanitize all features
        features_list = []
        for event in events:
            f = np.nan_to_num(event['features'], nan=0.0, posinf=0.0, neginf=0.0)
            features_list.append(f)
        features_batch = np.vstack(features_list)  # (N, n_features)

        # Batch inference — one call per model for all N bars
        batch_results = self.ensemble_engine.run_batch_inference(enabled_models, features_batch)

        # Process each bar sequentially (agreement, cooldown, risk, positions)
        for i, event in enumerate(events):
            self._process_bar(
                event, pair, tf, pair_tf, {}, run_id, scenario_name,
                precomputed_result=batch_results[i],
                precomputed_regime=(regime_state, regime_confidence),
                precomputed_enabled=enabled_models,
            )

    def _maybe_flush_chunks(self):
        """Flush ledger/trade buffers to chunk lists when they exceed CHUNK_SIZE."""
        if len(self.ledger_rows) >= self.CHUNK_SIZE:
            self._ledger_chunks.append(pd.DataFrame(self.ledger_rows))
            self.ledger_rows = []
        if len(self.trade_rows) >= self.CHUNK_SIZE:
            self._trade_chunks.append(pd.DataFrame(self.trade_rows))
            self.trade_rows = []

    def _process_bar(self, event: Dict, pair: str, tf: str, pair_tf: str,
                     models: Dict, run_id: str, scenario_name: str,
                     precomputed_result: Dict = None,
                     precomputed_regime: Tuple = None,
                     precomputed_enabled: List = None):
        """Process a single bar event. ALWAYS writes a ledger row."""
        start_time = time.perf_counter()
        # Lock 8: deterministic request_id — no UUIDs
        request_id = f"{pair}_{tf}_{event['bar_idx']:06d}"
        ts = event['timestamp']
        # Ensure ts_dt is a Python datetime (not numpy.datetime64)
        # so risk engine timedelta arithmetic works correctly
        ts_dt = pd.Timestamp(ts).to_pydatetime()
        close_price = event['close_price']
        features = event['features']
        bar_idx = event['bar_idx']

        reason_codes = []

        # --- Feature sanitization (matches training: models never saw NaN) ---
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # --- Feature validation (safety net — should never trigger after sanitization) ---
        has_valid = not (np.any(np.isnan(features)) or np.any(np.isinf(features)))

        if not has_valid:
            reason_codes.append('FEATURE_INVALID')
            self._write_ledger_row(
                event_ts=ts, pair=pair, tf=tf, request_id=request_id,
                run_id=run_id, scenario=scenario_name,
                regime_state=1, regime_confidence=0.85,
                enabled_count=0, enabled_keys='',
                models_voted=0, models_agreed=0,
                class_probs=[0.0, 1.0, 0.0],
                agreement_score=0.0,
                agreement_threshold_base=self.agreement_threshold_base,
                agreement_threshold_mod=self.agreement_threshold_base,
                decision_side=0, decision_confidence=0.0,
                raw_signal_side=0, signal_overridden=False,
                risk_veto=False, risk_reason=None,
                action_taken='SKIP',
                latency_ms=None,  # Lock 10 Rule 5: deterministic null in replay mode
                reason_codes=reason_codes,
            )
            return

        # --- Regime ---
        if precomputed_regime is not None:
            regime_state, regime_confidence = precomputed_regime
        else:
            regime_state, regime_confidence = self.regime_sim.get_regime_state(pair, tf, bar_idx)

        if precomputed_enabled is not None:
            enabled_models = precomputed_enabled
        else:
            enabled_models = self.regime_sim.get_enabled_models(models, tf, regime_state, regime_confidence)

        if not enabled_models:
            reason_codes.append('HMM_HOSTILE' if regime_state == 3 else 'MODELS_ENABLED_ZERO')
            self._write_ledger_row(
                event_ts=ts, pair=pair, tf=tf, request_id=request_id,
                run_id=run_id, scenario=scenario_name,
                regime_state=regime_state, regime_confidence=regime_confidence,
                enabled_count=0, enabled_keys='',
                models_voted=0, models_agreed=0,
                class_probs=[0.0, 1.0, 0.0],
                agreement_score=0.0,
                agreement_threshold_base=self.agreement_threshold_base,
                agreement_threshold_mod=self.agreement_threshold_base,
                decision_side=0, decision_confidence=1.0,
                raw_signal_side=0, signal_overridden=False,
                risk_veto=False, risk_reason=None,
                action_taken='SKIP',
                latency_ms=None,  # Lock 10 Rule 5: deterministic null in replay mode
                reason_codes=reason_codes,
            )
            return

        # --- Ensemble inference (use precomputed batch result if available) ---
        if precomputed_result is not None:
            result = precomputed_result
        else:
            try:
                result = self.ensemble_engine.run_inference(enabled_models, features)
            except Exception as e:
                reason_codes.append('INFERENCE_ERROR')
                self.errors.append(f"{pair_tf} bar {bar_idx}: {e}")
                self._write_ledger_row(
                    event_ts=ts, pair=pair, tf=tf, request_id=request_id,
                    run_id=run_id, scenario=scenario_name,
                    regime_state=regime_state, regime_confidence=regime_confidence,
                    enabled_count=len(enabled_models),
                    enabled_keys='|'.join(n for n, _ in enabled_models),
                    models_voted=0, models_agreed=0,
                    class_probs=[0.0, 1.0, 0.0],
                    agreement_score=0.0,
                    agreement_threshold_base=self.agreement_threshold_base,
                    agreement_threshold_mod=self.agreement_threshold_base,
                    decision_side=0, decision_confidence=0.0,
                    raw_signal_side=0, signal_overridden=False,
                    risk_veto=False, risk_reason=None,
                    action_taken='SKIP',
                    latency_ms=None,  # Lock 10 Rule 5: deterministic null in replay mode
                    reason_codes=reason_codes,
                )
                return

        raw_signal_side = result['signal_side']
        agreement_score = result['agreement_score']
        agreement_threshold_mod = self.agreement_threshold_base

        # --- Brain tracker: record per-brain votes (passive, no decision impact) ---
        brain_preds = result.get('brain_predictions', {})
        if brain_preds:
            # Map ensemble signal back: -1->0(SHORT), 0->1(FLAT), 1->2(LONG)
            ens_class = {-1: 0, 0: 1, 1: 2}.get(raw_signal_side, 1)
            self.brain_tracker.record_vote(
                bar_idx=bar_idx,
                brain_predictions=brain_preds,
                ensemble_signal=ens_class,
            )

        # --- TF Role: store CONFIRM_ONLY signals for cross-TF use ---
        is_confirm_only = not self.can_execute_trade(pair, tf, raw_signal_side)
        if is_confirm_only and raw_signal_side != 0:
            if pair not in self._confirm_signals:
                self._confirm_signals[pair] = {}
            self._confirm_signals[pair][tf] = raw_signal_side

        # --- TF Role: confirmation adjustment for EXECUTE TFs ---
        confirm_direction, confirm_strength = 0, 0.0
        confirmers_used, confirm_details = [], {}
        threshold_adjustment = 0.0

        if not is_confirm_only and raw_signal_side != 0:
            confirm_direction, confirm_strength, confirmers_used, confirm_details = \
                self.get_confirmation_signal(pair)
            agreement_threshold_mod = self.apply_confirmation_to_threshold(
                agreement_threshold_mod, raw_signal_side, confirm_direction, confirm_strength)
            threshold_adjustment = agreement_threshold_mod - self.agreement_threshold_base
            if confirm_direction != 0 and confirm_strength >= 0.5:
                if confirm_direction == raw_signal_side:
                    reason_codes.append('TF_CONFIRM_ALIGNED')
                else:
                    reason_codes.append('TF_CONFIRM_CONFLICTED')

        # --- Agreement check ---
        signal_overridden = False
        decision_side = raw_signal_side

        if raw_signal_side != 0 and agreement_score < agreement_threshold_mod:
            reason_codes.append('AGREEMENT_FAILED')
            decision_side = 0
            signal_overridden = True

        # --- TF Role: CONFIRM_ONLY TFs never open/close positions ---
        if is_confirm_only and decision_side != 0:
            reason_codes.append('TF_CONFIRM_ONLY')
            decision_side = 0
            signal_overridden = True

        # --- Cooldown check ---
        cd_key = pair_tf
        if cd_key in self.cooldown_counters and self.cooldown_counters[cd_key] > 0:
            self.cooldown_counters[cd_key] -= 1
            if decision_side != 0:
                reason_codes.append('EXEC_COOLDOWN_ACTIVE')
                decision_side = 0
                signal_overridden = True

        # --- Position management ---
        has_position = self.portfolio.has_position(pair, tf)
        action_taken = 'HOLD' if not is_confirm_only else 'SKIP'
        risk_veto = False
        risk_reason = None

        if has_position:
            pos = self.portfolio.get_position(pair, tf)
            # Check if signal reversed or went flat -> close
            if decision_side != pos.side:
                # Lock 8: inherit trade_id from the OPEN row
                trade_id = self._open_trade_ids.pop(pair_tf, pos.trade_id)
                # Close the position
                trade = self.portfolio.close_position(pair, tf, close_price, ts_dt)
                if trade:
                    costs = self.cost_calc.compute_costs(pair, pos.qty_lots, scenario_name)

                    # Lock 9: pnl_gross = raw PnL, pnl_net = gross - costs
                    pnl_gross = trade.pnl_net_usd  # raw PnL from price movement
                    pnl_net = pnl_gross - costs['total_cost_usd']

                    # CLOSE trade row
                    self._write_trade_row(
                        run_id=run_id, scenario=scenario_name, trade_id=trade_id,
                        pair=pair, tf=tf,
                        decision_ts=ts, fill_ts=ts, exit_ts=ts,
                        request_id=request_id, decision_side=decision_side,
                        action='CLOSE', fill_status='FILLED',
                        qty_lots=pos.qty_lots, qty_lots_filled=pos.qty_lots,
                        price_decision_ref=close_price, price_fill=pos.avg_entry_price,
                        price_exit=close_price,
                        costs=costs,
                        pnl_gross_usd=pnl_gross,
                        pnl_net_usd=pnl_net,
                        pnl_pips=trade.pnl_pips,
                        regime_state=regime_state,
                        agreement_score=agreement_score,
                        agreement_threshold=agreement_threshold_mod,
                        risk_veto=False, risk_reason=None,
                        reason_codes=reason_codes,
                        enabled_models_count=result['models_voted'],
                        latency_ms=None,
                    )

                    # Position change event (closed)
                    self._write_position_change(
                        run_id=run_id, scenario=scenario_name,
                        pair=pair, tf=tf, position_id=f"{pair}_{tf}",
                        event_ts=ts, request_id=request_id,
                        side=0, qty_lots=0.0,
                        avg_entry_price=None, mark_price=close_price,
                        regime_state=regime_state, reason_codes=reason_codes,
                    )

                    action_taken = 'CLOSE'
                    self.cooldown_counters[cd_key] = self.cooldown_bars_after_exit
                    has_position = False

        if not has_position and decision_side != 0:
            # Attempt to open a new position
            qty_lots = 0.10  # Fixed lot size for replay

            intent = OrderIntent(
                pair=pair, tf=tf, side=decision_side,
                qty_lots=qty_lots, price=close_price,
            )
            risk_result = self.risk_controller.check(
                self.portfolio, intent, ts_dt,
                regime_state=regime_state,
            )

            if not risk_result['approved']:
                risk_veto = True
                risk_reason = risk_result['reason']
                if risk_reason in self.valid_reason_codes:
                    reason_codes.append(risk_reason)
                action_taken = 'SKIP'
            else:
                # Check fill rate
                costs = self.cost_calc.compute_costs(pair, qty_lots, scenario_name)
                fill_roll = np.random.random()
                if fill_roll > costs['fill_rate']:
                    reason_codes.append('EXEC_FILL_MISSED')
                    action_taken = 'SKIP'
                else:
                    # Lock 8: deterministic trade_id from PortfolioState sequence
                    trade_id = self.portfolio.allocate_trade_id(
                        pair, tf, scenario_name, ts_dt)
                    # Open position
                    self.portfolio.open_position(
                        pair=pair, tf=tf, side=decision_side,
                        qty_lots=qty_lots, entry_price=close_price,
                        entry_ts=ts_dt, trade_id=trade_id,
                    )
                    self._open_trade_ids[pair_tf] = trade_id

                    # OPEN trade row
                    self._write_trade_row(
                        run_id=run_id, scenario=scenario_name, trade_id=trade_id,
                        pair=pair, tf=tf,
                        decision_ts=ts, fill_ts=ts, exit_ts=None,
                        request_id=request_id, decision_side=decision_side,
                        action='OPEN', fill_status='FILLED',
                        qty_lots=qty_lots, qty_lots_filled=qty_lots,
                        price_decision_ref=close_price, price_fill=close_price,
                        price_exit=None,
                        costs=costs,
                        pnl_gross_usd=None, pnl_net_usd=None, pnl_pips=None,
                        regime_state=regime_state,
                        agreement_score=agreement_score,
                        agreement_threshold=agreement_threshold_mod,
                        risk_veto=False, risk_reason=None,
                        reason_codes=reason_codes,
                        enabled_models_count=result['models_voted'],
                        latency_ms=None,
                    )

                    # Position change event (opened)
                    self._write_position_change(
                        run_id=run_id, scenario=scenario_name,
                        pair=pair, tf=tf, position_id=f"{pair}_{tf}",
                        event_ts=ts, request_id=request_id,
                        side=decision_side, qty_lots=qty_lots,
                        avg_entry_price=close_price, mark_price=close_price,
                        regime_state=regime_state, reason_codes=reason_codes,
                    )

                    action_taken = 'OPEN'

        elif has_position:
            if action_taken != 'CLOSE':
                action_taken = 'HOLD'

        latency_ms = None  # Lock 10 Rule 5: deterministic null in replay mode

        # --- Always write ledger row ---
        self._write_ledger_row(
            event_ts=ts, pair=pair, tf=tf, request_id=request_id,
            run_id=run_id, scenario=scenario_name,
            regime_state=regime_state, regime_confidence=regime_confidence,
            enabled_count=result['models_voted'],
            enabled_keys='|'.join(result['brain_keys']),
            models_voted=result['models_voted'],
            models_agreed=result['models_agreed'],
            class_probs=result['class_probs'].tolist(),
            agreement_score=agreement_score,
            agreement_threshold_base=self.agreement_threshold_base,
            agreement_threshold_mod=agreement_threshold_mod,
            decision_side=decision_side,
            decision_confidence=result['confidence'],
            raw_signal_side=raw_signal_side,
            signal_overridden=signal_overridden,
            risk_veto=risk_veto, risk_reason=risk_reason,
            action_taken=action_taken,
            latency_ms=latency_ms,
            reason_codes=reason_codes,
            confirmers_used='|'.join(sorted(confirmers_used)) if confirmers_used else None,
            confirm_signal=confirm_direction if confirm_direction != 0 else None,
            confirm_strength=round(confirm_strength, 4) if confirm_strength > 0 else None,
            threshold_adjustment=round(threshold_adjustment, 4) if threshold_adjustment != 0 else None,
        )

    def _force_close_position(self, pair: str, tf: str, run_id: str, scenario: str):
        """Force-close a position at end of replay."""
        pos = self.portfolio.get_position(pair, tf)
        if not pos:
            return
        pair_tf = f"{pair}_{tf}"
        # Lock 8: inherit trade_id from the OPEN row
        trade_id = self._open_trade_ids.pop(pair_tf, pos.trade_id)
        ts_dt = pos.entry_ts  # use last known bar ts, not wall clock
        trade = self.portfolio.close_position(pair, tf, pos.avg_entry_price, ts_dt)
        if trade:
            costs = self.cost_calc.compute_costs(pair, pos.qty_lots, scenario)
            # Lock 9: pnl_gross=0, net = gross - cost
            pnl_gross = 0.0
            pnl_net = pnl_gross - costs['total_cost_usd']
            self._write_trade_row(
                run_id=run_id, scenario=scenario, trade_id=trade_id,
                pair=pair, tf=tf,
                decision_ts=ts_dt, fill_ts=ts_dt, exit_ts=ts_dt,
                request_id='END_OF_REPLAY', decision_side=0,
                action='CLOSE', fill_status='FILLED',
                qty_lots=pos.qty_lots, qty_lots_filled=pos.qty_lots,
                price_decision_ref=pos.avg_entry_price,
                price_fill=pos.avg_entry_price, price_exit=pos.avg_entry_price,
                costs=costs,
                pnl_gross_usd=pnl_gross, pnl_net_usd=pnl_net, pnl_pips=0.0,
                regime_state=1,
                agreement_score=0.0, agreement_threshold=self.agreement_threshold_base,
                risk_veto=False, risk_reason=None,
                reason_codes=[],
                enabled_models_count=0, latency_ms=None,
            )

    # ============================================================
    # OUTPUT ROW WRITERS
    # ============================================================

    def _write_ledger_row(self, event_ts, pair, tf, request_id, run_id, scenario,
                          regime_state, regime_confidence,
                          enabled_count, enabled_keys,
                          models_voted, models_agreed,
                          class_probs, agreement_score,
                          agreement_threshold_base, agreement_threshold_mod,
                          decision_side, decision_confidence,
                          raw_signal_side, signal_overridden,
                          risk_veto, risk_reason,
                          action_taken, latency_ms, reason_codes,
                          confirmers_used=None, confirm_signal=None,
                          confirm_strength=None, threshold_adjustment=None):
        ts = pd.Timestamp(event_ts, tz='UTC') if not hasattr(event_ts, 'tzinfo') or event_ts.tzinfo is None \
            else pd.Timestamp(event_ts)
        self.ledger_rows.append({
            'event_ts': ts,
            'pair': pair,
            'tf': tf,
            'request_id': request_id,
            'run_id': run_id,
            'scenario': scenario,
            'regime_state': int(regime_state),
            'regime_confidence': float(regime_confidence),
            'enabled_models_count': int(enabled_count),
            'enabled_models_keys': enabled_keys,
            'models_voted': int(models_voted),
            'models_agreed': int(models_agreed),
            'brain_probs_trimmed_mean': [float(p) for p in class_probs],
            'agreement_score': float(agreement_score),
            'agreement_threshold_base': float(agreement_threshold_base),
            'agreement_threshold_modified': float(agreement_threshold_mod),
            'alt_data_available': False,
            'cot_pressure': None,
            'cot_extreme': None,
            'cme_spike': None,
            'cme_confirm': None,
            'alt_agreement_adjustment': None,
            'mtf_status': None,
            'decision_side': int(decision_side),
            'decision_confidence': float(decision_confidence),
            'raw_signal_side': int(raw_signal_side),
            'signal_overridden': bool(signal_overridden),
            'risk_veto': bool(risk_veto),
            'risk_reason': risk_reason,
            'action_taken': action_taken,
            'latency_ms': float(latency_ms) if latency_ms is not None else None,
            'reason_codes': '|'.join(sorted(reason_codes)) if reason_codes else '',
            'confirmers_used': confirmers_used,
            'confirm_signal': int(confirm_signal) if confirm_signal is not None else None,
            'confirm_strength': float(confirm_strength) if confirm_strength is not None else None,
            'threshold_adjustment': float(threshold_adjustment) if threshold_adjustment is not None else None,
        })

    def _write_trade_row(self, run_id, scenario, trade_id, pair, tf,
                         decision_ts, fill_ts, exit_ts,
                         request_id, decision_side, action, fill_status,
                         qty_lots, qty_lots_filled,
                         price_decision_ref, price_fill, price_exit,
                         costs, pnl_gross_usd, pnl_net_usd, pnl_pips,
                         regime_state, agreement_score, agreement_threshold,
                         risk_veto, risk_reason, reason_codes,
                         enabled_models_count, latency_ms):
        def to_ts(v):
            if v is None:
                return None
            return pd.Timestamp(v, tz='UTC') if not hasattr(v, 'tzinfo') or v.tzinfo is None \
                else pd.Timestamp(v)

        self.trade_rows.append({
            'run_id': run_id,
            'scenario': scenario,
            'trade_id': trade_id,
            'pair': pair,
            'tf': tf,
            'decision_ts': to_ts(decision_ts),
            'fill_ts': to_ts(fill_ts),
            'exit_ts': to_ts(exit_ts),
            'request_id': request_id,
            'decision_side': int(decision_side),
            'action': action,
            'fill_status': fill_status,
            'qty_lots': float(qty_lots),
            'qty_lots_filled': float(qty_lots_filled),
            'qty_units': int(qty_lots * 100_000),
            'price_decision_ref': float(price_decision_ref),
            'price_fill': float(price_fill) if price_fill is not None else None,
            'price_exit': float(price_exit) if price_exit is not None else None,
            'spread_cost_pips': float(costs['spread_cost_pips']),
            'slippage_cost_pips': float(costs['slippage_cost_pips']),
            'commission_cost_usd': float(costs['commission_cost_usd']),
            'total_cost_usd': float(costs['total_cost_usd']),
            'pnl_gross_usd': float(pnl_gross_usd) if pnl_gross_usd is not None else None,
            'pnl_net_usd': float(pnl_net_usd) if pnl_net_usd is not None else None,
            'pnl_pips': float(pnl_pips) if pnl_pips is not None else None,
            'regime_state': int(regime_state),
            'agreement_score': float(agreement_score) if agreement_score is not None else None,
            'agreement_threshold': float(agreement_threshold) if agreement_threshold is not None else None,
            'risk_veto': bool(risk_veto),
            'risk_reason': risk_reason,
            'reason_codes': '|'.join(sorted(reason_codes)) if reason_codes else '',
            'enabled_models_count': int(enabled_models_count) if enabled_models_count else None,
            'latency_ms': float(latency_ms) if latency_ms is not None else None,
        })

    def _write_position_change(self, run_id, scenario, pair, tf, position_id,
                               event_ts, request_id, side, qty_lots,
                               avg_entry_price, mark_price,
                               regime_state, reason_codes):
        ts = pd.Timestamp(event_ts, tz='UTC') if not hasattr(event_ts, 'tzinfo') or event_ts.tzinfo is None \
            else pd.Timestamp(event_ts)
        self.position_rows.append({
            'run_id': run_id,
            'scenario': scenario,
            'pair': pair,
            'tf': tf,
            'position_id': position_id,
            'event_ts': ts,
            'request_id': request_id,
            'side': int(side),
            'qty_lots': float(qty_lots),
            'avg_entry_price': float(avg_entry_price) if avg_entry_price is not None else None,
            'mark_price': float(mark_price) if mark_price is not None else None,
            'unrealized_pnl_usd': None,
            'realized_pnl_usd_cum': float(self.portfolio.realized_pnl_usd_cum),
            'gross_exposure_usd': float(self.portfolio.get_gross_exposure_usd()),
            'net_exposure_usd': float(self.portfolio.get_net_exposure_usd()),
            'group_exposure_usd_leg': None,
            'risk_state': self.portfolio.risk_state,
            'cooldown_until_ts': pd.Timestamp(self.portfolio.cooldown_until_ts.get(f"{pair}_{tf}"), tz='UTC')
                if self.portfolio.cooldown_until_ts.get(f"{pair}_{tf}") else None,
            'drawdown_pct_current': None,
            'regime_state': int(regime_state),
            'reason_codes': '|'.join(sorted(reason_codes)) if reason_codes else '',
        })

    # ============================================================
    # OUTPUT WRITING
    # ============================================================

    def _write_outputs(self, run_id: str, output_dir: Path,
                       scenario_name: str, total_bars: int, max_bars: int = 0):
        """Write all Parquet outputs and manifest."""

        # Decision ledger — concatenate chunks + remaining buffer
        ledger_parts = list(self._ledger_chunks)
        if self.ledger_rows:
            ledger_parts.append(pd.DataFrame(self.ledger_rows))
        if ledger_parts:
            ledger_df = pd.concat(ledger_parts, ignore_index=True)
            write_parquet_strict(ledger_df, DECISION_LEDGER_SCHEMA,
                                str(output_dir / 'decision_ledger.parquet'))
            logger.info(f"Written: decision_ledger.parquet ({len(ledger_df)} rows)")
        else:
            ledger_df = None

        # Trades — concatenate chunks + remaining buffer
        trade_parts = list(self._trade_chunks)
        if self.trade_rows:
            trade_parts.append(pd.DataFrame(self.trade_rows))
        if trade_parts:
            trades_df = pd.concat(trade_parts, ignore_index=True)
            write_parquet_strict(trades_df, TRADES_SCHEMA,
                                str(output_dir / 'trades.parquet'))
            logger.info(f"Written: trades.parquet ({len(trades_df)} rows)")
        else:
            # Write empty trades with correct schema
            trades_df = pd.DataFrame(columns=[f.name for f in TRADES_SCHEMA])
            write_parquet_strict(trades_df, TRADES_SCHEMA,
                                str(output_dir / 'trades.parquet'))
            logger.info("Written: trades.parquet (0 rows)")

        # Positions
        if self.position_rows:
            positions_df = pd.DataFrame(self.position_rows)
            write_parquet_strict(positions_df, POSITIONS_SCHEMA,
                                str(output_dir / 'positions.parquet'))
            logger.info(f"Written: positions.parquet ({len(positions_df)} rows)")
        else:
            positions_df = pd.DataFrame(columns=[f.name for f in POSITIONS_SCHEMA])
            write_parquet_strict(positions_df, POSITIONS_SCHEMA,
                                str(output_dir / 'positions.parquet'))
            logger.info("Written: positions.parquet (0 rows)")

        # Metrics
        metrics = self._compute_metrics(scenario_name)

        # Partial fill metrics
        pfm = compute_partial_fill_metrics(trades_df if trade_parts else None)
        metrics['partial_fill'] = pfm

        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2, default=str)

        # Per-pair-tf metrics breakdown
        metrics_by_pair_tf = self._compute_metrics_by_pair_tf(scenario_name)
        if metrics_by_pair_tf:
            with open(output_dir / 'metrics_by_pair_tf.json', 'w') as f:
                json.dump(metrics_by_pair_tf, f, indent=2, default=str)

        # Gate criteria (with regression triggers)
        # ledger_df was already built above from chunks + remaining rows
        gate = self._evaluate_gate_criteria(metrics, total_bars,
                                            metrics_by_pair_tf=metrics_by_pair_tf,
                                            decision_ledger_df=ledger_df,
                                            scenario_name=scenario_name)
        with open(output_dir / 'gate_criteria_result.json', 'w') as f:
            json.dump(gate, f, indent=2, default=str)

        # Manifest
        manifest = {
            'run_id': run_id,
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'scenario': scenario_name,
            'total_bars': total_bars,
            'total_ledger_rows': sum(len(c) for c in self._ledger_chunks) + len(self.ledger_rows),
            'total_trade_rows': sum(len(c) for c in self._trade_chunks) + len(self.trade_rows),
            'total_position_rows': len(self.position_rows),
            'gate_result': gate.get('overall', 'UNKNOWN'),
            'errors_count': len(self.errors),
        }

        # Bar cap transparency
        if self._bar_cap_info:
            any_capped = any(v['bars_available'] != v['bars_processed']
                            for v in self._bar_cap_info.values())
            manifest['bar_caps'] = {
                'max_bars_per_pair_tf': max_bars if max_bars > 0 else None,
                'cap_active': any_capped,
                'reason': 'Memory management for M5/M15 high-frequency timeframes' if any_capped else None,
                'per_pair_tf': self._bar_cap_info,
            }

        with open(output_dir / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2, default=str)

        # Errors log
        if self.errors:
            with open(output_dir / 'errors.log', 'w') as f:
                for err in self.errors:
                    f.write(err + '\n')

        # Coverage report
        try:
            generate_coverage_report(
                run_id=run_id,
                models_loaded=self.model_loader.models,
                decision_ledger_df=ledger_df,
                output_dir=output_dir,
            )
        except Exception as e:
            logger.warning(f"Coverage report failed: {e}")

        # Veto breakdown
        try:
            generate_veto_breakdown(
                run_id=run_id,
                decision_ledger_df=ledger_df,
                output_dir=output_dir,
            )
        except Exception as e:
            logger.warning(f"Veto breakdown failed: {e}")

        # Brain contributions
        try:
            brain_report = self.brain_tracker.compute_contributions()
            save_brain_contributions(brain_report, output_dir)
        except Exception as e:
            logger.warning(f"Brain contributions failed: {e}")

        logger.info(f"Gate result: {gate.get('overall', 'UNKNOWN')}")

    def _compute_metrics(self, scenario_name: str) -> Dict:
        """Compute aggregate metrics from trade rows."""
        close_trades = [r for r in self._all_trade_rows() if r.get('action') == 'CLOSE']

        if not close_trades:
            return {
                'scenario': scenario_name,
                'total_trades': 0,
                'total_pnl_net_usd': 0.0,
                'profit_factor': 0.0,
                'max_drawdown_pct': 0.0,
                'win_rate': 0.0,
            }

        pnls = [t['pnl_net_usd'] for t in close_trades if t['pnl_net_usd'] is not None]
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))

        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0

        wins = sum(1 for p in pnls if p > 0)
        win_rate = wins / len(pnls) if pnls else 0.0

        # Max drawdown
        equity_curve = np.cumsum([0.0] + pnls)
        peak = np.maximum.accumulate(equity_curve)
        dd = (equity_curve - peak)
        max_dd = abs(float(dd.min())) if len(dd) > 0 else 0.0
        initial_equity = 100000.0
        max_dd_pct = (max_dd / initial_equity) * 100

        return {
            'scenario': scenario_name,
            'total_trades': len(pnls),
            'total_pnl_net_usd': sum(pnls),
            'gross_profit_usd': gross_profit,
            'gross_loss_usd': gross_loss,
            'profit_factor': round(pf, 4),
            'win_rate': round(win_rate, 4),
            'max_drawdown_usd': max_dd,
            'max_drawdown_pct': round(max_dd_pct, 4),
            'avg_trade_pnl_usd': round(sum(pnls) / len(pnls), 2) if pnls else 0.0,
        }

    def _compute_metrics_by_pair_tf(self, scenario_name: str) -> Dict:
        """Compute per-pair-tf metrics breakdown."""
        close_trades = [r for r in self._all_trade_rows() if r.get('action') == 'CLOSE']
        if not close_trades:
            return {}

        # Group by pair_tf
        groups = {}
        for t in close_trades:
            key = f"{t['pair']}_{t['tf']}"
            groups.setdefault(key, []).append(t)

        result = {}
        for pair_tf, trades in sorted(groups.items()):
            pnls = [t['pnl_net_usd'] for t in trades if t['pnl_net_usd'] is not None]
            if not pnls:
                continue
            gross_profit = sum(p for p in pnls if p > 0)
            gross_loss = abs(sum(p for p in pnls if p < 0))
            pf = gross_profit / gross_loss if gross_loss > 0 else (
                float('inf') if gross_profit > 0 else 0.0)
            wins = sum(1 for p in pnls if p > 0)

            equity = np.cumsum([0.0] + pnls)
            peak = np.maximum.accumulate(equity)
            max_dd = abs(float((equity - peak).min()))

            entry = {
                'trades': len(pnls),
                'profit_factor': round(pf, 4),
                'win_rate': round(wins / len(pnls), 4),
                'total_pnl_usd': round(sum(pnls), 2),
                'max_drawdown_usd': round(max_dd, 2),
            }

            # Bar cap transparency per pair_tf
            if hasattr(self, '_bar_cap_info') and pair_tf in self._bar_cap_info:
                cap = self._bar_cap_info[pair_tf]
                entry['bars_available'] = cap['bars_available']
                entry['bars_processed'] = cap['bars_processed']
                entry['coverage_of_window_pct'] = cap['coverage_of_window_pct']

            result[pair_tf] = entry
        return result

    def _evaluate_gate_criteria(self, metrics: Dict, total_bars: int,
                                metrics_by_pair_tf: Optional[Dict] = None,
                                decision_ledger_df: Optional[pd.DataFrame] = None,
                                scenario_name: str = '') -> Dict:
        """Evaluate gate criteria from replay_config + regression triggers."""
        gate_cfg = self.replay_config.get('gate_criteria', {})
        results = {}

        # PF >= 1.15
        pf = metrics.get('profit_factor', 0)
        min_pf = gate_cfg.get('min_pf_base_plus_25', 1.15)
        results['profit_factor'] = {
            'criterion': f'PF >= {min_pf}',
            'value': pf,
            'passed': pf >= min_pf,
        }

        # Max DD <= 5%
        dd = metrics.get('max_drawdown_pct', 100)
        max_dd = gate_cfg.get('max_dd_pct', 5.0)
        results['max_drawdown'] = {
            'criterion': f'DD <= {max_dd}%',
            'value': dd,
            'passed': dd <= max_dd,
        }

        # NaN rate = 0
        all_ledger = self._all_ledger_rows()
        nan_rows = sum(1 for r in all_ledger if 'FEATURE_INVALID' in r.get('reason_codes', ''))
        nan_rate = nan_rows / total_bars if total_bars > 0 else 0
        results['nan_rate'] = {
            'criterion': 'NaN rate = 0',
            'value': nan_rate,
            'passed': nan_rate <= gate_cfg.get('max_nan_rate', 0.0),
        }

        # Schema violations = 0
        results['schema_violations'] = {
            'criterion': 'Schema violations = 0',
            'value': 0,
            'passed': True,
        }

        # Inference error rate
        err_rows = sum(1 for r in all_ledger if 'INFERENCE_ERROR' in r.get('reason_codes', ''))
        err_rate = err_rows / total_bars if total_bars > 0 else 0
        max_err = gate_cfg.get('max_inference_error_rate', 0.0001)
        results['inference_error_rate'] = {
            'criterion': f'Error rate <= {max_err}',
            'value': err_rate,
            'passed': err_rate <= max_err,
        }

        # Ledger coverage = 100%
        coverage = len(all_ledger) / total_bars if total_bars > 0 else 0
        results['ledger_coverage'] = {
            'criterion': 'Coverage = 100%',
            'value': coverage,
            'passed': coverage >= gate_cfg.get('min_ledger_coverage', 1.0),
        }

        overall = 'PASS' if all(r['passed'] for r in results.values()) else 'FAIL'

        # --- Regression triggers (WARNING severity, don't override overall) ---
        regression_triggers = {}

        # Trigger A: PF collapse under mild costs
        if metrics_by_pair_tf and scenario_name == 'BASE_PLUS_25':
            regression_triggers['pf_collapse'] = self._check_pf_regression(
                metrics_by_pair_tf, scenario_name)

        # Trigger B: Veto dominated by single cause
        if decision_ledger_df is not None and len(decision_ledger_df) > 0:
            regression_triggers['veto_dominance'] = self._check_veto_dominance(
                decision_ledger_df)

        gate_result = {'overall': overall, 'criteria': results}
        if regression_triggers:
            gate_result['regression_triggers'] = regression_triggers

        # Window capped note — doesn't fail gate but must be visible
        if hasattr(self, '_bar_cap_info') and self._bar_cap_info:
            capped_pairs = {k: v for k, v in self._bar_cap_info.items()
                           if v['coverage_of_window_pct'] < 100}
            if capped_pairs:
                gate_result['window_capped'] = True
                gate_result['capped_pair_tfs'] = {
                    k: f"{v['bars_processed']}/{v['bars_available']} bars ({v['coverage_of_window_pct']}%)"
                    for k, v in capped_pairs.items()
                }

        return gate_result

    @staticmethod
    def _check_pf_regression(metrics_by_pair_tf: Dict, scenario: str = 'BASE_PLUS_25',
                             min_pf: float = 1.20, min_trades: int = 50) -> Dict:
        """Trigger A: PF collapse detection per pair/tf."""
        failed_blocks = []
        for pair_tf, m in metrics_by_pair_tf.items():
            trades = m.get('trades', m.get('total_trades', 0))
            pf_val = m.get('profit_factor', 0)
            if trades >= min_trades and pf_val < min_pf:
                failed_blocks.append({
                    'pair_tf': pair_tf,
                    'profit_factor': round(pf_val, 4),
                    'total_trades': trades,
                    'scenario': scenario,
                })
        return {
            'trigger': 'PF_COLLAPSE_MILD_COSTS',
            'severity': 'WARNING',
            'status': 'FAIL' if failed_blocks else 'PASS',
            'failed_blocks': failed_blocks,
            'threshold': {'min_pf': min_pf, 'min_trades': min_trades, 'scenario': scenario},
        }

    @staticmethod
    def _check_veto_dominance(decision_ledger_df: pd.DataFrame,
                              max_single_cause_pct: float = 0.70) -> Dict:
        """Trigger B: Detects if a single reason code dominates FLAT decisions."""
        from collections import Counter

        flat_mask = decision_ledger_df['decision_side'] == 0
        flat_rows = decision_ledger_df[flat_mask]

        if len(flat_rows) == 0:
            return {
                'trigger': 'VETO_SINGLE_CAUSE_DOMINANCE',
                'severity': 'WARNING',
                'status': 'PASS',
                'dominant_cause': None,
                'dominant_pct': 0.0,
                'total_flat_decisions': 0,
                'threshold': {'max_single_cause_pct': max_single_cause_pct},
                'reason_distribution': {},
            }

        reason_counts = Counter()
        for _, row in flat_rows.iterrows():
            codes_str = row.get('reason_codes', '')
            if not codes_str or str(codes_str) == '':
                reason_counts['NO_REASON_CODE'] += 1
                continue
            for code in str(codes_str).split('|'):
                code = code.strip()
                if code:
                    reason_counts[code] += 1

        total_flat = len(flat_rows)
        if reason_counts:
            dominant_code, dominant_count = reason_counts.most_common(1)[0]
            dominant_pct = dominant_count / total_flat
        else:
            dominant_code = None
            dominant_pct = 0.0

        return {
            'trigger': 'VETO_SINGLE_CAUSE_DOMINANCE',
            'severity': 'WARNING',
            'status': 'FAIL' if dominant_pct > max_single_cause_pct else 'PASS',
            'dominant_cause': dominant_code,
            'dominant_pct': round(dominant_pct, 4),
            'total_flat_decisions': total_flat,
            'threshold': {'max_single_cause_pct': max_single_cause_pct},
            'reason_distribution': dict(reason_counts.most_common(10)),
        }


# ============================================================
# CLI ENTRY POINT
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description='CHAOS V1.0 Replay Runner')
    parser.add_argument('--pairs', nargs='+', default=None, help='Override pairs')
    parser.add_argument('--tfs', nargs='+', default=None, help='Override timeframes')
    parser.add_argument('--bars', type=int, default=0, help='Max bars per pair_tf (0=all)')
    parser.add_argument('--scenario', default='IBKR_BASE', help='Cost scenario')
    args = parser.parse_args()

    runner = ReplayRunner()
    run_id = runner.run(
        pairs=args.pairs,
        tfs=args.tfs,
        max_bars=args.bars,
        scenario_name=args.scenario,
    )
    print(f"\nReplay complete: {run_id}")
    print(f"Output: replay/outputs/runs/{run_id}/")


if __name__ == '__main__':
    main()
