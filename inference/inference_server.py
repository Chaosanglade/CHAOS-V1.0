"""
CHAOS V1.0 Inference Server
============================
Receives ZeroMQ requests from MT5 EA, runs ensemble inference, returns decisions.

# MT5 NEVER decides model logic. MT5 only:
# 1. Assembles features from price data
# 2. Sends ZeroMQ request
# 3. Receives decision
# 4. Executes with CoreArb DLL + risk constraints

Architecture:
  Layer 1: Brain Voting (trimmed mean aggregation per pair/tf)
  Layer 2: Timeframe Confirmation (M30 primary, H1 required bias)
  Layer 3: Cross-Pair Correlation (USD-beta capping)
  Layer 4: Regime Gating (pre-signal, applied BEFORE inference)
"""
import zmq
import json
import time
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('chaos_inference')

# Import backends from onnx_export module
import sys
sys.path.insert(0, str(Path(__file__).parent))
from onnx_export import OnnxBackend, SklearnBackend


class ChaosInferenceServer:

    def __init__(self, schema_path, models_dir, regime_policy_path,
                 ensemble_config_path, port=5555):
        """
        Args:
            schema_path: path to schema/feature_schema.json
            models_dir: path to directory containing .onnx and .joblib models
            regime_policy_path: path to regime/regime_policy.json
            ensemble_config_path: path to ensemble/ensemble_config.json
            port: ZeroMQ REP socket port
        """
        self.port = port
        self.schema = self._load_schema(schema_path)
        self.regime_policy = self._load_json(regime_policy_path)
        self.ensemble_config = self._load_json(ensemble_config_path)
        self.models = self._load_models(models_dir)

    def _load_schema(self, path):
        with open(path) as f:
            schema = json.load(f)
        logger.info(f"Schema loaded: {schema['feature_count']} features, v{schema['version']}")
        return schema

    def _load_json(self, path):
        with open(path) as f:
            return json.load(f)

    def _load_models(self, models_dir):
        """
        Load all models into memory, organized by symbol_timeframe.
        Returns dict: {
            'EURUSD_M30': {
                'lgb_optuna': OnnxBackend(...),
                'transformer_optuna': OnnxBackend(...),
                'rf_optuna': SklearnBackend(...),
                ...
            }
        }

        - Scans models_dir for .onnx files (GPU brains) and .joblib files (RF/ET)
        - Parses model_key from filename: {PAIR}_{TF}_{BRAIN}.{ext}
        - Instantiates OnnxBackend for .onnx, SklearnBackend for RF/ET .joblib
        - Groups by PAIR_TF
        - Skips M1, W1, MN1 models (not production-viable)
        """
        models_path = Path(models_dir)
        models = {}

        skip_tfs = {'M1', 'W1', 'MN1'}
        rf_et_brains = {'rf_optuna', 'et_optuna'}
        production_tfs = {'M5', 'M15', 'M30', 'H1', 'H4', 'D1'}

        # Load ONNX models
        for onnx_file in sorted(models_path.glob('*.onnx')):
            parts = onnx_file.stem.split('_')
            if len(parts) < 3:
                continue
            pair = parts[0]
            tf = parts[1]
            brain = '_'.join(parts[2:])

            if tf in skip_tfs or tf not in production_tfs:
                continue

            pair_tf = f"{pair}_{tf}"
            if pair_tf not in models:
                models[pair_tf] = {}

            try:
                # Check if corresponding .pt or .joblib has a scaler
                scaler = self._load_scaler(models_path, pair, tf, brain)
                models[pair_tf][brain] = OnnxBackend(str(onnx_file), scaler=scaler)
                logger.info(f"  Loaded ONNX: {pair_tf}/{brain}")
            except Exception as e:
                logger.warning(f"  Failed to load {onnx_file}: {e}")

        # Load RF/ET joblib models (these are NOT converted to ONNX)
        for joblib_file in sorted(models_path.glob('*.joblib')):
            parts = joblib_file.stem.split('_')
            if len(parts) < 3:
                continue
            pair = parts[0]
            tf = parts[1]
            brain = '_'.join(parts[2:])

            if tf in skip_tfs or tf not in production_tfs:
                continue

            # Only load RF/ET via SklearnBackend
            if brain not in rf_et_brains:
                continue

            pair_tf = f"{pair}_{tf}"
            if pair_tf not in models:
                models[pair_tf] = {}

            try:
                models[pair_tf][brain] = SklearnBackend(str(joblib_file))
                logger.info(f"  Loaded sklearn: {pair_tf}/{brain}")
            except Exception as e:
                logger.warning(f"  Failed to load {joblib_file}: {e}")

        total_models = sum(len(v) for v in models.values())
        logger.info(f"Total: {total_models} models across {len(models)} pair/tf combos")
        return models

    def _load_scaler(self, models_path, pair, tf, brain):
        """Try to load the scaler from the original checkpoint."""
        import joblib as jl
        import torch

        # Try .pt first (neural network models)
        pt_path = models_path / f"{pair}_{tf}_{brain}.pt"
        if pt_path.exists():
            try:
                checkpoint = torch.load(str(pt_path), map_location='cpu', weights_only=False)
                return checkpoint.get('scaler', None)
            except Exception:
                return None

        # Try .joblib (tree models)
        jl_path = models_path / f"{pair}_{tf}_{brain}.joblib"
        if jl_path.exists():
            try:
                loaded = jl.load(str(jl_path))
                if isinstance(loaded, dict):
                    return loaded.get('scaler', None)
            except Exception:
                return None

        return None

    def _validate_request(self, request):
        """Validate incoming request against schema."""
        features = np.array(request['features'])
        if features.shape[0] != self.schema['feature_count']:
            raise ValueError(f"Feature count {features.shape[0]} != {self.schema['feature_count']}")
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            raise ValueError("Features contain NaN or Inf")
        return features.reshape(1, -1)

    def _get_enabled_models(self, symbol, timeframe, regime_state, regime_confidence):
        """
        Apply regime gating to determine which models are enabled.
        Returns list of (brain_name, backend) tuples.
        """
        pair_tf_key = f"{symbol}_{timeframe}"
        available = self.models.get(pair_tf_key, {})

        if not available:
            return []

        # Check regime policy
        regime_key = f"REGIME_{regime_state}"
        regime_states = self.regime_policy.get('regime_states', {})
        policy = regime_states.get(regime_key, {})

        # If confidence below threshold, degrade to REGIME_2 (conservative)
        conf_threshold = self.regime_policy.get('confidence_threshold', 0.6)
        if regime_confidence < conf_threshold:
            policy = regime_states.get('REGIME_2', policy)

        allowed_tfs = policy.get('allow', [])
        denied_tfs = policy.get('deny', [])

        # If timeframe is in deny list, return empty (force FLAT)
        if timeframe in denied_tfs:
            return []

        # If allow list is empty (crisis regime), return empty
        if not allowed_tfs and denied_tfs:
            return []

        # If timeframe not in allow list (and allow list is not empty), deny
        if allowed_tfs and timeframe not in allowed_tfs:
            return []

        enabled = [(brain_name, backend) for brain_name, backend in available.items()]
        return enabled

    def _ensemble_aggregate(self, predictions, brain_names, symbol, timeframe):
        """
        Layer 1: Brain Voting — Trimmed mean aggregation.

        Trimmed mean with 10% trim is a standard robust location estimator
        (Wilcox, 2012 — Introduction to Robust Estimation and Hypothesis Testing).
        It reduces sensitivity to outlier models while preserving ensemble diversity.

        Args:
            predictions: list of np.ndarray shape (1, 3)
            brain_names: list of brain name strings
            symbol: pair name
            timeframe: tf name

        Returns:
            aggregated_probs: np.ndarray shape (3,)
            models_voted: int
            models_agreed: int
        """
        if not predictions:
            return np.array([0.0, 1.0, 0.0]), 0, 0  # FLAT if no models

        probs = np.vstack(predictions)  # shape (n_models, 3)
        n_models = probs.shape[0]

        # Load trim fraction from config (default 0.1 = 10%)
        trim_frac = self.ensemble_config.get('layer_1_brain_voting', {}).get('trim_fraction', 0.1)
        trim_count = max(1, int(n_models * trim_frac))

        aggregated = np.zeros(3)
        for cls in range(3):
            sorted_probs = np.sort(probs[:, cls])
            if n_models > 2 * trim_count:
                trimmed = sorted_probs[trim_count:-trim_count]
            else:
                trimmed = sorted_probs  # Not enough models to trim
            aggregated[cls] = trimmed.mean()

        # Normalize to sum to 1.0
        total = aggregated.sum()
        if total > 0:
            aggregated /= total
        else:
            aggregated = np.array([0.0, 1.0, 0.0])

        # Count agreement
        ensemble_signal = int(np.argmax(aggregated))
        individual_signals = np.argmax(probs, axis=1)
        models_agreed = int(np.sum(individual_signals == ensemble_signal))

        # Check minimum agreement ratio
        min_agreement = self.ensemble_config.get('layer_1_brain_voting', {}).get('minimum_agreement_ratio', 0.6)
        min_models = self.ensemble_config.get('layer_1_brain_voting', {}).get('minimum_models_required', 5)

        if n_models < min_models:
            logger.warning(f"Only {n_models} models available for {symbol}_{timeframe}, minimum is {min_models}")

        if n_models > 0 and (models_agreed / n_models) < min_agreement:
            # Insufficient consensus — degrade to FLAT
            aggregated = np.array([0.0, 1.0, 0.0])
            models_agreed = 0

        return aggregated, n_models, models_agreed

    def process_request(self, request_json):
        """Process a single inference request. Returns JSON response string."""
        start = time.perf_counter()
        request = json.loads(request_json)

        try:
            features = self._validate_request(request)

            # Layer 4: Regime gating (pre-signal)
            enabled = self._get_enabled_models(
                request['symbol'], request['timeframe'],
                request['regime_state'], request['regime_confidence']
            )

            if not enabled:
                # Regime denies this timeframe — force FLAT
                response = {
                    'request_id': request['request_id'],
                    'ensemble_key': f"{request['symbol']}_{request['timeframe']}_ensemble",
                    'signal': 'FLAT',
                    'class_probs': [0.0, 1.0, 0.0],
                    'confidence': 1.0,
                    'models_voted': 0,
                    'models_agreed': 0,
                    'agreement_ratio': 0.0,
                    'latency_ms': round((time.perf_counter() - start) * 1000, 2)
                }
                return json.dumps(response)

            # Layer 1: Run all enabled models and aggregate via trimmed mean
            predictions = []
            brain_names = []
            for brain_name, backend in enabled:
                try:
                    probs = backend.predict_proba(features)
                    predictions.append(probs)
                    brain_names.append(brain_name)
                except Exception as e:
                    logger.warning(f"Model {brain_name} failed: {e}")
                    continue

            # Ensemble aggregation (trimmed mean + agreement check)
            aggregated, n_voted, n_agreed = self._ensemble_aggregate(
                predictions, brain_names, request['symbol'], request['timeframe']
            )

            signal_map = {0: 'SHORT', 1: 'FLAT', 2: 'LONG'}
            signal = signal_map[int(np.argmax(aggregated))]

            elapsed = (time.perf_counter() - start) * 1000

            response = {
                'request_id': request['request_id'],
                'ensemble_key': f"{request['symbol']}_{request['timeframe']}_ensemble",
                'signal': signal,
                'class_probs': [round(float(p), 6) for p in aggregated],
                'confidence': round(float(np.max(aggregated)), 6),
                'models_voted': n_voted,
                'models_agreed': n_agreed,
                'agreement_ratio': round(n_agreed / n_voted, 4) if n_voted > 0 else 0.0,
                'latency_ms': round(elapsed, 2)
            }

            return json.dumps(response)

        except Exception as e:
            logger.error(f"Request {request.get('request_id', '?')} failed: {e}")
            return json.dumps({
                'request_id': request.get('request_id', -1),
                'error': str(e),
                'signal': 'FLAT',
                'class_probs': [0.0, 1.0, 0.0],
                'confidence': 0.0,
                'models_voted': 0,
                'models_agreed': 0,
                'agreement_ratio': 0.0,
                'latency_ms': round((time.perf_counter() - start) * 1000, 2)
            })

    def serve(self):
        """Start ZeroMQ REP server."""
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://*:{self.port}")
        socket.setsockopt(zmq.RCVTIMEO, 30000)  # 30s receive timeout

        logger.info(f"CHAOS Inference Server started on port {self.port}")
        logger.info(f"Models loaded: {sum(len(v) for v in self.models.values())} across {len(self.models)} pair/tf combos")

        while True:
            try:
                message = socket.recv_string()
                response = self.process_request(message)
                socket.send_string(response)
            except zmq.Again:
                continue  # Timeout, loop back
            except KeyboardInterrupt:
                logger.info("Server shutting down")
                break

        socket.close()
        context.term()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='CHAOS V1.0 Inference Server')
    parser.add_argument('--schema', default='G:/My Drive/chaos_v1.0/schema/feature_schema.json')
    parser.add_argument('--models', default='G:/My Drive/chaos_v1.0/models')
    parser.add_argument('--regime', default='G:/My Drive/chaos_v1.0/regime/regime_policy.json')
    parser.add_argument('--ensemble', default='G:/My Drive/chaos_v1.0/ensemble/ensemble_config.json')
    parser.add_argument('--port', type=int, default=5555)
    args = parser.parse_args()

    server = ChaosInferenceServer(
        schema_path=args.schema,
        models_dir=args.models,
        regime_policy_path=args.regime,
        ensemble_config_path=args.ensemble,
        port=args.port
    )
    server.serve()
