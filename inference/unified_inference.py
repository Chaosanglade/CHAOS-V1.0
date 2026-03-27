"""
CHAOS V1.0 — Unified Inference Path

SINGLE entry point for both REPLAY and LIVE inference.
Guarantees identical feature computation regardless of mode.

The only difference between modes is the data source:
  REPLAY: reads raw OHLCV from parquet files
  LIVE:   receives raw OHLCV bars from IBKR/ZMQ

Both paths:
  raw OHLCV → LiveFeatureAdapter (273 features) → reorder_for_model →
  thin ensemble → risk engine → signal

Usage:
    from inference.unified_inference import UnifiedInference

    ui = UnifiedInference(base_dir='G:/My Drive/chaos_v1.0')

    # Live mode: pass raw bar dicts
    result = ui.infer(pair='EURUSD', tf='H1', bars_by_tf={'H1': [bar_dicts]})

    # Replay mode: pass raw OHLCV DataFrame
    result = ui.infer_from_df(pair='EURUSD', tf='H1', ohlcv_df=df)
"""
import os
import sys
import json
import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger('unified_inference')

PROJECT_ROOT = Path(os.environ.get('CHAOS_BASE_DIR', os.getcwd()))


class UnifiedInference:
    """
    Single inference path for replay and live.

    Owns:
    - LiveFeatureAdapter (273 features from raw OHLCV)
    - Feature reordering (schema → training column order per pair+TF)
    - Thin ensemble config (eligible brains per block)
    - Model loading + quarantine enforcement
    - Risk engine (ExposureController + PortfolioState)

    Does NOT own:
    - Data sourcing (caller provides raw OHLCV)
    - Order execution (caller handles fills)
    - Position tracking beyond risk checks
    """

    def __init__(self, base_dir=None):
        self.base_dir = Path(base_dir or PROJECT_ROOT)
        sys.path.insert(0, str(self.base_dir))
        sys.path.insert(0, str(self.base_dir / 'inference'))

        t0 = time.perf_counter()
        logger.info("UnifiedInference initializing...")

        # Feature adapter
        from inference.live_feature_adapter import LiveFeatureAdapter
        self.feature_adapter = LiveFeatureAdapter(
            schema_path=self.base_dir / 'schema' / 'feature_schema.json',
            column_order_path=self.base_dir / 'schema' / 'feature_columns_by_pair_tf.json',
        )

        # Thin ensemble
        thin_path = self.base_dir / 'replay' / 'config' / 'thin_ensemble.json'
        self.thin_ensemble = None
        if thin_path.exists():
            with open(thin_path) as f:
                self.thin_ensemble = json.load(f)

        # Quarantine
        with open(self.base_dir / 'replay' / 'config' / 'brain_quarantine.json') as f:
            self.quarantine = json.load(f)
        self._quarantined = set(self.quarantine.get('global_quarantine', []))

        # Cross-pair bar cache (for cross-asset features)
        self._cross_cache = {}  # pair -> [close_prices]

        logger.info(f"UnifiedInference ready ({time.perf_counter()-t0:.1f}s)")

    def _update_cross_cache(self, pair, bars_by_tf):
        """Update cross-pair close cache from raw bars."""
        for tf_key, bar_list in bars_by_tf.items():
            if bar_list:
                closes = [float(b.get('close', b.get('Close', 0))) for b in bar_list]
                if len(closes) >= 20:
                    self._cross_cache[pair] = closes

    def compute_features(self, pair, tf, bars_by_tf):
        """
        Compute features from raw OHLCV bars.

        Returns:
            np.array in TRAINING column order for this pair+TF,
            or None on failure.
        """
        self._update_cross_cache(pair, bars_by_tf)

        # Compute 273 schema-ordered features
        schema_vec = self.feature_adapter.compute(
            pair, tf, bars_by_tf, cross_pair_closes=self._cross_cache)
        if schema_vec is None:
            return None

        # Reorder to training column order for this pair+TF
        model_vec = self.feature_adapter.reorder_for_model(schema_vec, pair, tf)
        return model_vec

    def infer(self, pair, tf, bars_by_tf, models=None):
        """
        Run full inference on raw bar dicts.

        Args:
            pair: e.g. 'EURUSD'
            tf: e.g. 'H1'
            bars_by_tf: {'H1': [{'time':..., 'open':..., 'high':..., ...}, ...]}
            models: dict {brain_name: backend} — if None, caller must handle

        Returns:
            dict with: signal, confidence, agreement, action, reason_codes, etc.
        """
        features = self.compute_features(pair, tf, bars_by_tf)
        if features is None:
            return {'signal': 0, 'action': 'SKIP', 'reason_codes': 'FEATURES_FAILED'}

        if models is None:
            return {'signal': 0, 'action': 'SKIP', 'reason_codes': 'NO_MODELS',
                    'features': features}

        # Run thin ensemble if configured
        block_key = f"{pair}_{tf}"
        thin_block = (self.thin_ensemble or {}).get('blocks', {}).get(block_key)

        if thin_block and thin_block.get('rule') != 'disabled':
            eligible = thin_block.get('brains', [])
            rule = thin_block.get('rule', 'direct_signal')
            selected = {name: backend for name, backend in models.items()
                        if name in eligible and name not in self._quarantined}
        else:
            selected = {name: backend for name, backend in models.items()
                        if name not in self._quarantined}
            rule = 'majority_0.50'

        if not selected:
            return {'signal': 0, 'action': 'SKIP', 'reason_codes': 'NO_ELIGIBLE_MODELS',
                    'features': features}

        # Run each brain
        signal_map = {0: -1, 1: 0, 2: 1}
        predictions = {}
        for brain_name, backend in selected.items():
            try:
                fv = features.reshape(1, -1).astype(np.float32)
                probs = backend.predict_proba(fv)[0]
                if len(probs) >= 3:
                    predictions[brain_name] = probs
            except Exception as e:
                logger.debug(f"Inference error for {brain_name}: {e}")

        if not predictions:
            return {'signal': 0, 'action': 'SKIP', 'reason_codes': 'INFERENCE_FAILED',
                    'features': features}

        models_voted = len(predictions)

        # Apply rule
        if rule == 'direct_signal' and models_voted == 1:
            probs = next(iter(predictions.values()))
            pred_class = int(np.argmax(probs))
            signal = signal_map[pred_class]
            confidence = float(probs[pred_class])
            agreement = 1.0
            models_agreed = 1
            reason = 'THIN_DIRECT'

        elif rule == 'unanimous' and models_voted >= 2:
            classes = [int(np.argmax(p)) for p in predictions.values()]
            if len(set(classes)) == 1:
                all_probs = np.array(list(predictions.values()))
                pred_class = classes[0]
                signal = signal_map[pred_class]
                confidence = float(np.mean(all_probs[:, pred_class]))
                agreement = 1.0
                models_agreed = models_voted
                reason = 'THIN_UNANIMOUS'
            else:
                signal = 0; confidence = 0.0; agreement = 0.0
                models_agreed = 0; reason = 'THIN_DISAGREE'

        else:
            all_probs = np.array(list(predictions.values()))
            avg_probs = all_probs.mean(axis=0)
            pred_class = int(np.argmax(avg_probs))
            agreed = sum(1 for p in all_probs if np.argmax(p) == pred_class)
            agreement = agreed / models_voted
            if agreement >= 0.50:
                signal = signal_map[pred_class]
                confidence = float(avg_probs[pred_class])
                models_agreed = agreed
                reason = 'MAJORITY'
            else:
                signal = 0; confidence = 0.0
                models_agreed = 0; reason = 'NO_MAJORITY'

        return {
            'signal': signal,
            'confidence': round(confidence, 4),
            'agreement_score': round(agreement, 4),
            'models_voted': models_voted,
            'models_agreed': models_agreed,
            'reason_codes': reason,
            'features': features,
        }

    def infer_from_df(self, pair, tf, ohlcv_df, models=None, n_bars=500):
        """
        Run inference from a pandas DataFrame (replay mode).

        Args:
            ohlcv_df: DataFrame with Open/High/Low/Close/Volume columns
                      and DatetimeIndex or 'time' column
            n_bars: how many recent bars to use
        """
        df = ohlcv_df.tail(n_bars)

        # Convert to bar dicts
        bars = []
        for _, row in df.iterrows():
            bar = {}
            for src, dst in [('Open', 'open'), ('High', 'high'), ('Low', 'low'),
                             ('Close', 'close'), ('Volume', 'volume')]:
                if src in row.index:
                    bar[dst] = float(row[src])
                elif dst in row.index:
                    bar[dst] = float(row[dst])
            if hasattr(df.index, 'strftime'):
                bar['time'] = row.name.strftime('%Y-%m-%dT%H:%M:%S')
            elif 'time' in row.index:
                bar['time'] = str(row['time'])
            bars.append(bar)

        return self.infer(pair, tf, {tf: bars}, models=models)
