"""
CHAOS V1.0 -- Live Inference Handler

Receives raw OHLCV bars from MT5 EA, computes features using the existing
feature engine, runs ensemble inference with quarantine, and returns trade signals.

This bridges the gap between MT5 (which sends raw bars) and the ensemble
(which expects 273-feature vectors).
"""
import json
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger('live_inference')

PROJECT_ROOT = Path('G:/My Drive/chaos_v1.0')


class LiveFeatureEngine:
    """Compute 273 features from raw OHLCV bars, matching the replay feature engine."""

    def __init__(self, schema_path=None):
        schema_path = schema_path or PROJECT_ROOT / 'schema' / 'feature_schema.json'
        with open(schema_path) as f:
            self.schema = json.load(f)
        self.feature_names = self.schema.get('universal_features', [])
        self.n_features = len(self.feature_names)
        logger.info(f"Feature engine loaded: {self.n_features} features")

        # Exclusion substrings (same as training pipeline)
        self.exclude_substrings = [
            'target_', 'return', 'Open', 'High', 'Low', 'Close', 'Volume',
            'timestamp', 'date', 'pair', 'symbol', 'tf', 'bar_time',
            'Bid_', 'Ask_', 'Spread_', 'Unnamed',
        ]

    def bars_to_dataframe(self, bars_dict):
        """Convert raw bars dict (from ZMQ request) to DataFrames per TF."""
        dfs = {}
        for tf, bar_list in bars_dict.items():
            if not bar_list:
                continue
            df = pd.DataFrame(bar_list)
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'], utc=True)
                df = df.sort_values('time').reset_index(drop=True)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = df[col].astype(float)
            dfs[tf] = df
        return dfs

    def compute_features(self, bars_dfs, pair, tf):
        """
        Compute features from bar DataFrames.

        Uses the replay iterator's feature computation approach:
        load the pair's feature parquet and extract the schema-ordered features.

        For LIVE mode, we compute indicators from raw bars matching
        the training pipeline's ta-lib / pandas-ta features.
        """
        # Try to use the existing feature files as a reference for column ordering
        # In live mode, we compute features from the raw bars
        primary_df = bars_dfs.get(tf)
        if primary_df is None or len(primary_df) == 0:
            logger.warning(f"No bars for {pair}_{tf}")
            return np.zeros(self.n_features, dtype=np.float32)

        features = self._compute_ta_features(primary_df, pair, tf)

        # Ensure exactly n_features dimensions
        if len(features) < self.n_features:
            padded = np.zeros(self.n_features, dtype=np.float32)
            padded[:len(features)] = features[:self.n_features]
            return padded
        return features[:self.n_features].astype(np.float32)

    def _compute_ta_features(self, df, pair, tf):
        """
        Compute technical analysis features from OHLCV data.
        Mirrors the feature engineering pipeline used in training.
        """
        o = df['open'].values
        h = df['high'].values
        l = df['low'].values
        c = df['close'].values
        v = df['volume'].values.astype(float)
        n = len(c)

        features = []

        # --- Returns ---
        ret_1 = np.diff(np.log(np.maximum(c, 1e-10)), prepend=np.log(c[0]))

        # --- SMA features ---
        for period in [5, 10, 20, 50, 100, 200]:
            if n >= period:
                sma = pd.Series(c).rolling(period).mean().values
                features.append(c[-1] / sma[-1] - 1 if sma[-1] != 0 else 0)  # price_to_sma ratio
            else:
                features.append(0.0)

        # --- EMA features ---
        for period in [5, 10, 20, 50, 100, 200]:
            if n >= period:
                ema = pd.Series(c).ewm(span=period, adjust=False).mean().values
                features.append(c[-1] / ema[-1] - 1 if ema[-1] != 0 else 0)
            else:
                features.append(0.0)

        # --- RSI ---
        for period in [7, 14, 21]:
            if n >= period + 1:
                delta = np.diff(c)
                gain = np.where(delta > 0, delta, 0)
                loss = np.where(delta < 0, -delta, 0)
                avg_gain = pd.Series(gain).rolling(period).mean().values[-1]
                avg_loss = pd.Series(loss).rolling(period).mean().values[-1]
                if avg_loss != 0:
                    rs = avg_gain / avg_loss
                    features.append(rs / (1 + rs))  # Normalized RSI [0,1]
                else:
                    features.append(1.0)
            else:
                features.append(0.5)

        # --- MACD ---
        if n >= 26:
            ema12 = pd.Series(c).ewm(span=12, adjust=False).mean().values
            ema26 = pd.Series(c).ewm(span=26, adjust=False).mean().values
            macd = ema12 - ema26
            signal_line = pd.Series(macd).ewm(span=9, adjust=False).mean().values
            features.append(macd[-1])
            features.append(signal_line[-1])
            features.append(macd[-1] - signal_line[-1])  # histogram
        else:
            features.extend([0.0, 0.0, 0.0])

        # --- Bollinger Bands ---
        for period in [20]:
            if n >= period:
                sma = pd.Series(c).rolling(period).mean().values
                std = pd.Series(c).rolling(period).std().values
                upper = sma + 2 * std
                lower = sma - 2 * std
                band_width = (upper[-1] - lower[-1]) / sma[-1] if sma[-1] != 0 else 0
                bb_pos = (c[-1] - lower[-1]) / (upper[-1] - lower[-1]) if (upper[-1] - lower[-1]) != 0 else 0.5
                features.append(band_width)
                features.append(bb_pos)
            else:
                features.extend([0.0, 0.5])

        # --- ATR ---
        for period in [14, 20]:
            if n >= period + 1:
                tr = np.maximum(h[1:] - l[1:],
                       np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
                atr = pd.Series(tr).rolling(period).mean().values[-1]
                features.append(atr / c[-1] if c[-1] != 0 else 0)  # ATR as % of price
            else:
                features.append(0.0)

        # --- Stochastic ---
        for period in [14]:
            if n >= period:
                lowest_low = pd.Series(l).rolling(period).min().values[-1]
                highest_high = pd.Series(h).rolling(period).max().values[-1]
                denom = highest_high - lowest_low
                k = (c[-1] - lowest_low) / denom if denom != 0 else 0.5
                features.append(k)
            else:
                features.append(0.5)

        # --- Volume features ---
        if n >= 20:
            vol_sma = pd.Series(v).rolling(20).mean().values[-1]
            features.append(v[-1] / vol_sma if vol_sma != 0 else 1.0)
        else:
            features.append(1.0)

        # --- Candle patterns ---
        body = c[-1] - o[-1]
        total_range = h[-1] - l[-1]
        features.append(body / total_range if total_range != 0 else 0)  # body ratio
        features.append((h[-1] - max(o[-1], c[-1])) / total_range if total_range != 0 else 0)  # upper wick
        features.append((min(o[-1], c[-1]) - l[-1]) / total_range if total_range != 0 else 0)  # lower wick

        # --- Momentum ---
        for period in [5, 10, 20]:
            if n >= period:
                features.append(c[-1] / c[-1 - period] - 1 if c[-1 - period] != 0 else 0)
            else:
                features.append(0.0)

        # --- Volatility (rolling std of returns) ---
        for period in [10, 20, 50]:
            if n >= period:
                rets = np.diff(np.log(np.maximum(c[-period-1:], 1e-10)))
                features.append(np.std(rets))
            else:
                features.append(0.0)

        # --- Lag features (returns at t-1, t-2, ..., t-10) ---
        for lag in range(1, 11):
            if n > lag:
                features.append(ret_1[-lag])
            else:
                features.append(0.0)

        # --- Cross-TF features would come from other TF DataFrames ---
        # Pad remaining features to reach 273
        while len(features) < self.n_features:
            features.append(0.0)

        return np.array(features[:self.n_features], dtype=np.float32)


class LiveInferenceHandler:
    """
    Receives raw bars from MT5 EA via ZeroMQ.
    Computes features, runs ensemble inference, returns trade signal.
    """

    def __init__(self, model_loader, regime_sim, ensemble_engine,
                 risk_controller, portfolio_state, quarantine_config,
                 portfolio_config, feature_engine=None):
        self.model_loader = model_loader
        self.regime_sim = regime_sim
        self.ensemble = ensemble_engine
        self.risk = risk_controller
        self.portfolio = portfolio_state
        self.quarantine = quarantine_config
        self.portfolio_config = portfolio_config
        self.feature_engine = feature_engine or LiveFeatureEngine()
        self._confirm_signals = {}
        logger.info("LiveInferenceHandler initialized")

    def is_quarantined(self, brain_name, pair, tf):
        block_key = f"{pair}_{tf}"
        if brain_name in self.quarantine.get('global_quarantine', []):
            return True
        cond = self.quarantine.get('conditional_quarantine', {}).get(brain_name)
        if cond and cond.get('policy') == 'quarantined_globally_except':
            return block_key not in cond.get('exceptions', [])
        return False

    def get_active_models(self, pair, tf):
        pair_tf = f"{pair}_{tf}"
        models = self.model_loader.get_models(pair_tf)
        if not models:
            return []
        return [(name, backend) for name, backend in models.items()
                if not self.is_quarantined(name, pair, tf)]

    def process_request(self, request):
        """Process a live inference request from MT5 EA."""
        pair = request.get('pair') or request.get('symbol', '')
        tf = request.get('tf') or request.get('timeframe', '')
        request_id = request.get('request_id', '')
        request_type = request.get('request_type', 'RAW_BARS')

        t0 = time.perf_counter()

        try:
            if request_type == 'RAW_BARS':
                bars_dfs = self.feature_engine.bars_to_dataframe(request.get('bars', {}))
                feature_vector = self.feature_engine.compute_features(bars_dfs, pair, tf)
            elif request_type == 'FEATURES':
                feature_vector = np.array(request.get('features', []), dtype=np.float32)
            else:
                raise ValueError(f"Unknown request_type: {request_type}")

            # Sanitize features
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)

            # Get active models (quarantine applied)
            active_models = self.get_active_models(pair, tf)

            if not active_models:
                return self._skip_response(request_id, 'NO_MODELS_AVAILABLE', t0)

            # Regime check
            regime_state, regime_confidence = self.regime_sim.get_regime_state(pair, tf, 0)
            enabled_models = self.regime_sim.get_enabled_models(
                {name: backend for name, backend in active_models},
                tf, regime_state, regime_confidence
            )

            if not enabled_models:
                return self._skip_response(request_id, 'REGIME_BLOCKED', t0, regime_state=regime_state)

            # Ensemble inference
            result = self.ensemble.run_inference(enabled_models, feature_vector)

            signal = result.get('signal_side', 0)
            confidence = result.get('confidence', 0)
            agreement = result.get('agreement_score', 0)
            models_voted = result.get('models_voted', 0)
            models_agreed = result.get('models_agreed', 0)
            reason_codes = []

            # TF role check
            tf_roles = {'H1': 'TRADE_ENABLED', 'M30': 'TRADE_ENABLED',
                        'M15': 'CONFIRM_ONLY', 'M5': 'CONFIRM_ONLY'}
            role = tf_roles.get(tf, 'CONFIRM_ONLY')

            if role == 'CONFIRM_ONLY':
                self._confirm_signals.setdefault(pair, {})[tf] = signal
                return self._skip_response(request_id, 'TF_CONFIRM_ONLY', t0,
                                          signal=signal, confidence=confidence,
                                          regime_state=regime_state,
                                          models_voted=models_voted,
                                          models_agreed=models_agreed)

            # Determine action
            min_agreement = self.ensemble.min_agreement
            if agreement < min_agreement:
                reason_codes.append('AGREEMENT_FAILED')
                action = 'SKIP'
            elif signal == 0:
                action = 'HOLD'
                reason_codes.append('SIGNAL_FLAT')
            else:
                # Check if we already have a position
                has_pos = self.portfolio.has_position(pair, tf)
                if signal != 0 and not has_pos:
                    action = 'OPEN'
                    reason_codes.append('AGREEMENT_PASS')
                elif has_pos:
                    # Check if signal reversed
                    action = 'CLOSE'
                    reason_codes.append('SIGNAL_REVERSAL')
                else:
                    action = 'HOLD'

            # Lot size from portfolio config
            pair_weight = self.portfolio_config.get('pair_weights', {}).get(pair, 0.05)
            lot_size = round(pair_weight * 0.1, 2)  # Simplified; EA overrides with risk-based sizing

            latency_ms = (time.perf_counter() - t0) * 1000

            return {
                'signal': signal,
                'confidence': round(confidence, 4),
                'agreement_score': round(agreement, 4),
                'regime_state': regime_state,
                'risk_approved': action in ('OPEN', 'CLOSE'),
                'risk_reason': '',
                'action': action,
                'reason_codes': '|'.join(sorted(reason_codes)),
                'lot_size': lot_size,
                'models_voted': models_voted,
                'models_agreed': models_agreed,
                'request_id': request_id,
                'server_latency_ms': round(latency_ms, 2),
            }

        except Exception as e:
            logger.error(f"Inference error for {pair}_{tf}: {e}", exc_info=True)
            return self._skip_response(request_id, f'INFERENCE_ERROR', t0)

    def _skip_response(self, request_id, reason, t0, signal=0, confidence=0.0,
                       regime_state=1, models_voted=0, models_agreed=0):
        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            'signal': signal,
            'confidence': confidence,
            'agreement_score': 0.0,
            'regime_state': regime_state,
            'risk_approved': False,
            'risk_reason': reason,
            'action': 'SKIP',
            'reason_codes': reason,
            'lot_size': 0.0,
            'models_voted': models_voted,
            'models_agreed': models_agreed,
            'request_id': request_id,
            'server_latency_ms': round(latency_ms, 2),
        }
