import warnings
import os
warnings.filterwarnings('ignore')
os.environ['ORT_LOG_SEVERITY_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'
"""
CHAOS V1.0 — IBKR Direct Execution Module

Pure Python → IBKR execution. No MT5, no ZeroMQ, no EA.
Uses ib_async (maintained fork). Connects to TWS or IB Gateway.

Usage:
    pip install ib_async
    python inference/ibkr_executor.py --paper
    python inference/ibkr_executor.py --live   (REAL MONEY)
"""
import sys
import json
import csv
import asyncio
import logging
import signal
import time as _time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(os.environ.get('CHAOS_BASE_DIR', os.getcwd()))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'inference'))

# Dual logging: console + file
_log_dir = PROJECT_ROOT / 'CHAOS_Logs'
_log_dir.mkdir(parents=True, exist_ok=True)
_log_file = _log_dir / f"ibkr_session_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.log"

_formatter = logging.Formatter('%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')

_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(_formatter)
_console_handler.setLevel(logging.INFO)

_file_handler = logging.FileHandler(str(_log_file), mode='a', encoding='utf-8')
_file_handler.setFormatter(_formatter)
_file_handler.setLevel(logging.DEBUG)

logging.basicConfig(level=logging.DEBUG, handlers=[_console_handler, _file_handler])
logger = logging.getLogger('ibkr_executor')
logger.info(f"Log file: {_log_file}")

import ib_async as ib
import numpy as np
import joblib

from inference.ibkr_bar_builder import IBKRBarBuilder


# ──────────────────────────────────────────────────────────────────────
# Lightweight model backend shims (avoid importing the full onnx_export)
# ──────────────────────────────────────────────────────────────────────

class _OnnxBackendLite:
    """Minimal wrapper around an onnxruntime session."""
    def __init__(self, session):
        self.session = session
        inp = session.get_inputs()[0]
        self.n_features_ = inp.shape[1] if len(inp.shape) > 1 else None

    def predict_proba(self, X):
        X = np.nan_to_num(X.astype(np.float32), nan=0, posinf=0, neginf=0)
        # Zero-pad if model expects more features than provided
        if self.n_features_ and X.shape[1] < self.n_features_:
            padded = np.zeros((X.shape[0], self.n_features_), dtype=np.float32)
            padded[:, :X.shape[1]] = X
            X = padded
        elif self.n_features_ and X.shape[1] > self.n_features_:
            X = X[:, :self.n_features_]
        results = self.session.run(None, {self.session.get_inputs()[0].name: X})
        if len(results) >= 2:
            probs_raw = results[1]
            if isinstance(probs_raw, list) and isinstance(probs_raw[0], dict):
                return np.array([[p.get(0, 0), p.get(1, 0), p.get(2, 0)] for p in probs_raw])
            probs = np.array(probs_raw)
        else:
            probs = np.array(results[0])
        if probs.ndim == 2 and not np.allclose(probs.sum(axis=1), 1.0, atol=0.05):
            exp_p = np.exp(probs - probs.max(axis=1, keepdims=True))
            probs = exp_p / exp_p.sum(axis=1, keepdims=True)
        return probs


class _SklearnBackendLite:
    """Minimal wrapper around a sklearn model."""
    def __init__(self, model):
        self.model = model
        self.n_features_ = getattr(model, 'n_features_in_', None)

    def predict_proba(self, X):
        X = np.nan_to_num(X.astype(np.float64), nan=0, posinf=0, neginf=0)
        # Zero-pad if model expects more features than provided
        if self.n_features_ and X.shape[1] < self.n_features_:
            padded = np.zeros((X.shape[0], self.n_features_), dtype=np.float64)
            padded[:, :X.shape[1]] = X
            X = padded
        elif self.n_features_ and X.shape[1] > self.n_features_:
            X = X[:, :self.n_features_]
        return self.model.predict_proba(X)


class _ModelLoaderShim:
    """Shim that looks like ModelLoader but uses our pre-loaded registry."""
    def __init__(self, registry):
        self.models = registry

    def get_models(self, pair_tf):
        return self.models.get(pair_tf, {})


# ──────────────────────────────────────────────────────────────────────
# Position & Order Tracking
# ──────────────────────────────────────────────────────────────────────

class PositionTracker:
    """Python-owned position state. Reconciles with IBKR on startup."""

    def __init__(self):
        self.positions: Dict[str, dict] = {}  # pair_tf -> {side, qty, entry_price, entry_ts, order_id}
        self.closed_trades: list = []
        self.daily_pnl: float = 0.0
        self.daily_trades: int = 0
        self.daily_wins: int = 0
        self._order_ids: set = set()  # Duplicate prevention

    def has_position(self, pair: str, tf: str) -> bool:
        return f"{pair}_{tf}" in self.positions

    def get_position(self, pair: str, tf: str) -> Optional[dict]:
        return self.positions.get(f"{pair}_{tf}")

    def open_position(self, pair: str, tf: str, side: int, qty: float,
                      price: float, order_id: int):
        key = f"{pair}_{tf}"
        self.positions[key] = {
            'pair': pair, 'tf': tf, 'side': side, 'qty': qty,
            'entry_price': price, 'entry_ts': datetime.now(timezone.utc),
            'order_id': order_id,
            'max_profit_pips': 0.0,   # For trailing stop
            'current_price': price,
            'unrealized_pnl': 0.0,
        }
        self._order_ids.add(order_id)
        self.daily_trades += 1
        logger.info(f"POSITION OPENED: {key} side={side} qty={qty} @ {price}")

    def close_position(self, pair: str, tf: str, exit_price: float, order_id: int,
                       usdjpy_rate: float = 150.0):
        key = f"{pair}_{tf}"
        pos = self.positions.pop(key, None)
        if not pos:
            return None
        self._order_ids.add(order_id)
        # Compute PnL in quote currency
        if pos['side'] == 1:  # LONG
            pnl_quote = (exit_price - pos['entry_price']) * pos['qty'] * 100000
        else:  # SHORT
            pnl_quote = (pos['entry_price'] - exit_price) * pos['qty'] * 100000
        # Convert to USD: JPY-denominated pairs need division by USDJPY rate
        if 'JPY' in pair:
            pnl = pnl_quote / usdjpy_rate
        else:
            pnl = pnl_quote
        trade = {**pos, 'exit_price': exit_price, 'exit_ts': datetime.now(timezone.utc),
                 'pnl_usd': pnl, 'pnl_quote': pnl_quote, 'close_order_id': order_id}
        self.closed_trades.append(trade)
        self.daily_pnl += pnl
        if pnl > 0:
            self.daily_wins += 1
        logger.info(f"POSITION CLOSED: {key} pnl=${pnl:.2f} "
                     f"{'('+str(round(pnl_quote,0))+' JPY) ' if 'JPY' in pair else ''}"
                     f"(daily=${self.daily_pnl:.2f})")
        return trade

    def is_duplicate_order(self, order_id: int) -> bool:
        return order_id in self._order_ids

    def open_count(self) -> int:
        return len(self.positions)

    def reset_daily(self):
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.daily_wins = 0

    async def reconcile(self, ib_client: ib.IB, pairs_map: dict):
        """
        Reconcile Python state with IBKR on connect/reconnect.
        Adopts ALL IBKR positions into self.positions so the executor
        knows about them and won't stack new orders on top.
        """
        ibkr_positions = ib_client.positions()
        logger.info(f"[RECONCILE] IBKR reports {len(ibkr_positions)} positions, "
                     f"Python tracks {len(self.positions)}")

        # Build reverse map: IBKR localSymbol -> CHAOS pair name
        # localSymbol examples: "EUR.USD", "GBP.JPY"
        ibkr_to_chaos = {}
        for chaos_pair, ibkr_pair in pairs_map.items():
            ibkr_to_chaos[ibkr_pair] = chaos_pair
            # Also handle localSymbol format without dot: "EURUSD" or "EUR.USD"
            ibkr_to_chaos[ibkr_pair.replace('.', '')] = chaos_pair

        adopted = 0
        for p in ibkr_positions:
            sym = p.contract.localSymbol or p.contract.symbol or ''
            qty = float(p.position)
            avg_price = float(p.avgCost)

            if abs(qty) < 0.01:
                continue  # Skip zero positions

            # Map to CHAOS pair name
            chaos_pair = ibkr_to_chaos.get(sym)
            if not chaos_pair:
                # Try stripping exchange suffix or matching symbol+currency
                combo = f"{p.contract.symbol}.{p.contract.currency}"
                chaos_pair = ibkr_to_chaos.get(combo)
            if not chaos_pair:
                # Try raw symbol concatenation
                raw = f"{p.contract.symbol}{p.contract.currency}"
                chaos_pair = ibkr_to_chaos.get(raw)
            if not chaos_pair:
                logger.warning(f"[RECONCILE] Unknown IBKR position: {sym} qty={qty} "
                                f"avg={avg_price} — cannot map to CHAOS pair")
                continue

            side = 1 if qty > 0 else -1  # LONG if positive, SHORT if negative
            lots = abs(qty) / 100000  # Convert units to lots

            # For IBKR FX, avgCost is the average price (not cost per unit)
            # It may need adjustment for JPY pairs
            if avg_price > 1000:
                # Likely JPY pair in wrong units — avgCost for FX is actual price
                pass

            # Adopt into tracker under H1 TF (we don't know original TF from IBKR)
            # Check if we already track this pair on any TF
            already_tracked = False
            for tf_check in ['H1', 'M30']:
                if self.has_position(chaos_pair, tf_check):
                    already_tracked = True
                    logger.info(f"[RECONCILE] {chaos_pair} already tracked as {chaos_pair}_{tf_check}, "
                                 f"updating price to {avg_price}")
                    # Update the tracked position's price
                    self.positions[f"{chaos_pair}_{tf_check}"]['entry_price'] = avg_price
                    self.positions[f"{chaos_pair}_{tf_check}"]['current_price'] = avg_price
                    break

            if not already_tracked:
                # Adopt as H1 position (default — we can't know the original TF)
                key = f"{chaos_pair}_H1"
                self.positions[key] = {
                    'pair': chaos_pair, 'tf': 'H1', 'side': side,
                    'qty': lots, 'entry_price': avg_price,
                    'entry_ts': datetime.now(timezone.utc),  # Unknown, use now
                    'order_id': -1,  # Reconciled, no order ID
                    'max_profit_pips': 0.0,
                    'current_price': avg_price,
                    'unrealized_pnl': 0.0,
                }
                side_str = 'LONG' if side == 1 else 'SHORT'
                logger.info(f"[RECONCILE] Adopted {chaos_pair} {side_str} "
                             f"{abs(qty):.0f} units ({lots:.2f} lots) "
                             f"@ {avg_price} from IBKR")
                adopted += 1

        logger.info(f"[RECONCILE] Done: {adopted} positions adopted, "
                     f"total tracked: {len(self.positions)}")


# ──────────────────────────────────────────────────────────────────────
# Decision Logger (CSV)
# ──────────────────────────────────────────────────────────────────────

class DecisionLogger:
    """Logs every decision to CSV (same columns as MT5 decision ledger)."""

    FIELDS = [
        'timestamp', 'pair', 'tf', 'request_id', 'signal', 'confidence',
        'agreement_score', 'regime_state', 'action', 'risk_approved',
        'risk_reason', 'reason_codes', 'models_voted', 'models_agreed',
        'lot_size', 'spread', 'latency_ms', 'equity', 'balance',
        'open_positions', 'daily_pnl', 'commission',
    ]

    def __init__(self, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        self._path = log_dir / f'decisions_{today}.csv'
        self._file = None
        self._writer = None
        self._open()

    def _open(self):
        is_new = not self._path.exists()
        self._file = open(self._path, 'a', newline='')
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS,
                                       extrasaction='ignore')
        if is_new:
            self._writer.writeheader()

    def log(self, row: dict):
        row.setdefault('timestamp', datetime.now(timezone.utc).isoformat())
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        if self._file:
            self._file.close()


# ──────────────────────────────────────────────────────────────────────
# Main Executor
# ──────────────────────────────────────────────────────────────────────

class IBKRExecutor:
    """
    Main execution engine. Connects to IBKR, subscribes to bar data,
    runs inference pipeline, executes trades.
    """

    def __init__(self, paper: bool = True, thin_only: bool = False, test_close: bool = False):
        self.paper = paper
        self.thin_only = thin_only
        self._test_close = test_close

        # Load configs
        with open(PROJECT_ROOT / 'inference' / 'ibkr_config.json') as f:
            self.ibkr_cfg = json.load(f)

        self.host = self.ibkr_cfg['host']
        self.port = self.ibkr_cfg['port_paper'] if paper else self.ibkr_cfg['port_live']
        self.client_id = self.ibkr_cfg['client_id']
        self.pairs_map = self.ibkr_cfg['pairs']
        self.kill_switch = self.ibkr_cfg['kill_switch']

        # IBKR client
        self.ib = ib.IB()

        # Position tracker
        self.tracker = PositionTracker()

        # Decision logger
        self.logger_csv = DecisionLogger(PROJECT_ROOT / 'logs' / 'ibkr')

        # Kill switch state
        self._halted = False
        self._defensive_active = False
        self._last_day = None

        # Inference handler (loaded in start())
        self._handler = None
        self._bar_builder = None

        # Qualified contracts (populated in Step 7, keyed by CHAOS pair name)
        self._qualified_contracts = {}
        # Live market data tickers (streaming, keyed by CHAOS pair name)
        self._live_tickers = {}

        # Signal tracking for cycle summary
        self._last_signals = {}  # "EURUSD_H1" -> {signal, action, reason, has_pos}

        # Next order ID
        self._next_order_id = 1

    def _load_inference_handler(self):
        """Load the full production inference pipeline with verbose per-model logging."""
        import glob as _glob

        # Step 1: Configs
        t_step = _time.perf_counter()
        logger.info("[STARTUP] Step 1/8: Loading configs...")
        with open(PROJECT_ROOT / 'replay' / 'config' / 'brain_quarantine.json') as f:
            quarantine_cfg = json.load(f)
        with open(PROJECT_ROOT / 'replay' / 'config' / 'portfolio_allocation.json') as f:
            portfolio_cfg = json.load(f)
        with open(PROJECT_ROOT / 'replay' / 'config' / 'defensive_mode.json') as f:
            defensive_cfg = json.load(f)
        with open(PROJECT_ROOT / 'risk' / 'config' / 'instrument_specs.json') as f:
            instrument_specs = json.load(f)
        logger.info(f"[STARTUP] Step 1/8: Loading configs... OK ({_time.perf_counter()-t_step:.1f}s)")

        # Step 2: Load models one by one with verbose output
        t_step = _time.perf_counter()
        quarantined_brains = set(quarantine_cfg.get('global_quarantine', []))
        pairs = list(self.pairs_map.keys())
        tfs = self.ibkr_cfg.get('timeframes', ['H1', 'M30'])

        # --thin-only: build set of brains we actually need
        thin_needed = None  # None means load everything
        if self.thin_only:
            thin_path = PROJECT_ROOT / 'replay' / 'config' / 'thin_ensemble.json'
            if thin_path.exists():
                with open(thin_path) as _f:
                    thin_cfg = json.load(_f)
                thin_needed = set()
                for block_key, block in thin_cfg.get('blocks', {}).items():
                    if block.get('rule') != 'disabled':
                        for b in block.get('brains', []):
                            thin_needed.add(b)
                logger.info(f"[STARTUP] Step 2/8: Loading models (--thin-only: {len(thin_needed)} brains)...")
            else:
                logger.warning("[STARTUP] --thin-only but thin_ensemble.json not found, loading all")
                logger.info("[STARTUP] Step 2/8: Loading models...")
        else:
            logger.info("[STARTUP] Step 2/8: Loading models...")

        # Discover model files (v2_retrained first, then originals)
        model_dirs = [PROJECT_ROOT / 'models' / 'v2_retrained', PROJECT_ROOT / 'models']
        model_files = []
        seen_keys = set()  # (pair, tf, brain) — prefer v2_retrained over original
        for mdir in model_dirs:
            if not mdir.exists():
                continue
            for pair in pairs:
                for tf in tfs:
                    for ext in ['onnx', 'joblib']:
                        for path in sorted(_glob.glob(str(mdir / f'{pair}_{tf}_*.{ext}'))):
                            basename = os.path.basename(path)
                            parts = basename.replace(f'.{ext}', '').split('_')
                            brain = '_'.join(parts[2:])
                            dedupe_key = (pair, tf, brain)
                            if dedupe_key not in seen_keys:
                                seen_keys.add(dedupe_key)
                                model_files.append((pair, tf, brain, ext, path))

        total = len(model_files)
        loaded = 0
        skipped = 0
        failed = 0
        # We'll load models into a dict structure for ModelLoader-compatible access
        model_registry = {}

        for idx, (pair, tf, brain, ext, path) in enumerate(model_files, 1):
            ptf = f"{pair}_{tf}"
            file_size = os.path.getsize(path) / (1024 * 1024)

            # Skip quarantined brains and tabnet (torch DLL unavailable on VPS)
            skip_brains = quarantined_brains | {'tabnet_optuna'}
            if brain in skip_brains:
                reason = 'quarantined' if brain in quarantined_brains else 'tabnet/torch'
                logger.debug(f"[{idx:>3}/{total}] {os.path.basename(path)}... SKIP ({reason})")
                skipped += 1
                continue

            # --thin-only: skip brains not in thin ensemble
            if thin_needed is not None and brain not in thin_needed:
                logger.debug(f"[{idx:>3}/{total}] {os.path.basename(path)}... SKIP (not in thin ensemble)")
                skipped += 1
                continue

            t_model = _time.perf_counter()
            try:
                if ext == 'onnx':
                    import onnxruntime as ort
                    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
                    backend = _OnnxBackendLite(sess)
                elif ext == 'joblib':
                    obj = joblib.load(path)
                    model = obj if not isinstance(obj, dict) else obj.get('model', obj.get('estimator', obj))
                    backend = _SklearnBackendLite(model)
                else:
                    continue

                if ptf not in model_registry:
                    model_registry[ptf] = {}
                # Prefer ONNX over joblib if both exist
                if brain not in model_registry[ptf] or ext == 'onnx':
                    model_registry[ptf][brain] = backend
                    loaded += 1

                elapsed = _time.perf_counter() - t_model
                logger.info(f"[{idx:>3}/{total}] Loading {os.path.basename(path)}... "
                            f"OK ({file_size:.1f} MB, {elapsed:.1f}s)")

            except Exception as e:
                elapsed = _time.perf_counter() - t_model
                logger.warning(f"[{idx:>3}/{total}] Loading {os.path.basename(path)}... "
                               f"FAIL ({file_size:.1f} MB, {elapsed:.1f}s: {str(e)[:60]})")
                failed += 1

        total_time = _time.perf_counter() - t_step
        logger.info(f"[STARTUP] Step 2/8: Loading models... Done. "
                     f"{loaded} loaded, {skipped} quarantined, {failed} failed. "
                     f"Total time: {total_time:.0f}s")

        # Build a ModelLoader-compatible wrapper
        model_loader = _ModelLoaderShim(model_registry)

        # Step 3: Feature adapter
        t_step = _time.perf_counter()
        logger.info("[STARTUP] Step 3/8: Initializing feature adapter...")
        from inference.live_inference_handler import LiveInferenceHandler, LiveFeatureEngine
        from inference.live_feature_adapter import LiveFeatureAdapter
        try:
            feature_adapter = LiveFeatureAdapter()
            logger.info(f"[STARTUP] Step 3/8: Initializing feature adapter... "
                        f"OK ({feature_adapter.n_features} features, "
                        f"{_time.perf_counter()-t_step:.1f}s)")
        except Exception as e:
            logger.warning(f"[STARTUP] Step 3/8: Feature adapter failed: {e}")
            feature_adapter = None

        # Step 4: Thin ensemble
        t_step = _time.perf_counter()
        logger.info("[STARTUP] Step 4/8: Loading thin ensemble...")
        thin_path = PROJECT_ROOT / 'replay' / 'config' / 'thin_ensemble.json'
        if thin_path.exists():
            with open(thin_path) as f:
                thin_cfg = json.load(f)
            n_blocks = sum(1 for b in thin_cfg.get('blocks', {}).values()
                           if b.get('rule') != 'disabled')
            n_brains = sum(len(b.get('brains', [])) for b in thin_cfg.get('blocks', {}).values())
            logger.info(f"[STARTUP] Step 4/8: Loading thin ensemble... "
                        f"OK ({n_blocks} blocks, {n_brains} brain assignments, "
                        f"{_time.perf_counter()-t_step:.1f}s)")
        else:
            thin_cfg = None
            logger.warning("[STARTUP] Step 4/8: thin_ensemble.json not found — using full ensemble")

        # Initialize remaining components
        from replay.runners.run_replay import RegimeSimulator, EnsembleEngine
        from risk.engine.portfolio_state import PortfolioState
        from risk.engine.exposure_controller import ExposureController

        t_step = _time.perf_counter()
        regime_sim = RegimeSimulator(str(PROJECT_ROOT / 'regime' / 'regime_policy.json'))
        ensemble_engine = EnsembleEngine(str(PROJECT_ROOT / 'ensemble' / 'ensemble_config.json'))
        risk_controller = ExposureController(str(PROJECT_ROOT / 'risk' / 'config' / 'risk_policy.json'))
        portfolio_state = PortfolioState(
            str(PROJECT_ROOT / 'risk' / 'config' / 'instrument_specs.json'),
            str(PROJECT_ROOT / 'risk' / 'config' / 'correlation_groups.json'),
        )
        logger.info(f"[STARTUP]          Regime + ensemble + risk initialized ({_time.perf_counter()-t_step:.1f}s)")

        self._handler = LiveInferenceHandler(
            model_loader=model_loader,
            regime_sim=regime_sim,
            ensemble_engine=ensemble_engine,
            risk_controller=risk_controller,
            portfolio_state=portfolio_state,
            quarantine_config=quarantine_cfg,
            portfolio_config=portfolio_cfg,
            feature_engine=LiveFeatureEngine(),
            defensive_config=defensive_cfg,
            instrument_specs=instrument_specs,
        )
        logger.info("[STARTUP]          Inference handler built")

    async def _on_bar_close(self, pair: str, tf: str, bars: list):
        """Called by bar builder when a bar closes. Runs full pipeline."""
        logger.info(f"[BAR_CLOSE] {pair}_{tf}: {len(bars)} bars received, "
                     f"last={bars[-1]['time'] if bars else '?'}")
        if self._halted:
            logger.warning(f"[BAR_CLOSE] HALTED — skipping {pair}_{tf}")
            return

        # Daily reset
        today = datetime.now(timezone.utc).date()
        if self._last_day != today:
            self._last_day = today
            self.tracker.reset_daily()
            logger.info("[DAILY_RESET] Daily stats reset")

        t0 = _time.perf_counter()

        # Build request in the format LiveInferenceHandler expects
        request = {
            'pair': pair,
            'tf': tf,
            'request_type': 'RAW_BARS',
            'bars': {tf: bars},
            'request_id': f"{pair}_{tf}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}",
            'mode': 'LIVE',
        }

        # Run inference
        response = self._handler.process_request(request)
        latency = (_time.perf_counter() - t0) * 1000

        action = response.get('action', 'SKIP')
        sig = response.get('signal', 0)
        confidence = response.get('confidence', 0)
        agreement = response.get('agreement_score', 0)
        lot_size = response.get('lot_size', 0)
        risk_approved = response.get('risk_approved', False)
        reason = response.get('reason_codes', '')
        models_voted = response.get('models_voted', 0)
        models_agreed = response.get('models_agreed', 0)

        sig_str = {-1: 'SHORT', 0: 'FLAT', 1: 'LONG'}.get(sig, str(sig))
        logger.info(f"[INFERENCE] {pair}_{tf}: signal={sig_str} conf={confidence:.3f} "
                     f"agree={agreement:.3f} voted={models_voted} agreed={models_agreed} "
                     f"regime={response.get('regime_state', '?')} action={action} "
                     f"risk_ok={risk_approved} lot={lot_size} reason={reason} "
                     f"latency={latency:.0f}ms")

        # ── DIAGNOSTIC: Feature + per-brain output dump ──
        self._log_diagnostic(pair, tf, bars)

        # Log to CSV + update handler equity
        account = await self.ib.accountSummaryAsync() if self.ib.isConnected() else []
        equity = 0.0
        for item in account:
            if item.tag == 'NetLiquidation' and item.currency == 'USD':
                equity = float(item.value)
                if self._handler:
                    self._handler.set_equity(equity)
                break

        self.logger_csv.log({
            'pair': pair, 'tf': tf, 'request_id': request['request_id'],
            'signal': sig, 'confidence': confidence, 'agreement_score': agreement,
            'regime_state': response.get('regime_state', 1),
            'action': action, 'risk_approved': risk_approved,
            'risk_reason': response.get('risk_reason', ''),
            'reason_codes': reason,
            'models_voted': models_voted, 'models_agreed': models_agreed,
            'lot_size': lot_size, 'spread': 0, 'latency_ms': round(latency, 1),
            'equity': equity, 'balance': equity,
            'open_positions': self.tracker.open_count(),
            'daily_pnl': self.tracker.daily_pnl, 'commission': 0,
        })

        # ── Signal-based close logic (executor-level, not handler-level) ──
        # The handler's portfolio state may not reflect our actual positions.
        # We check the executor's own tracker for close decisions.
        has_pos = self.tracker.has_position(pair, tf)
        pos = self.tracker.get_position(pair, tf)

        if has_pos and sig != 0 and pos['side'] != sig:
            # Opposite signal → close immediately
            logger.info(f"[CLOSE] SIGNAL_REVERSAL: {pair}_{tf} was "
                         f"{'LONG' if pos['side']==1 else 'SHORT'} → signal={sig_str}")
            await self._close_position(pair, tf)
            # After close, proceed to potentially open the new direction below

        if has_pos and sig == 0 and confidence > 0.8:
            # FLAT with high confidence — close if position is stale (2+ hours)
            age = (datetime.now(timezone.utc) - pos['entry_ts']).total_seconds() / 3600
            if age >= 2.0:
                logger.info(f"[CLOSE] FLAT_HIGH_CONF: {pair}_{tf} flat conf={confidence:.2f} "
                             f"age={age:.1f}h")
                await self._close_position(pair, tf)

        # Re-check position after potential close
        has_pos = self.tracker.has_position(pair, tf)

        # Skip/Hold: no further action
        if action == 'SKIP' or (action == 'HOLD' and not (sig != 0 and not has_pos)):
            # Store signal for cycle summary
            self._last_signals[f"{pair}_{tf}"] = {
                'signal': sig_str, 'action': action, 'reason': reason,
                'has_pos': has_pos}
            return

        # Kill switch checks
        if self.tracker.daily_pnl < -self.kill_switch['max_daily_loss_usd']:
            logger.error(f"KILL SWITCH: daily loss ${self.tracker.daily_pnl:.2f} "
                          f"exceeds -${self.kill_switch['max_daily_loss_usd']}")
            self._halted = True
            return
        if self.tracker.daily_trades >= self.kill_switch['max_daily_trades']:
            logger.error(f"KILL SWITCH: {self.tracker.daily_trades} trades today "
                          f"(max {self.kill_switch['max_daily_trades']})")
            self._halted = True
            return

        # Open new position if signal is directional and no position exists
        if sig != 0 and not has_pos:
            # Also check if ANY TF on this pair has a tracked position
            pair_has_any = any(self.tracker.has_position(pair, t) for t in ['H1', 'M30'])
            if pair_has_any:
                logger.info(f"[BLOCKED] {pair}_{tf}: pair already has position on another TF")
                self._last_signals[f"{pair}_{tf}"] = {
                    'signal': sig_str, 'action': 'BLOCKED', 'reason': 'PAIR_HAS_POSITION',
                    'has_pos': False}
                return
            if self.tracker.open_count() >= self.kill_switch['max_open_positions']:
                logger.warning(f"[BLOCKED] {pair}_{tf}: max positions "
                                f"({self.kill_switch['max_open_positions']})")
                self._last_signals[f"{pair}_{tf}"] = {
                    'signal': sig_str, 'action': 'BLOCKED', 'reason': 'MAX_POSITIONS',
                    'has_pos': False}
                return
            lot = lot_size if lot_size > 0 else 0.02
            await self._place_order(pair, tf, sig, lot)

        elif action == 'CLOSE' and has_pos:
            await self._close_position(pair, tf)

        # Store signal for cycle summary
        self._last_signals[f"{pair}_{tf}"] = {
            'signal': sig_str, 'action': action, 'reason': reason,
            'has_pos': self.tracker.has_position(pair, tf)}

    def _log_diagnostic(self, pair, tf, bars):
        """Log detailed feature + per-brain diagnostic for debugging FLAT signals."""
        try:
            from inference.live_feature_adapter import LiveFeatureAdapter

            # Get or reuse the handler's adapter
            adapter = getattr(self._handler, '_feature_adapter', None)
            if adapter is None:
                adapter = LiveFeatureAdapter()

            # Compute features
            cross_closes = {}
            if hasattr(self._handler, '_bar_cache'):
                cross_closes = self._handler._bar_cache.get_all_pair_closes(tf)

            schema_vec = adapter.compute(pair, tf, {tf: bars}, cross_pair_closes=cross_closes)
            if schema_vec is None:
                logger.warning(f"[DIAG] {pair}_{tf}: feature adapter returned None")
                return

            model_vec = adapter.reorder_for_model(schema_vec, pair, tf)

            # Feature stats
            n_zero = int(np.sum(model_vec == 0))
            n_nan = int(np.sum(np.isnan(model_vec)))
            n_inf = int(np.sum(np.isinf(model_vec)))
            names = adapter.get_feature_names()

            logger.info(f"[DIAG] {pair}_{tf}: features shape={model_vec.shape} "
                         f"zeros={n_zero} nan={n_nan} inf={n_inf} "
                         f"nonzero={len(model_vec)-n_zero}/{len(model_vec)}")

            # First 10 features
            first_10 = ', '.join(f"{names[i] if i < len(names) else f'f{i}'}={model_vec[i]:.4f}"
                                 for i in range(min(10, len(model_vec))))
            logger.info(f"[DIAG] {pair}_{tf} first10: {first_10}")

            # Last 10 features
            n = len(model_vec)
            last_10 = ', '.join(f"{names[i] if i < len(names) else f'f{i}'}={model_vec[i]:.4f}"
                                for i in range(max(0, n - 10), n))
            logger.info(f"[DIAG] {pair}_{tf} last10: {last_10}")

            # Per-brain predictions
            block_key = f"{pair}_{tf}"
            models = self._handler.model_loader.get_models(block_key) or {}
            thin_block = (self._handler._thin_ensemble or {}).get('blocks', {}).get(block_key, {})
            eligible = thin_block.get('brains', [])

            class_names = {0: 'SHORT', 1: 'FLAT', 2: 'LONG'}
            votes = {0: 0, 1: 0, 2: 0}
            brain_results = []

            for brain_name, backend in models.items():
                if eligible and brain_name not in eligible:
                    continue
                try:
                    fv = model_vec.reshape(1, -1).astype(np.float32)
                    probs = backend.predict_proba(fv)[0]
                    pred = int(np.argmax(probs))
                    votes[pred] += 1
                    brain_results.append((brain_name, pred, probs))
                except Exception as e:
                    brain_results.append((brain_name, -1, str(e)[:40]))

            for brain_name, pred, probs in brain_results:
                if isinstance(probs, np.ndarray) and len(probs) >= 3:
                    logger.info(f"[DIAG] {pair}_{tf} {brain_name:<25} → "
                                 f"{class_names.get(pred, '?'):<5} "
                                 f"P(S)={probs[0]:.3f} P(F)={probs[1]:.3f} P(L)={probs[2]:.3f}")
                else:
                    logger.info(f"[DIAG] {pair}_{tf} {brain_name:<25} → ERROR: {probs}")

            logger.info(f"[DIAG] {pair}_{tf} VOTES: SHORT={votes[0]} FLAT={votes[1]} "
                         f"LONG={votes[2]} (total={sum(votes.values())})")

        except Exception as e:
            logger.error(f"[DIAG] {pair}_{tf} diagnostic error: {e}", exc_info=True)

    def _check_ibkr_position(self, pair):
        """Check if IBKR already has a position on this pair (any TF)."""
        ibkr_positions = self.ib.positions()
        ibkr_pair = self.pairs_map.get(pair, '')
        for p in ibkr_positions:
            sym = p.contract.localSymbol or ''
            combo = f"{p.contract.symbol}.{p.contract.currency}"
            raw = f"{p.contract.symbol}{p.contract.currency}"
            if ibkr_pair in (sym, combo, raw, sym.replace('.', '')):
                if abs(float(p.position)) > 0.01:
                    return float(p.position)
        return 0.0

    async def _place_order(self, pair: str, tf: str, side: int, lot_size: float):
        """Place a market order on IBKR using pre-qualified contract."""
        contract = self._get_qualified_contract(pair)
        if not contract:
            logger.error(f"[ORDER] REJECTED: no qualified contract for {pair}")
            return
        ibkr_pair = self.pairs_map.get(pair, pair)

        # CRITICAL: Check IBKR directly for existing position on this pair
        # Prevents stacking if tracker is out of sync
        existing_qty = self._check_ibkr_position(pair)
        if abs(existing_qty) > 0.01:
            existing_side = 'LONG' if existing_qty > 0 else 'SHORT'
            new_side = 'LONG' if side == 1 else 'SHORT'
            # Allow if closing (opposite direction) but block if stacking (same direction)
            if (existing_qty > 0 and side == 1) or (existing_qty < 0 and side == -1):
                logger.warning(f"[ORDER] BLOCKED: {pair} already has IBKR position "
                                f"{existing_side} {abs(existing_qty):.0f} units — "
                                f"would stack {new_side}")
                # Adopt the position if not tracked
                if not any(self.tracker.has_position(pair, t) for t in ['H1', 'M30']):
                    logger.info(f"[ORDER] Adopting untracked IBKR position for {pair}")
                    self.tracker.positions[f"{pair}_{tf}"] = {
                        'pair': pair, 'tf': tf,
                        'side': 1 if existing_qty > 0 else -1,
                        'qty': abs(existing_qty) / 100000,
                        'entry_price': 0, 'entry_ts': datetime.now(timezone.utc),
                        'order_id': -1, 'max_profit_pips': 0.0,
                        'current_price': 0, 'unrealized_pnl': 0.0,
                    }
                return

        units = round(lot_size * self.ibkr_cfg.get('lot_to_units', 100000))
        if units < 1:
            units = 20000

        order_action = 'BUY' if side == 1 else 'SELL'
        order = ib.MarketOrder(order_action, units)
        order.tif = 'IOC'

        # Get pre-trade price for slippage calculation
        pre_price = await self._get_current_price(pair)
        logger.info(f"[ORDER] SUBMITTING: {order_action} {units} {ibkr_pair} "
                     f"lot={lot_size} pre_price={pre_price}")

        trade = self.ib.placeOrder(contract, order)
        max_wait = 100 if not self.paper else 50
        for _ in range(max_wait):
            await asyncio.sleep(0.1)
            if trade.orderStatus.status in ('Filled', 'Cancelled', 'Inactive'):
                break

        status = trade.orderStatus.status
        if status == 'Filled':
            fill_price = trade.orderStatus.avgFillPrice
            order_id = trade.order.orderId
            slippage = 0.0
            if pre_price and pre_price > 0:
                slippage = abs(fill_price - pre_price) / self._get_pip_size(pair)
            self.tracker.open_position(pair, tf, side, lot_size, fill_price, order_id)
            logger.info(f"[ORDER] FILLED: {order_action} {units} {ibkr_pair} "
                         f"@ {fill_price} (slippage={slippage:.1f} pips, "
                         f"orderId={order_id})")
        else:
            logger.warning(f"[ORDER] NOT_FILLED: {order_action} {units} {ibkr_pair} "
                            f"status={status}")

    async def _close_position(self, pair: str, tf: str):
        """Close an existing position."""
        pos = self.tracker.get_position(pair, tf)
        if not pos:
            return

        contract = self._get_qualified_contract(pair)
        if not contract:
            logger.error(f"[CLOSE] REJECTED: no qualified contract for {pair}")
            return
        ibkr_pair = self.pairs_map.get(pair, pair)

        units = round(pos['qty'] * self.ibkr_cfg.get('lot_to_units', 100000))
        close_action = 'SELL' if pos['side'] == 1 else 'BUY'
        order = ib.MarketOrder(close_action, units)
        order.tif = 'IOC'

        side_str = 'LONG' if pos['side'] == 1 else 'SHORT'
        logger.info(f"[CLOSE] SUBMITTING: {close_action} {units} {ibkr_pair} "
                     f"(was {side_str} @ {pos['entry_price']})")

        trade = self.ib.placeOrder(contract, order)
        for _ in range(100):
            await asyncio.sleep(0.1)
            if trade.orderStatus.status in ('Filled', 'Cancelled', 'Inactive'):
                break

        if trade.orderStatus.status == 'Filled':
            fill_price = trade.orderStatus.avgFillPrice
            order_id = trade.order.orderId
            usdjpy = await self._get_current_price('USDJPY') or 150.0
            result = self.tracker.close_position(pair, tf, fill_price, order_id,
                                                  usdjpy_rate=usdjpy)
            pnl = result['pnl_usd'] if result else 0
            logger.info(f"[CLOSE] FILLED: {pair}_{tf} exit={fill_price} "
                         f"pnl=${pnl:.2f} orderId={order_id}")
        else:
            logger.warning(f"[CLOSE] NOT_FILLED: {pair}_{tf} status={trade.orderStatus.status}")

    # ──────────────────────────────────────────────────────────────
    # Position Management Loop (SL / TP / trailing / stale / kill)
    # ──────────────────────────────────────────────────────────────

    def _get_pip_size(self, pair):
        return 0.01 if 'JPY' in pair else 0.0001

    def _get_pip_value(self, pair):
        specs = {'EURUSD': 10, 'GBPUSD': 10, 'USDJPY': 9.1, 'AUDUSD': 10,
                 'USDCAD': 7.7, 'USDCHF': 10.3, 'NZDUSD': 10, 'EURJPY': 9.1, 'GBPJPY': 9.1}
        return specs.get(pair, 10.0)

    def _price_to_pips(self, pair, price_diff):
        return price_diff / self._get_pip_size(pair)

    def _get_qualified_contract(self, pair):
        """Get the qualified (with conId) contract."""
        return self._qualified_contracts.get(pair)

    async def _get_current_price(self, pair):
        """Get current price from streaming market data ticker."""
        ticker = self._live_tickers.get(pair)
        if not ticker:
            logger.debug(f"{pair} ticker: no subscription")
            return None
        try:
            # marketPrice() returns best available: last, close, or mid
            mp = ticker.marketPrice()
            if mp and mp == mp and mp > 0:  # not NaN, not zero
                return float(mp)
            # Fallback: midpoint
            mid = ticker.midpoint()
            if mid and mid == mid and mid > 0:
                return float(mid)
            # Fallback: bid/ask average
            if ticker.bid > 0 and ticker.ask > 0:
                return float((ticker.bid + ticker.ask) / 2)
            logger.debug(f"{pair} ticker: no valid price (bid={ticker.bid} ask={ticker.ask} "
                         f"last={ticker.last} close={ticker.close})")
        except Exception as e:
            logger.debug(f"{pair} ticker error: {e}")
        return None

    async def _position_management_loop(self):
        """60-second loop: SL, TP, trailing stop, stale exit, kill switch."""
        pm = self.ibkr_cfg.get('position_management', {})
        sl_pips = pm.get('stop_loss_pips', 50)
        tp_pips = pm.get('take_profit_pips', 100)
        trail_start = pm.get('trailing_start_pips', 30)
        trail_dist = pm.get('trailing_distance_pips', 20)
        stale_hours = pm.get('stale_position_hours', 24)
        stale_min_pips = pm.get('stale_position_min_profit_pips', 5)
        interval = pm.get('check_interval_seconds', 60)
        spread_mult = pm.get('spread_spike_multiplier', 2.5)
        baseline_spreads = self.ibkr_cfg.get('baseline_spreads_pips', {})

        while True:
            await asyncio.sleep(interval)
            if not self.ib.isConnected() or not self.tracker.positions:
                continue

            now = datetime.now(timezone.utc)
            total_unrealized = 0.0

            # Iterate over a snapshot (positions may be modified during iteration)
            for key, pos in list(self.tracker.positions.items()):
                pair = pos['pair']
                tf = pos['tf']

                # Fetch current price from streaming market data
                current = await self._get_current_price(pair)
                if current is None:
                    logger.warning(f"[POS_MGMT] {key}: no live price available, skipping checks")
                    continue
                logger.info(f"[POS_MGMT] {key}: live={current:.5f} entry={pos['entry_price']:.5f}")

                pos['current_price'] = current
                pip_size = self._get_pip_size(pair)
                pip_val = self._get_pip_value(pair)

                # Compute PnL in pips
                if pos['side'] == 1:  # LONG
                    pnl_pips = (current - pos['entry_price']) / pip_size
                else:  # SHORT
                    pnl_pips = (pos['entry_price'] - current) / pip_size

                # Convert pips to USD — JPY pairs need USDJPY conversion
                if 'JPY' in pair:
                    usdjpy = await self._get_current_price('USDJPY') or 150.0
                    # For JPY pairs: PnL in USD = price_diff * units / USDJPY
                    price_diff = (current - pos['entry_price']) * pos['side']
                    pnl_usd = price_diff * pos['qty'] * 100000 / usdjpy
                else:
                    pnl_usd = pnl_pips * pip_val * pos['qty']
                pos['unrealized_pnl'] = pnl_usd
                total_unrealized += pnl_usd

                # Track max profit for trailing stop
                if pnl_pips > pos.get('max_profit_pips', 0):
                    pos['max_profit_pips'] = pnl_pips

                age = now - pos['entry_ts']
                age_hours = age.total_seconds() / 3600

                # ── Stop Loss ──
                if pnl_pips <= -sl_pips:
                    logger.warning(f"[CLOSE] SL_HIT pair={pair} tf={tf} pips={pnl_pips:.1f} "
                                    f"limit=-{sl_pips}")
                    await self._close_position(pair, tf)
                    self.logger_csv.log({'timestamp': now.isoformat(), 'pair': pair, 'tf': tf,
                                         'action': 'CLOSE', 'reason_codes': 'STOP_LOSS_HIT',
                                         'signal': 0, 'confidence': 0})
                    continue

                # ── Take Profit ──
                if pnl_pips >= tp_pips:
                    logger.info(f"[CLOSE] TP_HIT pair={pair} tf={tf} pips=+{pnl_pips:.1f} "
                                 f"limit=+{tp_pips}")
                    await self._close_position(pair, tf)
                    self.logger_csv.log({'timestamp': now.isoformat(), 'pair': pair, 'tf': tf,
                                         'action': 'CLOSE', 'reason_codes': 'TAKE_PROFIT_HIT',
                                         'signal': 0, 'confidence': 0})
                    continue

                # ── Trailing Stop ──
                max_p = pos.get('max_profit_pips', 0)
                if max_p >= trail_start and pnl_pips <= (max_p - trail_dist):
                    logger.info(f"[CLOSE] TRAILING_HIT pair={pair} tf={tf} "
                                 f"max_pips=+{max_p:.1f} now=+{pnl_pips:.1f} "
                                 f"trail_start={trail_start} trail_dist={trail_dist}")
                    await self._close_position(pair, tf)
                    self.logger_csv.log({'timestamp': now.isoformat(), 'pair': pair, 'tf': tf,
                                         'action': 'CLOSE', 'reason_codes': 'TRAILING_STOP_HIT',
                                         'signal': 0, 'confidence': 0})
                    continue

                # ── Time-based stale exit ──
                if age_hours >= stale_hours and pnl_pips < stale_min_pips:
                    logger.info(f"[CLOSE] TIME_EXIT pair={pair} tf={tf} "
                                 f"age={age_hours:.1f}h pips={pnl_pips:.1f}")
                    await self._close_position(pair, tf)
                    self.logger_csv.log({'timestamp': now.isoformat(), 'pair': pair, 'tf': tf,
                                         'action': 'CLOSE', 'reason_codes': 'TIME_EXIT_STALE',
                                         'signal': 0, 'confidence': 0})
                    continue

            # ── Kill switch: realized + unrealized ──
            total_pnl = self.tracker.daily_pnl + total_unrealized
            if total_pnl < -self.kill_switch['max_daily_loss_usd']:
                logger.error(f"KILL_SWITCH_ACTIVATED: daily PnL ${total_pnl:.2f} "
                              f"(realized ${self.tracker.daily_pnl:.2f} + "
                              f"unrealized ${total_unrealized:.2f})")
                self._halted = True
                # Close all positions
                for key, pos in list(self.tracker.positions.items()):
                    await self._close_position(pos['pair'], pos['tf'])
                self.logger_csv.log({'timestamp': now.isoformat(), 'pair': 'ALL', 'tf': 'ALL',
                                     'action': 'CLOSE_ALL', 'reason_codes': 'KILL_SWITCH_ACTIVATED',
                                     'signal': 0, 'confidence': 0})

            # ── Spread spike detection (from streaming tickers) ──
            try:
                any_spiked = False
                for pair in self.pairs_map:
                    ticker = self._live_tickers.get(pair)
                    if not ticker:
                        continue
                    if ticker.ask > 0 and ticker.bid > 0:
                        spread = (ticker.ask - ticker.bid) / self._get_pip_size(pair)
                        baseline = baseline_spreads.get(pair, 1.0)
                        if spread > baseline * spread_mult:
                            any_spiked = True
                            if not self._defensive_active:
                                logger.warning(f"[DEFENSIVE] ACTIVATED: {pair} "
                                               f"spread={spread:.1f} pips (baseline={baseline})")
                if any_spiked and not self._defensive_active:
                    self._defensive_active = True
                    if self._handler:
                        self._handler.set_defensive_mode(True)
                elif not any_spiked and self._defensive_active:
                    logger.info("[DEFENSIVE] DEACTIVATED: all spreads normalized")
                    self._defensive_active = False
                    if self._handler:
                        self._handler.set_defensive_mode(False)
            except Exception as e:
                logger.debug(f"Spread check error: {e}")

    async def _dashboard_loop(self):
        """Print console dashboard every N seconds with live PnL."""
        interval = self.ibkr_cfg.get('dashboard_interval_sec', 30)
        while True:
            await asyncio.sleep(interval)
            if not self.ib.isConnected():
                continue  # Reconnect loop handles logging; don't spam

            now = datetime.now(timezone.utc)
            now_str = now.strftime('%H:%M:%S')
            mode = "PAPER" if self.paper else "** LIVE **"
            halted = " ** HALTED **" if self._halted else ""

            total_unrealized = sum(p.get('unrealized_pnl', 0) for p in self.tracker.positions.values())

            print(f"\n{'='*78}")
            print(f"  CHAOS V1.0 IBKR Executor  [{mode}]  {now_str} UTC{halted}")
            print(f"{'='*78}")
            print(f"  Connected: {self.ib.isConnected()}")
            print(f"  Open positions: {self.tracker.open_count()}/{self.kill_switch['max_open_positions']}")
            print(f"  Daily trades: {self.tracker.daily_trades}/{self.kill_switch['max_daily_trades']}")
            print(f"  Realized PnL:   ${self.tracker.daily_pnl:>8.2f}")
            print(f"  Unrealized PnL: ${total_unrealized:>8.2f}")
            print(f"  Total PnL:      ${self.tracker.daily_pnl + total_unrealized:>8.2f}  "
                  f"(kill @ -${self.kill_switch['max_daily_loss_usd']})")
            wr = (self.tracker.daily_wins / max(self.tracker.daily_trades, 1) * 100)
            print(f"  Win rate: {wr:.0f}%")

            if self.tracker.positions:
                print(f"\n[POSITIONS] {now_str} UTC")
                for key, pos in self.tracker.positions.items():
                    side_str = 'LONG ' if pos['side'] == 1 else 'SHORT'
                    current = pos.get('current_price', pos['entry_price'])
                    pnl = pos.get('unrealized_pnl', 0)
                    age = now - pos['entry_ts']
                    hours = int(age.total_seconds() // 3600)
                    mins = int((age.total_seconds() % 3600) // 60)
                    pnl_str = f'+${pnl:.2f}' if pnl >= 0 else f'-${abs(pnl):.2f}'
                    # Determine price format
                    prec = 3 if 'JPY' in pos['pair'] else 5
                    print(f"  {key:<14} {side_str} {pos['qty']:.2f} "
                          f"@ {pos['entry_price']:.{prec}f}  "
                          f"current: {current:.{prec}f}  "
                          f"PnL: {pnl_str:<10} age: {hours}h {mins:02d}m")

            # Cycle summary
            if self._last_signals:
                directional = sum(1 for s in self._last_signals.values()
                                  if s['signal'] not in ('FLAT', '0'))
                total_sigs = len(self._last_signals)
                print(f"\n[CYCLE SUMMARY] {now_str} UTC")
                for pair_name in sorted(set(p.split('_')[0] for p in self._last_signals)):
                    parts = []
                    for tf_name in ['H1', 'M30']:
                        k = f"{pair_name}_{tf_name}"
                        s = self._last_signals.get(k)
                        if s:
                            tag = s['signal']
                            if s.get('has_pos'):
                                tag += '(open)'
                            elif s.get('action') == 'BLOCKED':
                                tag += f"(blocked:{s.get('reason','')})"
                            parts.append(f"{tf_name}={tag}")
                    if parts:
                        print(f"  {pair_name:<8} {' '.join(parts)}")
                print(f"  Signals: {directional} directional / {total_sigs} total  "
                      f"Positions: {self.tracker.open_count()}/{self.kill_switch['max_open_positions']}")
            print()

    async def start(self):
        """Main entry point with step-by-step verbose logging."""
        t_total = _time.perf_counter()
        mode_str = "PAPER" if self.paper else "LIVE"
        logger.info(f"{'='*60}")
        logger.info(f"  CHAOS V1.0 IBKR Executor — {mode_str} mode")
        logger.info(f"  Target: {self.host}:{self.port} (clientId={self.client_id})")
        logger.info(f"{'='*60}")

        # Steps 1-4 happen inside _load_inference_handler
        self._load_inference_handler()

        # Step 5: Connect to IBKR
        t_step = _time.perf_counter()
        logger.info("[STARTUP] Step 5/8: Connecting to IBKR...")
        try:
            await self.ib.connectAsync(
                self.host, self.port, clientId=self.client_id, timeout=60)
        except Exception as e:
            logger.error(f"[STARTUP] Step 5/8: FAILED — {e}")
            logger.error("Make sure TWS/IB Gateway is running and API connections are enabled")
            return
        sv = self.ib.client.serverVersion()
        logger.info(f"[STARTUP] Step 5/8: Connecting to IBKR... "
                     f"OK (server v{sv}, {_time.perf_counter()-t_step:.1f}s)")

        # Post-connect settle
        logger.info("[STARTUP]          Settling 5s...")
        await asyncio.sleep(5)

        # Step 6: Account summary
        t_step = _time.perf_counter()
        logger.info("[STARTUP] Step 6/8: Account summary...")
        equity_str = "unknown"
        try:
            await self.ib.reqAccountSummaryAsync()
            await asyncio.sleep(2)
            acct = await self.ib.accountSummaryAsync()
            for item in acct:
                if item.tag == 'NetLiquidation' and item.currency == 'USD':
                    equity_val = float(item.value)
                    equity_str = f"${equity_val:,.2f}"
                    if self._handler:
                        self._handler.set_equity(equity_val)
                        logger.info(f"[STARTUP]          Handler equity set to ${equity_val:,.2f}")
                    break
            logger.info(f"[STARTUP] Step 6/8: Account summary... "
                        f"OK (equity: {equity_str}, {_time.perf_counter()-t_step:.1f}s)")
        except Exception as e:
            logger.warning(f"[STARTUP] Step 6/8: Account summary... WARN ({e})")

        # Reconcile positions
        try:
            await self.tracker.reconcile(self.ib, self.pairs_map)
        except Exception as e:
            logger.warning(f"[STARTUP]          Position reconciliation: {e}")
        await asyncio.sleep(2)

        # Step 7: Subscribe pairs
        t_step = _time.perf_counter()
        logger.info("[STARTUP] Step 7/8: Subscribing pairs...")
        tfs = self.ibkr_cfg.get('timeframes', ['H1', 'M30'])
        self._bar_builder = IBKRBarBuilder(
            ib_client=self.ib,
            pairs_map=self.pairs_map,
            timeframes=tfs,
            bars_count=self.ibkr_cfg.get('bars_to_request', 500),
            pacing_delay=self.ibkr_cfg.get('pacing_delay_sec', 2),
            on_bar_close=self._on_bar_close,
        )

        qualified_count = 0
        total_pairs = len(self.pairs_map)
        for i, (chaos_pair, ibkr_pair) in enumerate(self.pairs_map.items(), 1):
            try:
                base, quote = ibkr_pair.split('.')
                contract = ib.Forex(base + quote, exchange='IDEALPRO')
                result = await self.ib.qualifyContractsAsync(contract)
                if result:
                    qc = result[0]
                    self._qualified_contracts[chaos_pair] = qc
                    self._bar_builder._contracts[chaos_pair] = qc
                    # Subscribe to streaming market data for live prices
                    self._live_tickers[chaos_pair] = self.ib.reqMktData(qc, '', False, False)
                    qualified_count += 1
                    logger.info(f"  [{i}/{total_pairs}] {chaos_pair} ({ibkr_pair})... "
                                f"OK (conId={qc.conId}, mktData subscribed)")
                else:
                    logger.warning(f"  [{i}/{total_pairs}] {chaos_pair} ({ibkr_pair})... FAILED")
            except Exception as e:
                logger.warning(f"  [{i}/{total_pairs}] {chaos_pair} ({ibkr_pair})... ERROR: {e}")
            await asyncio.sleep(2)

        logger.info(f"[STARTUP] Step 7/8: Subscribing pairs... "
                     f"{qualified_count}/{total_pairs} OK ({_time.perf_counter()-t_step:.1f}s)")

        if qualified_count == 0:
            logger.error("[STARTUP] No contracts qualified — cannot proceed")
            self.ib.disconnect()
            return

        # Step 8: Start bar builder
        logger.info(f"[STARTUP] Step 8/8: Starting bar builder...")
        logger.info(f"[STARTUP] Step 8/8: Starting bar builder... OK")
        total_startup = _time.perf_counter() - t_total
        logger.info(f"[STARTUP] READY. Total startup: {total_startup:.0f}s. "
                     f"Monitoring {qualified_count} pairs on {tfs}. "
                     f"Waiting for bar closes.")

        # --test-close: close oldest position to verify close flow
        if self._test_close and self.tracker.positions:
            oldest_key = min(self.tracker.positions,
                             key=lambda k: self.tracker.positions[k]['entry_ts'])
            pos = self.tracker.positions[oldest_key]
            logger.info(f"[TEST_CLOSE] Closing oldest position: {oldest_key} "
                         f"({'LONG' if pos['side']==1 else 'SHORT'} "
                         f"@ {pos['entry_price']})")
            await self._close_position(pos['pair'], pos['tf'])
            logger.info(f"[TEST_CLOSE] Done. Positions remaining: {self.tracker.open_count()}")
        elif self._test_close:
            logger.info("[TEST_CLOSE] No open positions to close")

        await self._run_with_reconnect()

    async def _subscribe_pairs(self):
        """(Re)subscribe to all pairs: qualify contracts + market data."""
        tfs = self.ibkr_cfg.get('timeframes', ['H1', 'M30'])
        if not self._bar_builder:
            self._bar_builder = IBKRBarBuilder(
                ib_client=self.ib,
                pairs_map=self.pairs_map,
                timeframes=tfs,
                bars_count=self.ibkr_cfg.get('bars_to_request', 500),
                pacing_delay=self.ibkr_cfg.get('pacing_delay_sec', 2),
                on_bar_close=self._on_bar_close,
            )
        else:
            self._bar_builder.ib = self.ib

        self._qualified_contracts.clear()
        self._live_tickers.clear()
        qualified_count = 0
        total_pairs = len(self.pairs_map)

        for i, (chaos_pair, ibkr_pair) in enumerate(self.pairs_map.items(), 1):
            try:
                base, quote = ibkr_pair.split('.')
                contract = ib.Forex(base + quote, exchange='IDEALPRO')
                result = await self.ib.qualifyContractsAsync(contract)
                if result:
                    qc = result[0]
                    self._qualified_contracts[chaos_pair] = qc
                    self._bar_builder._contracts[chaos_pair] = qc
                    self._live_tickers[chaos_pair] = self.ib.reqMktData(qc, '', False, False)
                    qualified_count += 1
                    logger.info(f"  [{i}/{total_pairs}] {chaos_pair} ({ibkr_pair})... OK")
                else:
                    logger.warning(f"  [{i}/{total_pairs}] {chaos_pair} ({ibkr_pair})... FAILED")
            except Exception as e:
                logger.warning(f"  [{i}/{total_pairs}] {chaos_pair} ({ibkr_pair})... ERROR: {e}")
            await asyncio.sleep(2)

        logger.info(f"Subscribed {qualified_count}/{total_pairs} pairs")
        return qualified_count

    async def _reconnect(self):
        """Reconnect to IBKR, re-subscribe pairs, reconcile positions."""
        max_retries = 10
        base_delay = self.ibkr_cfg.get('reconnect_delay_sec', 30)

        for attempt in range(1, max_retries + 1):
            delay = min(base_delay * (2 ** (attempt - 1)), 300)  # Exponential, cap 5 min
            logger.info(f"[RECONNECT] Attempt {attempt}/{max_retries} in {delay}s...")

            # Countdown (log every 10s instead of spamming)
            remaining = delay
            while remaining > 0:
                await asyncio.sleep(min(10, remaining))
                remaining -= 10
                if remaining > 0:
                    logger.info(f"[RECONNECT] Reconnecting in {remaining}s...")

            # Disconnect cleanly if still half-open
            try:
                if self.ib.isConnected():
                    self.ib.disconnect()
            except Exception:
                pass
            await asyncio.sleep(1)

            try:
                logger.info(f"[RECONNECT] Connecting to {self.host}:{self.port}...")
                await self.ib.connectAsync(
                    self.host, self.port, clientId=self.client_id, timeout=60)
                logger.info(f"[RECONNECT] Connected (server v{self.ib.client.serverVersion()})")

                await asyncio.sleep(3)

                # Re-subscribe pairs
                logger.info("[RECONNECT] Re-subscribing pairs...")
                n = await self._subscribe_pairs()
                if n == 0:
                    logger.warning("[RECONNECT] No pairs qualified, will retry")
                    continue

                # Reconcile positions
                logger.info("[RECONNECT] Reconciling positions...")
                try:
                    await self.tracker.reconcile(self.ib, self.pairs_map)
                except Exception as e:
                    logger.warning(f"[RECONNECT] Reconciliation: {e}")

                logger.info(f"[RECONNECT] SUCCESS — resumed trading with {n} pairs")
                return True

            except Exception as e:
                logger.error(f"[RECONNECT] Attempt {attempt} failed: {e}")

        # All retries exhausted — IB Gateway is likely fully down
        logger.critical(f"[CRITICAL] IB Gateway appears to be down — "
                         f"all {max_retries} reconnect attempts failed")

        # Attempt to restart IB Gateway
        try:
            import subprocess
            # Common IB Gateway paths on Windows
            gw_paths = [
                r'C:\Jts\ibgateway\1021\ibgateway.exe',
                r'C:\Jts\ibgateway\ibgateway.exe',
                os.path.expanduser(r'~\Jts\ibgateway\1021\ibgateway.exe'),
            ]
            for gw_path in gw_paths:
                if os.path.exists(gw_path):
                    logger.info(f"[CRITICAL] Attempting to restart IB Gateway: {gw_path}")
                    subprocess.Popen([gw_path], creationflags=getattr(subprocess, 'DETACHED_PROCESS', 0))
                    logger.info("[CRITICAL] IB Gateway restart command sent. "
                                 "Waiting 60s for it to initialize...")
                    await asyncio.sleep(60)

                    # One last reconnect attempt
                    try:
                        await self.ib.connectAsync(
                            self.host, self.port, clientId=self.client_id, timeout=60)
                        logger.info("[CRITICAL] IB Gateway restarted successfully!")
                        await asyncio.sleep(5)
                        n = await self._subscribe_pairs()
                        if n > 0:
                            try:
                                await self.tracker.reconcile(self.ib, self.pairs_map)
                            except Exception:
                                pass
                            logger.info(f"[CRITICAL] Recovered with {n} pairs after Gateway restart")
                            return True
                    except Exception as e:
                        logger.error(f"[CRITICAL] Gateway restart reconnect failed: {e}")
                    break
            else:
                logger.critical("[CRITICAL] IB Gateway executable not found — manual restart required")
        except Exception as e:
            logger.critical(f"[CRITICAL] Failed to restart IB Gateway: {e}")

        logger.critical("[CRITICAL] Entering dormant mode. Manual intervention required.")
        logger.critical("[CRITICAL] Restart IB Gateway, then restart the executor.")
        self._halted = True
        return False

    async def _run_with_reconnect(self):
        """Run the main loops with automatic reconnection on disconnect."""
        while not self._halted:
            try:
                await asyncio.gather(
                    self._bar_builder.run(),
                    self._dashboard_loop(),
                    self._position_management_loop(),
                )
            except KeyboardInterrupt:
                logger.info("[SHUTDOWN] KeyboardInterrupt — stopping")
                break
            except (ConnectionError, asyncio.IncompleteReadError, OSError) as e:
                logger.error(f"[DISCONNECT] Connection lost: {e}")
                self._bar_builder.stop()
                if not await self._reconnect():
                    break
                # Reset bar builder running state for re-entry
                self._bar_builder._running = False
            except Exception as e:
                err_str = str(e).lower()
                if 'peer closed' in err_str or 'connection' in err_str or 'reset' in err_str:
                    logger.error(f"[DISCONNECT] Connection lost: {e}")
                    self._bar_builder.stop()
                    if not await self._reconnect():
                        break
                    self._bar_builder._running = False
                else:
                    logger.error(f"[ERROR] Unexpected: {e}", exc_info=True)
                    break

        # Cleanup
        if self._bar_builder:
            self._bar_builder.stop()
        self.logger_csv.close()
        if self.ib.isConnected():
            self.ib.disconnect()
        logger.info("[SHUTDOWN] IBKR Executor stopped")


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description='CHAOS V1.0 IBKR Direct Executor')
    parser.add_argument('--paper', action='store_true', default=True,
                        help='Connect to paper trading (port 7497)')
    parser.add_argument('--live', action='store_true',
                        help='Connect to live trading (port 7496) — REAL MONEY')
    parser.add_argument('--thin-only', action='store_true',
                        help='Only load models in thin_ensemble.json (~30 instead of ~400)')
    parser.add_argument('--test-close', action='store_true',
                        help='Close oldest open position on startup to verify close flow')
    parser.add_argument('--host', default=None, help='Override IBKR host')
    parser.add_argument('--port', type=int, default=None, help='Override IBKR port')
    parser.add_argument('--client-id', type=int, default=None, help='Override client ID')
    args = parser.parse_args()

    is_paper = not args.live

    if args.live:
        logger.warning("=" * 60)
        logger.warning("  *** LIVE TRADING MODE — REAL MONEY ***")
        logger.warning("=" * 60)

    executor = IBKRExecutor(paper=is_paper, thin_only=args.thin_only,
                            test_close=args.test_close)

    if args.host:
        executor.host = args.host
    if args.port:
        executor.port = args.port
    if args.client_id:
        executor.client_id = args.client_id

    asyncio.run(executor.start())


if __name__ == '__main__':
    main()
