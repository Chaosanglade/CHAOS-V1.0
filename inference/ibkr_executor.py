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

_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')

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
        return self.model.predict_proba(np.nan_to_num(X.astype(np.float64), nan=0, posinf=0, neginf=0))


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

    def close_position(self, pair: str, tf: str, exit_price: float, order_id: int):
        key = f"{pair}_{tf}"
        pos = self.positions.pop(key, None)
        if not pos:
            return None
        self._order_ids.add(order_id)
        # Compute PnL
        if pos['side'] == 1:  # LONG
            pnl = (exit_price - pos['entry_price']) * pos['qty'] * 100000
        else:  # SHORT
            pnl = (pos['entry_price'] - exit_price) * pos['qty'] * 100000
        trade = {**pos, 'exit_price': exit_price, 'exit_ts': datetime.now(timezone.utc),
                 'pnl_usd': pnl, 'close_order_id': order_id}
        self.closed_trades.append(trade)
        self.daily_pnl += pnl
        if pnl > 0:
            self.daily_wins += 1
        logger.info(f"POSITION CLOSED: {key} pnl=${pnl:.2f} (daily=${self.daily_pnl:.2f})")
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
        """Reconcile Python state with IBKR on connect/reconnect."""
        ibkr_positions = ib_client.positions()
        logger.info(f"IBKR reports {len(ibkr_positions)} positions, "
                     f"Python tracks {len(self.positions)}")
        for p in ibkr_positions:
            sym = p.contract.localSymbol if p.contract.localSymbol else ''
            logger.info(f"  IBKR position: {sym} qty={p.position} avg={p.avgCost}")


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

    def __init__(self, paper: bool = True):
        self.paper = paper

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
        logger.info("[STARTUP] Step 2/8: Loading models...")
        quarantined_brains = set(quarantine_cfg.get('global_quarantine', []))
        pairs = list(self.pairs_map.keys())
        tfs = self.ibkr_cfg.get('timeframes', ['H1', 'M30'])

        # Discover all model files
        model_files = []
        for pair in pairs:
            for tf in tfs:
                for ext in ['onnx', 'joblib']:
                    for path in sorted(_glob.glob(str(PROJECT_ROOT / 'models' / f'{pair}_{tf}_*.{ext}'))):
                        basename = os.path.basename(path)
                        parts = basename.replace(f'.{ext}', '').split('_')
                        brain = '_'.join(parts[2:])
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
                logger.info(f"[{idx:>3}/{total}] Loading {os.path.basename(path)}... "
                            f"SKIP ({reason})")
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
        if self._halted:
            logger.warning(f"HALTED — skipping {pair}_{tf}")
            return

        # Daily reset
        today = datetime.now(timezone.utc).date()
        if self._last_day != today:
            self._last_day = today
            self.tracker.reset_daily()
            logger.info("Daily stats reset")

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

        logger.info(f"{pair}_{tf}: action={action} signal={sig} conf={confidence:.2f} "
                     f"agree={agreement:.2f} lot={lot_size} reason={reason} "
                     f"latency={latency:.0f}ms")

        # Log to CSV
        account = await self.ib.accountSummaryAsync() if self.ib.isConnected() else []
        equity = 0.0
        for item in account:
            if item.tag == 'NetLiquidation' and item.currency == 'USD':
                equity = float(item.value)
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

        # Execute
        if action == 'SKIP' or action == 'HOLD':
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

        if action == 'OPEN' and risk_approved and sig != 0:
            if self.tracker.open_count() >= self.kill_switch['max_open_positions']:
                logger.warning(f"Max open positions ({self.kill_switch['max_open_positions']}) — skipping")
                return
            await self._place_order(pair, tf, sig, lot_size)

        elif action == 'CLOSE':
            await self._close_position(pair, tf)

    async def _place_order(self, pair: str, tf: str, side: int, lot_size: float):
        """Place a market order on IBKR using pre-qualified contract."""
        contract = self._get_qualified_contract(pair)
        if not contract:
            logger.error(f"No qualified contract for {pair}")
            return
        ibkr_pair = self.pairs_map.get(pair, pair)

        units = round(lot_size * self.ibkr_cfg.get('lot_to_units', 100000))
        if units < 1:
            units = 20000  # Minimum IBKR FX order

        order_action = 'BUY' if side == 1 else 'SELL'
        order = ib.MarketOrder(order_action, units)
        order.tif = 'IOC'

        logger.info(f"PLACING ORDER: {order_action} {units} {ibkr_pair} (lot={lot_size})")

        if self.paper:
            # Paper mode: place on paper account
            trade = self.ib.placeOrder(contract, order)
            # Wait for fill
            for _ in range(50):  # Up to 5 seconds
                await asyncio.sleep(0.1)
                if trade.orderStatus.status in ('Filled', 'Cancelled', 'Inactive'):
                    break

            if trade.orderStatus.status == 'Filled':
                fill_price = trade.orderStatus.avgFillPrice
                self.tracker.open_position(pair, tf, side, lot_size,
                                           fill_price, trade.order.orderId)
                logger.info(f"FILLED: {order_action} {units} {ibkr_pair} @ {fill_price}")
            else:
                logger.warning(f"Order not filled: status={trade.orderStatus.status}")
        else:
            # Live mode: same flow
            trade = self.ib.placeOrder(contract, order)
            for _ in range(100):
                await asyncio.sleep(0.1)
                if trade.orderStatus.status in ('Filled', 'Cancelled', 'Inactive'):
                    break
            if trade.orderStatus.status == 'Filled':
                fill_price = trade.orderStatus.avgFillPrice
                self.tracker.open_position(pair, tf, side, lot_size,
                                           fill_price, trade.order.orderId)
                logger.info(f"FILLED: {order_action} {units} {ibkr_pair} @ {fill_price}")
            else:
                logger.warning(f"Order not filled: status={trade.orderStatus.status}")

    async def _close_position(self, pair: str, tf: str):
        """Close an existing position."""
        pos = self.tracker.get_position(pair, tf)
        if not pos:
            logger.debug(f"No position to close for {pair}_{tf}")
            return

        contract = self._get_qualified_contract(pair)
        if not contract:
            logger.error(f"No qualified contract for {pair}")
            return
        ibkr_pair = self.pairs_map.get(pair, pair)

        units = round(pos['qty'] * self.ibkr_cfg.get('lot_to_units', 100000))
        close_action = 'SELL' if pos['side'] == 1 else 'BUY'
        order = ib.MarketOrder(close_action, units)
        order.tif = 'IOC'

        logger.info(f"CLOSING: {close_action} {units} {ibkr_pair}")

        trade = self.ib.placeOrder(contract, order)
        for _ in range(100):
            await asyncio.sleep(0.1)
            if trade.orderStatus.status in ('Filled', 'Cancelled', 'Inactive'):
                break

        if trade.orderStatus.status == 'Filled':
            fill_price = trade.orderStatus.avgFillPrice
            self.tracker.close_position(pair, tf, fill_price, trade.order.orderId)
        else:
            logger.warning(f"Close order not filled: status={trade.orderStatus.status}")

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
                    logger.warning(f"STOP_LOSS_HIT: {key} at {pnl_pips:.1f} pips (limit: -{sl_pips})")
                    await self._close_position(pair, tf)
                    self.logger_csv.log({'timestamp': now.isoformat(), 'pair': pair, 'tf': tf,
                                         'action': 'CLOSE', 'reason_codes': 'STOP_LOSS_HIT',
                                         'signal': 0, 'confidence': 0})
                    continue

                # ── Take Profit ──
                if pnl_pips >= tp_pips:
                    logger.info(f"TAKE_PROFIT_HIT: {key} at {pnl_pips:.1f} pips (limit: {tp_pips})")
                    await self._close_position(pair, tf)
                    self.logger_csv.log({'timestamp': now.isoformat(), 'pair': pair, 'tf': tf,
                                         'action': 'CLOSE', 'reason_codes': 'TAKE_PROFIT_HIT',
                                         'signal': 0, 'confidence': 0})
                    continue

                # ── Trailing Stop ──
                max_p = pos.get('max_profit_pips', 0)
                if max_p >= trail_start and pnl_pips <= (max_p - trail_dist):
                    logger.info(f"TRAILING_STOP_HIT: {key} max={max_p:.1f} now={pnl_pips:.1f} "
                                f"(trail from {trail_start}, dist {trail_dist})")
                    await self._close_position(pair, tf)
                    self.logger_csv.log({'timestamp': now.isoformat(), 'pair': pair, 'tf': tf,
                                         'action': 'CLOSE', 'reason_codes': 'TRAILING_STOP_HIT',
                                         'signal': 0, 'confidence': 0})
                    continue

                # ── Time-based stale exit ──
                if age_hours >= stale_hours and pnl_pips < stale_min_pips:
                    logger.info(f"TIME_EXIT_STALE: {key} age={age_hours:.1f}h pnl={pnl_pips:.1f} pips")
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
                for pair in self.pairs_map:
                    ticker = self._live_tickers.get(pair)
                    if not ticker:
                        continue
                    if ticker.ask > 0 and ticker.bid > 0:
                        spread = (ticker.ask - ticker.bid) / self._get_pip_size(pair)
                        baseline = baseline_spreads.get(pair, 1.0)
                        if spread > baseline * spread_mult:
                            if not self._defensive_active:
                                logger.warning(f"DEFENSIVE_MODE_SPREAD_SPIKE: {pair} "
                                               f"spread={spread:.1f} pips (baseline={baseline})")
                                if self._handler:
                                    self._handler.set_defensive_mode(True)
                                self._defensive_active = True
                        elif self._defensive_active:
                            # Check if all spreads normalized
                            pass  # Will auto-deactivate when spreads normalize
            except Exception as e:
                logger.debug(f"Spread check error: {e}")

    async def _dashboard_loop(self):
        """Print console dashboard every N seconds with live PnL."""
        interval = self.ibkr_cfg.get('dashboard_interval_sec', 30)
        while True:
            await asyncio.sleep(interval)
            if not self.ib.isConnected():
                print("\n[DASHBOARD] DISCONNECTED from IBKR")
                continue

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
                    equity_str = f"${float(item.value):,.2f}"
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

        try:
            await asyncio.gather(
                self._bar_builder.run(),
                self._dashboard_loop(),
                self._position_management_loop(),
            )
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        except Exception as e:
            logger.error(f"Executor error: {e}", exc_info=True)
        finally:
            self._bar_builder.stop()
            self.logger_csv.close()
            if self.ib.isConnected():
                self.ib.disconnect()
            logger.info("IBKR Executor stopped")


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
    parser.add_argument('--host', default=None, help='Override IBKR host')
    parser.add_argument('--port', type=int, default=None, help='Override IBKR port')
    parser.add_argument('--client-id', type=int, default=None, help='Override client ID')
    args = parser.parse_args()

    is_paper = not args.live

    if args.live:
        logger.warning("=" * 60)
        logger.warning("  *** LIVE TRADING MODE — REAL MONEY ***")
        logger.warning("=" * 60)

    executor = IBKRExecutor(paper=is_paper)

    if args.host:
        executor.host = args.host
    if args.port:
        executor.port = args.port
    if args.client_id:
        executor.client_id = args.client_id

    asyncio.run(executor.start())


if __name__ == '__main__':
    main()
