"""
CHAOS V1.0 — IBKR Direct Execution Module

Pure Python → IBKR execution. No MT5, no ZeroMQ, no EA.
Uses ib_async (maintained fork). Connects to TWS or IB Gateway.

Pipeline per bar close:
  1. Fetch 500 historical bars from IBKR
  2. Compute 273 features via LiveFeatureAdapter
  3. Reorder to training column order
  4. Run thin ensemble (eligible brains only)
  5. Risk engine checks
  6. Place market order on IBKR if signal fires

Usage:
    pip install ib_async
    python inference/ibkr_executor.py --paper
    python inference/ibkr_executor.py --live   (REAL MONEY)
"""
import os
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

os.environ['ORT_LOG_SEVERITY_LEVEL'] = '3'

PROJECT_ROOT = Path(os.environ.get('CHAOS_BASE_DIR', os.getcwd()))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'inference'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger('ibkr_executor')

import ib_async as ib
import numpy as np

from inference.ibkr_bar_builder import IBKRBarBuilder


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
        self._last_day = None

        # Inference handler (loaded in start())
        self._handler = None
        self._bar_builder = None

        # Next order ID
        self._next_order_id = 1

    def _load_inference_handler(self):
        """Load the full production inference pipeline."""
        from replay.runners.run_replay import ModelLoader, RegimeSimulator, EnsembleEngine
        from risk.engine.portfolio_state import PortfolioState
        from risk.engine.exposure_controller import ExposureController
        from inference.live_inference_handler import LiveInferenceHandler, LiveFeatureEngine

        logger.info("Loading production inference pipeline...")

        with open(PROJECT_ROOT / 'replay' / 'config' / 'brain_quarantine.json') as f:
            quarantine_cfg = json.load(f)
        with open(PROJECT_ROOT / 'replay' / 'config' / 'portfolio_allocation.json') as f:
            portfolio_cfg = json.load(f)
        with open(PROJECT_ROOT / 'replay' / 'config' / 'defensive_mode.json') as f:
            defensive_cfg = json.load(f)
        with open(PROJECT_ROOT / 'risk' / 'config' / 'instrument_specs.json') as f:
            instrument_specs = json.load(f)

        pairs = list(self.pairs_map.keys())
        tfs = self.ibkr_cfg.get('timeframes', ['H1', 'M30'])

        logger.info(f"Loading models for {len(pairs)} pairs x {len(tfs)} TFs...")
        model_loader = ModelLoader(str(PROJECT_ROOT / 'models'), pairs=pairs, tfs=tfs)

        # Apply quarantine + verbose counting
        total_loaded = 0
        quarantined_count = 0
        for pair_tf, models in sorted(model_loader.models.items()):
            to_remove = [b for b in models
                         if b in quarantine_cfg.get('global_quarantine', [])]
            for b in to_remove:
                del models[b]
                quarantined_count += 1
            n = len(models)
            total_loaded += n
            brains = ', '.join(sorted(models.keys()))
            logger.info(f"  [{total_loaded:>4}] {pair_tf:<14} {n:>2} brains: {brains}")

        logger.info(f"Models loaded: {total_loaded} active, {quarantined_count} quarantined")

        logger.info("Initializing regime simulator...")
        regime_sim = RegimeSimulator(str(PROJECT_ROOT / 'regime' / 'regime_policy.json'))
        logger.info("Initializing ensemble engine...")
        ensemble_engine = EnsembleEngine(str(PROJECT_ROOT / 'ensemble' / 'ensemble_config.json'))
        logger.info("Initializing risk controller...")
        risk_controller = ExposureController(str(PROJECT_ROOT / 'risk' / 'config' / 'risk_policy.json'))
        logger.info("Initializing portfolio state...")
        portfolio_state = PortfolioState(
            str(PROJECT_ROOT / 'risk' / 'config' / 'instrument_specs.json'),
            str(PROJECT_ROOT / 'risk' / 'config' / 'correlation_groups.json'),
        )

        logger.info("Building inference handler...")
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
        logger.info("Inference pipeline ready")

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
        account = self.ib.accountSummary() if self.ib.isConnected() else []
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
        """Place a market order on IBKR."""
        ibkr_pair = self.pairs_map.get(pair)
        if not ibkr_pair:
            logger.error(f"No IBKR mapping for {pair}")
            return

        base, quote = ibkr_pair.split('.')
        contract = ib.Forex(base + quote, exchange='IDEALPRO')
        await self.ib.qualifyContractsAsync(contract)

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

        ibkr_pair = self.pairs_map.get(pair)
        if not ibkr_pair:
            return

        base, quote = ibkr_pair.split('.')
        contract = ib.Forex(base + quote, exchange='IDEALPRO')
        await self.ib.qualifyContractsAsync(contract)

        units = round(pos['qty'] * self.ibkr_cfg.get('lot_to_units', 100000))
        # Reverse the side to close
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

    async def _dashboard_loop(self):
        """Print console dashboard every N seconds."""
        interval = self.ibkr_cfg.get('dashboard_interval_sec', 30)
        while True:
            await asyncio.sleep(interval)
            if not self.ib.isConnected():
                print("\n[DASHBOARD] DISCONNECTED from IBKR")
                continue

            now = datetime.now(timezone.utc).strftime('%H:%M:%S')
            mode = "PAPER" if self.paper else "** LIVE **"
            halted = " ** HALTED **" if self._halted else ""

            print(f"\n{'='*60}")
            print(f"  CHAOS V1.0 IBKR Executor  [{mode}]  {now} UTC{halted}")
            print(f"{'='*60}")
            print(f"  Connected: {self.ib.isConnected()}")
            print(f"  Open positions: {self.tracker.open_count()}/{self.kill_switch['max_open_positions']}")
            print(f"  Daily trades: {self.tracker.daily_trades}/{self.kill_switch['max_daily_trades']}")
            print(f"  Daily PnL: ${self.tracker.daily_pnl:.2f} "
                  f"(kill @ -${self.kill_switch['max_daily_loss_usd']})")
            wr = (self.tracker.daily_wins / max(self.tracker.daily_trades, 1) * 100)
            print(f"  Win rate: {wr:.0f}%")

            if self.tracker.positions:
                print(f"\n  {'Position':<16} {'Side':>5} {'Qty':>6} {'Entry':>10}")
                print(f"  {'-'*42}")
                for key, pos in self.tracker.positions.items():
                    side_str = 'LONG' if pos['side'] == 1 else 'SHORT'
                    print(f"  {key:<16} {side_str:>5} {pos['qty']:>6.2f} {pos['entry_price']:>10.5f}")
            print()

    async def start(self):
        """Main entry point."""
        mode_str = "PAPER" if self.paper else "LIVE"
        logger.info(f"CHAOS V1.0 IBKR Executor starting in {mode_str} mode")
        logger.info(f"Connecting to {self.host}:{self.port} (clientId={self.client_id})")

        # Load inference pipeline
        self._load_inference_handler()

        # Connect to IBKR with extended timeout
        try:
            logger.info(f"Connecting to IBKR at {self.host}:{self.port}...")
            await self.ib.connectAsync(
                self.host, self.port, clientId=self.client_id, timeout=60)
        except Exception as e:
            logger.error(f"IBKR connection failed: {e}")
            logger.error("Make sure TWS/IB Gateway is running and API connections are enabled")
            return

        logger.info(f"Connected to IBKR (server version {self.ib.client.serverVersion()})")

        # Post-connect settling time — let IBKR finish its handshake
        logger.info("Waiting 5s for IBKR to settle...")
        await asyncio.sleep(5)

        # Request account info first (lightweight, warms up the connection)
        logger.info("Requesting account summary...")
        try:
            self.ib.reqAccountSummary()
            await asyncio.sleep(2)
            acct = self.ib.accountSummary()
            for item in acct:
                if item.tag == 'NetLiquidation' and item.currency == 'USD':
                    logger.info(f"  Account equity: ${float(item.value):,.2f}")
                    break
        except Exception as e:
            logger.warning(f"Account summary failed (non-fatal): {e}")

        # Reconcile positions
        logger.info("Reconciling positions with IBKR...")
        try:
            await self.tracker.reconcile(self.ib, self.pairs_map)
        except Exception as e:
            logger.warning(f"Position reconciliation failed (non-fatal): {e}")
        await asyncio.sleep(2)

        # Build bar builder
        tfs = self.ibkr_cfg.get('timeframes', ['H1', 'M30'])
        self._bar_builder = IBKRBarBuilder(
            ib_client=self.ib,
            pairs_map=self.pairs_map,
            timeframes=tfs,
            bars_count=self.ibkr_cfg.get('bars_to_request', 500),
            pacing_delay=self.ibkr_cfg.get('pacing_delay_sec', 2),
            on_bar_close=self._on_bar_close,
        )

        # Qualify contracts one at a time with pacing
        logger.info("Qualifying FX contracts (one at a time)...")
        qualified_count = 0
        for chaos_pair, ibkr_pair in self.pairs_map.items():
            try:
                base, quote = ibkr_pair.split('.')
                contract = ib.Forex(base + quote, exchange='IDEALPRO')
                result = await self.ib.qualifyContractsAsync(contract)
                if result:
                    self._bar_builder._contracts[chaos_pair] = result[0]
                    qualified_count += 1
                    logger.info(f"  Subscribing {chaos_pair} ({ibkr_pair})... OK "
                                f"(conId={result[0].conId})")
                else:
                    logger.warning(f"  Subscribing {chaos_pair} ({ibkr_pair})... FAILED")
            except Exception as e:
                logger.warning(f"  Subscribing {chaos_pair} ({ibkr_pair})... ERROR: {e}")
            await asyncio.sleep(2)  # IBKR pacing

        logger.info(f"Qualified {qualified_count}/{len(self.pairs_map)} contracts")

        if qualified_count == 0:
            logger.error("No contracts qualified — cannot proceed")
            self.ib.disconnect()
            return

        # Run concurrently: bar builder + dashboard
        logger.info("Starting bar builder + dashboard loops...")
        logger.info(f"Monitoring {qualified_count} pairs on {tfs}")
        logger.info("Waiting for first bar close...")

        try:
            await asyncio.gather(
                self._bar_builder.run(),
                self._dashboard_loop(),
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
