"""
CHAOS V1.0 — IBKR Bar Builder

Builds H1 and M30 bars from IBKR historical data with strict bar-close rules.
Only emits "bar closed" events when a bar is fully complete.

Uses ib_async (maintained fork of ib_insync).
Respects IBKR pacing limits: one historical request at a time with delays.
"""
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Callable, Dict, List, Optional

import ib_async as ib

logger = logging.getLogger('ibkr_bar_builder')

# IBKR bar size strings
TF_TO_IBKR = {
    'H1': '1 hour',
    'M30': '30 mins',
}

# Bar duration in minutes
TF_MINUTES = {
    'H1': 60,
    'M30': 30,
}


def _bar_close_time(tf: str, now: datetime) -> datetime:
    """Return the close time of the current bar for this TF."""
    mins = TF_MINUTES[tf]
    minute_slot = (now.hour * 60 + now.minute) // mins * mins
    h, m = divmod(minute_slot, 60)
    close = now.replace(hour=h, minute=m, second=0, microsecond=0) + timedelta(minutes=mins)
    return close


def _prev_bar_close(tf: str, now: datetime) -> datetime:
    """Return the close time of the most recently completed bar."""
    return _bar_close_time(tf, now) - timedelta(minutes=TF_MINUTES[tf])


class IBKRBarBuilder:
    """
    Manages bar subscriptions and historical data for all pairs + TFs.

    On each bar close, calls the on_bar_close callback with:
        (pair: str, tf: str, bars: list[dict])
    where bars is a list of the last N completed bars.
    """

    def __init__(self, ib_client: ib.IB, pairs_map: Dict[str, str],
                 timeframes: List[str], bars_count: int = 500,
                 pacing_delay: float = 2.0,
                 on_bar_close: Optional[Callable] = None):
        """
        Args:
            ib_client: connected ib_async.IB instance
            pairs_map: {"EURUSD": "EUR.USD", ...}
            timeframes: ["H1", "M30"]
            bars_count: how many historical bars to fetch per request
            pacing_delay: seconds between historical data requests (IBKR pacing)
            on_bar_close: async callback(pair, tf, bars_list)
        """
        self.ib = ib_client
        self.pairs_map = pairs_map
        self.timeframes = timeframes
        self.bars_count = bars_count
        self.pacing_delay = pacing_delay
        self.on_bar_close = on_bar_close

        # Contracts
        self._contracts: Dict[str, ib.Forex] = {}
        for chaos_pair, ibkr_pair in pairs_map.items():
            base, quote = ibkr_pair.split('.')
            self._contracts[chaos_pair] = ib.Forex(base + quote, exchange='IDEALPRO')

        # Track last processed bar close per pair+TF
        self._last_bar_close: Dict[str, datetime] = {}

        # Pacing lock: one historical request at a time
        self._pacing_lock = asyncio.Lock()

        self._running = False

    async def qualify_contracts(self):
        """Qualify all FX contracts with IBKR.
        Note: IBKRExecutor now qualifies one-by-one with pacing.
        This method is kept for standalone usage."""
        contracts = list(self._contracts.values())
        if not contracts:
            return []
        qualified = await self.ib.qualifyContractsAsync(*contracts)
        for c in qualified:
            logger.info(f"  Qualified: {c.localSymbol} (conId={c.conId})")
        return qualified

    async def get_historical_bars(self, pair: str, tf: str,
                                  count: int = None) -> List[dict]:
        """
        Fetch last N completed bars from IBKR historical data.
        Respects pacing limits via lock + delay.

        Returns list of bar dicts with: time, open, high, low, close, volume
        """
        count = count or self.bars_count
        if not self.ib.isConnected():
            return []
        contract = self._contracts.get(pair)
        if not contract:
            logger.warning(f"No contract for {pair}")
            return []

        bar_size = TF_TO_IBKR.get(tf)
        if not bar_size:
            logger.warning(f"Unknown TF: {tf}")
            return []

        # Duration string: enough to cover count bars
        mins = TF_MINUTES[tf]
        total_hours = (count * mins) / 60
        if total_hours <= 24:
            duration = '1 D'
        elif total_hours <= 168:
            duration = '1 W'
        else:
            days = int(total_hours / 24) + 1
            duration = f'{min(days, 365)} D'

        async with self._pacing_lock:
            try:
                bars = await self.ib.reqHistoricalDataAsync(
                    contract,
                    endDateTime='',  # Now
                    durationStr=duration,
                    barSizeSetting=bar_size,
                    whatToShow='MIDPOINT',
                    useRTH=False,
                    formatDate=2,  # UTC datetime
                )
                await asyncio.sleep(self.pacing_delay)
            except Exception as e:
                logger.error(f"Historical data error for {pair}_{tf}: {e}")
                return []

        if not bars:
            return []

        # Convert to dicts, exclude the last bar (may be incomplete)
        result = []
        for b in bars[:-1]:  # Drop last bar (current, incomplete)
            result.append({
                'time': b.date.strftime('%Y-%m-%dT%H:%M:%S') if hasattr(b.date, 'strftime')
                        else str(b.date),
                'open': float(b.open),
                'high': float(b.high),
                'low': float(b.low),
                'close': float(b.close),
                'volume': float(b.volume) if b.volume > 0 else 1.0,
            })

        return result[-count:]  # Last N completed bars

    async def run(self):
        """
        Main loop: check for new bar closes every 5 seconds.
        When a bar closes, fetch historical data and call on_bar_close.
        """
        self._running = True
        logger.info(f"Bar builder started: {len(self.pairs_map)} pairs x "
                     f"{len(self.timeframes)} TFs, checking every 5s")

        while self._running:
            now = datetime.now(timezone.utc)

            # Skip all work if disconnected — no error spam
            if not self.ib.isConnected():
                await asyncio.sleep(5)
                continue

            for pair in self.pairs_map:
                for tf in self.timeframes:
                    key = f"{pair}_{tf}"
                    expected_close = _prev_bar_close(tf, now)

                    # Have we already processed this bar close?
                    last = self._last_bar_close.get(key)
                    if last and last >= expected_close:
                        continue

                    # Is the bar actually closed? Need to be past the close time.
                    if now < expected_close + timedelta(seconds=5):
                        continue

                    # New bar closed — fetch historical data
                    # Skip if disconnected (reconnect loop will handle it)
                    if not self.ib.isConnected():
                        continue

                    logger.info(f"Bar closed: {key} at {expected_close}")
                    self._last_bar_close[key] = expected_close

                    try:
                        bars = await self.get_historical_bars(pair, tf)
                        if bars and self.on_bar_close:
                            await self.on_bar_close(pair, tf, bars)
                    except Exception as e:
                        logger.error(f"Bar close handler error for {key}: {e}",
                                     exc_info=True)

            await asyncio.sleep(5)

    def stop(self):
        self._running = False
