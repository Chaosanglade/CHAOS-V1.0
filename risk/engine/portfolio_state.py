"""
Portfolio state tracker for the risk engine.

Tracks all open positions, exposure calculations, PnL, and risk state.
Pure data class — no trade logic, no file I/O.
"""
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger('portfolio_state')


@dataclass
class OpenPosition:
    pair: str
    tf: str
    side: int  # -1, 0, +1
    qty_lots: float
    avg_entry_price: float
    entry_ts: datetime
    position_id: str = ""
    trade_id: str = ""
    unrealized_pnl_usd: float = 0.0

    def __post_init__(self):
        if not self.position_id:
            self.position_id = f"{self.pair}_{self.tf}"


@dataclass
class ClosedTrade:
    pair: str
    tf: str
    side: int
    pnl_pips: float
    pnl_net_usd: float
    close_ts: datetime
    trade_id: str = ""


class PortfolioState:
    """
    Tracks all open positions and risk state.

    Thread-safe design (no shared mutable state across threads).
    """

    def __init__(self, instrument_specs_path, correlation_groups_path):
        import time as _time
        for fpath, attr in [(instrument_specs_path, 'instrument_specs'),
                            (correlation_groups_path, 'correlation_groups')]:
            for attempt in range(3):
                try:
                    with open(fpath) as f:
                        setattr(self, attr, json.load(f))
                    break
                except (FileNotFoundError, OSError, PermissionError) as e:
                    if attempt < 2:
                        logger.warning(f"Retry {attempt+1}/3 reading {fpath}: {e}")
                        _time.sleep(2.0 * (attempt + 1))
                    else:
                        raise

        self.positions: Dict[str, OpenPosition] = {}  # key = position_id
        self.closed_trades: List[ClosedTrade] = []
        self.risk_state: str = "NORMAL"  # NORMAL | CIRCUIT_BREAKER (global only)
        self.cooldown_until_ts: Dict[str, Optional[datetime]] = {}  # pair_tf -> until
        self.realized_pnl_usd_cum: float = 0.0
        self._consecutive_losses: Dict[str, int] = {}  # pair_tf -> count
        self._last_entry_ts: Dict[str, datetime] = {}  # pair_tf -> last entry timestamp
        self._trade_seq: Dict[str, int] = {}  # {pair}_{tf}_{scenario} -> sequence counter

    def allocate_trade_id(self, pair: str, tf: str, scenario: str, entry_ts: datetime) -> str:
        """
        Lock 8: Deterministic trade ID allocation.
        Format: {pair}_{tf}_{scenario}_{entry_ts_iso}_{seq:06d}
        Sequence increments per {pair}_{tf}_{scenario} key.
        """
        seq_key = f"{pair}_{tf}_{scenario}"
        self._trade_seq[seq_key] = self._trade_seq.get(seq_key, 0) + 1
        seq = self._trade_seq[seq_key]
        ts_iso = entry_ts.strftime('%Y%m%dT%H%M%S') if hasattr(entry_ts, 'strftime') else str(entry_ts)
        return f"{pair}_{tf}_{scenario}_{ts_iso}_{seq:06d}"

    def open_position(self, pair, tf, side, qty_lots, entry_price, entry_ts, trade_id=""):
        pos_id = f"{pair}_{tf}"
        if pos_id in self.positions:
            raise ValueError(f"Position already open: {pos_id}")

        self.positions[pos_id] = OpenPosition(
            pair=pair, tf=tf, side=side, qty_lots=qty_lots,
            avg_entry_price=entry_price, entry_ts=entry_ts,
            position_id=pos_id, trade_id=trade_id
        )
        self._last_entry_ts[f"{pair}_{tf}"] = entry_ts
        return pos_id

    def close_position(self, pair, tf, exit_price, exit_ts, trade_id=""):
        pos_id = f"{pair}_{tf}"
        if pos_id not in self.positions:
            return None

        pos = self.positions.pop(pos_id)

        # Compute PnL
        pip_size = self.instrument_specs.get('pip_size', {}).get(pair, 0.0001)
        pip_value = self.instrument_specs.get('pip_value_usd_per_standard_lot', {}).get(pair, 10.0)

        if pos.side == 1:  # LONG
            pnl_pips = (exit_price - pos.avg_entry_price) / pip_size
        elif pos.side == -1:  # SHORT
            pnl_pips = (pos.avg_entry_price - exit_price) / pip_size
        else:
            pnl_pips = 0.0

        pnl_usd = pnl_pips * pip_value * pos.qty_lots

        self.realized_pnl_usd_cum += pnl_usd

        # Track consecutive losses per (pair, tf)
        key = f"{pair}_{tf}"
        if pnl_usd < 0:
            self._consecutive_losses[key] = self._consecutive_losses.get(key, 0) + 1
        else:
            self._consecutive_losses[key] = 0

        trade = ClosedTrade(
            pair=pair, tf=tf, side=pos.side, pnl_pips=pnl_pips,
            pnl_net_usd=pnl_usd, close_ts=exit_ts, trade_id=trade_id
        )
        self.closed_trades.append(trade)

        return trade

    def get_gross_exposure_usd(self):
        total = 0.0
        for pos in self.positions.values():
            total += pos.qty_lots * 100000
        return total

    def get_net_exposure_usd(self):
        total = 0.0
        for pos in self.positions.values():
            total += pos.side * pos.qty_lots * 100000
        return total

    def get_group_exposure(self, group_name):
        group_pairs = self.correlation_groups.get(group_name, [])
        total = 0.0
        for pos in self.positions.values():
            if pos.pair in group_pairs:
                total += pos.qty_lots * 100000
        return total

    def get_position_count(self):
        return len(self.positions)

    def get_positions_for_pair(self, pair):
        return [p for p in self.positions.values() if p.pair == pair]

    def get_positions_for_tf(self, tf):
        return [p for p in self.positions.values() if p.tf == tf]

    def get_consecutive_losses(self, pair_tf=None):
        if pair_tf:
            return self._consecutive_losses.get(pair_tf, 0)
        return max(self._consecutive_losses.values()) if self._consecutive_losses else 0

    def get_last_entry_ts(self, pair):
        return self._last_entry_ts.get(pair)

    def get_today_pnl(self, current_ts):
        today_trades = [t for t in self.closed_trades if t.close_ts.date() == current_ts.date()]
        return sum(t.pnl_net_usd for t in today_trades)

    def set_risk_state(self, state, cooldown_until=None, pair_tf=None):
        if state == "CIRCUIT_BREAKER":
            self.risk_state = state
        elif state == "COOLDOWN" and pair_tf:
            self.cooldown_until_ts[pair_tf] = cooldown_until
        elif state == "NORMAL":
            if pair_tf:
                self.cooldown_until_ts.pop(pair_tf, None)
            else:
                self.risk_state = "NORMAL"
                self.cooldown_until_ts = {}

    def has_position(self, pair, tf):
        return f"{pair}_{tf}" in self.positions

    def get_position(self, pair, tf):
        return self.positions.get(f"{pair}_{tf}")

    def to_snapshot_dict(self, run_id, scenario, event_ts, regime_state, reason_codes=""):
        """Generate positions.parquet rows for each open position."""
        snapshots = []
        for pos in self.positions.values():
            snapshots.append({
                'run_id': run_id,
                'scenario': scenario,
                'pair': pos.pair,
                'tf': pos.tf,
                'position_id': pos.position_id,
                'event_ts': event_ts,
                'request_id': None,
                'side': pos.side,
                'qty_lots': pos.qty_lots,
                'avg_entry_price': pos.avg_entry_price,
                'mark_price': None,
                'unrealized_pnl_usd': pos.unrealized_pnl_usd,
                'realized_pnl_usd_cum': self.realized_pnl_usd_cum,
                'gross_exposure_usd': self.get_gross_exposure_usd(),
                'net_exposure_usd': self.get_net_exposure_usd(),
                'group_exposure_usd_leg': None,
                'risk_state': self.risk_state,
                'cooldown_until_ts': self.cooldown_until_ts.get(f"{pos.pair}_{pos.tf}"),
                'drawdown_pct_current': None,
                'regime_state': regime_state,
                'reason_codes': reason_codes,
            })
        return snapshots
