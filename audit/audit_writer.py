"""
Decision Audit Ledger Writer.

Writes immutable decision records to parquet files.
Designed for both replay engine and live inference server.

Files are written as daily partitions:
    audit/ledger/YYYY-MM-DD.parquet

For replay, all events go to a single file:
    audit/ledger/replay_{pair}_{tf}_{scenario}.parquet
"""
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger('audit_writer')


class AuditWriter:
    """
    Writes decision audit records to parquet files.

    Buffers records in memory and flushes periodically or on demand.
    """

    REASON_CODES = {
        'none': 0,
        'regime_gated': 1,
        'low_agreement': 2,
        'risk_drawdown_breaker': 3,
        'risk_exposure_limit': 4,
        'risk_cooldown': 5,
        'risk_correlated_positions': 6,
        'missed_fill': 7,
    }

    def __init__(self, output_dir='G:/My Drive/chaos_v1.0/audit/ledger',
                 buffer_size=500, mode='live'):
        """
        Args:
            output_dir: directory for ledger parquet files
            buffer_size: flush to disk every N records
            mode: 'live' (daily partitioned) or 'replay' (single file)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.buffer_size = buffer_size
        self.mode = mode
        self.buffer = []

    def write_row(self, entry: Dict):
        """
        Write a single audit record.

        Args:
            entry: dict conforming to decision_ledger_schema.json
        """
        # Generate entry_id if not present
        if 'entry_id' not in entry:
            ts = entry.get('timestamp', datetime.now().isoformat())
            pair = entry.get('pair', 'UNKNOWN')
            tf = entry.get('timeframe', 'UNKNOWN')
            entry['entry_id'] = f"{pair}_{tf}_{ts}"

        # Encode reason as both string and numeric code
        reason = entry.get('override_reason', 'none')
        entry['override_reason_code'] = self.REASON_CODES.get(reason, -1)

        self.buffer.append(entry)

        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self, filename=None):
        """Flush buffer to parquet file."""
        if not self.buffer:
            return

        df = pd.DataFrame(self.buffer)

        if filename:
            path = self.output_dir / filename
        elif self.mode == 'live':
            # Daily partition
            date_str = datetime.now().strftime('%Y-%m-%d')
            path = self.output_dir / f"{date_str}.parquet"
        else:
            # Replay mode — append to existing or create new
            path = self.output_dir / 'replay_ledger.parquet'

        if path.exists():
            existing = pd.read_parquet(path)
            df = pd.concat([existing, df], ignore_index=True)

        df.to_parquet(path, index=False)
        logger.info(f"Audit ledger flushed: {len(self.buffer)} records -> {path}")
        self.buffer = []

    def close(self):
        """Flush remaining buffer and close."""
        self.flush()

    def get_stats(self):
        """Return summary stats from current buffer."""
        if not self.buffer:
            return {'buffered_records': 0}

        signals = [e.get('signal', 'UNKNOWN') for e in self.buffer]
        overrides = [e.get('signal_overridden', False) for e in self.buffer]

        return {
            'buffered_records': len(self.buffer),
            'signal_distribution': {s: signals.count(s) for s in set(signals)},
            'override_rate': sum(overrides) / len(overrides) if overrides else 0
        }
