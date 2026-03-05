"""
PyArrow-based audit writer for the decision ledger.

Writes decision ledger rows to Parquet with explicit schema enforcement.
Supports buffered writes for memory efficiency during long replays.
"""
import logging
import pandas as pd
import pyarrow as pa
from pathlib import Path
from typing import Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from replay.runners.parquet_schemas import (
    DECISION_LEDGER_SCHEMA, write_parquet_strict
)

logger = logging.getLogger('audit_writer')


class AuditWriter:
    """
    Buffered audit writer for decision ledger Parquet.

    Usage:
        writer = AuditWriter(output_path)
        for bar in replay:
            writer.append(ledger_row_dict)
        writer.flush()
    """

    def __init__(self, output_path: str, buffer_size: int = 10000):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.buffer_size = buffer_size
        self._buffer: List[Dict] = []
        self._flushed_count = 0

    def append(self, row: Dict):
        """Add a ledger row to the buffer."""
        self._buffer.append(row)
        if len(self._buffer) >= self.buffer_size:
            self.flush()

    def flush(self):
        """Write buffered rows to Parquet."""
        if not self._buffer:
            return

        df = pd.DataFrame(self._buffer)
        if self._flushed_count == 0:
            # First write — create file
            write_parquet_strict(df, DECISION_LEDGER_SCHEMA, str(self.output_path))
        else:
            # Append — read existing, concat, rewrite
            existing = pd.read_parquet(self.output_path)
            combined = pd.concat([existing, df], ignore_index=True)
            write_parquet_strict(combined, DECISION_LEDGER_SCHEMA, str(self.output_path))

        self._flushed_count += len(self._buffer)
        logger.debug(f"Flushed {len(self._buffer)} rows (total: {self._flushed_count})")
        self._buffer.clear()

    @property
    def total_rows(self) -> int:
        return self._flushed_count + len(self._buffer)

    def close(self):
        """Flush remaining buffer and finalize."""
        self.flush()
        logger.info(f"Audit writer closed: {self._flushed_count} total rows -> {self.output_path}")

    def validate_schema(self) -> bool:
        """Validate that the output file conforms to DECISION_LEDGER_SCHEMA."""
        if not self.output_path.exists():
            return False

        import pyarrow.parquet as pq
        table = pq.read_table(self.output_path)
        expected_names = set(f.name for f in DECISION_LEDGER_SCHEMA)
        actual_names = set(table.column_names)

        if expected_names != actual_names:
            missing = expected_names - actual_names
            extra = actual_names - expected_names
            logger.error(f"Schema mismatch — missing: {missing}, extra: {extra}")
            return False

        return True
