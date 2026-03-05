"""
Bar-by-bar event emitter for the replay engine.

Reads feature parquet files, filters to the replay window,
sorts by timestamp, and yields one event per bar close.

Dual-mode contract:
  REPLAY mode (default):
    - Derives features from pair-specific parquet using training exclusion logic
    - Feature count matches what models were trained on (pair-native, e.g. 275 for EURUSD)
  LIVE mode:
    - Enforces 273 universal boundary from feature_schema.json
    - Used for MT5 / production deployment

Common:
  - Reads {pair}_{tf}_features.parquet
  - Filters to [date_start, date_end]
  - Sorts by timestamp
  - Applies 1-bar execution lag to features
  - Yields dicts with: timestamp, close_price, features, feature_count, feature_names, mode
"""
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Generator, Dict, List, Optional

logger = logging.getLogger('replay_iterator')


class ReplayIterator:
    """
    Yields one event per bar for a single pair/timeframe.

    Events are yielded in strict timestamp order.
    Features are lagged by 1 bar (decision at bar N uses features from bar N-1).
    """

    def __init__(self, pair: str, tf: str, features_dir: str,
                 schema_path: str = None, date_start: str = '2024-01-01',
                 date_end: str = '2024-12-31', mode: str = 'REPLAY'):
        self.pair = pair
        self.tf = tf
        self.features_dir = Path(features_dir)
        self.date_start = pd.Timestamp(date_start)
        self.date_end = pd.Timestamp(date_end)
        self.mode = mode.upper()

        if self.mode == 'LIVE':
            # LIVE mode: enforce 273 universal boundary from schema
            if schema_path is None:
                raise ValueError("schema_path is required in LIVE mode")
            with open(schema_path) as f:
                schema = json.load(f)
            self.feature_names = [feat['name'] for feat in schema['features']]
            assert len(self.feature_names) == 273, \
                f"Expected 273 features, got {len(self.feature_names)}"
        else:
            # REPLAY mode: derive features from parquet (pair-native)
            self.feature_names = None  # derived in _load_and_prepare

        # Load and prepare data
        self._df = self._load_and_prepare()

    @staticmethod
    def _get_feature_columns(df: pd.DataFrame) -> List[str]:
        """Replicate training feature selection logic from chaos_rf_et_training.py.

        Excludes targets, returns, metadata, and non-numeric columns.
        """
        exclude_patterns = [
            'target_', 'return', 'Open', 'High', 'Low', 'Close',
            'Volume', 'timestamp', 'date', 'pair', 'symbol', 'tf',
            'bar_time', 'Bid_', 'Ask_', 'Spread_', 'Unnamed',
        ]
        valid_dtypes = {'int64', 'int32', 'float64', 'float32', 'Float64'}

        feature_cols = []
        for col in df.columns:
            if any(p in col for p in exclude_patterns):
                continue
            if str(df[col].dtype) not in valid_dtypes:
                continue
            feature_cols.append(col)
        return feature_cols

    def _load_and_prepare(self) -> pd.DataFrame:
        """Load parquet, apply lag, filter, sort.

        Memory-optimized: reads only needed columns to avoid OOM on large M5 files.
        """
        parquet_path = self.features_dir / f"{self.pair}_{self.tf}_features.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Feature file not found: {parquet_path}")

        # Step 1: Read minimal columns to discover schema + derive feature names
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(str(parquet_path))
        all_columns = pf.schema_arrow.names

        # Identify timestamp and close columns from schema
        ts_col = None
        for c in ['timestamp', 'bar_time', 'date', 'datetime', 'time']:
            if c in all_columns:
                ts_col = c
                break

        close_col = None
        for c in ['Close', 'close', 'CLOSE']:
            if c in all_columns:
                close_col = c
                break
        if close_col is None:
            for c in all_columns:
                if 'close' in c.lower() and 'pct' not in c.lower() and 'change' not in c.lower():
                    close_col = c
                    break

        # Step 2: Derive feature names from Arrow schema (no data read needed)
        if self.feature_names is None:
            import pyarrow as pa
            schema = pf.schema_arrow
            exclude_patterns = [
                'target_', 'return', 'Open', 'High', 'Low', 'Close',
                'Volume', 'timestamp', 'date', 'pair', 'symbol', 'tf',
                'bar_time', 'Bid_', 'Ask_', 'Spread_', 'Unnamed',
            ]
            numeric_arrow_types = {pa.int32(), pa.int64(), pa.float32(), pa.float64()}
            self.feature_names = []
            for i in range(len(schema)):
                field = schema.field(i)
                col = field.name
                if any(p in col for p in exclude_patterns):
                    continue
                if field.type in numeric_arrow_types:
                    self.feature_names.append(col)
            logger.info(f"{self.pair}_{self.tf}: REPLAY mode — {len(self.feature_names)} "
                        f"pair-native features derived")

        # Step 3: Check if ts_col is the pandas index (stored as index, not a column)
        pandas_meta = pf.schema_arrow.pandas_metadata
        index_columns = pandas_meta.get('index_columns', []) if pandas_meta else []
        ts_is_index = ts_col in index_columns if ts_col else False

        # Build list of columns to read (excluding index — pandas loads it automatically)
        available_features = [f for f in self.feature_names if f in all_columns]
        needed_cols = []
        if ts_col and not ts_is_index:
            needed_cols.append(ts_col)
        if close_col:
            needed_cols.append(close_col)
        needed_cols.extend(available_features)
        # Deduplicate while preserving order
        seen = set()
        unique_cols = []
        for c in needed_cols:
            if c not in seen:
                seen.add(c)
                unique_cols.append(c)

        # Step 4: Read timestamp FIRST (lightweight) to determine sort + date filter
        if ts_is_index:
            ts_table = pf.read(columns=[ts_col])
            ts_all = pd.to_datetime(ts_table.column(ts_col).to_numpy())
            del ts_table
        else:
            ts_table = pf.read(columns=[ts_col])
            ts_all = pd.to_datetime(ts_table.column(ts_col).to_numpy())
            del ts_table

        if close_col is None:
            raise ValueError(f"Cannot find close price column in {all_columns[:20]}")

        # Sort + apply date filter + execution lag BEFORE reading feature columns
        sort_idx = np.argsort(ts_all)
        ts_sorted = ts_all[sort_idx].values
        # Apply execution lag: row N uses features from row N-1, so effective start is row 1
        # Date filter on lagged timestamps (skip row 0 which has NaN features from lag)
        ts_lagged = ts_sorted[1:]  # timestamps for rows after lag warmup
        date_mask = (pd.to_datetime(ts_lagged) >= self.date_start) & \
                    (pd.to_datetime(ts_lagged) <= self.date_end)
        # Indices into the sorted array: for each output row i, features come from sort_idx[i]
        # (the row before it in sorted order, due to lag)
        filtered_indices = np.where(date_mask)[0]  # indices into ts_lagged (= sort_idx[1:])
        n_filtered = len(filtered_indices)

        ts_vals = ts_lagged[date_mask]
        self._total_bars_in_parquet = n_filtered  # for bar cap reporting

        missing_features = [f for f in self.feature_names if f not in all_columns]
        if missing_features:
            logger.warning(f"{self.pair}_{self.tf}: {len(missing_features)} features missing, "
                          f"using {len(available_features)}/{len(self.feature_names)}")

        # Step 5: Read columns ONE AT A TIME from Arrow (avoids loading all 275
        # columns simultaneously which OOMs on M5 with 48 models in memory)

        # Close price
        close_table = pf.read(columns=[close_col])
        close_all = close_table.column(close_col).to_numpy()
        close_vals = close_all[sort_idx][1:][date_mask].astype(np.float64)
        del close_table, close_all

        # Build feature matrix ONLY for filtered rows (much smaller than full parquet)
        n_feats = len(self.feature_names)
        feat_matrix = np.full((n_filtered, n_feats), np.nan, dtype=np.float32)
        # For each filtered output row, features come from the PREVIOUS sorted row (lag)
        lag_source_idx = sort_idx[filtered_indices]

        # Read feature columns one at a time to minimize peak memory
        for i, feat in enumerate(self.feature_names):
            if feat in all_columns:
                col_table = pf.read(columns=[feat])
                all_vals = col_table.column(feat).to_numpy()
                feat_matrix[:, i] = all_vals[lag_source_idx].astype(np.float32)
                del col_table, all_vals

        # Build lightweight output DF (only timestamp + close, features stay in matrix)
        out = pd.DataFrame({'timestamp': ts_vals, 'close_price': close_vals})

        # Pre-computed feature matrix for fast iteration
        self._feature_matrix = feat_matrix

        logger.info(f"{self.pair}_{self.tf}: {len(out)} bars, {len(self.feature_names)} features "
                    f"in {self.mode} mode")
        return out

    def _find_timestamp_col(self, df: pd.DataFrame) -> str:
        """Find the timestamp column."""
        # Check index first
        if df.index.name and ('time' in df.index.name.lower() or 'date' in df.index.name.lower()):
            return '__index__'

        for col in ['timestamp', 'bar_time', 'date', 'datetime', 'time']:
            if col in df.columns:
                return col

        # Check index dtype
        if hasattr(df.index, 'dtype') and 'datetime' in str(df.index.dtype):
            return '__index__'

        raise ValueError(f"Cannot find timestamp column in {list(df.columns[:10])}")

    def _find_close_col(self, df: pd.DataFrame) -> str:
        """Find the close price column."""
        for col in ['Close', 'close', 'CLOSE']:
            if col in df.columns:
                return col

        # Fallback: column containing 'close' but not 'pct'/'change'
        for col in df.columns:
            if 'close' in col.lower() and 'pct' not in col.lower() and 'change' not in col.lower():
                return col

        raise ValueError(f"Cannot find close price column in {list(df.columns[:20])}")

    def __len__(self) -> int:
        return len(self._df)

    def __iter__(self) -> Generator[Dict, None, None]:
        """Yield one event dict per bar."""
        timestamps = self._df['timestamp'].values
        close_prices = self._df['close_price'].values
        for idx in range(len(self._df)):
            yield {
                'timestamp': timestamps[idx],
                'close_price': float(close_prices[idx]),
                'features': self._feature_matrix[idx],
                'feature_count': len(self.feature_names),
                'feature_names': self.feature_names,
                'bar_idx': idx,
                'mode': self.mode,
                'pair': self.pair,
                'tf': self.tf,
            }

    def has_valid_features(self, features: np.ndarray) -> bool:
        """Check if features contain NaN or Inf."""
        return not (np.any(np.isnan(features)) or np.any(np.isinf(features)))
