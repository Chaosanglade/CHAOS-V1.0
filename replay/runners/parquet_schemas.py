"""
Exact PyArrow schemas for all replay output files.
These are the canonical column definitions — no deviation permitted.
All Parquet writes must use these schemas explicitly via pq.write_table().

Dictionary encoding is used for low-cardinality columns (pair, tf, scenario, action, etc.)
to achieve byte-stable, memory-efficient output.
"""
import pyarrow as pa

# ============================================================
# GLOBAL PARQUET WRITE SETTINGS
# ============================================================
PARQUET_WRITE_KWARGS = {
    'compression': 'zstd',
    'use_dictionary': True,
    'write_statistics': True,
    'coerce_timestamps': 'us',
    'version': '2.6',
    'data_page_size': 1048576,
    'row_group_size': 50_000,
}

# ============================================================
# trades.parquet — Row-per-trade-lifecycle event
# Two-row method: one OPEN row, one CLOSE row per trade.
# trade_id links them.
# ============================================================
TRADES_SCHEMA = pa.schema([
    # Identifiers
    pa.field("run_id", pa.string()),
    pa.field("scenario", pa.dictionary(pa.int8(), pa.string())),
    pa.field("trade_id", pa.string()),
    pa.field("pair", pa.dictionary(pa.int8(), pa.string())),
    pa.field("tf", pa.dictionary(pa.int8(), pa.string())),

    # Timing
    pa.field("decision_ts", pa.timestamp("us", tz="UTC")),
    pa.field("fill_ts", pa.timestamp("us", tz="UTC")),
    pa.field("exit_ts", pa.timestamp("us", tz="UTC"), nullable=True),

    # Decision linkage
    pa.field("request_id", pa.string()),
    pa.field("decision_side", pa.int8()),   # -1 = SHORT, 0 = FLAT, +1 = LONG
    pa.field("action", pa.dictionary(pa.int8(), pa.string())),  # OPEN|CLOSE|REVERSE|SKIP
    pa.field("fill_status", pa.dictionary(pa.int8(), pa.string())),  # FILLED|PARTIAL|REJECTED|SKIPPED

    # Sizes
    pa.field("qty_lots", pa.float64()),
    pa.field("qty_lots_filled", pa.float64()),
    pa.field("qty_units", pa.int64(), nullable=True),

    # Prices
    pa.field("price_decision_ref", pa.float64()),
    pa.field("price_fill", pa.float64(), nullable=True),
    pa.field("price_exit", pa.float64(), nullable=True),

    # Costs (per-trade)
    pa.field("spread_cost_pips", pa.float32()),
    pa.field("slippage_cost_pips", pa.float32()),
    pa.field("commission_cost_usd", pa.float64()),
    pa.field("total_cost_usd", pa.float64()),

    # PnL (only on CLOSE rows)
    pa.field("pnl_gross_usd", pa.float64(), nullable=True),
    pa.field("pnl_net_usd", pa.float64(), nullable=True),
    pa.field("pnl_pips", pa.float32(), nullable=True),

    # Context
    pa.field("regime_state", pa.int16()),
    pa.field("agreement_score", pa.float32(), nullable=True),
    pa.field("agreement_threshold", pa.float32(), nullable=True),
    pa.field("risk_veto", pa.bool_()),
    pa.field("risk_reason", pa.string(), nullable=True),
    pa.field("reason_codes", pa.string()),  # pipe-joined, e.g. "ALT_COT_EXTREME_CONFIRM|MTF_PASS"

    # Optional diagnostics
    pa.field("enabled_models_count", pa.int16(), nullable=True),
    pa.field("latency_ms", pa.float32(), nullable=True),
])

# ============================================================
# positions.parquet — Row-per-position-change event
# Written only when: position opens, size changes, position closes, reversal occurs.
# ============================================================
POSITIONS_SCHEMA = pa.schema([
    # Identifiers
    pa.field("run_id", pa.string()),
    pa.field("scenario", pa.dictionary(pa.int8(), pa.string())),
    pa.field("pair", pa.dictionary(pa.int8(), pa.string())),
    pa.field("tf", pa.dictionary(pa.int8(), pa.string())),
    pa.field("position_id", pa.string()),  # Format: {pair}_{tf} (single position per pair_tf)

    # Timing
    pa.field("event_ts", pa.timestamp("us", tz="UTC")),
    pa.field("request_id", pa.string(), nullable=True),

    # State
    pa.field("side", pa.int8()),  # -1, 0, +1
    pa.field("qty_lots", pa.float64()),
    pa.field("avg_entry_price", pa.float64(), nullable=True),
    pa.field("mark_price", pa.float64(), nullable=True),
    pa.field("unrealized_pnl_usd", pa.float64(), nullable=True),
    pa.field("realized_pnl_usd_cum", pa.float64()),

    # Exposure
    pa.field("gross_exposure_usd", pa.float64()),
    pa.field("net_exposure_usd", pa.float64()),
    pa.field("group_exposure_usd_leg", pa.float64(), nullable=True),

    # Risk controls
    pa.field("risk_state", pa.dictionary(pa.int8(), pa.string())),  # NORMAL|COOLDOWN|CIRCUIT_BREAKER
    pa.field("cooldown_until_ts", pa.timestamp("us", tz="UTC"), nullable=True),
    pa.field("drawdown_pct_current", pa.float32(), nullable=True),

    # Context
    pa.field("regime_state", pa.int16()),
    pa.field("reason_codes", pa.string()),
])

# ============================================================
# decision_ledger.parquet — Row-per-inference event
# One row per bar per pair/timeframe. ALWAYS written, even on errors.
# ============================================================
DECISION_LEDGER_SCHEMA = pa.schema([
    # Primary key: (event_ts, pair, tf, request_id)
    pa.field("event_ts", pa.timestamp("us", tz="UTC")),
    pa.field("pair", pa.dictionary(pa.int8(), pa.string())),
    pa.field("tf", pa.dictionary(pa.int8(), pa.string())),
    pa.field("request_id", pa.string()),
    pa.field("run_id", pa.string()),
    pa.field("scenario", pa.dictionary(pa.int8(), pa.string())),

    # Regime
    pa.field("regime_state", pa.int16()),
    pa.field("regime_confidence", pa.float32()),

    # Models
    pa.field("enabled_models_count", pa.int16()),
    pa.field("enabled_models_keys", pa.string()),  # pipe-joined list
    pa.field("models_voted", pa.int16()),
    pa.field("models_agreed", pa.int16()),

    # Ensemble
    pa.field("brain_probs_trimmed_mean", pa.list_(pa.float32(), 3)),  # [P(SHORT), P(FLAT), P(LONG)]
    pa.field("agreement_score", pa.float32()),
    pa.field("agreement_threshold_base", pa.float32()),
    pa.field("agreement_threshold_modified", pa.float32()),

    # Alt-data
    pa.field("alt_data_available", pa.bool_()),
    pa.field("cot_pressure", pa.float32(), nullable=True),
    pa.field("cot_extreme", pa.bool_(), nullable=True),
    pa.field("cme_spike", pa.bool_(), nullable=True),
    pa.field("cme_confirm", pa.bool_(), nullable=True),
    pa.field("alt_agreement_adjustment", pa.float32(), nullable=True),

    # MTF confirmation
    pa.field("mtf_status", pa.dictionary(pa.int8(), pa.string()), nullable=True),  # PASS|FAIL

    # Decision
    pa.field("decision_side", pa.int8()),  # -1, 0, +1
    pa.field("decision_confidence", pa.float32()),
    pa.field("raw_signal_side", pa.int8()),  # Before risk/agreement override
    pa.field("signal_overridden", pa.bool_()),

    # Risk
    pa.field("risk_veto", pa.bool_()),
    pa.field("risk_reason", pa.string(), nullable=True),

    # Execution
    pa.field("action_taken", pa.dictionary(pa.int8(), pa.string())),  # OPEN|CLOSE|HOLD|SKIP
    pa.field("latency_ms", pa.float32(), nullable=True),

    # Reason codes (pipe-joined)
    pa.field("reason_codes", pa.string()),

    # TF Role confirmation fields
    pa.field("confirmers_used", pa.string(), nullable=True),
    pa.field("confirm_signal", pa.int8(), nullable=True),
    pa.field("confirm_strength", pa.float32(), nullable=True),
    pa.field("threshold_adjustment", pa.float32(), nullable=True),
])


def write_parquet_strict(table_or_df, schema, path):
    """
    Write Parquet with explicit schema enforcement.

    Args:
        table_or_df: pa.Table or pd.DataFrame
        schema: pa.Schema (one of TRADES_SCHEMA, POSITIONS_SCHEMA, DECISION_LEDGER_SCHEMA)
        path: output file path
    """
    import pyarrow.parquet as pq

    if not isinstance(table_or_df, pa.Table):
        import pandas as pd
        if isinstance(table_or_df, pd.DataFrame):
            # Sort by timestamp column before writing
            ts_cols = [c for c in table_or_df.columns if 'ts' in c.lower() or 'time' in c.lower()]
            if ts_cols:
                table_or_df = table_or_df.sort_values(ts_cols[0])
            table = pa.Table.from_pandas(table_or_df, schema=schema, preserve_index=False)
        else:
            raise TypeError(f"Expected pa.Table or pd.DataFrame, got {type(table_or_df)}")
    else:
        table = table_or_df

    pq.write_table(table, str(path), **PARQUET_WRITE_KWARGS)
