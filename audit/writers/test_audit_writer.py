"""
Test: Audit writer produces Parquet conforming to DECISION_LEDGER_SCHEMA.
"""
import os
import sys
import tempfile
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime

os.environ["PYTHONHASHSEED"] = "42"

PROJECT_ROOT = Path('G:/My Drive/chaos_v1.0')
sys.path.insert(0, str(PROJECT_ROOT))

from audit.writers.audit_writer import AuditWriter
from replay.runners.parquet_schemas import DECISION_LEDGER_SCHEMA


def make_sample_row(idx=0):
    """Create a sample ledger row."""
    return {
        'event_ts': pd.Timestamp('2024-06-01 12:00:00', tz='UTC') + pd.Timedelta(minutes=idx * 30),
        'pair': 'EURUSD',
        'tf': 'M30',
        'request_id': f'req_{idx:04d}',
        'run_id': '20240601T120000Z',
        'scenario': 'IBKR_BASE',
        'regime_state': 1,
        'regime_confidence': 0.85,
        'enabled_models_count': 4,
        'enabled_models_keys': 'rf_optuna|et_optuna|lgb_optuna|transformer_optuna',
        'models_voted': 4,
        'models_agreed': 3,
        'brain_probs_trimmed_mean': [0.15, 0.35, 0.50],
        'agreement_score': 0.75,
        'agreement_threshold_base': 0.6,
        'agreement_threshold_modified': 0.6,
        'alt_data_available': False,
        'cot_pressure': None,
        'cot_extreme': None,
        'cme_spike': None,
        'cme_confirm': None,
        'alt_agreement_adjustment': None,
        'mtf_status': None,
        'decision_side': 1,
        'decision_confidence': 0.5,
        'raw_signal_side': 1,
        'signal_overridden': False,
        'risk_veto': False,
        'risk_reason': None,
        'action_taken': 'OPEN',
        'latency_ms': 2.5,
        'reason_codes': '',
    }


def test_audit_writer_produces_valid_parquet():
    """Test that the audit writer produces schema-conformant Parquet."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, 'test_ledger.parquet')
        writer = AuditWriter(output_path, buffer_size=5)

        # Write 10 rows
        for i in range(10):
            writer.append(make_sample_row(i))

        writer.close()

        # Validate
        assert os.path.exists(output_path), "Output file not created"
        assert writer.validate_schema(), "Schema validation failed"

        # Read back and check
        table = pq.read_table(output_path)
        assert table.num_rows == 10, f"Expected 10 rows, got {table.num_rows}"

        expected_cols = set(f.name for f in DECISION_LEDGER_SCHEMA)
        actual_cols = set(table.column_names)
        assert expected_cols == actual_cols, f"Column mismatch: missing={expected_cols - actual_cols}, extra={actual_cols - expected_cols}"

    return True


def run_all_tests():
    tests = [
        ("Audit writer schema compliance", test_audit_writer_produces_valid_parquet),
    ]

    passed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")

    print(f"\nAudit writer tests: {passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == '__main__':
    run_all_tests()
