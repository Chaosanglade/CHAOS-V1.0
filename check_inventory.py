"""Quick file inventory check."""
import os
from pathlib import Path

ROOT = Path('.')
FILES = [
    # Core Training
    'chaos_gpu_training.py', 'chaos_rf_et_training.py', 'chaos_v2_evaluation.py',
    'chaos_v1_0_core_FIXED.py', 'chaos_test_suite.py', 'preflight_check.py', 'run_verification.py',
    # Schema & Contracts
    'schema/feature_schema.json', 'contracts/inference_request.json', 'contracts/inference_response.json',
    # Inference Pipeline
    'inference/onnx_export.py', 'inference/inference_server.py',
    'inference/build_model_registry.py', 'inference/export_available_models.py',
    # Regime & Ensemble
    'regime/regime_policy.json', 'ensemble/ensemble_config.json',
    # Alt-Data
    'alt_data/alt_data_provider.py', 'alt_data/cot_pipeline.py', 'alt_data/cme_pipeline.py',
    # Replay Engine
    'replay/config/universe.json', 'replay/config/replay_config.json',
    'replay/config/execution_cost_scenarios.json', 'replay/config/defensive_mode.json',
    'replay/runners/run_replay.py', 'replay/runners/replay_iterator.py',
    'replay/runners/mt5_gate_export.py', 'replay/runners/parquet_schemas.py',
    'replay/runners/report_generators.py', 'replay/runners/brain_tracker.py',
    'replay/tests/run_scaling_tests.py', 'replay/tests/run_v35_production_verification.py',
    'replay/tests/test_determinism.py', 'replay/tests/test_no_lookahead.py',
    'replay/tests/test_replay_alignment.py',
    # Audit
    'audit/schemas/decision_ledger_schema.json', 'audit/schemas/reason_codes.json',
    'audit/writers/audit_writer.py',
    # Risk Engine
    'risk/config/risk_policy.json', 'risk/config/correlation_groups.json',
    'risk/config/instrument_specs.json', 'risk/engine/exposure_controller.py',
    'risk/engine/portfolio_state.py', 'risk/engine/test_risk_engine.py',
    'risk/engine/volatility_targeter.py', 'risk/engine/edge_decay_monitor.py',
    # Ops
    'ops/build_edge_baselines_by_regime.py', 'ops/retrain_trigger.py',
    'ops/retrain_runner.py', 'ops/promotion_gate.py', 'ops/canary_deployer.py',
    'ops/run_self_heal_verification.py',
    'ops/tests/test_baseline_builder.py', 'ops/tests/test_retrain_trigger.py',
    'ops/tests/test_promotion_gate.py', 'ops/tests/test_canary_deployer.py',
    # Docs
    'docs/RL_META_CONTROLLER_ARCHITECTURE.md',
]

print("FILE INVENTORY CHECK")
found = 0
missing = 0
for f in FILES:
    p = ROOT / f
    if p.exists():
        size = p.stat().st_size
        if size > 1024:
            sz = f"{size // 1024} KB"
        else:
            sz = f"{size} B"
        print(f"[OK]      {f} ({sz})")
        found += 1
    else:
        print(f"[MISSING] {f}")
        missing += 1

print(f"\nTotal: {found}/{found+missing} files found")
if missing:
    print(f"WARNING: {missing} files MISSING")
