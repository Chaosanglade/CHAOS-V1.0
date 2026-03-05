"""
Self-Healing Infrastructure — Full Verification Suite

Runs all tests and prints summary table + schemas + sample objects.
"""
import json
import sys
import os
import subprocess
import hashlib
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def run_test_file(test_path: Path) -> tuple:
    """Run a pytest file and return (passed, failed, output)."""
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', str(test_path), '-v', '--tb=short', '-q'],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=120
    )
    output = result.stdout + result.stderr

    # Parse results
    passed = output.count(' PASSED')
    failed = output.count(' FAILED')
    errors = output.count(' ERROR')

    return passed, failed + errors, output


def run_baselines_build() -> tuple:
    """Build baselines and verify output."""
    from ops.build_edge_baselines_by_regime import build_baselines
    baselines = build_baselines(PROJECT_ROOT)

    output_path = PROJECT_ROOT / 'ops' / 'edge_baselines_by_regime.json'
    with open(output_path, 'w') as f:
        json.dump(baselines, f, indent=2, default=str)

    blocks = baselines.get('blocks', {})
    scenarios_seen = set()
    for b in blocks.values():
        for s in b.get('scenarios', {}):
            scenarios_seen.add(s)

    checks = len(blocks) > 0 and 'IBKR_BASE' in scenarios_seen and 'STRESS_PLUS_75' in scenarios_seen
    return checks, len(blocks), scenarios_seen


def run_trigger_check() -> tuple:
    """Run triggers and verify output."""
    from ops.retrain_trigger import run_triggers

    run_dir = PROJECT_ROOT / 'replay' / 'outputs' / 'runs' / '20260304T162601Z'
    trades_path = run_dir / 'trades.parquet'
    ledger_path = run_dir / 'decision_ledger.parquet'
    baselines_path = PROJECT_ROOT / 'ops' / 'edge_baselines_by_regime.json'
    output_path = PROJECT_ROOT / 'ops' / 'retrain_jobs.jsonl'

    results = run_triggers(trades_path, ledger_path, baselines_path, output_path)

    counts = {'GREEN': 0, 'ORANGE': 0, 'RED': 0}
    for r in results:
        counts[r['level']] = counts.get(r['level'], 0) + 1

    # RED should only fire on eligible blocks (H1/M30, regime 0/1/2)
    for r in results:
        if r['level'] == 'RED':
            assert r['tf'] in ['H1', 'M30'], f"RED on ineligible TF: {r['tf']}"
            regime_int = int(r['effective_regime'].replace('REGIME_', ''))
            assert regime_int in [0, 1, 2], f"RED on REGIME_3: {r['block_key']}"

    return True, counts


def run_retrain_runner_check() -> tuple:
    """Run retrain runner and verify K=3 diversified families."""
    from ops.retrain_runner import run_retrain_jobs, FAMILY_POOLS

    manifests = run_retrain_jobs(PROJECT_ROOT)

    for m in manifests:
        regime = m.get('effective_regime', 'REGIME_2')
        pool = FAMILY_POOLS.get(regime, [])
        families = m.get('families_selected', [])

        # K=3 or len(pool) if pool < 3
        expected_k = min(3, len(pool))
        assert len(families) == expected_k, \
            f"Expected {expected_k} families for {regime}, got {len(families)}"

        # All families from correct pool
        for f in families:
            assert f in pool, f"Family {f} not in pool for {regime}: {pool}"

        # Diversified (no duplicates)
        assert len(set(families)) == len(families), f"Duplicate families: {families}"

    return True, len(manifests)


def run_promotion_gate_check() -> tuple:
    """Run promotion gate evaluation."""
    from ops.promotion_gate import evaluate_candidate, select_winner

    # Test with synthetic passing candidate
    candidate = {'candidate_id': 'synth_001'}
    champion = {'stress_pf': 1.20, 'stress_avg_r': 1.5, 'vol_spike_max_dd': 80.0}
    metrics = {
        'IBKR_BASE': {'pf': 2.0, 'avg_r': 3.0, 'max_dd': 50, 'trade_count': 300,
                       'inference_error_rate': 0, 'schema_violations': 0, 'turnover_multiplier': 1.0},
        'STRESS_PLUS_75': {'pf': 1.50, 'avg_r': 2.0, 'max_dd': 70, 'trade_count': 300,
                           'inference_error_rate': 0, 'schema_violations': 0, 'turnover_multiplier': 1.0},
        'VOLATILITY_SPIKE': {'pf': 1.10, 'avg_r': 1.0, 'max_dd': 90, 'trade_count': 300,
                             'inference_error_rate': 0, 'schema_violations': 0, 'turnover_multiplier': 1.0},
    }

    result = evaluate_candidate(candidate, champion, metrics)

    # Test determinism: run twice
    result2 = evaluate_candidate(candidate, champion, metrics)
    assert result['decision'] == result2['decision'], "Promotion not deterministic"
    assert result['gates_passed'] == result2['gates_passed'], "Gates not deterministic"

    return result['decision'] == 'PASS', result


def run_canary_deployer_check() -> tuple:
    """Test canary deployer lifecycle."""
    import tempfile
    from ops.canary_deployer import CanaryDeployer

    with tempfile.TemporaryDirectory() as tmp:
        deployer = CanaryDeployer(
            registry_path=Path(tmp) / 'registry.json',
            audit_dir=Path(tmp) / 'audit'
        )

        # Full lifecycle
        deployer.deploy_shadow('TEST|H1|REGIME_0', 'c001',
                              {'family': 'lgb'}, {'pf': 5.0, 'max_dd': 50})
        deployer.advance_stage('TEST|H1|REGIME_0')  # CANARY_10

        # Check no rollback with good metrics
        r = deployer.check_rollback('TEST|H1|REGIME_0', 4.5, 40.0, 60)
        assert not r['rollback']

        deployer.advance_stage('TEST|H1|REGIME_0')  # CANARY_50
        deployer.advance_stage('TEST|H1|REGIME_0')  # PROMOTE_100

        champion = deployer.promote_to_champion('TEST|H1|REGIME_0')
        assert champion['champion_id'] == 'c001'

        # Verify rollback path
        deployer.deploy_shadow('TEST2|H1|REGIME_1', 'c002',
                              {'family': 'xgb'}, {'pf': 3.0, 'max_dd': 40})
        deployer.advance_stage('TEST2|H1|REGIME_1')  # CANARY_10
        r = deployer.check_rollback('TEST2|H1|REGIME_1', 2.0, 55.0, 60)
        assert r['rollback']

        audit_files = list(Path(tmp, 'audit').glob('*.json'))
        assert len(audit_files) >= 3

    return True, 'lifecycle + rollback OK'


def main():
    print("=" * 70)
    print("SELF-HEALING INFRASTRUCTURE — VERIFICATION SUITE")
    print("=" * 70)
    print()

    results = []

    # Test 1: Unit tests - baseline builder
    print("Running Test 1: Baseline builder unit tests...")
    p, f, out = run_test_file(PROJECT_ROOT / 'ops' / 'tests' / 'test_baseline_builder.py')
    results.append(('Test 1', 'PASS' if f == 0 else 'FAIL',
                    f'Baseline builder unit tests', f'{p}/{p+f}'))
    if f > 0:
        print(out)

    # Test 2: Build baselines from Run 4
    print("Running Test 2: Build baselines from Run 4 data...")
    ok, n_blocks, scenarios = run_baselines_build()
    results.append(('Test 2', 'PASS' if ok else 'FAIL',
                    f'Baselines created for all blocks ({n_blocks} blocks, {scenarios})',
                    f'{n_blocks} blocks'))

    # Test 3: Unit tests - retrain trigger
    print("Running Test 3: Retrain trigger unit tests...")
    p, f, out = run_test_file(PROJECT_ROOT / 'ops' / 'tests' / 'test_retrain_trigger.py')
    results.append(('Test 3', 'PASS' if f == 0 else 'FAIL',
                    f'Retrain trigger unit tests', f'{p}/{p+f}'))
    if f > 0:
        print(out)

    # Test 4: Run triggers on Run 4 data
    print("Running Test 4: Trigger evaluation + effective regime mapping...")
    ok, counts = run_trigger_check()
    results.append(('Test 4', 'PASS' if ok else 'FAIL',
                    f'Trigger RED only on eligible blocks (G:{counts["GREEN"]} O:{counts["ORANGE"]} R:{counts["RED"]})',
                    f'{sum(counts.values())} blocks'))

    # Test 5: Retrain runner
    print("Running Test 5: K=3 challengers with diversified families...")
    ok, n_jobs = run_retrain_runner_check()
    results.append(('Test 5', 'PASS' if ok else 'FAIL',
                    f'K=3 challengers created with diversified families',
                    f'{n_jobs} jobs'))

    # Test 6: Unit tests - promotion gate
    print("Running Test 6: Promotion gate unit tests...")
    p, f, out = run_test_file(PROJECT_ROOT / 'ops' / 'tests' / 'test_promotion_gate.py')
    results.append(('Test 6', 'PASS' if f == 0 else 'FAIL',
                    f'Promotion gate unit tests', f'{p}/{p+f}'))
    if f > 0:
        print(out)

    # Test 7: Promotion determinism
    print("Running Test 7: Promotion decision deterministic...")
    ok, promo_result = run_promotion_gate_check()
    results.append(('Test 7', 'PASS' if ok else 'FAIL',
                    f'Promotion decision deterministic', promo_result.get('decision', '?')))

    # Test 8: Unit tests - canary deployer
    print("Running Test 8: Canary deployer unit tests...")
    p, f, out = run_test_file(PROJECT_ROOT / 'ops' / 'tests' / 'test_canary_deployer.py')
    results.append(('Test 8', 'PASS' if f == 0 else 'FAIL',
                    f'Canary deployer unit tests', f'{p}/{p+f}'))
    if f > 0:
        print(out)

    # Test 9: Canary lifecycle + rollback
    print("Running Test 9: Registry update + rollback logic...")
    ok, detail = run_canary_deployer_check()
    results.append(('Test 9', 'PASS' if ok else 'FAIL',
                    f'Registry update + rollback logic', detail))

    # Print results table
    print("\n")
    print("=" * 70)
    print(f"{'Test':<8} {'Status':<7} {'Description':<45} {'Detail'}")
    print("-" * 70)
    for test_id, status, desc, detail in results:
        print(f"{test_id:<8} {status:<7} {desc:<45} {detail}")
    print("=" * 70)

    total_pass = sum(1 for r in results if r[1] == 'PASS')
    total_fail = sum(1 for r in results if r[1] == 'FAIL')
    print(f"\nTotal: {total_pass} PASS, {total_fail} FAIL out of {len(results)} tests")

    # ═══════════════════════════════════════════════════════════════
    # REQUIRED OUTPUT: Schemas, Samples, Files, Run Command
    # ═══════════════════════════════════════════════════════════════

    print("\n\n")
    print("=" * 70)
    print("JSON SCHEMAS")
    print("=" * 70)

    print("\n1) edge_baselines_by_regime.json fields:")
    print("   metadata: {baseline_version, generated_at, source_runs, confidence_threshold,")
    print("              eligible_tfs, eligible_regimes, regime_policy_hash, schema_hash, total_blocks}")
    print("   blocks.<key>: {pair, tf, effective_regime, scenarios}")
    print("   blocks.<key>.scenarios.<name>: {pf, avg_r, win_rate, max_dd, wl_ratio,")
    print("                                   avg_winner, avg_loser, spread_mean, trade_count,")
    print("                                   gross_win, gross_loss}")

    print("\n2) retrain_jobs.jsonl fields (per line):")
    print("   {job_id, status, created_at, block_key, pair, tf, effective_regime,")
    print("    trigger_level, trigger_reason, trigger_details, baseline_scenario}")

    print("\n3) promotion_decision.json fields:")
    print("   {job_id, block_key, effective_regime, evaluated_at, champion_exists,")
    print("    candidates_evaluated, candidates_passed, candidates_failed, winner,")
    print("    evaluations[{candidate_id, decision, gates_passed, gates_failed,")
    print("                  stress_pf, stress_avg_r, stress_max_dd, total_holdout_trades}],")
    print("    gates_config}")

    print("\n4) model_registry.json block entry fields:")
    print("   deployments.<key>: {candidate_id, block_key, stage, deployed_at,")
    print("                       candidate_spec, champion_metrics, trade_count,")
    print("                       current_pf, current_dd, stage_history[], rollback, rollback_reason}")
    print("   registry.<key>: {champion_id, family, artifact_dir, promoted_at,")
    print("                    promotion_source, stress_pf, vol_spike_max_dd}")

    # Sample objects
    print("\n\n")
    print("=" * 70)
    print("SAMPLE JSON OBJECTS")
    print("=" * 70)

    # Load actual baselines for sample
    baselines_path = PROJECT_ROOT / 'ops' / 'edge_baselines_by_regime.json'
    if baselines_path.exists():
        with open(baselines_path) as f:
            baselines = json.load(f)
        first_key = next(iter(baselines.get('blocks', {})), None)
        if first_key:
            sample_baseline = {first_key: baselines['blocks'][first_key]}
            print(f"\n1) edge_baselines_by_regime.json sample block:")
            print(json.dumps(sample_baseline, indent=2))

    print(f"\n2) retrain_jobs.jsonl sample line:")
    sample_job = {
        "job_id": "a1b2c3d4e5f6",
        "status": "QUEUED",
        "created_at": "2026-03-05T12:00:00+00:00",
        "block_key": "EURUSD|H1|REGIME_1",
        "pair": "EURUSD",
        "tf": "H1",
        "effective_regime": "REGIME_1",
        "trigger_level": "RED",
        "trigger_reason": "slow_pf_critical: 1.20 < 0.60*5.15",
        "trigger_details": {"current_pf": 1.20, "baseline_pf": 5.15},
        "baseline_scenario": "IBKR_BASE"
    }
    print(json.dumps(sample_job, indent=2))

    print(f"\n3) promotion_decision.json sample:")
    sample_promotion = {
        "job_id": "a1b2c3d4e5f6",
        "block_key": "EURUSD|H1|REGIME_1",
        "effective_regime": "REGIME_1",
        "evaluated_at": "2026-03-05T12:30:00+00:00",
        "champion_exists": True,
        "candidates_evaluated": 3,
        "candidates_passed": 1,
        "candidates_failed": 2,
        "winner": {"candidate_id": "cand_xyz", "decision": "PASS", "stress_pf": 1.65}
    }
    print(json.dumps(sample_promotion, indent=2))

    print(f"\n4) model_registry.json deployment entry sample:")
    sample_deploy = {
        "candidate_id": "cand_xyz",
        "block_key": "EURUSD|H1|REGIME_1",
        "stage": "CANARY_10",
        "deployed_at": "2026-03-05T13:00:00+00:00",
        "champion_metrics": {"pf": 5.15, "max_dd": 54.46},
        "trade_count": 25,
        "rollback": False
    }
    print(json.dumps(sample_deploy, indent=2))

    # Files list
    print("\n\n")
    print("=" * 70)
    print("FILES CREATED/MODIFIED")
    print("=" * 70)
    files = [
        'ops/__init__.py',
        'ops/build_edge_baselines_by_regime.py',
        'ops/retrain_trigger.py',
        'ops/retrain_runner.py',
        'ops/promotion_gate.py',
        'ops/canary_deployer.py',
        'ops/run_self_heal_verification.py',
        'ops/edge_baselines_by_regime.json',
        'ops/retrain_jobs.jsonl',
        'ops/tests/__init__.py',
        'ops/tests/test_baseline_builder.py',
        'ops/tests/test_retrain_trigger.py',
        'ops/tests/test_promotion_gate.py',
        'ops/tests/test_canary_deployer.py',
    ]
    for f in files:
        exists = (PROJECT_ROOT / f).exists()
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {f}")

    print("\n\n")
    print("=" * 70)
    print("RUN COMMAND")
    print("=" * 70)
    print(f"\n  cd \"{PROJECT_ROOT}\" && python ops/run_self_heal_verification.py\n")

    return total_fail == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
