"""
Promotion Gate

Evaluates candidate models vs champion for the same (pair, tf, effective_regime) block.
Multi-scenario replay with strict constraints.

Promotion rules:
  - IBKR_BASE PF >= 1.30
  - STRESS_PLUS_75 PF >= 1.15
  - VOL_SPIKE PF >= 1.05 (or DD <= 1.2x champion)
  - Must beat champion on STRESS_PLUS_75 PF by >= +5% (or avg_r by >= +5%)
  - min_trades_holdout >= 200
  - inference_error_rate <= 1e-4
  - schema_violations == 0
  - turnover_multiplier <= 1.30
"""
import json
import logging
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Optional, List

logger = logging.getLogger('promotion_gate')

# Gate thresholds
GATES = {
    'IBKR_BASE_PF_MIN': 1.30,
    'STRESS_PLUS_75_PF_MIN': 1.15,
    'VOL_SPIKE_PF_MIN': 1.05,
    'VOL_SPIKE_DD_RATIO_MAX': 1.20,
    'CHAMPION_BEAT_PCT': 0.05,  # Must beat by 5%
    'MIN_TRADES_HOLDOUT': 200,
    'MAX_INFERENCE_ERROR_RATE': 1e-4,
    'MAX_SCHEMA_VIOLATIONS': 0,
    'MAX_TURNOVER_MULTIPLIER': 1.30,
}


def evaluate_candidate(
    candidate: dict,
    champion: Optional[dict],
    scenario_metrics: dict,
) -> dict:
    """
    Evaluate a single candidate against promotion gates.

    Args:
        candidate: Candidate spec dict
        champion: Current champion metrics (or None if no champion)
        scenario_metrics: Dict of scenario -> {pf, avg_r, max_dd, trade_count,
                          inference_error_rate, schema_violations, turnover_multiplier}

    Returns:
        Decision dict with PASS/FAIL and rationale
    """
    gates_passed = []
    gates_failed = []

    # Gate 1: IBKR_BASE PF
    ibkr = scenario_metrics.get('IBKR_BASE', {})
    ibkr_pf = ibkr.get('pf', 0)
    if ibkr_pf >= GATES['IBKR_BASE_PF_MIN']:
        gates_passed.append(f"IBKR_BASE PF {ibkr_pf:.2f} >= {GATES['IBKR_BASE_PF_MIN']}")
    else:
        gates_failed.append(f"IBKR_BASE PF {ibkr_pf:.2f} < {GATES['IBKR_BASE_PF_MIN']}")

    # Gate 2: STRESS_PLUS_75 PF
    stress = scenario_metrics.get('STRESS_PLUS_75', {})
    stress_pf = stress.get('pf', 0)
    if stress_pf >= GATES['STRESS_PLUS_75_PF_MIN']:
        gates_passed.append(f"STRESS_PLUS_75 PF {stress_pf:.2f} >= {GATES['STRESS_PLUS_75_PF_MIN']}")
    else:
        gates_failed.append(f"STRESS_PLUS_75 PF {stress_pf:.2f} < {GATES['STRESS_PLUS_75_PF_MIN']}")

    # Gate 3: VOL_SPIKE PF (or DD fallback)
    vol = scenario_metrics.get('VOLATILITY_SPIKE', {})
    vol_pf = vol.get('pf', 0)
    vol_dd = vol.get('max_dd', float('inf'))
    champion_dd = champion.get('vol_spike_max_dd', float('inf')) if champion else float('inf')

    if vol_pf >= GATES['VOL_SPIKE_PF_MIN']:
        gates_passed.append(f"VOL_SPIKE PF {vol_pf:.2f} >= {GATES['VOL_SPIKE_PF_MIN']}")
    elif champion and champion_dd > 0 and vol_dd <= GATES['VOL_SPIKE_DD_RATIO_MAX'] * champion_dd:
        gates_passed.append(f"VOL_SPIKE DD {vol_dd:.2f} <= {GATES['VOL_SPIKE_DD_RATIO_MAX']}x champion DD {champion_dd:.2f}")
    else:
        gates_failed.append(f"VOL_SPIKE PF {vol_pf:.2f} < {GATES['VOL_SPIKE_PF_MIN']} and DD check failed")

    # Gate 4: Beat champion on STRESS_PLUS_75
    if champion:
        champ_stress_pf = champion.get('stress_pf', 0)
        champ_stress_avg_r = champion.get('stress_avg_r', 0)
        stress_avg_r = stress.get('avg_r', 0)

        beat_pf = (stress_pf >= champ_stress_pf * (1 + GATES['CHAMPION_BEAT_PCT'])) if champ_stress_pf > 0 else True
        beat_avg_r = (stress_avg_r >= champ_stress_avg_r * (1 + GATES['CHAMPION_BEAT_PCT'])) if champ_stress_avg_r > 0 else True

        if beat_pf or beat_avg_r:
            gates_passed.append(f"Beats champion: PF {stress_pf:.2f} vs {champ_stress_pf:.2f}, avg_r {stress_avg_r:.4f} vs {champ_stress_avg_r:.4f}")
        else:
            gates_failed.append(f"Does not beat champion by {GATES['CHAMPION_BEAT_PCT']:.0%}: PF {stress_pf:.2f} vs {champ_stress_pf:.2f}")
    else:
        gates_passed.append("No champion to beat (first model for block)")

    # Gate 5: Min trades on holdout
    total_trades = sum(s.get('trade_count', 0) for s in scenario_metrics.values())
    if total_trades >= GATES['MIN_TRADES_HOLDOUT']:
        gates_passed.append(f"Holdout trades {total_trades} >= {GATES['MIN_TRADES_HOLDOUT']}")
    else:
        gates_failed.append(f"Holdout trades {total_trades} < {GATES['MIN_TRADES_HOLDOUT']}")

    # Gate 6: Execution sanity
    for sc_name, sc_metrics in scenario_metrics.items():
        err_rate = sc_metrics.get('inference_error_rate', 0)
        violations = sc_metrics.get('schema_violations', 0)
        turnover = sc_metrics.get('turnover_multiplier', 1.0)

        if err_rate > GATES['MAX_INFERENCE_ERROR_RATE']:
            gates_failed.append(f"{sc_name} inference_error_rate {err_rate} > {GATES['MAX_INFERENCE_ERROR_RATE']}")
        if violations > GATES['MAX_SCHEMA_VIOLATIONS']:
            gates_failed.append(f"{sc_name} schema_violations {violations} > {GATES['MAX_SCHEMA_VIOLATIONS']}")
        if turnover > GATES['MAX_TURNOVER_MULTIPLIER']:
            gates_failed.append(f"{sc_name} turnover {turnover:.2f} > {GATES['MAX_TURNOVER_MULTIPLIER']}")

    passed = len(gates_failed) == 0

    return {
        'candidate_id': candidate.get('candidate_id', 'unknown'),
        'decision': 'PASS' if passed else 'FAIL',
        'gates_passed': gates_passed,
        'gates_failed': gates_failed,
        'stress_pf': stress_pf,
        'stress_avg_r': stress.get('avg_r', 0),
        'stress_max_dd': stress.get('max_dd', 0),
        'total_holdout_trades': total_trades,
    }


def select_winner(evaluations: List[dict]) -> Optional[dict]:
    """
    Select the best passing candidate.
    Maximize STRESS_PLUS_75 PF, tie-break by lower MaxDD.
    """
    passing = [e for e in evaluations if e['decision'] == 'PASS']
    if not passing:
        return None

    passing.sort(key=lambda e: (-e['stress_pf'], e['stress_max_dd']))
    return passing[0]


def run_promotion_gate(
    job_manifest_path: Path,
    model_registry_path: Path,
    output_dir: Path,
) -> dict:
    """
    Run promotion evaluation for all candidates in a job.
    """
    with open(job_manifest_path) as f:
        manifest = json.load(f)

    job_id = manifest['job_id']
    block_key = manifest['block_key']

    # Load champion from registry (if exists)
    champion = None
    if model_registry_path.exists():
        with open(model_registry_path) as f:
            registry = json.load(f)
        # Look for champion in registry
        champion_entry = registry.get('registry', {}).get(block_key)
        if champion_entry:
            champion = champion_entry

    # Evaluate each candidate
    evaluations = []
    candidates_dir = job_manifest_path.parent
    for cid in manifest.get('candidates', []):
        spec_path = candidates_dir / cid / 'candidate_spec.json'
        if not spec_path.exists():
            continue

        with open(spec_path) as f:
            spec = json.load(f)

        # In production: run replay engine for each scenario
        # Here: use scenario_metrics from spec or generate placeholder
        scenario_metrics = spec.get('scenario_metrics') or {
            'IBKR_BASE': {'pf': 0, 'avg_r': 0, 'max_dd': 0, 'trade_count': 0,
                          'inference_error_rate': 0, 'schema_violations': 0, 'turnover_multiplier': 1.0},
            'STRESS_PLUS_75': {'pf': 0, 'avg_r': 0, 'max_dd': 0, 'trade_count': 0,
                               'inference_error_rate': 0, 'schema_violations': 0, 'turnover_multiplier': 1.0},
            'VOLATILITY_SPIKE': {'pf': 0, 'avg_r': 0, 'max_dd': 0, 'trade_count': 0,
                                 'inference_error_rate': 0, 'schema_violations': 0, 'turnover_multiplier': 1.0},
        }

        evaluation = evaluate_candidate(spec, champion, scenario_metrics)
        evaluation['family'] = spec.get('family', 'unknown')
        evaluations.append(evaluation)

    winner = select_winner(evaluations)

    decision = {
        'job_id': job_id,
        'block_key': block_key,
        'effective_regime': manifest.get('effective_regime'),
        'evaluated_at': datetime.now(timezone.utc).isoformat(),
        'champion_exists': champion is not None,
        'candidates_evaluated': len(evaluations),
        'candidates_passed': sum(1 for e in evaluations if e['decision'] == 'PASS'),
        'candidates_failed': sum(1 for e in evaluations if e['decision'] == 'FAIL'),
        'winner': winner,
        'evaluations': evaluations,
        'gates_config': dict(GATES),
    }

    # Save decision
    output_dir.mkdir(parents=True, exist_ok=True)
    decision_path = output_dir / f"{job_id}.json"
    with open(decision_path, 'w') as f:
        json.dump(decision, f, indent=2, default=str)

    logger.info(f"Promotion decision for {block_key}: "
                f"{decision['candidates_passed']}/{decision['candidates_evaluated']} passed, "
                f"winner={'YES' if winner else 'NONE'}")

    return decision


def main():
    logging.basicConfig(level=logging.INFO)
    project_root = Path('.')
    model_registry_path = project_root / 'models' / 'model_registry.json'
    output_dir = project_root / 'ops' / 'promotion_decisions'

    # Process all job manifests
    candidates_dir = project_root / 'models' / 'candidates'
    if not candidates_dir.exists():
        print("No candidates directory found")
        return

    decisions = []
    for job_dir in sorted(candidates_dir.iterdir()):
        manifest_path = job_dir / 'manifest.json'
        if manifest_path.exists():
            decision = run_promotion_gate(manifest_path, model_registry_path, output_dir)
            decisions.append(decision)

    print(f"\nPROMOTION GATE RESULTS")
    for d in decisions:
        status = "WINNER" if d['winner'] else "NO WINNER"
        print(f"  {d['block_key']}: {d['candidates_passed']}/{d['candidates_evaluated']} passed -> {status}")


if __name__ == '__main__':
    main()
