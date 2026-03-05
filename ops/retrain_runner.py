"""
Retrain Runner

Consumes retrain_jobs.jsonl (QUEUED jobs) and generates K=3 challengers
per triggered block with diversified model families based on regime.

Family pools by regime:
  REGIME_0 -> [catboost, lgb, rf, et, tabnet]
  REGIME_1 -> [transformer, tcn, lstm, gru, xgb, lgb]
  REGIME_2 -> [lgb, xgb, catboost]

Produces candidate artifacts in models/candidates/<job_id>/
"""
import json
import logging
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict

logger = logging.getLogger('retrain_runner')

K_CHALLENGERS = 3

# Model family pools by regime
FAMILY_POOLS = {
    'REGIME_0': ['catboost', 'lgb', 'rf', 'et', 'tabnet'],
    'REGIME_1': ['transformer', 'tcn', 'lstm', 'gru', 'xgb', 'lgb'],
    'REGIME_2': ['lgb', 'xgb', 'catboost'],
}

# Default hyperparameter seeds for deterministic family selection
FAMILY_HPARAM_DEFAULTS = {
    'catboost':    {'iterations': 500, 'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 3},
    'lgb':         {'n_estimators': 500, 'max_depth': 7, 'learning_rate': 0.05, 'reg_lambda': 1.0},
    'rf':          {'n_estimators': 300, 'max_depth': 12, 'min_samples_leaf': 10},
    'et':          {'n_estimators': 300, 'max_depth': 12, 'min_samples_leaf': 10},
    'tabnet':      {'n_steps': 3, 'n_a': 32, 'n_d': 32, 'lambda_sparse': 1e-3},
    'xgb':         {'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.05, 'reg_lambda': 1.0},
    'transformer': {'d_model': 64, 'nhead': 4, 'num_layers': 2, 'dropout': 0.1},
    'tcn':         {'num_channels': [32, 32], 'kernel_size': 3, 'dropout': 0.1},
    'lstm':        {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.1},
    'gru':         {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.1},
}


def select_families(regime: str, k: int = K_CHALLENGERS, seed: int = 42) -> List[str]:
    """
    Select K diversified families for the given regime.
    Uses deterministic selection with maximum diversity.
    """
    pool = FAMILY_POOLS.get(regime, FAMILY_POOLS['REGIME_2'])

    if len(pool) <= k:
        return pool[:k]

    # Deterministic diversified selection
    rng = np.random.RandomState(seed)
    selected = list(rng.choice(pool, size=k, replace=False))
    return sorted(selected)


def generate_candidate_spec(
    job: dict,
    family: str,
    candidate_idx: int,
    output_dir: Path
) -> dict:
    """
    Generate a candidate training specification.

    In production this would invoke the actual training pipeline.
    Here we produce the spec + placeholder metrics for the promotion gate.
    """
    candidate_id = hashlib.sha256(
        f"{job['job_id']}|{family}|{candidate_idx}".encode()
    ).hexdigest()[:12]

    candidate_dir = output_dir / candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=True)

    hparams = dict(FAMILY_HPARAM_DEFAULTS.get(family, {}))

    spec = {
        'candidate_id': candidate_id,
        'job_id': job['job_id'],
        'block_key': job['block_key'],
        'pair': job['pair'],
        'tf': job['tf'],
        'effective_regime': job['effective_regime'],
        'family': family,
        'candidate_idx': candidate_idx,
        'hyperparameters': hparams,
        'status': 'SPEC_GENERATED',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'artifact_dir': str(candidate_dir),
        'holdout_metrics': None,
        'scenario_metrics': None,
    }

    # Save spec
    spec_path = candidate_dir / 'candidate_spec.json'
    with open(spec_path, 'w') as f:
        json.dump(spec, f, indent=2, default=str)

    return spec


def process_job(job: dict, project_root: Path) -> dict:
    """
    Process a single retrain job: generate K challenger specs.
    """
    regime = job['effective_regime']
    families = select_families(regime, K_CHALLENGERS)

    output_dir = project_root / 'models' / 'candidates' / job['job_id']
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    for idx, family in enumerate(families):
        spec = generate_candidate_spec(job, family, idx, output_dir)
        candidates.append(spec)
        logger.info(f"  Candidate {idx}: {family} -> {spec['candidate_id']}")

    # Save job manifest
    manifest = {
        'job_id': job['job_id'],
        'block_key': job['block_key'],
        'effective_regime': regime,
        'families_selected': families,
        'k_challengers': len(candidates),
        'candidates': [c['candidate_id'] for c in candidates],
        'status': 'CANDIDATES_GENERATED',
        'processed_at': datetime.now(timezone.utc).isoformat(),
    }

    manifest_path = output_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)

    return manifest


def run_retrain_jobs(project_root: Path = None) -> List[dict]:
    """
    Read retrain_jobs.jsonl and process all QUEUED jobs.
    """
    if project_root is None:
        project_root = Path('.')

    jobs_path = project_root / 'ops' / 'retrain_jobs.jsonl'
    if not jobs_path.exists():
        logger.warning(f"No retrain_jobs.jsonl found at {jobs_path}")
        return []

    jobs = []
    with open(jobs_path) as f:
        for line in f:
            line = line.strip()
            if line:
                jobs.append(json.loads(line))

    queued = [j for j in jobs if j.get('status') == 'QUEUED']
    logger.info(f"Found {len(queued)} QUEUED jobs out of {len(jobs)} total")

    manifests = []
    for job in queued:
        logger.info(f"Processing job {job['job_id']}: {job['block_key']}")
        manifest = process_job(job, project_root)
        manifests.append(manifest)

    print(f"\nRETRAIN RUNNER RESULTS")
    print(f"  Jobs processed: {len(manifests)}")
    for m in manifests:
        print(f"  {m['block_key']}: {m['families_selected']} -> {m['candidates']}")

    return manifests


def main():
    logging.basicConfig(level=logging.INFO)
    run_retrain_jobs(Path('.'))


if __name__ == '__main__':
    main()
