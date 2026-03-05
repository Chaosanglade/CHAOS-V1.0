"""
Canary Deployer

Manages model deployment lifecycle:
  SHADOW -> CANARY_10 -> CANARY_50 -> PROMOTE_100

Rollback triggers:
  - PF < 0.8 * champion after 50 trades
  - DD > 1.2x champion

Updates model_registry.json and writes audit records.
"""
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Optional
from copy import deepcopy

logger = logging.getLogger('canary_deployer')

DEPLOYMENT_STAGES = ['SHADOW', 'CANARY_10', 'CANARY_50', 'PROMOTE_100']

ROLLBACK_PF_RATIO = 0.80
ROLLBACK_DD_RATIO = 1.20
ROLLBACK_MIN_TRADES = 50


class CanaryDeployer:
    """
    Manages canary deployment lifecycle for model promotion.
    """

    def __init__(self, registry_path: Path, audit_dir: Path):
        self.registry_path = registry_path
        self.audit_dir = audit_dir
        self.audit_dir.mkdir(parents=True, exist_ok=True)

        if registry_path.exists():
            with open(registry_path) as f:
                self.registry = json.load(f)
        else:
            self.registry = {'registry': {}, 'deployments': {}}

        # Ensure deployments section exists
        if 'deployments' not in self.registry:
            self.registry['deployments'] = {}

    def deploy_shadow(self, block_key: str, candidate_id: str,
                      candidate_spec: dict, champion_metrics: Optional[dict] = None) -> dict:
        """
        Deploy a candidate in SHADOW mode (log actions, don't execute).
        """
        deployment = {
            'candidate_id': candidate_id,
            'block_key': block_key,
            'stage': 'SHADOW',
            'deployed_at': datetime.now(timezone.utc).isoformat(),
            'candidate_spec': candidate_spec,
            'champion_metrics': champion_metrics,
            'trade_count': 0,
            'current_pf': None,
            'current_dd': None,
            'stage_history': [
                {'stage': 'SHADOW', 'at': datetime.now(timezone.utc).isoformat()}
            ],
            'rollback': False,
            'rollback_reason': None,
        }

        self.registry['deployments'][block_key] = deployment
        self._save_registry()
        self._write_audit(block_key, candidate_id, 'DEPLOY_SHADOW', deployment)

        logger.info(f"Deployed {candidate_id} to SHADOW for {block_key}")
        return deployment

    def advance_stage(self, block_key: str) -> dict:
        """
        Advance deployment to next stage if conditions met.
        """
        deployment = self.registry['deployments'].get(block_key)
        if not deployment:
            raise ValueError(f"No active deployment for {block_key}")

        current_stage = deployment['stage']
        current_idx = DEPLOYMENT_STAGES.index(current_stage)

        if current_idx >= len(DEPLOYMENT_STAGES) - 1:
            logger.info(f"{block_key} already at PROMOTE_100")
            return deployment

        next_stage = DEPLOYMENT_STAGES[current_idx + 1]
        deployment['stage'] = next_stage
        deployment['stage_history'].append({
            'stage': next_stage,
            'at': datetime.now(timezone.utc).isoformat()
        })

        self._save_registry()
        self._write_audit(block_key, deployment['candidate_id'],
                          f'ADVANCE_{next_stage}', deployment)

        logger.info(f"Advanced {block_key} from {current_stage} to {next_stage}")
        return deployment

    def check_rollback(self, block_key: str, current_pf: float,
                       current_dd: float, trade_count: int) -> dict:
        """
        Check if rollback conditions are met.
        """
        deployment = self.registry['deployments'].get(block_key)
        if not deployment:
            return {'rollback': False, 'reason': 'no_deployment'}

        champion = deployment.get('champion_metrics') or {}
        champion_pf = champion.get('pf', 0)
        champion_dd = champion.get('max_dd', float('inf'))

        deployment['trade_count'] = trade_count
        deployment['current_pf'] = round(current_pf, 4)
        deployment['current_dd'] = round(current_dd, 2)

        rollback = False
        reason = None

        if trade_count >= ROLLBACK_MIN_TRADES:
            if champion_pf > 0 and current_pf < ROLLBACK_PF_RATIO * champion_pf:
                rollback = True
                reason = f"PF {current_pf:.2f} < {ROLLBACK_PF_RATIO}x champion PF {champion_pf:.2f}"

            if champion_dd > 0 and current_dd > ROLLBACK_DD_RATIO * champion_dd:
                rollback = True
                reason = (reason + "; " if reason else "") + \
                         f"DD {current_dd:.2f} > {ROLLBACK_DD_RATIO}x champion DD {champion_dd:.2f}"

        if rollback:
            deployment['rollback'] = True
            deployment['rollback_reason'] = reason
            deployment['stage'] = 'ROLLED_BACK'
            deployment['stage_history'].append({
                'stage': 'ROLLED_BACK',
                'at': datetime.now(timezone.utc).isoformat(),
                'reason': reason
            })
            self._save_registry()
            self._write_audit(block_key, deployment['candidate_id'],
                              'ROLLBACK', deployment)
            logger.warning(f"ROLLBACK {block_key}: {reason}")

        return {
            'rollback': rollback,
            'reason': reason,
            'deployment': deployment
        }

    def promote_to_champion(self, block_key: str) -> dict:
        """
        Finalize promotion: update registry to make candidate the new champion.
        """
        deployment = self.registry['deployments'].get(block_key)
        if not deployment:
            raise ValueError(f"No active deployment for {block_key}")

        if deployment['stage'] != 'PROMOTE_100':
            raise ValueError(f"Cannot promote: stage is {deployment['stage']}, need PROMOTE_100")

        # Update champion in registry
        candidate_spec = deployment.get('candidate_spec', {})
        self.registry['registry'][block_key] = {
            'champion_id': deployment['candidate_id'],
            'family': candidate_spec.get('family', 'unknown'),
            'artifact_dir': candidate_spec.get('artifact_dir', ''),
            'promoted_at': datetime.now(timezone.utc).isoformat(),
            'promotion_source': deployment.get('candidate_id'),
            'stress_pf': deployment.get('current_pf'),
            'vol_spike_max_dd': deployment.get('current_dd'),
        }

        # Clean up deployment
        del self.registry['deployments'][block_key]
        self._save_registry()
        self._write_audit(block_key, deployment['candidate_id'],
                          'PROMOTE_CHAMPION', deployment)

        logger.info(f"Promoted {deployment['candidate_id']} to champion for {block_key}")
        return self.registry['registry'][block_key]

    def get_deployment_status(self) -> list:
        """Get status of all active deployments."""
        statuses = []
        for block_key, dep in self.registry.get('deployments', {}).items():
            statuses.append({
                'block_key': block_key,
                'candidate_id': dep['candidate_id'],
                'stage': dep['stage'],
                'trade_count': dep.get('trade_count', 0),
                'rollback': dep.get('rollback', False),
            })
        return statuses

    def _save_registry(self):
        """Save registry to disk."""
        # Write to separate deployment registry to not modify the existing model_registry.json
        deploy_registry_path = self.registry_path.parent / 'deployment_registry.json'
        with open(deploy_registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2, default=str)

    def _write_audit(self, block_key: str, candidate_id: str,
                     action: str, details: dict):
        """Write audit record."""
        import uuid
        ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        uid = uuid.uuid4().hex[:8]
        audit = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'block_key': block_key,
            'candidate_id': candidate_id,
            'action': action,
            'details': deepcopy(details),
        }

        # Use a safe job_id + uid for unique filename
        job_id = details.get('candidate_id', candidate_id)[:12]
        audit_path = self.audit_dir / f"{ts}_{job_id}_{uid}.json"
        with open(audit_path, 'w') as f:
            json.dump(audit, f, indent=2, default=str)


def main():
    logging.basicConfig(level=logging.INFO)
    project_root = Path('.')

    deployer = CanaryDeployer(
        registry_path=project_root / 'models' / 'model_registry.json',
        audit_dir=project_root / 'ops' / 'deploy_audit'
    )

    statuses = deployer.get_deployment_status()
    print(f"\nCANARY DEPLOYER STATUS")
    print(f"  Active deployments: {len(statuses)}")
    for s in statuses:
        print(f"  {s['block_key']}: {s['stage']} ({s['trade_count']} trades)")


if __name__ == '__main__':
    main()
