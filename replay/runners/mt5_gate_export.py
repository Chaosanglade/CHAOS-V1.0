"""
MT5 Gate Export — Creates zip bundle for MT5 paper trading authorization.

Reads the latest replay run results and creates:
    audit/exports/mt5_audit_bundle_{run_id}.zip
      ├── manifest.json
      ├── metrics.json
      ├── gate_criteria_result.json
      ├── decision_ledger.parquet
      └── risk_report.json
"""
import json
import shutil
import logging
import zipfile
from pathlib import Path

logger = logging.getLogger('mt5_gate_export')

PROJECT_ROOT = Path('G:/My Drive/chaos_v1.0')


def find_latest_run() -> Path:
    """Find the most recent replay run directory."""
    runs_dir = PROJECT_ROOT / 'replay' / 'outputs' / 'runs'
    if not runs_dir.exists():
        raise FileNotFoundError(f"No runs directory: {runs_dir}")

    run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], reverse=True)
    if not run_dirs:
        raise FileNotFoundError("No replay runs found")

    return run_dirs[0]


def export_mt5_bundle(run_dir: Path = None) -> Path:
    """
    Create MT5 audit bundle zip from a replay run.

    Args:
        run_dir: Path to replay run directory. If None, uses latest.

    Returns:
        Path to created zip file.
    """
    if run_dir is None:
        run_dir = find_latest_run()

    run_id = run_dir.name
    logger.info(f"Exporting MT5 bundle for run: {run_id}")

    # Verify required files exist
    required_files = ['manifest.json', 'metrics.json', 'gate_criteria_result.json']
    for fname in required_files:
        if not (run_dir / fname).exists():
            raise FileNotFoundError(f"Missing required file: {run_dir / fname}")

    # Create exports directory
    exports_dir = PROJECT_ROOT / 'audit' / 'exports'
    exports_dir.mkdir(parents=True, exist_ok=True)

    # Build risk report
    risk_report = _build_risk_report(run_dir, run_id)
    risk_report_path = run_dir / 'risk_report.json'
    with open(risk_report_path, 'w') as f:
        json.dump(risk_report, f, indent=2, default=str)

    # Create zip
    zip_path = exports_dir / f"mt5_audit_bundle_{run_id}.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # JSON files
        for fname in ['manifest.json', 'metrics.json', 'gate_criteria_result.json', 'risk_report.json']:
            fpath = run_dir / fname
            if fpath.exists():
                zf.write(fpath, fname)

        # Decision ledger parquet
        ledger_path = run_dir / 'decision_ledger.parquet'
        if ledger_path.exists():
            zf.write(ledger_path, 'decision_ledger.parquet')

    logger.info(f"MT5 bundle created: {zip_path}")
    return zip_path


def _build_risk_report(run_dir: Path, run_id: str) -> dict:
    """Build risk report from run outputs."""
    report = {
        'run_id': run_id,
        'report_type': 'mt5_gate_risk_assessment',
    }

    # Load metrics
    metrics_path = run_dir / 'metrics.json'
    if metrics_path.exists():
        with open(metrics_path) as f:
            report['metrics'] = json.load(f)

    # Load gate criteria
    gate_path = run_dir / 'gate_criteria_result.json'
    if gate_path.exists():
        with open(gate_path) as f:
            gate = json.load(f)
            report['gate_result'] = gate.get('overall', 'UNKNOWN')
            report['gate_criteria'] = gate.get('criteria', {})

    # Load manifest
    manifest_path = run_dir / 'manifest.json'
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
            report['total_bars'] = manifest.get('total_bars', 0)
            report['total_trades'] = manifest.get('total_trade_rows', 0)
            report['errors_count'] = manifest.get('errors_count', 0)

    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description='MT5 Gate Export')
    parser.add_argument('--run-dir', default=None, help='Path to replay run directory')
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else None
    zip_path = export_mt5_bundle(run_dir)
    print(f"MT5 audit bundle: {zip_path}")


if __name__ == '__main__':
    main()
