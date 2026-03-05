"""
Scans the models/ directory and builds a comprehensive registry of all
trained models, their export status, and readiness for inference.

Output: models/model_registry.json

For each pair/tf/brain:
  - artifact_path: path to .pt, .joblib, or .onnx file
  - artifact_type: 'onnx' | 'pt' | 'joblib'
  - export_status: 'ready' | 'needs_export' | 'missing'
  - expected_features: int (from model inspection)
  - scaler_path: str or null (for neural nets with StandardScaler in checkpoint)
"""
import json
import logging
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger('model_registry')

# All 21 brain types
ALL_BRAINS = [
    'lgb_optuna', 'lgb_v2_optuna', 'xgb_optuna', 'xgb_v2_optuna',
    'cat_optuna', 'cat_v2_optuna',
    'rf_optuna', 'et_optuna',
    'mlp_optuna', 'residual_mlp_optuna', 'ensemble_nn_optuna',
    'lstm_optuna', 'gru_optuna',
    'cnn1d_optuna', 'tcn_optuna', 'wavenet_optuna',
    'attention_net_optuna', 'transformer_optuna', 'tft_optuna',
    'nbeats_optuna', 'tabnet_optuna'
]

# Brains that stay as .joblib (no ONNX needed)
SKLEARN_BRAINS = {'rf_optuna', 'et_optuna'}

# Brains that are .joblib but CAN be exported to ONNX
TREE_BRAINS_ONNX = {'lgb_optuna', 'lgb_v2_optuna', 'xgb_optuna', 'xgb_v2_optuna',
                     'cat_optuna', 'cat_v2_optuna'}

# Brains that are .pt and need ONNX export
PYTORCH_BRAINS = set(ALL_BRAINS) - SKLEARN_BRAINS - TREE_BRAINS_ONNX

PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY', 'GBPJPY']
TIMEFRAMES = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1', 'MN1']
PRODUCTION_TFS = ['M5', 'M15', 'M30', 'H1']


def build_registry(models_dir='G:/My Drive/chaos_v1.0/models',
                   output_path='G:/My Drive/chaos_v1.0/models/model_registry.json'):
    """
    Scan models directory and build comprehensive registry.
    """
    models_dir = Path(models_dir)

    # Index all files
    onnx_files = {f.stem: f for f in models_dir.glob('*.onnx')}
    pt_files = {f.stem: f for f in models_dir.glob('*.pt')}
    joblib_files = {f.stem: f for f in models_dir.glob('*.joblib')}

    registry = {}
    stats = {
        'total_expected': 0,
        'total_found': 0,
        'ready_for_inference': 0,
        'needs_onnx_export': 0,
        'missing': 0,
        'production_ready': 0,
        'production_needs_export': 0,
    }

    for pair in PAIRS:
        for tf in TIMEFRAMES:
            pair_tf = f"{pair}_{tf}"
            registry[pair_tf] = {}

            for brain in ALL_BRAINS:
                model_key = f"{pair}_{tf}_{brain}"
                stats['total_expected'] += 1

                entry = {
                    'pair': pair,
                    'tf': tf,
                    'brain': brain,
                    'model_key': model_key,
                    'artifact_path': None,
                    'artifact_type': None,
                    'onnx_path': None,
                    'export_status': 'missing',
                    'expected_features': None,
                    'scaler_path': None,
                    'production_tf': tf in PRODUCTION_TFS,
                }

                # Check for ONNX first (highest priority — inference-ready)
                if model_key in onnx_files:
                    entry['artifact_path'] = str(onnx_files[model_key])
                    entry['onnx_path'] = str(onnx_files[model_key])
                    entry['artifact_type'] = 'onnx'
                    entry['export_status'] = 'ready'
                    stats['total_found'] += 1
                    stats['ready_for_inference'] += 1
                    if tf in PRODUCTION_TFS:
                        stats['production_ready'] += 1

                # Check for .joblib (sklearn brains are ready as-is, tree brains may need ONNX)
                elif model_key in joblib_files:
                    entry['artifact_path'] = str(joblib_files[model_key])
                    entry['artifact_type'] = 'joblib'
                    stats['total_found'] += 1

                    if brain in SKLEARN_BRAINS:
                        entry['export_status'] = 'ready'  # RF/ET stay as joblib
                        stats['ready_for_inference'] += 1
                        if tf in PRODUCTION_TFS:
                            stats['production_ready'] += 1
                    elif brain in TREE_BRAINS_ONNX:
                        # Check if ONNX version exists
                        onnx_key = model_key
                        if onnx_key in onnx_files:
                            entry['onnx_path'] = str(onnx_files[onnx_key])
                            entry['export_status'] = 'ready'
                            stats['ready_for_inference'] += 1
                            if tf in PRODUCTION_TFS:
                                stats['production_ready'] += 1
                        else:
                            entry['export_status'] = 'needs_export'
                            stats['needs_onnx_export'] += 1
                            if tf in PRODUCTION_TFS:
                                stats['production_needs_export'] += 1

                # Check for .pt (PyTorch — needs ONNX export)
                elif model_key in pt_files:
                    entry['artifact_path'] = str(pt_files[model_key])
                    entry['artifact_type'] = 'pt'
                    stats['total_found'] += 1

                    # Check if ONNX version already exists
                    if model_key in onnx_files:
                        entry['onnx_path'] = str(onnx_files[model_key])
                        entry['export_status'] = 'ready'
                        stats['ready_for_inference'] += 1
                        if tf in PRODUCTION_TFS:
                            stats['production_ready'] += 1
                    else:
                        entry['export_status'] = 'needs_export'
                        stats['needs_onnx_export'] += 1
                        if tf in PRODUCTION_TFS:
                            stats['production_needs_export'] += 1

                else:
                    stats['missing'] += 1

                registry[pair_tf][brain] = entry

    output = {
        'generated': str(Path(output_path)),
        'stats': stats,
        'production_summary': {
            'total_production_models': len(PAIRS) * len(PRODUCTION_TFS) * len(ALL_BRAINS),
            'production_ready': stats['production_ready'],
            'production_needs_export': stats['production_needs_export'],
            'production_coverage': round(
                stats['production_ready'] / (len(PAIRS) * len(PRODUCTION_TFS) * len(ALL_BRAINS)), 4
            ) if (len(PAIRS) * len(PRODUCTION_TFS) * len(ALL_BRAINS)) > 0 else 0,
        },
        'registry': registry,
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"Model registry built: {output_path}")
    logger.info(f"  Total expected: {stats['total_expected']}")
    logger.info(f"  Total found: {stats['total_found']}")
    logger.info(f"  Ready for inference: {stats['ready_for_inference']}")
    logger.info(f"  Needs ONNX export: {stats['needs_onnx_export']}")
    logger.info(f"  Missing: {stats['missing']}")
    logger.info(f"  Production ready: {stats['production_ready']}/{len(PAIRS) * len(PRODUCTION_TFS) * len(ALL_BRAINS)}")

    return output


def generate_registry_summary(registry_path='G:/My Drive/chaos_v1.0/models/model_registry.json',
                               output_path='G:/My Drive/chaos_v1.0/models/model_registry_summary.json'):
    """Compact summary of model registry for quick status checks."""
    with open(registry_path) as f:
        registry = json.load(f)

    pair_tf_total = 0
    pair_tf_ready_21 = 0
    pair_tf_ready_partial = 0
    missing_by_pair_tf = {}

    for pair_tf, brains in registry.get('registry', {}).items():
        pair, tf = pair_tf.split('_', 1)
        if tf not in PRODUCTION_TFS:
            continue

        pair_tf_total += 1
        ready_count = sum(1 for b in brains.values() if b.get('export_status') == 'ready')
        needs_export = sum(1 for b in brains.values() if b.get('export_status') == 'needs_export')
        missing = sum(1 for b in brains.values() if b.get('export_status') == 'missing')

        if ready_count == 21:
            pair_tf_ready_21 += 1
        elif ready_count > 0:
            pair_tf_ready_partial += 1

        if needs_export > 0 or missing > 0:
            missing_by_pair_tf[pair_tf] = {
                'ready': ready_count,
                'needs_export': needs_export,
                'missing': missing,
                'gap': 21 - ready_count,
            }

    top_missing = sorted(missing_by_pair_tf.items(), key=lambda x: -x[1]['gap'])[:10]

    summary = {
        'production_timeframes': PRODUCTION_TFS,
        'pair_tf_total': pair_tf_total,
        'pair_tf_ready_for_21_brain': pair_tf_ready_21,
        'pair_tf_partial_coverage': pair_tf_ready_partial,
        'pair_tf_zero_coverage': pair_tf_total - pair_tf_ready_21 - pair_tf_ready_partial,
        'brains_ready': registry.get('stats', {}).get('ready_for_inference', 0),
        'brains_needs_export': registry.get('stats', {}).get('needs_onnx_export', 0),
        'brains_missing': registry.get('stats', {}).get('missing', 0),
        'production_coverage_pct': round(
            registry.get('production_summary', {}).get('production_coverage', 0) * 100, 1
        ),
        'top_10_missing_pair_tf': [
            {'pair_tf': k, **v} for k, v in top_missing
        ],
    }

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Registry summary saved: {output_path}")
    return summary


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    result = build_registry()

    # Print actionable summary
    print("\n" + "=" * 60)
    print("MODEL REGISTRY SUMMARY")
    print("=" * 60)
    print(f"Total models found: {result['stats']['total_found']}/{result['stats']['total_expected']}")
    print(f"Ready for inference: {result['stats']['ready_for_inference']}")
    print(f"Needs ONNX export: {result['stats']['needs_onnx_export']}")
    print(f"Missing (not yet trained): {result['stats']['missing']}")
    print(f"\nProduction TFs ({', '.join(PRODUCTION_TFS)}):")
    print(f"  Ready: {result['production_summary']['production_ready']}")
    print(f"  Needs export: {result['production_summary']['production_needs_export']}")
    print(f"  Coverage: {result['production_summary']['production_coverage']:.1%}")

    # List models that can be ONNX-exported RIGHT NOW
    print(f"\n{'='*60}")
    print("MODELS AVAILABLE FOR IMMEDIATE ONNX EXPORT")
    print("=" * 60)
    exportable = []
    for pair_tf, brains in result['registry'].items():
        pair, tf = pair_tf.split('_', 1)
        if tf not in PRODUCTION_TFS:
            continue
        for brain, entry in brains.items():
            if entry['export_status'] == 'needs_export' and entry['artifact_path'] is not None:
                exportable.append(entry)

    if exportable:
        for e in exportable[:20]:  # Show first 20
            print(f"  {e['model_key']}: {e['artifact_type']} -> needs ONNX")
        if len(exportable) > 20:
            print(f"  ... and {len(exportable) - 20} more")
    else:
        print("  None — all available models are already exported or use sklearn backend")
