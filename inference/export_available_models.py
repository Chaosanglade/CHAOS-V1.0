"""
Exports all available .pt and .joblib (tree) models to ONNX format.
Only exports models that exist AND don't already have ONNX versions.

Uses the onnx_export.py infrastructure already built.
Focuses on production timeframes (M5, M15, M30, H1) first.

This script can be run incrementally — it's safe to run multiple times.
Already-exported models are skipped.
"""
import json
import logging
from pathlib import Path

logger = logging.getLogger('export_models')


def export_available(registry_path='G:/My Drive/chaos_v1.0/models/model_registry.json',
                     models_dir='G:/My Drive/chaos_v1.0/models',
                     production_only=True):
    """
    Export all models marked 'needs_export' in the registry.

    Exports in priority order:
    1. Production TFs (M5, M15, M30, H1)
    2. Tree models first (LGB, XGB, CatBoost — faster export)
    3. PyTorch models second (need architecture reconstruction)
    """
    import sys
    sys.path.insert(0, 'G:/My Drive/chaos_v1.0/inference')
    from onnx_export import (
        export_pytorch_to_onnx, export_lgb_to_onnx,
        export_xgb_to_onnx, export_catboost_to_onnx
    )

    with open(registry_path) as f:
        registry = json.load(f)

    PRODUCTION_TFS = ['M5', 'M15', 'M30', 'H1']
    models_dir = Path(models_dir)

    exported = 0
    failed = 0
    skipped = 0

    # Collect all exportable models
    to_export = []
    for pair_tf, brains in registry['registry'].items():
        pair, tf = pair_tf.split('_', 1)
        if production_only and tf not in PRODUCTION_TFS:
            continue

        for brain, entry in brains.items():
            if entry['export_status'] != 'needs_export':
                continue
            if entry['artifact_path'] is None:
                continue
            to_export.append(entry)

    logger.info(f"Models to export: {len(to_export)}")

    for entry in to_export:
        model_key = entry['model_key']
        artifact_path = entry['artifact_path']
        artifact_type = entry['artifact_type']
        onnx_path = str(models_dir / f"{model_key}.onnx")

        # Skip if ONNX already exists
        if Path(onnx_path).exists():
            skipped += 1
            continue

        logger.info(f"Exporting {model_key} ({artifact_type})...")

        try:
            if artifact_type == 'joblib':
                brain = entry['brain']
                if 'lgb' in brain:
                    export_lgb_to_onnx(artifact_path, onnx_path)
                elif 'xgb' in brain:
                    export_xgb_to_onnx(artifact_path, onnx_path)
                elif 'cat' in brain:
                    export_catboost_to_onnx(artifact_path, onnx_path)
                else:
                    logger.warning(f"  Unknown joblib brain type: {brain}")
                    failed += 1
                    continue

            elif artifact_type == 'pt':
                # PyTorch models need architecture reconstruction
                # The onnx_export.py already handles this via checkpoint inspection
                # Need to determine input size from the parquet
                pair = entry['pair']
                tf = entry['tf']

                # Try to determine feature count from a parquet file
                try:
                    import pandas as pd
                    parquet_path = f"G:/My Drive/chaos_v1.0/{pair}_{tf}_features.parquet"
                    df = pd.read_parquet(parquet_path, columns=None)
                    # Count features using same logic as training
                    exclude_prefixes = ['target_', 'returns_']
                    feature_cols = [c for c in df.columns
                                   if not any(c.startswith(p) for p in exclude_prefixes)
                                   and c not in ['target_3class_8', 'target_return_8']]
                    input_size = len(feature_cols)
                except Exception as e:
                    logger.warning(f"  Cannot determine input size for {model_key}: {e}")
                    input_size = 277  # Fallback

                # Export — this may fail if architecture can't be reconstructed
                # The export function handles checkpoint inspection internally
                export_pytorch_to_onnx(artifact_path, onnx_path, input_size=input_size)

            exported += 1
            logger.info(f"  Exported: {onnx_path}")

        except Exception as e:
            failed += 1
            logger.error(f"  FAILED to export {model_key}: {e}")

    logger.info(f"\nExport complete: {exported} exported, {failed} failed, {skipped} skipped (already exist)")

    # Rebuild registry to reflect new exports
    if exported > 0:
        logger.info("Rebuilding registry with new exports...")
        from build_model_registry import build_registry
        build_registry()

    return exported, failed, skipped


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    exported, failed, skipped = export_available(production_only=True)

    print(f"\n{'='*60}")
    print(f"EXPORT SUMMARY")
    print(f"{'='*60}")
    print(f"  Exported: {exported}")
    print(f"  Failed: {failed}")
    print(f"  Skipped (already exist): {skipped}")
