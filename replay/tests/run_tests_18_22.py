"""Quick runner for Tests 18-22 only."""
import os, sys, json, time, numpy as np, random, pandas as pd, traceback
from pathlib import Path

os.environ['PYTHONHASHSEED'] = '42'
np.random.seed(42)
random.seed(42)

PROJECT_ROOT = Path('G:/My Drive/chaos_v1.0')
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'inference'))

RESULTS = []

def record(test_num, description, passed, details=''):
    status = 'PASS' if passed else 'FAIL'
    RESULTS.append((test_num, description, status, details))
    print(f'  Test {test_num:2d} | {status} | {description}' +
          (f' | {details}' if details else ''))


def test_18():
    """Feature compat: 10-bar inference with 0 errors."""
    try:
        from replay.runners.replay_iterator import ReplayIterator
        from replay.runners.run_replay import ModelLoader, EnsembleEngine

        iterator = ReplayIterator(
            pair='EURUSD', tf='M30',
            features_dir=str(PROJECT_ROOT / 'features'),
            date_start='2024-01-01', date_end='2024-06-30',
            mode='REPLAY',
        )
        replay_feat_count = len(iterator.feature_names)

        loader = ModelLoader(str(PROJECT_ROOT / 'models'))
        models = loader.get_models('EURUSD_M30')
        if not models:
            record(18, 'Replay feature compatibility', False, 'No EURUSD_M30 models')
            return

        model_info = []
        for brain, backend in models.items():
            expected = getattr(backend, 'expected_features', None)
            model_info.append(f'{brain}={expected}')
        detail = f'Replay={replay_feat_count} | {", ".join(model_info)}'

        ensemble = EnsembleEngine(
            str(PROJECT_ROOT / 'ensemble' / 'ensemble_config.json'))
        errors = 0
        bars_tested = 0
        for event in iterator:
            if bars_tested >= 10:
                break
            try:
                features = np.nan_to_num(
                    event['features'], nan=0.0, posinf=0.0, neginf=0.0)
                result = ensemble.run_inference(list(models.items()), features)
                bars_tested += 1
            except Exception as e:
                errors += 1
                detail += f' | Err@{event["bar_idx"]}: {e}'

        ok = errors == 0 and bars_tested == 10
        detail += f' | {bars_tested} bars, {errors} errors'
        record(18, 'Replay feature compatibility (A)', ok, detail)
    except Exception as e:
        record(18, 'Replay feature compatibility (A)', False, traceback.format_exc())


def test_19():
    """REPLAY mode yields > 273 pair-native features."""
    try:
        from replay.runners.replay_iterator import ReplayIterator
        iterator = ReplayIterator(
            pair='EURUSD', tf='M30',
            features_dir=str(PROJECT_ROOT / 'features'),
            date_start='2024-01-01', date_end='2024-01-31',
            mode='REPLAY',
        )
        feat_count = len(iterator.feature_names)
        event = next(iter(iterator))
        event_feat_count = event['feature_count']
        ok = feat_count > 273 and event_feat_count == feat_count
        record(19, 'REPLAY > 273 features (B)', ok,
               f'features={feat_count}, event={event_feat_count}')
    except Exception as e:
        record(19, 'REPLAY > 273 features (B)', False, traceback.format_exc())


def test_20():
    """LIVE mode enforces exactly 273."""
    try:
        from replay.runners.replay_iterator import ReplayIterator
        iterator = ReplayIterator(
            pair='EURUSD', tf='M30',
            features_dir=str(PROJECT_ROOT / 'features'),
            schema_path=str(PROJECT_ROOT / 'schema' / 'feature_schema.json'),
            date_start='2024-01-01', date_end='2024-01-31',
            mode='LIVE',
        )
        feat_count = len(iterator.feature_names)
        event = next(iter(iterator))
        event_feat_count = event['feature_count']
        ok = feat_count == 273 and event_feat_count == 273
        record(20, 'LIVE = 273 boundary (C)', ok,
               f'features={feat_count}, event={event_feat_count}')
    except Exception as e:
        record(20, 'LIVE = 273 boundary (C)', False, traceback.format_exc())


def test_21():
    """500-bar smoke: trades > 0, inference errors = 0."""
    try:
        os.environ['PYTHONHASHSEED'] = '42'
        np.random.seed(42)
        random.seed(42)

        from replay.runners.run_replay import ReplayRunner
        runner = ReplayRunner()
        run_id = runner.run(pairs=['EURUSD'], tfs=['M30'], max_bars=500)

        run_dir = PROJECT_ROOT / 'replay' / 'outputs' / 'runs' / run_id
        trades_df = pd.read_parquet(run_dir / 'trades.parquet')
        ledger_df = pd.read_parquet(run_dir / 'decision_ledger.parquet')

        trade_count = len(trades_df[trades_df['action'] == 'CLOSE']) \
            if len(trades_df) > 0 else 0

        inference_errors = 0
        feature_invalid = 0
        agreement_failed = 0
        if 'reason_codes' in ledger_df.columns:
            for codes in ledger_df['reason_codes'].dropna():
                cs = str(codes)
                if 'INFERENCE_ERROR' in cs:
                    inference_errors += 1
                if 'FEATURE_INVALID' in cs:
                    feature_invalid += 1
                if 'AGREEMENT_FAILED' in cs:
                    agreement_failed += 1

        flat_raw = int((ledger_df['raw_signal_side'] == 0).sum()) \
            if 'raw_signal_side' in ledger_df.columns else -1
        flat_dec = int((ledger_df['decision_side'] == 0).sum()) \
            if 'decision_side' in ledger_df.columns else -1
        voted_mean = ledger_df['models_voted'].mean() \
            if 'models_voted' in ledger_df.columns else -1

        ok = trade_count > 0 and inference_errors == 0
        detail = (f'trades={trade_count} bars={len(ledger_df)} '
                  f'inf_err={inference_errors} feat_inv={feature_invalid} '
                  f'agree_fail={agreement_failed} '
                  f'raw_flat={flat_raw}/{len(ledger_df)} '
                  f'dec_flat={flat_dec}/{len(ledger_df)} '
                  f'voted_mean={voted_mean:.1f}')

        with open(run_dir / 'metrics.json') as f:
            metrics = json.load(f)
        detail += f' PF={metrics.get("profit_factor", 0)} WR={metrics.get("win_rate", 0)}'

        record(21, '500-bar smoke (D)', ok, detail)
    except Exception as e:
        record(21, '500-bar smoke (D)', False, traceback.format_exc())


def test_22():
    """Ledger schema unchanged."""
    try:
        from replay.runners.parquet_schemas import DECISION_LEDGER_SCHEMA
        expected_fields = [
            'event_ts', 'pair', 'tf', 'request_id', 'run_id', 'scenario',
            'regime_state', 'regime_confidence',
            'enabled_models_count', 'enabled_models_keys',
            'models_voted', 'models_agreed',
            'brain_probs_trimmed_mean',
            'agreement_score', 'agreement_threshold_base',
            'agreement_threshold_modified',
            'alt_data_available', 'cot_pressure', 'cot_extreme',
            'cme_spike', 'cme_confirm',
            'alt_agreement_adjustment', 'mtf_status',
            'decision_side', 'decision_confidence',
            'raw_signal_side', 'signal_overridden',
            'risk_veto', 'risk_reason',
            'action_taken', 'latency_ms', 'reason_codes',
        ]
        schema_fields = [f.name for f in DECISION_LEDGER_SCHEMA]
        missing = set(expected_fields) - set(schema_fields)
        extra = set(schema_fields) - set(expected_fields)
        ok = len(missing) == 0
        detail = f'{len(schema_fields)} fields'
        if missing:
            detail += f', missing: {missing}'
        if extra:
            detail += f', extra: {extra}'
        record(22, 'Ledger schema unchanged (E)', ok, detail)
    except Exception as e:
        record(22, 'Ledger schema unchanged (E)', False, traceback.format_exc())


if __name__ == '__main__':
    print('=' * 80)
    print('CHAOS V3.5 — VERIFICATION TESTS 18-22')
    print('=' * 80)
    print()

    for t in [test_18, test_19, test_20, test_21, test_22]:
        t()

    print()
    passed = sum(1 for _, _, s, _ in RESULTS if s == 'PASS')
    failed = sum(1 for _, _, s, _ in RESULTS if s == 'FAIL')
    print(f'TOTAL: {passed} PASS / {failed} FAIL / {len(RESULTS)} TOTAL')
    print('=' * 80)
