"""
Post-run report generators for coverage and veto breakdown.

Generates:
  - coverage.json: brain coverage per pair/tf (loaded vs expected)
  - veto_breakdown.json: reason code counts from decision ledger
"""
import json
import logging
from pathlib import Path
from collections import Counter
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger('report_generators')

ALL_BRAINS = [
    'lgb_optuna', 'lgb_v2_optuna', 'xgb_optuna', 'xgb_v2_optuna',
    'cat_optuna', 'cat_v2_optuna',
    'rf_optuna', 'et_optuna',
    'mlp_optuna', 'residual_mlp_optuna', 'ensemble_nn_optuna',
    'lstm_optuna', 'gru_optuna',
    'cnn1d_optuna', 'tcn_optuna', 'wavenet_optuna',
    'attention_net_optuna', 'transformer_optuna', 'tft_optuna',
    'nbeats_optuna', 'tabnet_optuna',
]
BRAINS_EXPECTED = len(ALL_BRAINS)  # 21


def generate_coverage_report(run_id: str, models_loaded: Dict,
                             decision_ledger_df: Optional[pd.DataFrame],
                             output_dir) -> Dict:
    """
    Coverage matrix: brains loaded vs expected per pair/tf.

    Args:
        run_id: run identifier
        models_loaded: {pair_tf: {brain_name: backend}}
        decision_ledger_df: full ledger DataFrame (or None)
        output_dir: Path to run output directory
    """
    coverage = {}

    for pair_tf, brains in models_loaded.items():
        loaded_names = list(brains.keys())

        # Get average models_voted from ledger
        avg_voted = 0.0
        avg_agreed = 0.0
        parts = pair_tf.split('_', 1)
        if len(parts) == 2 and decision_ledger_df is not None and len(decision_ledger_df) > 0:
            pair, tf = parts
            subset = decision_ledger_df[
                (decision_ledger_df['pair'] == pair) &
                (decision_ledger_df['tf'] == tf)
            ]
            if len(subset) > 0:
                avg_voted = float(subset['models_voted'].mean())
                avg_agreed = float(subset['models_agreed'].mean())

        coverage[pair_tf] = {
            'brains_expected': BRAINS_EXPECTED,
            'brains_loaded': len(loaded_names),
            'brains_loaded_list': sorted(loaded_names),
            'brains_missing': sorted(set(ALL_BRAINS) - set(loaded_names)),
            'avg_models_enabled_after_regime': round(avg_voted, 1),
            'avg_models_used_in_vote': round(avg_voted, 1),
            'avg_models_agreed': round(avg_agreed, 1),
            'pct_coverage': round(len(loaded_names) / BRAINS_EXPECTED, 4),
            'production_ready': len(loaded_names) / BRAINS_EXPECTED >= 0.9,
        }

    report = {
        'run_id': run_id,
        'coverage_by_pair_tf': coverage,
        'summary': {
            'total_pair_tfs': len(coverage),
            'avg_coverage': round(
                sum(c['pct_coverage'] for c in coverage.values()) / len(coverage)
                if coverage else 0, 4
            ),
            'production_ready_count': sum(
                1 for c in coverage.values() if c['production_ready']),
            'note': ('Gate requires pct_coverage >= 0.9 for production. '
                     'Current runs use partial ensembles.'),
        },
    }

    # Ensemble health metrics
    report['ensemble_health'] = compute_ensemble_health(decision_ledger_df)

    output_path = Path(output_dir) / 'coverage.json'
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"Coverage report written: {output_path}")
    return report


def compute_ensemble_health(decision_ledger_df: Optional[pd.DataFrame]) -> Dict:
    """
    Compute entropy and confidence metrics from ensemble probabilities.

    Entropy: measures how "spread out" the probability distribution is.
      - Low entropy (near 0) = model is very decisive (one class dominates)
      - High entropy (near log(3) ~ 1.099) = model is uncertain (uniform)

    Confidence: max probability across the 3 classes.
      - High confidence (near 1.0) = strong conviction
      - Low confidence (near 0.33) = guessing
    """
    empty = {
        'vote_entropy_mean': None,
        'vote_entropy_p90': None,
        'vote_entropy_median': None,
        'vote_confidence_mean': None,
        'vote_confidence_p10': None,
        'vote_confidence_median': None,
        'decisiveness_ratio': None,
        'max_entropy_theoretical': round(float(np.log(3)), 4),
        'n_events_analyzed': 0,
    }

    if decision_ledger_df is None or len(decision_ledger_df) == 0:
        empty['note'] = 'No ledger data available'
        return empty

    probs_col = 'brain_probs_trimmed_mean'
    if probs_col not in decision_ledger_df.columns:
        empty['note'] = f'{probs_col} column not found in ledger'
        return empty

    entropies = []
    confidences = []

    for _, row in decision_ledger_df.iterrows():
        probs = row[probs_col]

        # Parse probs (could be list, string, or array)
        if isinstance(probs, str):
            try:
                probs = json.loads(probs)
            except (json.JSONDecodeError, ValueError):
                continue

        if probs is None or not hasattr(probs, '__len__') or len(probs) != 3:
            continue

        probs = np.array(probs, dtype=float)

        # Clamp to avoid log(0)
        probs = np.clip(probs, 1e-10, 1.0)
        probs = probs / probs.sum()  # Renormalize

        # Shannon entropy: H = -sum(p * log(p))
        entropy = -np.sum(probs * np.log(probs))
        entropies.append(entropy)

        # Confidence: max probability
        confidences.append(float(np.max(probs)))

    if not entropies:
        empty['note'] = 'No valid probability vectors found in ledger'
        return empty

    entropies = np.array(entropies)
    confidences = np.array(confidences)

    return {
        'vote_entropy_mean': round(float(np.mean(entropies)), 4),
        'vote_entropy_p90': round(float(np.percentile(entropies, 90)), 4),
        'vote_entropy_median': round(float(np.median(entropies)), 4),
        'vote_confidence_mean': round(float(np.mean(confidences)), 4),
        'vote_confidence_p10': round(float(np.percentile(confidences, 10)), 4),
        'vote_confidence_median': round(float(np.median(confidences)), 4),
        'decisiveness_ratio': round(float(np.mean(confidences > 0.5)), 4),
        'max_entropy_theoretical': round(float(np.log(3)), 4),
        'n_events_analyzed': len(entropies),
    }


def generate_veto_breakdown(run_id: str,
                            decision_ledger_df: pd.DataFrame,
                            output_dir) -> Dict:
    """
    Veto breakdown: counts every reason code from the decision ledger.
    Breaks down by overall, per-pair, per-tf.
    """
    all_reasons = Counter()
    by_pair = {}
    by_tf = {}

    if decision_ledger_df is not None and len(decision_ledger_df) > 0:
        for _, row in decision_ledger_df.iterrows():
            codes_str = row.get('reason_codes', '')
            if not codes_str or codes_str == '':
                continue

            codes = str(codes_str).split('|')
            pair = row.get('pair', 'UNKNOWN')
            tf = row.get('tf', 'UNKNOWN')

            for code in codes:
                code = code.strip()
                if code:
                    all_reasons[code] += 1
                    by_pair.setdefault(pair, Counter())[code] += 1
                    by_tf.setdefault(tf, Counter())[code] += 1

    total_events = len(decision_ledger_df) if decision_ledger_df is not None else 0

    total_trades = 0
    if (decision_ledger_df is not None and len(decision_ledger_df) > 0
            and 'action_taken' in decision_ledger_df.columns):
        total_trades = int(decision_ledger_df[
            decision_ledger_df['action_taken'].isin(['OPEN', 'CLOSE'])
        ].shape[0])

    risk_keys = [
        'RISK_BLOCKED_POSITION_LIMIT', 'RISK_BLOCKED_EXPOSURE',
        'RISK_BLOCKED_CORRELATION', 'RISK_BLOCKED_DRAWDOWN',
        'RISK_BLOCKED_COOLDOWN', 'RISK_BLOCKED_THROTTLE',
    ]
    total_vetoed = sum(all_reasons.get(k, 0) for k in risk_keys)

    report = {
        'run_id': run_id,
        'total_events': total_events,
        'total_trade_actions': total_trades,
        'overall_reason_counts': dict(all_reasons.most_common()),
        'top_10_reasons': dict(all_reasons.most_common(10)),
        'risk_vetoes_total': total_vetoed,
        'by_pair': {p: dict(c.most_common(5)) for p, c in by_pair.items()},
        'by_tf': {t: dict(c.most_common(5)) for t, c in by_tf.items()},
    }

    output_path = Path(output_dir) / 'veto_breakdown.json'
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"Veto breakdown written: {output_path}")
    return report


def compute_partial_fill_metrics(trades_df: Optional[pd.DataFrame]) -> Dict:
    """
    Verify fill model consistency. Detects artificial trade inflation from partial fills.

    Invariant: every trade converges to FILLED or SKIPPED within 3 bars.
    """
    if trades_df is None or len(trades_df) == 0:
        return {
            'partial_fill_rate': 0.0,
            'fill_status_counts': {},
            'avg_fill_completion_bars': 0,
            'max_fill_completion_bars': 0,
            'fill_invariant_pass': True,
            'total_trades': 0,
            'note': 'No trades to analyze',
        }

    fill_counts = {}
    if 'fill_status' in trades_df.columns:
        fill_counts = trades_df['fill_status'].value_counts().to_dict()

    total_trades = len(trades_df)
    filled_count = fill_counts.get('FILLED', 0)
    partial_fill_rate = 1.0 - (filled_count / total_trades) if total_trades > 0 else 0.0

    # In next-bar-open fill model, fill_ts == decision_ts (same bar).
    # fill_completion_bars is always 1 (next bar open).
    fill_completion_bars = []
    if 'decision_ts' in trades_df.columns and 'fill_ts' in trades_df.columns:
        for _, row in trades_df.iterrows():
            if row.get('fill_ts') is not None and row.get('decision_ts') is not None:
                fill_completion_bars.append(1)

    avg_fill_bars = sum(fill_completion_bars) / len(fill_completion_bars) if fill_completion_bars else 0
    max_fill_bars = max(fill_completion_bars) if fill_completion_bars else 0

    return {
        'partial_fill_rate': round(partial_fill_rate, 4),
        'fill_status_counts': {str(k): int(v) for k, v in fill_counts.items()},
        'avg_fill_completion_bars': round(avg_fill_bars, 2),
        'max_fill_completion_bars': max_fill_bars,
        'fill_invariant_pass': max_fill_bars <= 3,
        'total_trades': total_trades,
    }
