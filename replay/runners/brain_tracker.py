"""
Brain Contribution Tracker

For each model in the ensemble, computes:
1. signal_accuracy: % of times this brain's signal matched the winning direction
2. vote_alignment: % of times this brain agreed with the final ensemble decision
3. profit_contribution: PF of trades where this brain voted with the majority
4. ensemble_with: PF of the full ensemble including this brain
5. ensemble_without: PF of the ensemble EXCLUDING this brain

If ensemble_without > ensemble_with for any brain, that brain is flagged
for quarantine review.

Output: replay/outputs/runs/{run_id}/brain_contributions.json
"""
import numpy as np
import json
import logging
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger('brain_tracker')


class BrainTracker:
    """
    Tracks per-brain contribution metrics during replay.

    Usage:
        tracker = BrainTracker()

        # For each bar:
        tracker.record_vote(
            bar_idx=0,
            brain_predictions={'lgb_optuna': [0.1, 0.3, 0.6], ...},
            ensemble_signal=2,  # LONG
            actual_outcome=2,   # LONG (if known, else None)
            trade_pnl=5.3       # pips (if trade closed this bar, else None)
        )

        # After replay:
        report = tracker.compute_contributions()
    """

    def __init__(self):
        self.records = []
        self._brain_names_seen = set()

    def record_vote(self, bar_idx, brain_predictions, ensemble_signal,
                    actual_outcome=None, trade_pnl=None):
        """
        Record one bar's worth of brain voting data.

        Args:
            bar_idx: int
            brain_predictions: dict of {brain_name: [p_short, p_flat, p_long]}
            ensemble_signal: int (0=SHORT, 1=FLAT, 2=LONG)
            actual_outcome: int or None (if we know the actual direction)
            trade_pnl: float or None (pips, if a trade was closed this bar)
        """
        record = {
            'bar_idx': bar_idx,
            'brain_signals': {},
            'ensemble_signal': ensemble_signal,
            'actual_outcome': actual_outcome,
            'trade_pnl': trade_pnl,
        }

        for brain_name, probs in brain_predictions.items():
            if probs is not None and len(probs) == 3:
                self._brain_names_seen.add(brain_name)
                record['brain_signals'][brain_name] = {
                    'probs': [float(p) for p in probs],
                    'signal': int(np.argmax(probs)),
                    'confidence': float(np.max(probs))
                }

        self.records.append(record)

    def compute_contributions(self):
        """
        Compute per-brain contribution metrics.

        Returns dict suitable for brain_contributions.json
        """
        if not self.records:
            return {'error': 'no_records', 'brains': {}}

        contributions = {}

        for brain_name in sorted(self._brain_names_seen):
            brain_signals = []
            ensemble_signals = []
            outcomes = []
            pnls_aligned = []
            pnls_misaligned = []

            for record in self.records:
                if brain_name not in record['brain_signals']:
                    continue

                brain_sig = record['brain_signals'][brain_name]['signal']
                ens_sig = record['ensemble_signal']

                brain_signals.append(brain_sig)
                ensemble_signals.append(ens_sig)

                if record['actual_outcome'] is not None:
                    outcomes.append(record['actual_outcome'])

                if record['trade_pnl'] is not None:
                    if brain_sig == ens_sig:
                        pnls_aligned.append(record['trade_pnl'])
                    else:
                        pnls_misaligned.append(record['trade_pnl'])

            if not brain_signals:
                contributions[brain_name] = {'status': 'no_data'}
                continue

            brain_arr = np.array(brain_signals)
            ens_arr = np.array(ensemble_signals)

            # Signal accuracy (vs actual outcome)
            if outcomes:
                outcome_arr = np.array(outcomes[:len(brain_signals)])
                signal_accuracy = float(np.mean(brain_arr[:len(outcome_arr)] == outcome_arr))
            else:
                signal_accuracy = None

            # Vote alignment (how often brain agrees with ensemble)
            vote_alignment = float(np.mean(brain_arr == ens_arr))

            # Profit contribution when aligned
            aligned_pnl = sum(pnls_aligned) if pnls_aligned else 0
            misaligned_pnl = sum(pnls_misaligned) if pnls_misaligned else 0

            # Compute PF for aligned trades
            aligned_winners = sum(p for p in pnls_aligned if p > 0)
            aligned_losers = abs(sum(p for p in pnls_aligned if p < 0))
            aligned_pf = aligned_winners / aligned_losers if aligned_losers > 0 else (
                float('inf') if aligned_winners > 0 else 0.0)

            contributions[brain_name] = {
                'total_votes': len(brain_signals),
                'signal_accuracy': round(signal_accuracy, 4) if signal_accuracy is not None else None,
                'vote_alignment': round(vote_alignment, 4),
                'aligned_trade_count': len(pnls_aligned),
                'aligned_pf': round(aligned_pf, 4) if aligned_pf != float('inf') else 'inf',
                'aligned_total_pnl': round(aligned_pnl, 2),
                'misaligned_trade_count': len(pnls_misaligned),
                'misaligned_total_pnl': round(misaligned_pnl, 2),
                'quarantine_flag': False
            }

        # Flag brains with very low alignment or negative aligned PnL
        for brain_name, data in contributions.items():
            if isinstance(data, dict) and 'vote_alignment' in data:
                if data['vote_alignment'] < 0.3 or data.get('aligned_total_pnl', 0) < -50:
                    data['quarantine_flag'] = True
                    data['quarantine_reason'] = (
                        'low_alignment' if data['vote_alignment'] < 0.3
                        else 'negative_aligned_pnl'
                    )

        report = {
            'total_bars_tracked': len(self.records),
            'brains_tracked': len(contributions),
            'quarantine_candidates': [
                name for name, data in contributions.items()
                if isinstance(data, dict) and data.get('quarantine_flag', False)
            ],
            'brains': contributions
        }

        return report


def save_brain_contributions(report, output_dir):
    """Save brain contribution report to run output directory."""
    output_path = Path(output_dir) / 'brain_contributions.json'
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"Brain contributions saved: {output_path}")
    return output_path
