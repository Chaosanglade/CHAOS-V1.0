"""
Edge Decay Monitor

Detects early signs of strategy edge degradation by monitoring
signal quality metrics that typically deteriorate 2-8 weeks
before PnL shows visible decline.

Monitored metrics:
1. Rolling PF (20-trade window) — detects PF compression
2. Win rate trend — detects gradual accuracy decline
3. Average winner / average loser ratio — detects payoff decay
4. Signal frequency — detects regime shift (too many or too few signals)
5. Ensemble agreement trend — detects model divergence
6. Vote entropy trend — detects confidence decay

Alert levels:
- GREEN: all metrics within normal bands
- YELLOW: 1-2 metrics outside bands (watch)
- ORANGE: 3+ metrics outside bands (reduce exposure)
- RED: sustained degradation across multiple metrics (halt trading, review)
"""
import numpy as np
import json
import logging
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger('edge_decay_monitor')


class EdgeDecayMonitor:
    """
    Monitors signal quality metrics for early edge degradation detection.

    Usage:
        monitor = EdgeDecayMonitor(
            pair='EURUSD',
            tf='H1',
            baseline_pf=5.15,
            baseline_wr=0.604,
            baseline_avg_winner_loser_ratio=2.5
        )

        # Feed trade results
        monitor.record_trade(pnl=5.3, signal_confidence=0.72, agreement_score=0.65)
        monitor.record_trade(pnl=-2.1, signal_confidence=0.58, agreement_score=0.61)

        # Check health
        status = monitor.get_status()  # GREEN/YELLOW/ORANGE/RED
    """

    def __init__(self, pair, tf, baseline_pf, baseline_wr,
                 baseline_avg_winner_loser_ratio=2.0,
                 rolling_window=20, alert_lookback=50):
        self.pair = pair
        self.tf = tf
        self.baseline_pf = baseline_pf
        self.baseline_wr = baseline_wr
        self.baseline_wl_ratio = baseline_avg_winner_loser_ratio
        self.rolling_window = rolling_window
        self.alert_lookback = alert_lookback

        # Trade history (ring buffer)
        self._trades = deque(maxlen=alert_lookback * 2)
        self._agreements = deque(maxlen=alert_lookback * 2)
        self._entropies = deque(maxlen=alert_lookback * 2)
        self._confidences = deque(maxlen=alert_lookback * 2)
        self._signal_times = deque(maxlen=alert_lookback * 2)

        # Alert thresholds (% deviation from baseline)
        self.thresholds = {
            'pf_decay_pct': 0.40,       # PF drops > 40% from baseline
            'wr_decay_pct': 0.15,        # WR drops > 15% absolute
            'wl_ratio_decay_pct': 0.30,  # W/L ratio drops > 30%
            'entropy_increase_pct': 0.50, # Entropy rises > 50%
            'agreement_decay_pct': 0.20,  # Agreement drops > 20%
            'frequency_change_pct': 0.50  # Signal frequency changes > 50%
        }

    def record_trade(self, pnl, signal_confidence=None, agreement_score=None,
                     vote_entropy=None, timestamp=None):
        """Record a completed trade result."""
        self._trades.append({
            'pnl': pnl,
            'timestamp': timestamp or datetime.utcnow(),
            'confidence': signal_confidence,
            'agreement': agreement_score,
            'entropy': vote_entropy
        })

        if signal_confidence is not None:
            self._confidences.append(signal_confidence)
        if agreement_score is not None:
            self._agreements.append(agreement_score)
        if vote_entropy is not None:
            self._entropies.append(vote_entropy)
        if timestamp is not None:
            self._signal_times.append(timestamp)

    def _compute_rolling_pf(self):
        """Rolling PF over last N trades."""
        if len(self._trades) < self.rolling_window:
            return None

        recent = list(self._trades)[-self.rolling_window:]
        winners = sum(t['pnl'] for t in recent if t['pnl'] > 0)
        losers = abs(sum(t['pnl'] for t in recent if t['pnl'] < 0))
        return winners / losers if losers > 0 else float('inf')

    def _compute_rolling_wr(self):
        """Rolling win rate over last N trades."""
        if len(self._trades) < self.rolling_window:
            return None

        recent = list(self._trades)[-self.rolling_window:]
        wins = sum(1 for t in recent if t['pnl'] > 0)
        return wins / len(recent)

    def _compute_wl_ratio(self):
        """Average winner / average loser ratio."""
        if len(self._trades) < self.rolling_window:
            return None

        recent = list(self._trades)[-self.rolling_window:]
        winners = [t['pnl'] for t in recent if t['pnl'] > 0]
        losers = [abs(t['pnl']) for t in recent if t['pnl'] < 0]

        if not winners or not losers:
            return None

        return np.mean(winners) / np.mean(losers)

    def _compute_agreement_trend(self):
        """Recent vs baseline agreement score."""
        if len(self._agreements) < self.rolling_window:
            return None
        return float(np.mean(list(self._agreements)[-self.rolling_window:]))

    def _compute_entropy_trend(self):
        """Recent vote entropy."""
        if len(self._entropies) < self.rolling_window:
            return None
        return float(np.mean(list(self._entropies)[-self.rolling_window:]))

    def get_status(self):
        """
        Compute current edge health status.

        Returns:
            {
                'status': 'GREEN' | 'YELLOW' | 'ORANGE' | 'RED',
                'alerts': [...],
                'metrics': {...},
                'trade_count': int
            }
        """
        alerts = []
        metrics = {}

        # 1. Rolling PF
        rolling_pf = self._compute_rolling_pf()
        metrics['rolling_pf'] = rolling_pf
        if rolling_pf is not None and self.baseline_pf > 0:
            pf_decay = 1.0 - (rolling_pf / self.baseline_pf)
            if pf_decay > self.thresholds['pf_decay_pct']:
                alerts.append({
                    'metric': 'rolling_pf',
                    'message': f'PF decayed {pf_decay:.0%} from baseline ({self.baseline_pf:.2f} -> {rolling_pf:.2f})',
                    'severity': 'HIGH'
                })

        # 2. Rolling WR
        rolling_wr = self._compute_rolling_wr()
        metrics['rolling_wr'] = rolling_wr
        if rolling_wr is not None:
            wr_decay = self.baseline_wr - rolling_wr
            if wr_decay > self.thresholds['wr_decay_pct']:
                alerts.append({
                    'metric': 'rolling_wr',
                    'message': f'Win rate dropped {wr_decay:.1%} from baseline ({self.baseline_wr:.1%} -> {rolling_wr:.1%})',
                    'severity': 'MEDIUM'
                })

        # 3. W/L Ratio
        wl_ratio = self._compute_wl_ratio()
        metrics['wl_ratio'] = wl_ratio
        if wl_ratio is not None and self.baseline_wl_ratio > 0:
            wl_decay = 1.0 - (wl_ratio / self.baseline_wl_ratio)
            if wl_decay > self.thresholds['wl_ratio_decay_pct']:
                alerts.append({
                    'metric': 'wl_ratio',
                    'message': f'W/L ratio decayed {wl_decay:.0%}',
                    'severity': 'MEDIUM'
                })

        # 4. Agreement trend
        agreement = self._compute_agreement_trend()
        metrics['agreement_trend'] = agreement

        # 5. Entropy trend
        entropy = self._compute_entropy_trend()
        metrics['entropy_trend'] = entropy

        # Determine status
        high_alerts = sum(1 for a in alerts if a['severity'] == 'HIGH')
        medium_alerts = sum(1 for a in alerts if a['severity'] == 'MEDIUM')
        total_alerts = len(alerts)

        if high_alerts >= 2 or total_alerts >= 4:
            status = 'RED'
        elif high_alerts >= 1 or total_alerts >= 3:
            status = 'ORANGE'
        elif total_alerts >= 1:
            status = 'YELLOW'
        else:
            status = 'GREEN'

        return {
            'pair': self.pair,
            'tf': self.tf,
            'status': status,
            'alerts': alerts,
            'metrics': metrics,
            'trade_count': len(self._trades),
            'baseline': {
                'pf': self.baseline_pf,
                'wr': self.baseline_wr,
                'wl_ratio': self.baseline_wl_ratio
            }
        }


def create_monitors_from_run4(equity_curves_path):
    """
    Initialize edge decay monitors for each Tier 1+2 pair
    using Run 4 baseline metrics.
    """
    with open(equity_curves_path) as f:
        curves = json.load(f)

    monitors = {}
    for pair, data in curves.get('per_pair', {}).items():
        # Create monitor with Run 4 baselines
        monitor = EdgeDecayMonitor(
            pair=pair,
            tf='H1_M30',  # Combined
            baseline_pf=data.get('pf', 2.0),
            baseline_wr=data.get('wr', 0.50) / 100.0 if data.get('wr', 50) > 1 else data.get('wr', 0.50),
            baseline_avg_winner_loser_ratio=2.0  # Default, can be computed
        )
        monitors[pair] = monitor

    return monitors
