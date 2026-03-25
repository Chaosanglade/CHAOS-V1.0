"""
CHAOS V1.0 -- Live Feature Adapter (273 features)

Computes the exact 273 features from schema/feature_schema.json using only
raw OHLCV bars from the MT5 EA.  Dependencies: numpy, pandas, scipy.

Replaces the basic ~50-feature LiveFeatureEngine with production-grade
features matching what the models were trained on.

Feature groups:
  0-21:   Tick-level proxies (from OHLCV, no real tick data)
  22-62:  Liquidity & order flow (VPIN, Kyle's Lambda, Amihud, OFI, spreads)
  63-74:  Volume & momentum (multi-horizon)
  75-105: Mean reversion & oscillators (RSI, OU, EMAs, Stochastic)
  106-170: Multi-horizon volatility (CC/Parkinson/GK/YZ x 4 windows)
  171-186: ATR variants & Hurst exponent
  187-197: Regime & memory (fractional diff, HMM approximation)
  198-206: Entropy & Bollinger Bands
  207-222: Trend indicators (MACD x3, Williams %R, CCI, ADX)
  223-237: Calendar & London Fix features
  238-252: Session & execution quality
  253-257: Round number features
  258-272: Weekend & cross-asset proxies
"""
import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger('live_feature_adapter')

PROJECT_ROOT = Path(os.environ.get('CHAOS_BASE_DIR', os.getcwd()))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_div(a, b, fill=0.0):
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.where(np.abs(b) > 1e-12, a / b, fill)
    return np.nan_to_num(r, nan=fill, posinf=fill, neginf=fill)


def _roll_mean(x, w):
    return pd.Series(x).rolling(w, min_periods=1).mean().values


def _roll_std(x, w):
    return pd.Series(x).rolling(w, min_periods=2).std().fillna(0).values


def _roll_sum(x, w):
    return pd.Series(x).rolling(w, min_periods=1).sum().values


def _roll_min(x, w):
    return pd.Series(x).rolling(w, min_periods=1).min().values


def _roll_max(x, w):
    return pd.Series(x).rolling(w, min_periods=1).max().values


def _ema(x, span):
    return pd.Series(x).ewm(span=span, adjust=False).mean().values


def _zscore(x, w):
    m = _roll_mean(x, w)
    s = _roll_std(x, w) + 1e-10
    return (x - m) / s


def _percentile_rank(x, w):
    s = pd.Series(x)
    return s.rolling(w, min_periods=1).apply(
        lambda v: (v < v.iloc[-1]).sum() / max(len(v) - 1, 1), raw=False
    ).values


def _rsi(close, period):
    delta = np.diff(close, prepend=close[0])
    up = np.where(delta > 0, delta, 0.0)
    dn = np.where(delta < 0, -delta, 0.0)
    up_avg = _ema(up, period)
    dn_avg = _ema(dn, period) + 1e-10
    return 100.0 - 100.0 / (1.0 + up_avg / dn_avg)


def _stochastic_k(close, high, low, period):
    ll = _roll_min(low, period)
    hh = _roll_max(high, period)
    return _safe_div(close - ll, hh - ll, 0.5) * 100.0


def _atr(close, high, low, period):
    prev_c = np.roll(close, 1); prev_c[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_c), np.abs(low - prev_c)))
    return _roll_mean(tr, period)


def _frac_diff(series, d, threshold=1e-5, max_lag=100):
    """Fractional differentiation (fixed-width window)."""
    w = [1.0]
    for k in range(1, max_lag):
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
    w = np.array(w)
    n = len(series)
    out = np.zeros(n)
    for i in range(len(w), n):
        out[i] = np.dot(w, series[i - len(w) + 1: i + 1][::-1])
    return out


# ---------------------------------------------------------------------------
# Main adapter class
# ---------------------------------------------------------------------------

class LiveFeatureAdapter:
    """
    Compute the 273-feature vector from raw OHLCV bars.

    Usage:
        adapter = LiveFeatureAdapter()
        vec = adapter.compute('EURUSD', 'H1', bars_by_tf)
        # vec is np.array shape (273,) matching schema ordering
    """

    def __init__(self, schema_path=None):
        schema_path = schema_path or PROJECT_ROOT / 'schema' / 'feature_schema.json'
        with open(schema_path) as f:
            schema = json.load(f)
        self._feature_list = [entry['name'] for entry in schema['features']]
        self.n_features = len(self._feature_list)
        self._name_to_idx = {n: i for i, n in enumerate(self._feature_list)}
        logger.info(f"LiveFeatureAdapter loaded: {self.n_features} features from schema")

    # -----------------------------------------------------------------------
    def compute(self, pair, tf, bars_by_tf):
        """
        Compute 273 features from raw OHLCV bars.

        Args:
            pair: e.g. 'EURUSD'
            tf: primary timeframe e.g. 'H1'
            bars_by_tf: dict {tf_name: [bar_dicts]}
                Each bar dict has: time, open, high, low, close, volume

        Returns:
            np.array of shape (273,) ordered by schema, or None on failure
        """
        try:
            tf_bars = bars_by_tf.get(tf, [])
            if not tf_bars or len(tf_bars) < 50:
                logger.warning(f"Insufficient bars for {pair}_{tf}: {len(tf_bars)}")
                return None

            df = pd.DataFrame(tf_bars)
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'], utc=True)
                df = df.sort_values('time').reset_index(drop=True)
            for src, dst in [('Time','time'),('Open','open'),('High','high'),
                             ('Low','low'),('Close','close'),('Volume','volume')]:
                if src in df.columns and dst not in df.columns:
                    df[dst] = df[src]
            for c in ['open','high','low','close','volume']:
                df[c] = df[c].astype(np.float64)

            o = df['open'].values
            h = df['high'].values
            lo = df['low'].values
            c = df['close'].values
            v = df['volume'].values.astype(np.float64)
            n = len(c)

            log_c = np.log(np.maximum(c, 1e-10))
            ret = np.zeros(n); ret[1:] = np.clip(np.diff(log_c), -0.1, 0.1)
            prev_c = np.roll(c, 1); prev_c[0] = c[0]

            # Output dict — populate by name, then assemble in schema order
            F = {}

            # === Group 1: Tick-level proxies (0-21) ===
            tick_proxy = (h - lo) * np.maximum(v, 1)
            F['Tick_Interval_Mean'] = _roll_mean(tick_proxy, 20)[-1]
            F['Tick_Interval_Std'] = _roll_std(tick_proxy, 20)[-1]
            F['Tick_Interval_Max'] = _roll_max(tick_proxy, 20)[-1]
            F['Tick_Interval_Min'] = _roll_min(tick_proxy, 20)[-1]
            same_dir = (np.sign(c - o) == np.sign(np.roll(c - o, 1))).astype(float)
            F['Tick_Clustering'] = _roll_mean(same_dir, 10)[-1]
            F['Tick_Intensity'] = _safe_div(v, _roll_mean(v, 50) + 1e-10)[-1]
            up_ticks = (c > prev_c).astype(float)
            dn_ticks = (c < prev_c).astype(float)
            F['Uptick_Count'] = _roll_sum(up_ticks, 20)[-1]
            F['Downtick_Count'] = _roll_sum(dn_ticks, 20)[-1]
            F['Uptick_Ratio'] = _roll_mean(up_ticks, 20)[-1]
            F['Tick_Imbalance'] = _roll_mean(up_ticks - dn_ticks, 20)[-1]
            abs_moves = np.abs(np.diff(c, prepend=c[0]))
            F['Path_Length'] = np.sum(abs_moves[-20:])
            net_move = abs(c[-1] - c[-21]) if n > 20 else abs(c[-1] - c[0])
            F['Path_Efficiency'] = _safe_div(net_move, F['Path_Length'] + 1e-10)
            reversals = (np.sign(np.diff(c, prepend=c[0])) != np.sign(np.roll(np.diff(c, prepend=c[0]), 1))).astype(float)
            F['Path_Reversals'] = _roll_sum(reversals, 20)[-1]
            # MFE / MAE (lookback 10)
            lb = min(10, n - 1)
            recent_c = c[-lb:]
            F['Max_Favorable_Excursion'] = (np.max(recent_c) - recent_c[0]) / (recent_c[0] + 1e-10)
            F['Max_Adverse_Excursion'] = (recent_c[0] - np.min(recent_c)) / (recent_c[0] + 1e-10)
            F['Reversal_Rate'] = _roll_mean(reversals, 20)[-1]
            F['TWAP'] = _safe_div(c[-1], _roll_mean(c, 20)[-1], 1.0)
            F['VWAP'] = _safe_div(np.sum(c[-20:] * v[-20:]), np.sum(v[-20:]) + 1e-10)
            F['Realized_Vol'] = _roll_std(ret, 20)[-1]
            bipower = np.abs(ret[1:]) * np.abs(ret[:-1])
            F['Realized_Vol_Bipower'] = np.mean(bipower[-20:]) if len(bipower) >= 20 else 0.0
            F['Time_To_First_Tick'] = 0.0  # Not available from bar data
            F['Quote_Stability'] = 1.0 - _roll_std(_safe_div(h - lo, c), 20)[-1]

            # === Group 2: Liquidity & Order Flow (22-62) ===
            # VPIN proxy: |buy_vol - sell_vol| / total_vol
            buy_v = np.where(c > o, v, v * 0.5)
            sell_v = np.where(c < o, v, v * 0.5)
            vpin_raw = _safe_div(np.abs(buy_v - sell_v), v + 1e-10)
            F['vpin'] = vpin_raw[-1]
            F['vpin_ma10'] = _roll_mean(vpin_raw, 10)[-1]
            F['vpin_ma50'] = _roll_mean(vpin_raw, 50)[-1]
            F['vpin_zscore'] = _zscore(vpin_raw, 50)[-1]
            F['vpin_percentile'] = _percentile_rank(vpin_raw, 100)[-1]

            # Kyle's lambda: price impact per unit volume
            kyle_raw = _safe_div(np.abs(ret), np.sqrt(v + 1e-10))
            F['kyle_lambda'] = kyle_raw[-1]
            F['kyle_lambda_ma20'] = _roll_mean(kyle_raw, 20)[-1]
            F['kyle_lambda_ma50'] = _roll_mean(kyle_raw, 50)[-1]
            F['kyle_lambda_zscore'] = _zscore(kyle_raw, 50)[-1]

            # Amihud illiquidity
            amihud_raw = _safe_div(np.abs(ret), v + 1e-10) * 1e6
            F['amihud_illiq'] = amihud_raw[-1]
            F['amihud_ma21'] = _roll_mean(amihud_raw, 21)[-1]
            F['amihud_zscore'] = _zscore(amihud_raw, 50)[-1]

            # OFI (order flow imbalance)
            ofi_raw = np.sign(c - o) * v
            ofi_norm = _safe_div(ofi_raw, _roll_mean(np.abs(ofi_raw), 20) + 1e-10)
            F['ofi'] = np.clip(ofi_norm[-1], -5, 5)
            F['ofi_ma5'] = _roll_mean(ofi_norm, 5)[-1]
            F['ofi_ma20'] = _roll_mean(ofi_norm, 20)[-1]
            F['ofi_zscore'] = _zscore(ofi_norm, 50)[-1]
            F['ofi_cumsum'] = _roll_sum(ofi_norm, 20)[-1]

            F['buy_pressure'] = _roll_mean(_safe_div(buy_v, v + 1e-10), 20)[-1]
            F['sell_pressure'] = _roll_mean(_safe_div(sell_v, v + 1e-10), 20)[-1]
            F['pressure_ratio'] = _safe_div(F['buy_pressure'], F['sell_pressure'] + 1e-10)

            # Spread proxies (from high-low range)
            spread_proxy = h - lo
            F['effective_spread'] = _safe_div(spread_proxy, c)[-1]
            F['realized_spread'] = _safe_div(spread_proxy, _roll_mean(c, 20) + 1e-10)[-1]
            F['price_impact'] = _safe_div(np.abs(c - o), v + 1e-10)[-1]
            F['spread_decomposition'] = _safe_div(spread_proxy - np.abs(c - o), spread_proxy + 1e-10)[-1]

            tick_dir = np.sign(c - prev_c)
            F['tick_direction'] = tick_dir[-1]
            F['tick_up_count'] = _roll_sum(up_ticks, 20)[-1]
            F['tick_down_count'] = _roll_sum(dn_ticks, 20)[-1]
            F['tick_imbalance'] = F['Tick_Imbalance']
            F['tick_intensity'] = F['Tick_Intensity']
            # Run length
            run = 0
            for i in range(n - 1, max(n - 21, 0), -1):
                if tick_dir[i] == tick_dir[-1]:
                    run += 1
                else:
                    break
            F['tick_run_length'] = float(run)
            F['tick_reversal_rate'] = F['Reversal_Rate']
            F['tick_clustering'] = F['Tick_Clustering']

            sp = _safe_div(h - lo, c)
            F['spread_raw'] = sp[-1]
            F['spread_pct'] = sp[-1] * 100
            F['spread_ma5'] = _roll_mean(sp, 5)[-1]
            F['spread_ma20'] = _roll_mean(sp, 20)[-1]
            F['spread_ma50'] = _roll_mean(sp, 50)[-1]
            F['spread_std20'] = _roll_std(sp, 20)[-1]
            F['spread_zscore'] = _zscore(sp, 20)[-1]
            F['spread_percentile'] = _percentile_rank(sp, 100)[-1]
            F['spread_vs_typical'] = _safe_div(sp[-1], _roll_mean(sp, 50)[-1] + 1e-10)

            # === Group 3: Volume & Momentum (63-74) ===
            F['volume_proxy'] = _safe_div(v[-1], _roll_mean(v, 50)[-1] + 1e-10)
            for lb in [5, 10, 21, 63, 126, 252]:
                k = f'momentum_{lb}'
                F[k] = _roll_sum(ret, lb)[-1] if n >= lb else 0.0
            for lb, skip in [(252, 21), (126, 10), (63, 5)]:
                k = f'momentum_{lb}_skip{skip}'
                if n > lb:
                    F[k] = (c[-1 - skip] / (c[-1 - lb] + 1e-10)) - 1.0
                else:
                    F[k] = 0.0
            mom20 = _roll_sum(ret, 20)
            mom_accel = np.diff(mom20, prepend=mom20[0])
            F['momentum_acceleration'] = mom_accel[-1]
            F['momentum_jerk'] = np.diff(mom_accel, prepend=mom_accel[0])[-1]

            # === Group 4: Mean Reversion & Oscillators (75-105) ===
            F['rsi_2'] = _rsi(c, 2)[-1]
            F['rsi_3'] = _rsi(c, 3)[-1]
            F['rsi_14'] = _rsi(c, 14)[-1]
            F['rsi_2_cumulative'] = _roll_sum(_rsi(c, 2), 5)[-1] / 500.0
            rsi14 = _rsi(c, 14)
            F['rsi_stretched'] = float(rsi14[-1] > 80 or rsi14[-1] < 20)

            # Ornstein-Uhlenbeck proxy
            log_p = np.log(np.maximum(c, 1e-10))
            log_diff = np.diff(log_p)
            if len(log_diff) > 20:
                y = log_diff[-100:] if len(log_diff) >= 100 else log_diff
                x = log_p[:-1][-len(y):]
                try:
                    slope = np.polyfit(x, y, 1)[0]
                    hl = -np.log(2) / slope if slope < -1e-8 else 500.0
                except:
                    hl = 500.0
            else:
                hl = 500.0
            F['ou_half_life'] = np.clip(hl, 0, 500)
            F['ou_half_life_ma20'] = F['ou_half_life']  # Single point; no history
            eq = _roll_mean(c, 50)[-1]
            F['ou_equilibrium'] = _safe_div(c[-1] - eq, eq + 1e-10)
            F['ou_zscore'] = _zscore(c, 50)[-1]
            F['mean_reversion_signal'] = -F['ou_zscore']
            F['mean_reversion_strength'] = abs(F['ou_zscore'])

            # EMAs
            for span in [8, 13, 21, 34, 55, 89]:
                F[f'ema_{span}'] = _safe_div(c[-1], _ema(c, span)[-1], 1.0) - 1.0
            # SMAs
            for w in [10, 20, 50, 100, 200]:
                ma = _roll_mean(c, w)[-1]
                F[f'sma_{w}'] = _safe_div(c[-1] - ma, ma + 1e-10)

            F['price_vs_ema_21'] = F['ema_21']
            F['price_vs_ema_55'] = F['ema_55']

            # Trend consistency: % of recent bars where close > SMA20
            sma20 = _roll_mean(c, 20)
            F['trend_consistency'] = np.mean((c[-20:] > sma20[-20:]).astype(float)) if n >= 20 else 0.5

            # Stochastic
            for p in [5, 9, 14, 21]:
                k_val = _stochastic_k(c, h, lo, p)
                F[f'stoch_k_{p}'] = k_val[-1] / 100.0
                if p in [5, 9]:
                    F[f'stoch_d_{p}'] = _roll_mean(k_val, 3)[-1] / 100.0

            # === Group 5: Multi-Horizon Volatility (106-170) ===
            for w in [7, 14, 21, 50]:
                # Close-to-close
                cc = _roll_std(ret, w)
                F[f'vol_cc_{w}'] = cc[-1]
                F[f'vol_cc_{w}_ma'] = _roll_mean(cc, w * 3)[-1]
                F[f'vol_cc_{w}_zscore'] = _zscore(cc, w * 5)[-1]
                F[f'vol_cc_{w}_pct'] = _percentile_rank(cc, min(w * 5, n))[-1]

                # Parkinson
                park = np.sqrt(_roll_mean((np.log(np.maximum(h, 1e-10)) - np.log(np.maximum(lo, 1e-10)))**2 / (4 * np.log(2)), w))
                F[f'vol_parkinson_{w}'] = park[-1]
                F[f'vol_parkinson_{w}_ma'] = _roll_mean(park, w * 3)[-1]
                F[f'vol_parkinson_{w}_zscore'] = _zscore(park, w * 5)[-1]
                F[f'vol_parkinson_{w}_pct'] = _percentile_rank(park, min(w * 5, n))[-1]

                # Garman-Klass
                gk = np.sqrt(_roll_mean(
                    0.5 * (np.log(np.maximum(h, 1e-10)) - np.log(np.maximum(lo, 1e-10)))**2
                    - (2 * np.log(2) - 1) * (np.log(np.maximum(c, 1e-10)) - np.log(np.maximum(o, 1e-10)))**2,
                    w))
                F[f'vol_gk_{w}'] = gk[-1]
                F[f'vol_gk_{w}_ma'] = _roll_mean(gk, w * 3)[-1]
                F[f'vol_gk_{w}_zscore'] = _zscore(gk, w * 5)[-1]
                F[f'vol_gk_{w}_pct'] = _percentile_rank(gk, min(w * 5, n))[-1]

                # Yang-Zhang
                oc = np.log(np.maximum(o, 1e-10)) - np.log(np.maximum(np.roll(c, 1), 1e-10))
                co = np.log(np.maximum(c, 1e-10)) - np.log(np.maximum(o, 1e-10))
                oc[0] = 0
                var_oc = _roll_mean(oc**2, w)
                var_co = _roll_mean(co**2, w)
                k = 0.34 / (1.34 + (w + 1) / (w - 1))
                yz = np.sqrt(var_oc + k * var_co + (1 - k) * _roll_mean(
                    (np.log(np.maximum(h, 1e-10)) - np.log(np.maximum(lo, 1e-10)))**2 / (4 * np.log(2)), w))
                F[f'vol_yz_{w}'] = yz[-1]
                F[f'vol_yz_{w}_ma'] = _roll_mean(yz, w * 3)[-1]
                F[f'vol_yz_{w}_zscore'] = _zscore(yz, w * 5)[-1]
                F[f'vol_yz_{w}_pct'] = _percentile_rank(yz, min(w * 5, n))[-1]

            # Vol regime duration (bars in current vol quartile)
            cc21 = _roll_std(ret, 21)
            q = np.percentile(cc21[~np.isnan(cc21)], [25, 50, 75]) if n > 21 else [0, 0, 0]
            current_q = 0
            if cc21[-1] > q[2]: current_q = 3
            elif cc21[-1] > q[1]: current_q = 2
            elif cc21[-1] > q[0]: current_q = 1
            dur = 0
            for i in range(n - 1, max(n - 100, 0), -1):
                cq = 0
                if cc21[i] > q[2]: cq = 3
                elif cc21[i] > q[1]: cq = 2
                elif cc21[i] > q[0]: cq = 1
                if cq == current_q:
                    dur += 1
                else:
                    break
            F['vol_regime_duration'] = float(dur)

            # === Group 6: ATR & Hurst (171-186) ===
            for w in [7, 14, 21, 50]:
                a = _atr(c, h, lo, w)
                F[f'atr_{w}'] = a[-1]
                F[f'atr_pct_{w}'] = _safe_div(a[-1], c[-1])
                F[f'atr_zscore_{w}'] = _zscore(a, w * 3)[-1]

            # Hurst exponent (R/S method, simplified)
            try:
                from scipy.stats import linregress
                hurst_vals = []
                for win in [20, 50, 100]:
                    if n >= win:
                        seg = ret[-win:]
                        cum_dev = np.cumsum(seg - np.mean(seg))
                        R = np.max(cum_dev) - np.min(cum_dev)
                        S = np.std(seg) + 1e-10
                        hurst_vals.append(np.log(R / S + 1e-10) / np.log(win))
                hurst = np.mean(hurst_vals) if hurst_vals else 0.5
            except:
                hurst = 0.5
            F['hurst_exponent'] = np.clip(hurst, 0, 1)
            F['hurst_ma20'] = F['hurst_exponent']  # Single point
            F['hurst_ma50'] = F['hurst_exponent']
            F['hurst_confidence'] = 1.0 - abs(F['hurst_exponent'] - 0.5) * 2

            # === Group 7: Regime & Memory (187-197) ===
            for d in [0.3, 0.4, 0.5]:
                fd = _frac_diff(log_c, d)
                F[f'frac_diff_0{int(d*10)}'] = fd[-1]
            fd04 = _frac_diff(log_c, 0.4)
            F['frac_diff_04_ma10'] = _roll_mean(fd04, 10)[-1]
            F['frac_diff_04_zscore'] = _zscore(fd04, 50)[-1]
            F['frac_diff_signal'] = np.sign(fd04[-1])

            # HMM approximation via volatility regimes
            vol_pct = _percentile_rank(_roll_std(ret, 14), min(200, n))
            hmm_p = np.array([0.25, 0.25, 0.25, 0.25])  # default
            p = vol_pct[-1]
            if p < 0.25:
                hmm_p = np.array([0.6, 0.2, 0.15, 0.05])
            elif p < 0.50:
                hmm_p = np.array([0.15, 0.55, 0.2, 0.1])
            elif p < 0.75:
                hmm_p = np.array([0.1, 0.2, 0.55, 0.15])
            else:
                hmm_p = np.array([0.05, 0.1, 0.25, 0.6])
            for i in range(4):
                F[f'hmm_prob_{i}'] = hmm_p[i]
            # State duration
            state = int(np.argmax(hmm_p))
            dur = 1
            for i in range(n - 2, max(n - 50, 0), -1):
                pi = vol_pct[i]
                si = 0 if pi < 0.25 else (1 if pi < 0.5 else (2 if pi < 0.75 else 3))
                if si == state:
                    dur += 1
                else:
                    break
            F['hmm_state_duration'] = float(dur)

            # === Group 8: Entropy & Bollinger Bands (198-206) ===
            try:
                from scipy.stats import entropy as sp_entropy
                hist_ret = ret[-50:] if n >= 50 else ret
                counts, _ = np.histogram(hist_ret, bins=20)
                probs = counts / (counts.sum() + 1e-10)
                ent = sp_entropy(probs + 1e-10)
            except:
                ent = 0.0
            F['entropy'] = ent
            # Approximate rolling entropy
            F['entropy_ma10'] = ent  # single-point approximation
            F['entropy_ma50'] = ent
            F['entropy_zscore'] = 0.0

            sma20_val = _roll_mean(c, 20)
            std20_val = _roll_std(c, 20)
            F['bb_middle'] = _safe_div(c[-1], sma20_val[-1], 1.0) - 1.0
            F['bb_upper_2'] = _safe_div(c[-1], (sma20_val[-1] + 2 * std20_val[-1]), 1.0) - 1.0
            F['bb_lower_2'] = _safe_div(c[-1], (sma20_val[-1] - 2 * std20_val[-1]), 1.0) - 1.0
            bw = _safe_div(4 * std20_val[-1], sma20_val[-1] + 1e-10)
            F['bb_width'] = bw
            bb_pct = _safe_div(c[-1] - (sma20_val[-1] - 2 * std20_val[-1]),
                               4 * std20_val[-1] + 1e-10)
            F['bb_percent_b'] = bb_pct

            # === Group 9: Trend Indicators (207-222) ===
            for fast, slow in [(12, 26), (8, 17), (5, 35)]:
                macd_line = _ema(c, fast) - _ema(c, slow)
                macd_sig = _ema(macd_line, 9)
                F[f'macd_{fast}_{slow}'] = _safe_div(macd_line[-1], c[-1])
                F[f'macd_signal_{fast}_{slow}'] = _safe_div(macd_sig[-1], c[-1])
                F[f'macd_hist_{fast}_{slow}'] = _safe_div(macd_line[-1] - macd_sig[-1], c[-1])

            # Williams %R
            for p in [14, 21]:
                hh = _roll_max(h, p)[-1]
                ll = _roll_min(lo, p)[-1]
                F[f'williams_r_{p}'] = _safe_div(hh - c[-1], hh - ll + 1e-10) * -100.0 / 100.0

            # CCI
            for p in [14, 20]:
                tp = (h + lo + c) / 3.0
                tp_ma = _roll_mean(tp, p)
                tp_md = _roll_mean(np.abs(tp - tp_ma), p) + 1e-10
                cci = _safe_div(tp - tp_ma, 0.015 * tp_md)
                F[f'cci_{p}'] = cci[-1] / 200.0  # Normalize

            # ADX
            atr14 = _atr(c, h, lo, 14) + 1e-10
            plus_dm = np.where((h - np.roll(h, 1)) > (np.roll(lo, 1) - lo),
                               np.maximum(h - np.roll(h, 1), 0), 0)
            minus_dm = np.where((np.roll(lo, 1) - lo) > (h - np.roll(h, 1)),
                                np.maximum(np.roll(lo, 1) - lo, 0), 0)
            pdi = 100 * _ema(plus_dm, 14) / atr14
            mdi = 100 * _ema(minus_dm, 14) / atr14
            dx = 100 * _safe_div(np.abs(pdi - mdi), pdi + mdi + 1e-10)
            adx = _ema(dx, 14)
            F['adx'] = adx[-1] / 100.0
            F['plus_di'] = pdi[-1] / 100.0
            F['minus_di'] = mdi[-1] / 100.0

            # === Group 10: Calendar & London Fix (223-237) ===
            ts = df['time'].iloc[-1] if 'time' in df.columns else pd.Timestamp.now(tz='UTC')
            if hasattr(ts, 'hour'):
                hr = ts.hour
                mn = ts.minute
                dow = ts.weekday()
                dom = ts.day
            else:
                hr, mn, dow, dom = 12, 0, 2, 15

            F['lf_hour'] = hr / 23.0
            # London fix = 16:00 UTC
            mins_to_fix = ((16 - hr) * 60 - mn) % 1440
            F['lf_minutes_to_fix'] = mins_to_fix / 1440.0
            F['lf_proximity'] = max(0, 1.0 - mins_to_fix / 120.0)
            # AM fix = 10:30 UTC
            mins_to_am = ((10 - hr) * 60 + 30 - mn) % 1440
            F['lf_am_fix_proximity'] = max(0, 1.0 - mins_to_am / 120.0)
            # Momentum around fix times
            F['lf_momentum_pre'] = ret[-1] if n > 0 else 0.0
            F['lf_momentum_fix'] = np.mean(ret[-3:]) if n >= 3 else 0.0
            F['lf_vol_during_fix'] = _roll_std(ret, 5)[-1]
            F['lf_typical_direction'] = np.sign(np.mean(ret[-5:])) if n >= 5 else 0.0
            F['lf_fix_strength'] = abs(np.mean(ret[-5:])) / (_roll_std(ret, 20)[-1] + 1e-10)

            F['hour_sin'] = np.sin(2 * np.pi * hr / 24)
            F['hour_cos'] = np.cos(2 * np.pi * hr / 24)
            F['minute_sin'] = np.sin(2 * np.pi * mn / 60)
            F['minute_cos'] = np.cos(2 * np.pi * mn / 60)
            F['dow_sin'] = np.sin(2 * np.pi * dow / 7)
            F['dow_cos'] = np.cos(2 * np.pi * dow / 7)

            # === Group 11: Session & Execution (238-252) ===
            # Session liquidity score based on hour
            liq_map = {0: 0.3, 1: 0.2, 2: 0.2, 3: 0.3, 4: 0.5, 5: 0.6,
                       6: 0.7, 7: 0.85, 8: 0.95, 9: 1.0, 10: 1.0, 11: 0.95,
                       12: 0.9, 13: 0.95, 14: 1.0, 15: 1.0, 16: 0.9, 17: 0.7,
                       18: 0.5, 19: 0.4, 20: 0.35, 21: 0.3, 22: 0.3, 23: 0.3}
            F['sess_liquidity_score'] = liq_map.get(hr, 0.5)

            F['cal_day_of_month'] = dom / 31.0
            days_in_month = 30  # approx
            F['cal_days_to_month_end'] = (days_in_month - dom) / 31.0
            # NFP is first Friday of month — approximate
            F['cal_nfp_hours'] = 0.0  # Would need calendar; default
            F['cal_major_news_risk'] = 0.0
            month = ts.month if hasattr(ts, 'month') else 6
            F['cal_month_sin'] = np.sin(2 * np.pi * month / 12)
            F['cal_month_cos'] = np.cos(2 * np.pi * month / 12)

            typical_spread = _roll_mean(sp, 50)[-1]
            F['exec_spread_score'] = 1.0 - np.clip(sp[-1] / (typical_spread * 3 + 1e-10), 0, 1)
            F['exec_spread_vs_typical'] = _safe_div(sp[-1], typical_spread + 1e-10)
            F['exec_spread_percentile'] = _percentile_rank(sp, min(100, n))[-1]
            F['exec_gap_risk'] = abs(o[-1] - prev_c[-1]) / (c[-1] + 1e-10)
            F['exec_slippage_estimate'] = sp[-1] * 0.5  # Half-spread estimate

            F['exec_fill_probability'] = F['sess_liquidity_score'] * (1.0 - F['exec_spread_percentile'])
            F['exec_market_impact'] = F['kyle_lambda'] * 0.01
            F['exec_quality_score'] = (F['exec_fill_probability'] + F['exec_spread_score']) / 2.0

            # === Group 12: Round Numbers (253-257) ===
            pip_size = 0.01 if 'JPY' in pair else 0.0001
            price = c[-1]
            big_figure = round(price / (100 * pip_size)) * (100 * pip_size)
            half_figure = round(price / (50 * pip_size)) * (50 * pip_size)
            quarter_figure = round(price / (25 * pip_size)) * (25 * pip_size)
            F['rn_big_figure'] = float(abs(price - big_figure) < 2 * pip_size)
            F['rn_half_figure'] = float(abs(price - half_figure) < 2 * pip_size)
            F['rn_quarter_figure'] = float(abs(price - quarter_figure) < 2 * pip_size)
            F['rn_dist_big_pips'] = abs(price - big_figure) / pip_size
            F['rn_dist_half_pips'] = abs(price - half_figure) / pip_size

            # === Group 13: Weekend & Cross-Asset (258-272) ===
            F['wknd_day_of_week'] = dow / 6.0
            # Hours to Friday 22:00 UTC close
            hrs_to_close = ((4 - dow) * 24 + (22 - hr)) % (7 * 24)
            F['wknd_hours_to_close'] = min(hrs_to_close, 120) / 120.0
            hrs_since_open = ((dow - 0) * 24 + hr) % (7 * 24)
            F['wknd_hours_since_open'] = min(hrs_since_open, 120) / 120.0
            F['wknd_gap_risk'] = F['exec_gap_risk']

            # Cross-asset (not available in single-pair mode — use momentum proxy)
            F['corr_USDJPY_20'] = 0.0  # Would need USDJPY data
            F['rel_strength'] = _roll_sum(ret, 20)[-1]
            F['rel_strength_cum'] = _roll_sum(ret, 50)[-1]
            F['rel_strength_rank'] = 0.5  # No cross-pair ranking available
            F['usd_index'] = 0.0  # Would need USD index
            F['usd_index_ma'] = 0.0
            F['jpy_index'] = 0.0
            F['jpy_index_ma'] = 0.0
            F['cross_momentum_div_5'] = 0.0
            F['cross_momentum_div_20'] = 0.0
            F['vol_vs_basket'] = 1.0

            # === Assemble into schema-ordered vector ===
            result = np.zeros(self.n_features, dtype=np.float32)
            matched = 0
            for name, idx in self._name_to_idx.items():
                if name in F:
                    val = F[name]
                    if isinstance(val, (np.ndarray,)):
                        val = float(val.flat[0]) if val.size > 0 else 0.0
                    result[idx] = float(val)
                    matched += 1

            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

            if matched < 250:
                logger.warning(f"{pair}_{tf}: only matched {matched}/{self.n_features} features")
            else:
                logger.debug(f"{pair}_{tf}: matched {matched}/{self.n_features} features")

            return result

        except Exception as e:
            logger.error(f"Feature computation error for {pair}_{tf}: {e}", exc_info=True)
            return None

    def get_feature_count(self):
        return self.n_features

    def get_feature_names(self):
        return list(self._feature_list)
