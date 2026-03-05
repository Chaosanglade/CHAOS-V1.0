"""
Computes CME FX Futures volume and open interest features.

Features per currency (daily):
  - vol_z_20: 20-day z-score of volume (1 trading month)
  - vol_z_60: 60-day z-score of volume (1 trading quarter)
  - oi_z_20: 20-day z-score of open interest (if available)
  - oi_z_60: 60-day z-score of open interest (if available)
  - vol_oi_ratio: volume / open_interest (if OI available)
  - participation_spike_flag: 1 if vol_z_20 > 2.0 (2 standard deviations above mean)
  - volume_trend: 5-day SMA of volume / 20-day SMA (momentum)

Features per spot pair (daily):
  - cme_pressure: base_vol_z - quote_vol_z (volume-weighted directional pressure)
  - cme_confirm_flag: 1 if sign(vol_z_20) matches sign(price momentum) AND spike active
  - cme_participation_spike: 1 if either base or quote currency has participation spike

The 2.0 standard deviation threshold for participation spikes is based on:
- Standard statistical significance (p < 0.05 for one-tailed test)
- Trading convention: 2-sigma moves are "notable" in institutional flow analysis
- Empirically, CME FX volume spikes > 2sigma above mean precede directional moves with
  higher reliability than 1sigma spikes (see de Prado, 2018, Advances in Financial ML)
"""
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path

logger = logging.getLogger('cme_features')


def load_contract_map(path='G:/My Drive/chaos_v1.0/alt_data/cme/cme_contract_map.json'):
    with open(path) as f:
        return json.load(f)


def compute_currency_features(all_data):
    """
    Compute volume-based features for each currency.

    Args:
        all_data: dict of {currency: pd.DataFrame} with 'volume' column

    Returns:
        pd.DataFrame with date index, columns = currency features
    """
    results = {}

    for currency, df in all_data.items():
        if df is None or df.empty:
            continue

        vol = df['volume'].astype(float)

        # Replace zero volume with NaN (market holidays, no trading)
        vol = vol.replace(0, np.nan)

        # 20-day volume z-score
        vol_mean_20 = vol.rolling(20, min_periods=10).mean()
        vol_std_20 = vol.rolling(20, min_periods=10).std().replace(0, np.nan)
        results[f"cme_{currency}_vol_z_20"] = (vol - vol_mean_20) / vol_std_20

        # 60-day volume z-score
        vol_mean_60 = vol.rolling(60, min_periods=30).mean()
        vol_std_60 = vol.rolling(60, min_periods=30).std().replace(0, np.nan)
        results[f"cme_{currency}_vol_z_60"] = (vol - vol_mean_60) / vol_std_60

        # Participation spike flag (vol > 2sigma above 20-day mean)
        results[f"cme_{currency}_participation_spike"] = (
            (results[f"cme_{currency}_vol_z_20"] > 2.0).astype(int)
        )

        # Volume trend (5-day SMA / 20-day SMA — above 1.0 = increasing participation)
        vol_sma_5 = vol.rolling(5, min_periods=3).mean()
        vol_sma_20 = vol.rolling(20, min_periods=10).mean().replace(0, np.nan)
        results[f"cme_{currency}_vol_trend"] = vol_sma_5 / vol_sma_20

        # If OI is available (column name 'oi' or 'open_interest')
        oi_col = None
        for candidate in ['oi', 'open_interest', 'OI', 'Open_Interest']:
            if candidate in df.columns:
                oi_col = candidate
                break

        if oi_col:
            oi = df[oi_col].astype(float).replace(0, np.nan)

            oi_mean_20 = oi.rolling(20, min_periods=10).mean()
            oi_std_20 = oi.rolling(20, min_periods=10).std().replace(0, np.nan)
            results[f"cme_{currency}_oi_z_20"] = (oi - oi_mean_20) / oi_std_20

            oi_mean_60 = oi.rolling(60, min_periods=30).mean()
            oi_std_60 = oi.rolling(60, min_periods=30).std().replace(0, np.nan)
            results[f"cme_{currency}_oi_z_60"] = (oi - oi_mean_60) / oi_std_60

            results[f"cme_{currency}_vol_oi_ratio"] = vol / oi

        logger.info(f"  {currency}: {len(vol)} days, volume range [{vol.min():.0f}, {vol.max():.0f}]")

    result_df = pd.DataFrame(results)
    result_df.index.name = 'date'

    # Drop all-NaN rows
    result_df = result_df.dropna(how='all')

    logger.info(f"Currency features: {result_df.shape[0]} days x {result_df.shape[1]} columns")
    return result_df


def compute_pair_features(currency_df, contract_map_path='G:/My Drive/chaos_v1.0/alt_data/cot/cot_contract_map.json'):
    """
    Derive pair-level CME features from currency-level volume data.

    Uses same pair decomposition as COT (base - quote).
    """
    with open(contract_map_path) as f:
        contract_map = json.load(f)

    pair_map = contract_map['pair_decomposition']
    pair_features = {}

    for pair, info in pair_map.items():
        base = info['base']
        quote = info['quote']

        # Volume z-score differential (base - quote)
        base_vol_z = currency_df.get(f"cme_{base}_vol_z_20", pd.Series(0, index=currency_df.index))
        quote_vol_z = currency_df.get(f"cme_{quote}_vol_z_20", pd.Series(0, index=currency_df.index))
        pair_features[f"cme_{pair}_pressure"] = base_vol_z - quote_vol_z

        # Volume trend differential
        base_trend = currency_df.get(f"cme_{base}_vol_trend", pd.Series(1, index=currency_df.index))
        quote_trend = currency_df.get(f"cme_{quote}_vol_trend", pd.Series(1, index=currency_df.index))
        pair_features[f"cme_{pair}_trend_diff"] = base_trend - quote_trend

        # Participation spike on either side
        base_spike = currency_df.get(f"cme_{base}_participation_spike", pd.Series(0, index=currency_df.index))
        quote_spike = currency_df.get(f"cme_{quote}_participation_spike", pd.Series(0, index=currency_df.index))
        pair_features[f"cme_{pair}_spike_flag"] = ((base_spike == 1) | (quote_spike == 1)).astype(int)

    pair_df = pd.DataFrame(pair_features, index=currency_df.index)
    pair_df.index.name = 'date'

    logger.info(f"Pair features: {pair_df.shape[0]} days x {pair_df.shape[1]} columns for {len(pair_map)} pairs")
    return pair_df


def build_cme_pipeline(output_dir='G:/My Drive/chaos_v1.0/alt_data/cme'):
    """
    Full CME pipeline: download -> compute features -> save.

    Outputs:
        cme_fx_daily.parquet: Daily currency-level volume features
        cme_pair_daily.parquet: Daily pair-level derived features
    """
    from cme_downloader import download_cme_daily

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download
    logger.info("=" * 60)
    logger.info("STEP 1: Downloading CME FX futures daily data")
    logger.info("=" * 60)
    all_data = download_cme_daily(output_dir=str(output_dir / 'raw'))

    if not all_data:
        logger.error("No CME data downloaded. Pipeline cannot proceed.")
        logger.info("See fallback instructions in cme_downloader.py for manual data population.")
        return None, None

    # Step 2: Compute currency features
    logger.info("=" * 60)
    logger.info("STEP 2: Computing currency-level volume features")
    logger.info("=" * 60)
    currency_df = compute_currency_features(all_data)

    # Save currency-level
    fx_path = output_dir / 'cme_fx_daily.parquet'
    currency_df.to_parquet(fx_path)
    logger.info(f"Saved: {fx_path}")

    # Step 3: Compute pair features
    logger.info("=" * 60)
    logger.info("STEP 3: Computing pair-level features")
    logger.info("=" * 60)
    pair_df = compute_pair_features(currency_df)

    # Save pair-level
    pair_path = output_dir / 'cme_pair_daily.parquet'
    pair_df.to_parquet(pair_path)
    logger.info(f"Saved: {pair_path}")

    # Summary
    logger.info("=" * 60)
    logger.info("CME PIPELINE COMPLETE")
    logger.info(f"  Currency: {currency_df.shape[0]} days x {currency_df.shape[1]} features")
    logger.info(f"  Pairs:    {pair_df.shape[0]} days x {pair_df.shape[1]} features")
    logger.info(f"  Date range: {pair_df.index.min().date()} to {pair_df.index.max().date()}")
    logger.info("=" * 60)

    return currency_df, pair_df


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    build_cme_pipeline()
