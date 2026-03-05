"""
Computes COT features from raw CFTC TFF data.

Features per currency (weekly):
  - net_noncommercial: Leveraged Funds net positioning (long - short)
  - net_percent_oi: net_noncommercial / total open interest
  - z_52w: 52-week z-score of net_percent_oi
  - pct_rank_156w: 156-week (3-year) percentile rank of net_percent_oi
  - wow_delta: week-over-week change in net_percent_oi
  - extreme_long_flag: 1 if pct_rank_156w > 0.95
  - extreme_short_flag: 1 if pct_rank_156w < 0.05

Features per spot pair (daily, forward-filled):
  - cot_pressure: base_currency_pressure - quote_currency_pressure
  - cot_extreme_flag: 1 if either currency at extreme positioning

LEAKAGE PREVENTION:
  COT report date = Tuesday. Released = Friday 3:30 PM ET.
  Data is only available starting the NEXT Monday bar (conservative).
  Implementation: shift report date forward to next Monday, then forward-fill daily.
"""
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path

logger = logging.getLogger('cot_features')


def load_contract_map(path='G:/My Drive/chaos_v1.0/alt_data/cot/cot_contract_map.json'):
    with open(path) as f:
        return json.load(f)


def extract_currency_positioning(raw_df, contract_map):
    """
    Extract net positioning per currency from raw TFF data.

    In TFF reports, the key columns are:
    - Lev_Money_Positions_Long / Short (Leveraged Funds = hedge funds)
    - Asset_Mgr_Positions_Long / Short (Asset Managers = institutional)

    We use Leveraged Funds (hedge funds) as the primary signal because:
    - They are the most speculative and trend-following
    - Extreme positioning historically precedes reversals (Brunnermeier et al., 2009)
    - Asset managers are more stable and less contrarian-useful

    Args:
        raw_df: combined raw COT dataframe from downloader
        contract_map: loaded cot_contract_map.json

    Returns:
        pd.DataFrame indexed by date, columns = currency features
    """
    # Identify column names (CFTC naming varies slightly across years)
    col_candidates = {
        'date': ['Report_Date_as_YYYY-MM-DD', 'As_of_Date_In_Form_YYMMDD', 'report_date'],
        'market': ['Market_and_Exchange_Names', 'Contract_Market_Name'],
        'lev_long': ['Lev_Money_Positions_Long_All', 'Lev_Money_Positions_Long'],
        'lev_short': ['Lev_Money_Positions_Short_All', 'Lev_Money_Positions_Short'],
        'oi': ['Open_Interest_All', 'Open_Interest (All)'],
    }

    def find_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
            # Try case-insensitive
            matches = [col for col in df.columns if col.lower().replace(' ', '_') == c.lower().replace(' ', '_')]
            if matches:
                return matches[0]
        return None

    date_col = find_col(raw_df, col_candidates['date'])
    market_col = find_col(raw_df, col_candidates['market'])
    lev_long_col = find_col(raw_df, col_candidates['lev_long'])
    lev_short_col = find_col(raw_df, col_candidates['lev_short'])
    oi_col = find_col(raw_df, col_candidates['oi'])

    if not all([date_col, market_col]):
        logger.error(f"Cannot identify required columns. Available: {list(raw_df.columns[:20])}")
        raise ValueError("COT column identification failed. Check raw data format.")

    logger.info(f"COT columns: date={date_col}, market={market_col}, "
                f"lev_long={lev_long_col}, lev_short={lev_short_col}, oi={oi_col}")

    # Parse dates
    raw_df[date_col] = pd.to_datetime(raw_df[date_col])

    # Filter to FX contracts only
    currency_map = contract_map['currency_to_cot']
    cot_names = {v['cot_name']: k for k, v in currency_map.items() if k != 'USD'}

    results = {}

    for cot_name, currency in cot_names.items():
        # Filter rows matching this contract
        mask = raw_df[market_col].str.contains(cot_name, case=False, na=False)
        ccy_df = raw_df[mask].copy()

        if len(ccy_df) == 0:
            logger.warning(f"No data found for {cot_name} ({currency})")
            continue

        ccy_df = ccy_df.sort_values(date_col).drop_duplicates(subset=[date_col], keep='last')
        ccy_df = ccy_df.set_index(date_col)

        # Compute net positioning
        if lev_long_col and lev_short_col:
            ccy_df['net_noncommercial'] = (
                pd.to_numeric(ccy_df[lev_long_col], errors='coerce') -
                pd.to_numeric(ccy_df[lev_short_col], errors='coerce')
            )
        else:
            # Fallback: try NonComm columns (Legacy format)
            noncomm_long = find_col(ccy_df, ['NonComm_Positions_Long_All', 'Noncommercial_Long'])
            noncomm_short = find_col(ccy_df, ['NonComm_Positions_Short_All', 'Noncommercial_Short'])
            if noncomm_long and noncomm_short:
                ccy_df['net_noncommercial'] = (
                    pd.to_numeric(ccy_df[noncomm_long], errors='coerce') -
                    pd.to_numeric(ccy_df[noncomm_short], errors='coerce')
                )
            else:
                logger.error(f"No positioning columns found for {currency}")
                continue

        # Net as percent of OI
        if oi_col:
            oi_values = pd.to_numeric(ccy_df[oi_col], errors='coerce')
            ccy_df['net_percent_oi'] = ccy_df['net_noncommercial'] / oi_values.replace(0, np.nan)
        else:
            ccy_df['net_percent_oi'] = ccy_df['net_noncommercial']  # Raw if no OI available

        # 52-week z-score (52 weekly observations)
        ccy_df['z_52w'] = (
            (ccy_df['net_percent_oi'] - ccy_df['net_percent_oi'].rolling(52, min_periods=26).mean()) /
            ccy_df['net_percent_oi'].rolling(52, min_periods=26).std().replace(0, np.nan)
        )

        # 156-week percentile rank (3 years of weekly data)
        ccy_df['pct_rank_156w'] = ccy_df['net_percent_oi'].rolling(156, min_periods=52).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )

        # Week-over-week delta
        ccy_df['wow_delta'] = ccy_df['net_percent_oi'].diff()

        # Extreme flags
        ccy_df['extreme_long_flag'] = (ccy_df['pct_rank_156w'] > 0.95).astype(int)
        ccy_df['extreme_short_flag'] = (ccy_df['pct_rank_156w'] < 0.05).astype(int)

        # Store with currency prefix
        feature_cols = ['net_noncommercial', 'net_percent_oi', 'z_52w', 'pct_rank_156w',
                       'wow_delta', 'extreme_long_flag', 'extreme_short_flag']

        for col in feature_cols:
            results[f"cot_{currency}_{col}"] = ccy_df[col]

    result_df = pd.DataFrame(results)
    result_df.index.name = 'date'

    # Drop rows where all values are NaN (warmup period)
    result_df = result_df.dropna(how='all')

    logger.info(f"Currency positioning: {result_df.shape[0]} weeks, {result_df.shape[1]} features, "
                f"date range: {result_df.index.min()} to {result_df.index.max()}")

    return result_df


def compute_pair_features(currency_df, contract_map):
    """
    Derive pair-level COT pressure from currency-level positioning.

    For direct pairs (EURUSD): cot_pressure = EUR_pressure - USD_pressure
    For inverse pairs (USDJPY): cot_pressure = USD_pressure - JPY_pressure
    For crosses (EURJPY): cot_pressure = EUR_pressure - JPY_pressure

    Since USD doesn't have a direct CME FX future in TFF (it's on ICE as DX),
    we use USD = 0 as baseline (all other currencies measured against USD implicitly).

    This means for EURUSD: cot_pressure ~ EUR net positioning (positive = EUR bullish = EURUSD bullish)
    For USDJPY: cot_pressure ~ -JPY net positioning (positive = JPY bearish = USDJPY bullish)
    """
    pair_map = contract_map['pair_decomposition']

    pair_features = {}

    for pair, info in pair_map.items():
        base = info['base']
        quote = info['quote']

        # Get base currency z-score (use z_52w as the primary pressure metric)
        base_z_col = f"cot_{base}_z_52w"
        quote_z_col = f"cot_{quote}_z_52w"

        base_z = currency_df.get(base_z_col, pd.Series(0, index=currency_df.index))
        quote_z = currency_df.get(quote_z_col, pd.Series(0, index=currency_df.index))

        # COT pressure = base pressure - quote pressure
        pair_features[f"cot_{pair}_pressure"] = base_z - quote_z

        # Percentile rank differential
        base_pct_col = f"cot_{base}_pct_rank_156w"
        quote_pct_col = f"cot_{quote}_pct_rank_156w"
        base_pct = currency_df.get(base_pct_col, pd.Series(0.5, index=currency_df.index))
        quote_pct = currency_df.get(quote_pct_col, pd.Series(0.5, index=currency_df.index))
        pair_features[f"cot_{pair}_pct_diff"] = base_pct - quote_pct

        # WoW delta differential
        base_wow_col = f"cot_{base}_wow_delta"
        quote_wow_col = f"cot_{quote}_wow_delta"
        base_wow = currency_df.get(base_wow_col, pd.Series(0, index=currency_df.index))
        quote_wow = currency_df.get(quote_wow_col, pd.Series(0, index=currency_df.index))
        pair_features[f"cot_{pair}_wow_diff"] = base_wow - quote_wow

        # Extreme flag (either currency at extreme)
        # fillna(0) required because bitwise OR fails on float NaN
        base_extreme_long = currency_df.get(f"cot_{base}_extreme_long_flag", pd.Series(0, index=currency_df.index)).fillna(0).astype(int)
        base_extreme_short = currency_df.get(f"cot_{base}_extreme_short_flag", pd.Series(0, index=currency_df.index)).fillna(0).astype(int)
        quote_extreme_long = currency_df.get(f"cot_{quote}_extreme_long_flag", pd.Series(0, index=currency_df.index)).fillna(0).astype(int)
        quote_extreme_short = currency_df.get(f"cot_{quote}_extreme_short_flag", pd.Series(0, index=currency_df.index)).fillna(0).astype(int)

        pair_features[f"cot_{pair}_extreme_flag"] = (
            (base_extreme_long | base_extreme_short | quote_extreme_long | quote_extreme_short).astype(int)
        )

    pair_df = pd.DataFrame(pair_features, index=currency_df.index)
    pair_df.index.name = 'date'

    logger.info(f"Pair features: {pair_df.shape[0]} weeks, {pair_df.shape[1]} features for {len(pair_map)} pairs")
    return pair_df


def expand_weekly_to_daily(weekly_df, start_date='2010-01-01', end_date='2025-12-31'):
    """
    Expand weekly COT data to daily timestamps with leakage-safe forward fill.

    LEAKAGE PREVENTION:
    - COT report_date = Tuesday (positions as of close of business Tuesday)
    - CFTC releases data = Friday 3:30 PM ET
    - Conservative availability = next Monday open

    Implementation:
    1. Shift each report date forward to the following Monday (+6 calendar days from Tuesday)
    2. Create daily date range
    3. Forward-fill from each Monday until next Monday

    This ensures NO model can see COT data before it was publicly available.
    """
    # Create daily date range
    daily_index = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days

    # Shift report dates to next Monday (report_date is Tuesday, available Monday = +6 days)
    shifted_index = weekly_df.index + pd.Timedelta(days=6)

    shifted_df = weekly_df.copy()
    shifted_df.index = shifted_index

    # Sort and deduplicate index
    shifted_df = shifted_df.sort_index()
    shifted_df = shifted_df[~shifted_df.index.duplicated(keep='last')]

    # Reindex to daily and forward-fill (use concat approach for pandas compatibility)
    daily_scaffold = pd.DataFrame(index=daily_index)
    combined = pd.concat([shifted_df, daily_scaffold], axis=0)
    combined = combined[~combined.index.duplicated(keep='first')]
    combined = combined.sort_index()
    daily_df = combined.reindex(daily_index).ffill()
    daily_df.index.name = 'date'

    # Drop NaN rows at the start (warmup period before first COT data)
    first_valid = daily_df.first_valid_index()
    if first_valid is not None:
        daily_df = daily_df.loc[first_valid:]

    logger.info(f"Daily expansion: {len(daily_df)} business days, "
                f"from {daily_df.index.min().date()} to {daily_df.index.max().date()}")

    return daily_df


def build_cot_pipeline(output_dir='G:/My Drive/chaos_v1.0/alt_data/cot'):
    """
    Full COT pipeline: download -> extract -> compute features -> expand to daily.

    Outputs:
        cot_weekly.parquet: Weekly currency-level features (7 currencies x 7 features = 49 columns)
        cot_pair_daily.parquet: Daily pair-level features (9 pairs x 4 features = 36 columns)
    """
    from cot_downloader import download_all_cot

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    contract_map = load_contract_map(str(output_dir / 'cot_contract_map.json'))

    # Step 1: Download
    logger.info("=" * 60)
    logger.info("STEP 1: Downloading COT TFF data")
    logger.info("=" * 60)
    raw_df = download_all_cot(start_year=2010, output_dir=str(output_dir / 'raw'))

    if raw_df is None:
        raise RuntimeError("COT download failed")

    # Step 2: Extract currency positioning
    logger.info("=" * 60)
    logger.info("STEP 2: Computing currency-level features")
    logger.info("=" * 60)
    currency_df = extract_currency_positioning(raw_df, contract_map)

    # Save weekly
    weekly_path = output_dir / 'cot_weekly.parquet'
    currency_df.to_parquet(weekly_path)
    logger.info(f"Saved: {weekly_path}")

    # Step 3: Compute pair features
    logger.info("=" * 60)
    logger.info("STEP 3: Computing pair-level features")
    logger.info("=" * 60)
    pair_weekly = compute_pair_features(currency_df, contract_map)

    # Step 4: Expand to daily with leakage-safe forward fill
    logger.info("=" * 60)
    logger.info("STEP 4: Expanding to daily (leakage-safe)")
    logger.info("=" * 60)
    pair_daily = expand_weekly_to_daily(pair_weekly)

    # Save daily
    daily_path = output_dir / 'cot_pair_daily.parquet'
    pair_daily.to_parquet(daily_path)
    logger.info(f"Saved: {daily_path}")

    # Summary
    logger.info("=" * 60)
    logger.info("COT PIPELINE COMPLETE")
    logger.info(f"  Weekly: {currency_df.shape[0]} weeks x {currency_df.shape[1]} features")
    logger.info(f"  Daily:  {pair_daily.shape[0]} days x {pair_daily.shape[1]} features")
    logger.info(f"  Date range: {pair_daily.index.min().date()} to {pair_daily.index.max().date()}")
    logger.info("=" * 60)

    return currency_df, pair_daily


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    build_cot_pipeline()
