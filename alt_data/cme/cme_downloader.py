"""
Downloads CME FX Futures daily volume and open interest data.

Primary source: CME Group's daily settlement/volume reports.
Alternative: Yahoo Finance or Quandl/Nasdaq Data Link for historical daily OHLCV+Volume+OI.

For historical data going back to 2010+, we use the free Yahoo Finance API
(via yfinance library) which provides daily OHLCV + Volume for CME FX futures.
Open Interest requires CME's own reports or Quandl.

Strategy:
1. Use yfinance for daily Volume (free, goes back 10+ years)
2. Attempt CME settlement reports for OI (may require scraping)
3. If OI unavailable, compute features from Volume only (still valuable)
"""
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path

logger = logging.getLogger('cme_downloader')

# Yahoo Finance continuous contract symbols for CME FX futures
# The '=F' suffix gets the continuous front-month contract
YAHOO_SYMBOLS = {
    'EUR': '6E=F',
    'GBP': '6B=F',
    'JPY': '6J=F',
    'AUD': '6A=F',
    'CAD': '6C=F',
    'CHF': '6S=F',
    'NZD': '6N=F',
}


def download_cme_daily(start_date='2010-01-01', end_date='2025-12-31',
                       output_dir='G:/My Drive/chaos_v1.0/alt_data/cme/raw'):
    """
    Download daily volume data for all CME FX futures contracts.

    Uses yfinance for free historical data. Falls back gracefully
    if yfinance is not installed or data is unavailable.

    Returns:
        dict of {currency: pd.DataFrame} with columns [date, open, high, low, close, volume]
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed. Run: pip install yfinance")
        logger.info("Attempting alternative download method...")
        return _download_alternative(start_date, end_date, output_dir)

    all_data = {}

    for currency, symbol in YAHOO_SYMBOLS.items():
        logger.info(f"Downloading {currency} ({symbol})...")
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval='1d')

            if df.empty:
                logger.warning(f"  No data for {symbol}")
                continue

            # Standardize columns
            df = df.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low',
                'Close': 'close', 'Volume': 'volume'
            })

            # Keep only what we need
            keep_cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
            df = df[keep_cols].copy()
            df.index = pd.to_datetime(df.index)
            df.index = df.index.tz_localize(None)  # Remove timezone
            df.index.name = 'date'

            # Save raw
            raw_path = output_dir / f"cme_{currency}_daily.csv"
            df.to_csv(raw_path)

            all_data[currency] = df
            logger.info(f"  {currency}: {len(df)} days, {df.index.min().date()} to {df.index.max().date()}")

        except Exception as e:
            logger.error(f"  Failed to download {currency}: {e}")
            continue

    return all_data


def _download_alternative(start_date, end_date, output_dir):
    """
    Fallback: create empty structure for manual data population.
    If yfinance fails, user can manually download CSV from TradingView
    or other free sources and place in the raw directory.
    """
    logger.info("Creating empty structure for manual CME data population.")
    logger.info("To populate manually:")
    logger.info("  1. Go to TradingView.com")
    logger.info("  2. Search for 6E1!, 6B1!, 6J1!, 6A1!, 6C1!, 6S1!, 6N1!")
    logger.info("  3. Export daily OHLCV data as CSV")
    logger.info("  4. Save to: G:/My Drive/chaos_v1.0/alt_data/cme/raw/")
    return {}


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    data = download_cme_daily()
    if data:
        for ccy, df in data.items():
            print(f"{ccy}: {len(df)} rows")
    else:
        print("No data downloaded")
