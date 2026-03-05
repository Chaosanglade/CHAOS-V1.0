"""
Downloads CFTC Commitment of Traders data from official historical compressed datasets.
Source: https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm
Report type: Traders in Financial Futures (TFF), Futures Only.

The CFTC publishes annual zip files containing CSV data for all contracts.
We download the TFF Futures Only files for each year.
"""
import os
import zipfile
import io
import pandas as pd
import requests
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('cot_downloader')

# CFTC Historical Compressed URLs
# TFF = Traders in Financial Futures
# Format: annual zip files containing a single CSV
CFTC_BASE_URL = "https://www.cftc.gov/files/dea/history"


# TFF Futures Only file naming pattern
# Current year: fut_fin_txt.zip
# Historical: fut_fin_txt_{year}.zip
def get_tff_url(year=None):
    """Get URL for TFF Futures Only data file."""
    if year is None or year >= 2026:
        return f"{CFTC_BASE_URL}/fut_fin_txt.zip"
    else:
        return f"{CFTC_BASE_URL}/fut_fin_txt_{year}.zip"


def download_cot_year(year, output_dir):
    """
    Download and parse one year of TFF Futures Only data.

    Args:
        year: int, e.g. 2020. None for current year.
        output_dir: directory to save raw CSV

    Returns:
        pd.DataFrame with columns from CFTC TFF report
    """
    url = get_tff_url(year)
    logger.info(f"Downloading COT TFF data: {url}")

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to download {url}: {e}")
        return None

    # Extract CSV from zip
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        csv_names = [n for n in zf.namelist() if n.endswith('.txt') or n.endswith('.csv')]
        if not csv_names:
            logger.error(f"No CSV/TXT file found in {url}")
            return None

        csv_name = csv_names[0]
        logger.info(f"  Extracting: {csv_name}")

        with zf.open(csv_name) as f:
            df = pd.read_csv(f, low_memory=False)

    # Save raw CSV for audit trail
    raw_path = Path(output_dir) / f"cot_tff_raw_{year or 'current'}.csv"
    df.to_csv(raw_path, index=False)
    logger.info(f"  Saved raw: {raw_path} ({len(df)} rows)")

    return df


def download_all_cot(start_year=2010, output_dir='G:/My Drive/chaos_v1.0/alt_data/cot/raw'):
    """
    Download all available TFF data from start_year to present.

    Our training data spans 2009-2025. COT TFF data is available from 2006+.
    Download 2010-current to have sufficient history for 156-week lookbacks.

    Args:
        start_year: first year to download
        output_dir: directory for raw CSV files

    Returns:
        pd.DataFrame with all years concatenated
    """
    os.makedirs(output_dir, exist_ok=True)

    all_dfs = []
    current_year = 2026  # Update if running in different year

    for year in range(start_year, current_year + 1):
        df = download_cot_year(year, output_dir)
        if df is not None:
            all_dfs.append(df)

    # Also try current year file (no year suffix)
    df_current = download_cot_year(None, output_dir)
    if df_current is not None:
        all_dfs.append(df_current)

    if not all_dfs:
        logger.error("No COT data downloaded")
        return None

    combined = pd.concat(all_dfs, ignore_index=True)

    # Remove duplicates (current year file may overlap with annual file)
    if 'Report_Date_as_YYYY-MM-DD' in combined.columns:
        date_col = 'Report_Date_as_YYYY-MM-DD'
    elif 'As_of_Date_In_Form_YYMMDD' in combined.columns:
        date_col = 'As_of_Date_In_Form_YYMMDD'
    else:
        # Find the date column dynamically
        date_cols = [c for c in combined.columns if 'date' in c.lower()]
        date_col = date_cols[0] if date_cols else combined.columns[0]

    # Get market name column
    market_cols = [c for c in combined.columns if 'market' in c.lower() and 'name' in c.lower()]
    market_col = market_cols[0] if market_cols else 'Market_and_Exchange_Names'

    combined = combined.drop_duplicates(subset=[date_col, market_col], keep='last')
    combined = combined.sort_values(date_col).reset_index(drop=True)

    logger.info(f"Combined COT data: {len(combined)} rows, {combined[date_col].nunique()} unique dates")
    return combined


if __name__ == '__main__':
    df = download_all_cot()
    if df is not None:
        print(f"Downloaded {len(df)} rows")
        print(f"Columns: {list(df.columns[:10])}...")
