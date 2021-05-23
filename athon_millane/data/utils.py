import yfinance as yf
import yahoo_fin.stock_info as ys

import pandas as pd
import os.path as path

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_PATH = "./data/data.csv"


def pull_sp500(force=False):
    """
    Pull S&P500 daily close prices for the past 20 years.

    :param force (bool): If true, overwrite existing data file.
    """

    # Check if data already exists.
    if path.exists(DATA_PATH) and not force:
        return

    # Pull tickers from yahoo API.
    tickers_sp500 = ys.tickers_sp500()

    # Pull close data using yfinance.
    price_data = yf.download(tickers_sp500,
                             start="2001-01-01", end="2021-01-01",
                             group_by='column')['Close']

    # Save to csv.
    price_data.to_csv(DATA_PATH)


def load_sp500() -> pd.DataFrame:
    """
    Load S&P500 data, construct simple portfolio and convert to returns.

    :return returns_df (pd.DataFrame): Portfolio returns.
    """

    prices_df = pd.read_csv(DATA_PATH).set_index('Date')

    # Cleaning. Remove assets from the dataset with missing prices during the
    # timeframe.
    prices_df.index = pd.to_datetime(prices_df.index)
    num_missing = prices_df.isna().sum()
    min_missing = num_missing.value_counts().index.min()
    prices_df = prices_df[(
        num_missing[num_missing == min_missing]).index.tolist()].dropna()

    # Convert close prices to returns. Need to drop the shifted row.
    returns_df = (prices_df / prices_df.shift(1) - 1).dropna()

    # Take mean across all assets as naïve portfolio.
    returns_df = returns_df.mean(axis=1).to_frame("r_t")

    return returns_df
