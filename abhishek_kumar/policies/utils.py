from yfinance import download
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np


def download_data(
    universe: list, filename: str = 'ticker',
    length: str = '10y', tick: str = '1d'
) -> DataFrame:
    '''
    Download stock data via Yahoo Finance API

    :param universe (list): list of stocks in universe
    :param filename (str): filename to save to
    :param length (str): overall period of data
    :param tick (str): time difference between datapoints

    :return adj_close_data (pd.DataFrame): pd.DataFrame of adjusted close
        prices of universe
    '''
    ticker_info = download(tickers=universe, period=length, interval=tick)
    adj_close_data = ticker_info['Adj Close']
    adj_close_data.to_csv(f'data/{filename}.csv')
    return adj_close_data


def calculate_pct_change(observations: np.array) -> np.array:
    '''
    Calculate Percent Change of a Time Series

    :param observations (np.array): original time series

    :return percent_change (np.array): percent change of time series
    '''
    percent_change = np.diff(observations)/observations[:-1]
    percent_change = np.insert(percent_change, 0, 0)
    return percent_change


def calculate_metrics(
    position: np.array, pct_change: np.array
) -> (np.array, np.array):
    '''
    Calculate Daily Returns and Equity Curve of Time Series

    :param position (np.array): one-hot encoding of when
        invested in portfolio
    :param pct_change (np.array): vector of return values over time

    :return daily_returns (np.array): array representing daily returns
        when holding
    :return equity_curve (np.array): array representing equity curve
    '''
    daily_returns = position * pct_change
    equity_curve = (pct_change + 1).cumprod() - 1
    return daily_returns, equity_curve


def generate_random_walk(
    mu: float = 0.0, eps: float = 1.0,
    length: int = 100
) -> np.array:
    '''
    Create a random walk as desrcibed in Proposition 1 based on parameters

    :param mu (float): mean of random walk
    :param eps (float): variance of random walk
    :param length (int): length of series

    :return walk (np.array): random walk of data with mean mu and variance eps
    '''
    std = np.sqrt(eps)
    walk = np.random.normal(loc=mu, scale=std, size=length)
    return walk


def generate_autoregressive(
    mu: float = 0.0, eps: float = 1.0,
    rho: float = 0.0, length: int = 100
) -> np.array:
    '''
    Create an autoregressive series as desrcibed in
    Proposition 2 based on parameters given

    :param mu (float): mean of series
    :param eps (float): variance of series
    :param rho (float): autogressive coefficent
    :param length (int): length of series

    :return ar_series (np.array): autoregressive series of power 1
    '''
    assert abs(rho) <= 1, 'Rho value must be in [-1, 1]'

    ar_series = generate_random_walk(mu, eps, length)
    for i in range(1, ar_series.shape[0]):
        ar_series[i] += rho * (ar_series[i-1] - mu)
    return ar_series


def generate_regime(
    mu: np.array = np.zeros(2), eps: np.array = np.zeros(2),
    A: np.array = np.eye(2), length: int = 100
):
    '''
    Create an autoregressive series as desrcibed in
    Proposition 2 based on parameters given

    :param mu (np.array): mean of both regimes
    :param eps (np.array): variance of regimes
    :param A (np.array): transition matrix
    :param length (int): length of series

    :return regime_series (np.array): regime series
    :return regime_state (np.array): one hot encoding to
        represent which regime is occuring
    '''
    assert mu.shape == (2, ) or mu.shape == (2, 1), 'Incorrect Mean Shape'
    assert eps.shape == (2, ) or eps.shape == (2, 1), 'Incorrect Var Shape'
    assert A.shape == (2, 2), 'Incorrect Transition Matrix Shape'

    regime_series = []
    regime_state = []
    std_dev = np.sqrt(eps)
    for i in range(0, length):
        state = regime_state[-1]
        value = np.random.normal(
            mu[state], eps[state]
        )
        regime_series.append(value)
        tranitions = A[1-state]
        regime_state.append(1 - int(np.random.rand() > tranitions[0]))
    return np.array(regime_series), np.array(regime_state)


def perform_numpy_policy(
    observations: np.array, lookback: int = 30,
    delta: float = 0.05, gamma: float = 0.3
) -> np.array:
    '''
    Peforms Stop Loss algorithm as defined in Definition 1

    :param observations (float): return values {r_t}
    :param lookback (float): lookback value for R
    :param delta (float): sell threshold
    :param gamma (int): stop loss threshold

    :return states (np.array): one-hot encoding representing
        when invested in portfolio
    '''
    pct_change = calculate_pct_change(observations)
    R = np.cumsum(pct_change)
    R[lookback:] = R[lookback:] - R[:-lookback]
    R = R[lookback-1:]
    states = [0]*(lookback-1)
    for i, R_t in enumerate(R):
        if states[-1] == 1 and R_t < gamma:
            states.append(0)
        elif states[-1] == 0 and pct_change[lookback+i-1] >= delta:
            states.append(1)
        else:
            states.append(states[-1])
    return np.array(states)


def plot_observations(observations, positions):
    '''
    Plots Equity Curve, Daily Returns, and Positions

    :param observations (float): true observed data
    :param position (np.array): one-hot encoding of when
        invested in portfolio
    '''
    length = len(observations)
    fig, axs = plt.subplots(3, figsize=(10, 6))
    fig.tight_layout(pad=1)
    pct_change = calculate_pct_change(observations)
    daily_returns, equity_curve = calculate_metrics(positions, pct_change)
    axs[0].plot(np.arange(length), equity_curve)
    axs[0].set_title('Equity Curve of Observations')
    axs[1].plot(np.arange(length), positions, color='orange')
    axs[1].set_title('Position of Simple Stop Loss on Observations')
    axs[2].plot(np.arange(length), daily_returns, color='r')
    axs[2].set_title('Daily Returns of Simple Stop Loss on Observations')
