from .utils import download_data, calculate_metrics
import matplotlib.pyplot as plt
import numpy as np


class MarketStopLoss:
    def __init__(
        self, stock_universe: list, length: str = '10y',
        tick: str = '1d', delta: float = 0.05, gamma: float = 0.3,
        J: int = 30, risk_free: float = .03
    ):
        self.set_universe(stock_universe, length, tick)
        self.delta = delta
        self.gamma = -1*gamma
        self.lookback = J
        self.positions = {}
        self.r_f = .03
        self.perform_policy()

    def set_universe(self, stock_universe, length, tick):
        '''
        Downlaods all data and sets global variables

        :param stock_universe (list): list of stocks in universe
        :param length (str): overall period of data
        :param tick (str): time difference between datapoints
        '''
        self.universe = download_data(stock_universe, 'simple', length, tick)
        self.returns = self.universe.pct_change()
        self.stocks = stock_universe
        self.period = length
        self.interval = tick

    def perform_policy_individual(self, stock: str):
        '''
        Performs policy as described in Definiton 1

        :param stock (str): name of stock
        '''
        if stock in self.positions:
            return self.positions[stock]
        data = self.returns[stock]
        R = data.rolling(self.lookback).sum()
        states = [0]*self.lookback
        for i, R_t in enumerate(R.iloc[self.lookback:]):
            if states[-1] == 1 and R_t < self.gamma:
                states.append(0)
            elif states[-1] == 0 and data[self.lookback+i] >= self.delta:
                states.append(1)
            else:
                states.append(states[-1])
        self.positions[stock] = np.array(states)

    def perform_policy(self):
        '''
        Performs policy on all stocks as described in Definiton 1
        '''
        for stock in self.stocks:
            self.perform_policy_individual(stock)

    def plot_individual(self, stock: str, i=0, axs=None):
        '''
        Plots Equity Curve, Daily Returns, and Positions

        :param stock (str): name of stock
        :param i (int): Subplot index
        :param axs (matplotlib.axe): Existing axes to plot on
        '''
        if axs is None:
            fig, axs = plt.subplots(3, figsize=(10, 6))
            fig.tight_layout(pad=1)
        daily_returns, equity_curve = calculate_metrics(
            self.positions[stock], self.returns[stock]
        )
        axs[i].plot(self.universe.index, equity_curve)
        axs[i].set_title(f'Equity Curve of {stock}')
        axs[i+1].plot(
            self.universe.index, self.positions[stock], color='orange'
        )
        axs[i+1].set_title(f'Position of Simple Stop Loss on {stock}')
        axs[i+2].plot(self.universe.index, daily_returns, color='r')
        axs[i+2].set_title(f'Daily Returns of Simple Stop Loss on {stock}')

    def plot(self):
        '''
        Plot all equities in the universe
        '''
        fig, axs = plt.subplots(
            3*len(self.stocks), figsize=(10, 6*len(self.stocks))
        )
        fig.tight_layout(pad=1)
        for i, stock in enumerate(self.stocks):
            self.plot_individual(stock, 3*i, axs)

    def stop_ratio_individual(self, stock: str) -> float:
        '''
        Calculate Stop Ratio for a single equity

        :param stock (str): name of stock

        :return stop_ratio (float): stop ratio as described
        '''
        position = self.positions[stock]
        not_held = 1 - position
        not_held_returns = not_held * self.returns[stock]
        return self.r_f-np.sum(not_held_returns)/np.sum(not_held_returns != 0)

    def stop_premium_individual(self, stock: str) -> float:
        '''
        Calculate Stop Premium for a single equity

        :param stock (str): name of stock

        :return var_diff (float): stop premium as described
        '''
        p0 = np.mean(1-self.positions[stock])
        return p0 * self.stop_ratio_individual(stock)

    def variance_difference_individual(self, stock: str) -> float:
        '''
        Calculate Varaince Difference for a single equity

        :param stock (str): name of stock

        :return var_diff (float): variance difference as described
        '''
        not_held = 1 - self.positions[stock]
        not_held_returns = not_held * self.returns[stock]
        daily_returns = self.positions[stock] * self.returns[stock]
        not_held_mean = np.sum(not_held_returns)/np.sum(not_held_returns != 0)
        p0 = np.mean(not_held)
        mu = np.sum(daily_returns)/np.sum(self.positions[stock] != 0)
        left_term = -p0*np.var(not_held_returns)
        right_term = (self.r_f-not_held_mean)**2
        right_term -= ((mu - not_held_mean)/(1-p0))**2
        return left_term + p0*(1-p0)*right_term
