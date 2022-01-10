import policies.utils as utils
import matplotlib.pyplot as plt
import numpy as np


class RandomWalkStopLoss:
    def __init__(
        self, mu: float = 100.0, eps: float = 5.0, length: int = 500,
        delta: float = 0.05, gamma: float = 0.3,
        J: int = 30, risk_free: float = .03
    ):
        '''
        :param mu (float): mean of random walk
        :param eps (float): variance of random walk
        :param length (int): length of series
        :param observations (float): return values {r_t}
        :param delta (float): sell threshold
        :param gamma (int): stop loss threshold
        :param J (float): lookback value for R
        :param risk_free (float): risk free rate when not holding portfolio
        '''
        self.observations = utils.generate_random_walk(mu, eps, length)
        self.mu = mu
        self.eps = eps
        self.length = length
        self.delta = delta
        self.gamma = -1*gamma
        self.lookback = J
        self.r_f = .03
        self.perform_policy()

    def perform_policy(self):
        '''
        Calculates positions based of Definiton 1
        '''
        self.positions = utils.perform_numpy_policy(
            self.observations, self.lookback, self.delta, self.gamma
        )
        self.p0 = np.mean(self.positions)

    def plot(self):
        '''
        Plots Equity Curve, Daily Returns, and Positions
        '''
        utils.plot_observations(self.observations, self.positions)

    def stop_ratio(self) -> float:
        '''
        Calculate Stop Ratio as defined in Proposition 1

        :return stop_ratio (float): stop_ratio as described
        '''
        return self.r_f-self.mu

    def stop_premium(self) -> float:
        '''
        Calculate Stop Premium as defined in Proposition 1

        :return stop_prem (float): stop premium as described
        '''
        return self.p0 * self.stop_ratio()

    def variance_difference(self) -> float:
        '''
        Calculate Varaince Difference as defined in Proposition 1

        :return var_diff (float): variance difference as described
        '''
        return -self.p0*self.eps + self.p0*(1-self.p0)*(self.stop_ratio()**2)


class ARStopLoss:
    def __init__(
        self, mu: float = 100.0, eps: float = 5.0,
        rho: float = 0.0, length: int = 500,
        delta: float = .05, gamma: float = .3,
        J: int = 30, risk_free: float = .03
    ):
        '''
        :param mu (float): mean of AR series
        :param eps (float): variance of AR series
        :param rho (float): autogressive coefficent
        :param length (int): length of series
        :param observations (float): return values {r_t}
        :param delta (float): sell threshold
        :param gamma (int): stop loss threshold
        :param J (float): lookback value for R
        :param risk_free (float): risk free rate when not holding portfolio
        '''
        self.observations = utils.generate_autoregressive(mu, eps, rho, length)
        self.rho = rho
        self.mu = mu
        self.eps = eps
        self.length = length
        self.delta = delta
        self.gamma = -1*gamma
        self.lookback = J
        self.r_f = .03
        self.perform_policy()

    def perform_policy(self):
        '''
        Calculates positions based of Definiton 1
        '''
        self.positions = utils.perform_numpy_policy(
            self.observations, self.lookback, self.delta, self.gamma
        )
        self.p0 = np.mean(self.positions)

    def plot(self):
        '''
        Plots Equity Curve, Daily Returns, and Positions
        '''
        utils.plot_observations(self.observations, self.positions)

    def stop_ratio(self) -> float:
        '''
        Calculate lower bound Stop Ratio as defined in Proposition 2

        :return stop_ratio (float): stop_ratio as described
        '''
        return self.r_f-self.mu + self.rho * np.sqrt(eps)

    def stop_premium(self) -> float:
        '''
        Calculate lower bound Stop Premium as defined in Proposition 2

        :return stop_prem (float): stop premium as described
        '''
        return self.p0 * self.stop_ratio()

    def reveal_pnl(self):
        '''
        Return PnL when hold and over entire series

        :return daily_returns (np.array): dail returns

        '''
        pct_change = utils.calculate_pct_change(self.observations)
        daily_returns, _ = utils.calculate_metrics(self.positions, pct_change)
        return daily_returns, pct_change


class RegimeStopLoss:
    def __init__(
        self, mu: np.array = np.zeros(2), eps: np.array = np.zeros(2),
        A: np.array = np.eye(2), length: int = 500,
        delta: float = .05, gamma: float = .3,
        J: int = 30, risk_free: float = .03
    ):
        '''
        :param mu (np.array): mean of both regimes
        :param eps (np.array): variance of both regimes
        :param A (np.array): transition matrix
        :param length (int): length of series
        :param observations (float): return values {r_t}
        :param delta (float): sell threshold
        :param gamma (int): stop loss threshold
        :param J (float): lookback value for R
        :param risk_free (float): risk free rate when not holding portfolio
        '''
        self.observations, self.regime_state = utils.generate_regime(
            mu, eps, A, length
        )
        self.A = A
        self.mu = mu
        self.eps = eps
        self.length = length
        self.delta = delta
        self.gamma = -1*gamma
        self.lookback = J
        self.r_f = .03
        self.perform_policy()

    def perform_policy(self):
        '''
        Calculates positions based of Definiton 1
        '''

        self.positions = utils.perform_numpy_policy(
            self.observations, self.lookback, self.delta, self.gamma
        )
        self.p0 = np.mean(self.positions)
        self.p_o = np.zeros(2)
        self.p_o[0] = np.sum(
            (self.positions == 0) & (self.regime_state == 1)
        )/self.length
        self.p_o[1] = np.sum(
            (self.positions == 0) & (self.regime_state == 0)
        )/self.length

    def plot(self):
        '''
        Plots Equity Curve, Daily Returns, and Positions
        '''
        utils.plot_observations(self.observations, self.positions)

    def stop_ratio(self) -> float:
        '''
        Calculate lower bound Stop Ratio as defined in Proposition 2

        :return stop_ratio (float): stop_ratio as described
        '''
        return self.r_f-self.mu + self.rho * np.sqrt(eps)

    def stop_premium(self) -> float:
        '''
        Calculate lower bound Stop Premium as defined in Proposition 2

        :return stop_prem (float): stop premium as described
        '''
        return self.p0 * self.stop_ratio()

    def variance_difference(self) -> float:
        '''
        Calculate Varaince Difference as defined in Proposition 2

        :return var_diff (float): variance difference as described
        '''
        return -self.p0*self.eps + self.p0*(1-self.p0)*(self.stop_ratio()**2)

    def stop_ratio(self) -> float:
        '''
        Calculate lower bound Stop Ratio as defined in Formula 20

        :return stop_ratio (float): stop_ratio as described
        '''
        return np.sum(self.p_o * (self.r_f-self.mu))

    def stop_premium(self) -> float:
        '''
        Calculate lower bound Stop Premium as defined in Formula 21

        :return stop_prem (float): stop premium as described
        '''
        return np.sum(self.p_o * (self.r_f-self.mu))
