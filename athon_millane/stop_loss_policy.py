import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

from vectorised_policy_utils import calculate_states_jit


class StopLossPolicy:
    """
    Implementation of the stop loss policy definition from the following paper.
    The class consists of methods to calculate a stop loss policy with desired
    parameters, apply that policy to a portfolio and calculate relevant
    performance measures.

    https://dspace.mit.edu/bitstream/handle/1721.1/114876/Lo_When%20Do%20Stop-Loss.pdf
    """

    def __init__(self, returns_df: pd.DataFrame):
        """
        Constructor. Initialises returns dataframe.

        :param returns_df: (pd.DataFrame): Portfolio returns dataframe.
        """

        self.rets_df = returns_df.copy()
        self.df = None

    def _init_policy_parameters(
            self,
            gamma: float,
            delta: float,
            j: float,
            kappa: float = 0,
            rolling_sd: bool = False):
        """
        Helper method to initialise parameters into the policy.
        If kappa is provided, transaction cost will be considered in all
        policy return calculations. If rolling_sd is set to True, gamma and
        delta are considered as multiples of a rolling standard deviation,
        otherwise their value is held constant.

        :param gamma: (float): The stopping threshold.
        :param delta: (float): The re-entry threshold.
        :param j: (float): The rolling window size for R(J) and rolling sigma.
        :param kappa: (float): The return penaly. Defaults to 0.
        :param rolling_sd: (bool): Flag to determine gamma and delta type.
        """

        self.df = self.rets_df.copy()
        self.j = int(j)
        self.kappa = kappa

        # If true, parameters need to be remapped according to the rolling
        # standard deviation of returns.
        if rolling_sd:
            self.df["gamma"] = self.df["r_t"].rolling(
                self.j).mean() + self.df["r_t"].rolling(self.j).std() * gamma

            self.df["delta"] = self.df["r_t"].rolling(
                self.j).mean() + self.df["r_t"].rolling(self.j).std() * delta
        else:
            self.df["gamma"] = gamma
            self.df["delta"] = delta

        # Calculate rolling cumulative return.
        self.df["R_j"] = self.df["r_t"].rolling(self.j).sum()

        # Drop NaNs created by rolling windows.
        self.df = self.df.dropna()

    def _calculate_states(self) -> list:
        """
        Helper method to calculate states iteratively.
        Follows Definition 1 from the source paper.

        :return (list) : List of states of the stopping policy.
        """

        # Iterate through df and calculate s_t.
        s0 = 1
        states = [s0]
        state = s0
        for i, row in self.df.iterrows():
            # exit
            if row["R_j"] < row["gamma"] and state == 1:
                state = 0

            # reenter
            elif row["r_t"] >= row["delta"] and state == 0:
                state = 1

            # stay in
            elif row["R_j"] >= row["gamma"] and state == 1:
                state = 1

            # stay out
            elif row["r_t"] < row["delta"] and state == 0:
                state = 0

            states.append(state)
        return states[:-1]

    def _error_check(self, check_applied=False):
        """
        Helper method to run error checks and ensure methods are called
        in the correct order.

        :param applied: (float): Flag to check whether policy has been applied.
        """

        if self.df is None:
            raise Exception("The stopping policy has not yet been calculated.",
                            "Please run calculate() before this method.")

        if check_applied and "r_st" not in self.df.columns:
            raise Exception("The stopping policy has not yet been applied.",
                            "Please run apply() before this method.")

    def calculate(
            self,
            gamma: float,
            delta: float,
            j: float,
            kappa: float = 0,
            rolling_sd: bool = False,
            vectorised: bool = True):
        """
        This method applies the stop loss policy to the portfolio
        and generates a state sequencyaccording to definition 1 of the paper.
        If kappa is provided, transaction cost will be considered in all
        policy return calculations. If rolling_sd is set to True, gamma and
        delta are considered as multiples of a rolling standard deviation,
        otherwise their value is held constant.

        :param gamma: (float): The stopping threshold.
        :param delta: (float): The re-entry threshold.
        :param j: (float): The rolling window size for R(J) and rolling sigma.
        :param kappa: (float): The return penaly. Defaults to 0.
        :param rolling_sd: (bool): Flag to determine gamma and delta type.
        :vectorised: (bool) : Flag for JIT compiled policy state calculation.
        """

        # Initialise policy parameters
        self._init_policy_parameters(gamma, delta, j, kappa, rolling_sd)

        # Calculate policy.
        if vectorised:
            states = calculate_states_jit(
                self.df["r_t"].to_numpy(),
                self.df["R_j"].to_numpy(),
                self.df["gamma"].to_numpy(),
                self.df["delta"].to_numpy())
        else:
            states = self._calculate_states()

        # Update df with states
        self.df["s_t"] = states

    def apply(self, r_f: pd.Series = None):
        """
        This method applies the calculated stop loss policy to the portfolio
        and calculates the return of the portfolio with stop loss applied.
        The risk free return is assumed to be constant at zero, however a
        a different return process may be provided as a pandas Series or
        as a constant expected return.

        :param r_f: (float): The risk free return.
        """

        self._error_check()

        if r_f is None:
            r_f = 0

        self.df["r_k"] = self.kappa * \
            ((self.df["s_t"] - self.df["s_t"].shift(1)).fillna(0)).abs()
        self.df["r_f"] = r_f
        self.df["r_st"] = (self.df["s_t"] * self.df["r_t"]) + (
            (1 - self.df["s_t"]) * self.df["r_f"]) - (self.df["r_k"])

    def visualise(self, show_params: bool = False):
        """
        Method to visualise the calculated policy against the portfolio.
        Optionally show gamma and delta overlaid on the portfolio to
        to demonstrate how stop policy is calculated.

        :param show_params: (bool): Flag to show gamma and delta on the plots.
        """

        self._error_check()

        fig, axs = plt.subplots(
            len(self.df.columns) - 2, 1,
            figsize=[12, 8], sharex=True)

        [self.df[col].plot(ax=axs[i], title=col)
         for i, col in enumerate(self.df.drop(["gamma", "delta"], axis=1))]

        if show_params:
            [self.df[col].plot(ax=axs[i], legend=True)
             for i, col in enumerate(self.df[["delta", "gamma"]])]

    def visualise_returns(self, compounding: bool = False):
        """
        Method to visualise impact of applying the policy to the portfolio.
        If compounding is set to true, results are obtained by multiplying
        by (1 + r_t), otherwise simple returns are summed.

        :param compounding: (bool): Flag for simple or compound returns.
        """

        self._error_check(check_applied=True)

        if compounding:
            self.df["cumulative_r_t"] = (1 + self.df["r_t"]).cumprod()
            self.df["cumulative_r_st"] = (1 + self.df["r_st"]).cumprod()
        else:
            self.df["cumulative_r_t"] = self.df["r_t"].cumsum()
            self.df["cumulative_r_st"] = self.df["r_st"].cumsum()

        fig, axs = plt.subplots(2, 1, figsize=[12, 6], sharex=True)
        [self.df[col].plot(ax=axs[i], title=col) for i, col in enumerate(
            self.df[["cumulative_r_t", "cumulative_r_st"]])]

    def calculate_performance_measures(self) -> pd.DataFrame:
        """
        This method calculates all performance measures as laid out in
        definitions 2 and 3 of the paper.

        :return (pd.DataFrame): Results table of performance measures.
        """

        self._error_check(check_applied=True)

        mu = self.df["r_t"].mean()
        mu_s = self.df["r_st"].mean()
        mu_f = self.df["r_f"].mean()
        var_s = self.df["r_st"].var()
        var = self.df["r_t"].var()
        sigma_s = self.df["r_st"].std()
        sigma = self.df["r_t"].std()

        N = len(self.df["s_t"])
        p_0 = (len(self.df["s_t"]) - np.count_nonzero(self.df["s_t"])) / N

        stopping_premium = mu_s - mu
        stopping_ratio = stopping_premium / p_0

        variance_difference = var_s - var
        sd_difference = sigma_s - sigma
        sharpe_difference = ((mu_s - mu_f) / sigma_s) - ((mu - mu_f) / sigma)

        measures = []

        measures.append(("Stopping Premium", stopping_premium))
        measures.append(("Stopping Ratio", stopping_ratio))
        measures.append(("Variance Difference", variance_difference))
        measures.append(("Standard Deviation Difference", sd_difference))
        measures.append(("Sharpe Difference", sharpe_difference))

        return pd.DataFrame(measures)
