import numpy as np
from numba import njit


@njit
def policy_step_jit(ret: float, rolling_j: float, state_: int,
                    gamma: float, delta: float) -> int:
    """
    Single policy step.

    :param ret (float): One-period return.
    :param rolling_j (float): The rolling window size for R(J).
    :param state_ (int): Policy state at time t-1.
    :param gamma (float): The stopping threshold.
    :param delta (float): The re-entry threshold.
    :return (int): Policy state at time t.
    """

    # Exit
    if rolling_j < gamma and state_ == 1:
        state = 0

    # Re-enter
    elif ret >= delta and state_ == 0:
        state = 1

    # Stay in
    elif rolling_j >= gamma and state_ == 1:
        state = 1

    # Stay out
    elif ret < delta and state_ == 0:
        state = 0

    return state


@njit()
def calculate_states_jit(col_return: np.ndarray,
                         col_rolling_j: np.ndarray,
                         gamma: np.ndarray,
                         delta: np.ndarray) -> np.ndarray:
    """
    Definition 1 from the paper.

    :param col_returns (np.ndarray): One-period return.
    :param col_rolling_j (np.ndarray): The rolling window size for R(J).
    :param gamma (np.ndarray): The stopping threshold.
    :param delta (np.ndarray): The re-entry threshold.
    :return (np.ndarray): Policy state.
    """

    n = len(col_return)
    states = np.empty(n, dtype="int32")

    # Initialise state at one.
    states[0] = 1
    for i in range(1, n):
        states[i] = policy_step_jit(col_return[i - 1],
                                    col_rolling_j[i - 1],
                                    states[i - 1],
                                    gamma[i - 1],
                                    delta[i - 1])

    return states
