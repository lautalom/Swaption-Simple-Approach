"""
Forward rate calculations for the Libor market model.

This module provides functions for calculating terms related to forward rates dynamics in the Libor market model framework.

"""

import numpy as np
from numba import njit


@njit
def mu_k(
    t: int, lam: float, forward_rates: np.ndarray, tenors: np.ndarray
) -> np.ndarray:
    """
    Calculate the term mu_k(t) for the forward rate F_k(t) under the spot measure.

    Args:
        t (int): The current time index
        lam (float): Volatility parameter (deterministic bounded function)
        forward_rates (np.ndarray): A numpy array of forward rates F_j(t)
        tenors (np.ndarray): A list of tenors corresponding to each forward rate

    Returns:
        np.ndarray: The mu terms for the forward rates F_k(t) for k > t

    """
    delta = np.diff(tenors)  # Calculate the day count fractions between tenor dates
    nrates = len(tenors) - 1
    mu_ti = np.zeros(nrates - (t + 1))
    mu_ti[0] = lam * (delta[t] * forward_rates[t]) / (1 + delta[t] * forward_rates[t])

    for j in range(t + 2, nrates):
        F_j = forward_rates[j]
        delta_j = delta[j - 1]
        mu_ti[j - (t + 1)] = mu_ti[j - (t + 2)] + lam * (delta_j * F_j) / (
            1 + delta_j * F_j
        )

    return mu_ti
