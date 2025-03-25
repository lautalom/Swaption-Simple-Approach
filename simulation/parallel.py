"""
Parallel simulation functions for the Libor market model.

This module provides Numba-optimized parallel implementations of simulation functions
for the Libor market model. These implementations leverage Numba's JIT compilation and
parallel capabilities to significantly improve performance for CPU-bound Monte Carlo
simulations.

Key optimizations:
- JIT compilation of simulation functions
- Parallel execution using Numba's prange
- Pre-allocation of arrays for results
- Vectorized operations where possible
"""

import os
import sys

import numpy as np
from numba import njit, prange

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.payoff import payer_swaption_payoff, receiver_swaption_payoff, zcb2
from core.rates import mu_k


@njit(parallel=True)
def simulate_paths_parallel(
    L_0: float,
    lam: float,
    tenors: np.ndarray,
    maturity: float,
    n_paths: int,
    european: bool = False,
    payer: bool = True,
) -> tuple:
    """
    Parallel implementation of forward rate simulation using Numba's prange.

    This function simulates forward rate paths in parallel using Numba's prange,
    which distributes the simulation workload across multiple CPU cores.

    Args:
        L_0 (float): Initial forward rate (e.g., 0.06 for 6% flat forward curve)
        lam (float): Volatility parameter for the forward rate process
        tenors (np.ndarray): Time points for the forward rates
        maturity (float): Maturity of the swaption (first exercise date)
        n_paths (int): Number of Monte Carlo simulation paths
        european (bool, optional): Whether the swaption is European (True) or Bermudan (False). Defaults to False.
        payer (bool, optional): Whether the swaption is a payer (True) or receiver (False). Defaults to True.

    Returns:
        tuple: A tuple containing:
            - values (np.ndarray): Intrinsic values for each simulation path at each exercise date
            - numeraire (np.ndarray): Numeraire values for each simulation path at each time step
    """
    n_rates = len(tenors) - 1

    # Pre-allocate arrays for results
    if european:
        values_dict = np.zeros((n_paths, 1))
    else:
        values_dict = np.zeros(
            (n_paths, len(tenors[np.searchsorted(tenors, maturity) : -1]))
        )
    numeraire_dict = np.ones((n_paths, n_rates + 1))

    # Initialize the first column of forward rates with the flat rate
    forward_rates_init = np.full(n_rates, L_0)
    zcbs = zcb2(0, forward_rates_init, tenors)
    numeraire_dict[:, 1] = 1 / zcbs[0]

    # Parallel simulation of paths
    for nsim in prange(n_paths):
        # Create local copy of forward rates for this path
        forward_rates = np.zeros((n_rates, n_rates))
        forward_rates[:, 0] = forward_rates_init

        for i in range(1, n_rates):
            dt = tenors[i] - tenors[i - 1]
            W = np.random.normal(0, 1)
            scaled_gauss = W * np.sqrt(dt)
            spot_measure_brownian = (
                mu_k(i - 1, lam, forward_rates[:, i - 1], tenors) - 0.5 * lam
            ) * dt + scaled_gauss
            exp = np.exp(lam * spot_measure_brownian)
            forward_rates[i:, i] = forward_rates[i:, i - 1] * exp
            zcbs = zcb2(tenors[i], forward_rates[:, i], tenors)

            if european and tenors[i] == maturity:
                if payer:
                    values_dict[nsim, 0] = payer_swaption_payoff(zcbs, L_0, tenors)
                else:
                    values_dict[nsim, 0] = receiver_swaption_payoff(zcbs, L_0, tenors)
            elif tenors[i] >= maturity and not european:
                if payer:
                    values_dict[
                        nsim, i - len(tenors[: np.searchsorted(tenors, maturity)])
                    ] = payer_swaption_payoff(zcbs, L_0, tenors)
                else:
                    values_dict[
                        nsim, i - len(tenors[: np.searchsorted(tenors, maturity)])
                    ] = receiver_swaption_payoff(zcbs, L_0, tenors)
            numeraire_dict[nsim, i + 1] = numeraire_dict[nsim, i] / zcbs[0]

    return values_dict, numeraire_dict


@njit(parallel=True)
def antithetic_simulate_paths_parallel(
    start_rate: float,
    lam: float,
    boundary: np.ndarray,
    n_paths: int,
    tenors: np.ndarray,
    maturity: float,
    payer: bool = True,
) -> tuple:
    """
    Parallel implementation of antithetic forward rate simulation using Numba's prange.

    This function simulates forward rate paths with antithetic variates in parallel
    using Numba's prange, which distributes the simulation workload across multiple CPU cores.

    Args:
        start_rate (float): Initial forward rate (e.g., 0.06 for 6% flat forward curve)
        lam (float): Volatility parameter for the forward rate process
        boundary (np.ndarray): Pre-calculated exercise boundary for each exercise date
        n_paths (int): Number of Monte Carlo simulation paths
        tenors (np.ndarray): Time points for the forward rates
        maturity (float): Maturity of the swaption (first exercise date)
        payer (bool, optional): Whether the swaption is a payer (True) or receiver (False). Defaults to True.

    Returns:
        tuple: A tuple containing:
            - payoffs (np.ndarray): Payoffs for each simulation path at each exercise date
            - numeraire (np.ndarray): Numeraire values for each simulation path
            - payoffs_a (np.ndarray): Antithetic payoffs for each simulation path
            - numeraire_a (np.ndarray): Antithetic numeraire values for each simulation path
    """
    n_rates = len(tenors) - 1

    # Pre-allocate arrays for results
    payoffs_dict = np.zeros(
        (n_paths, len(tenors[np.searchsorted(tenors, maturity) : -1]))
    )
    payoffs_dict_a = np.zeros(
        (n_paths, len(tenors[np.searchsorted(tenors, maturity) : -1]))
    )
    numeraire_dict = np.ones((n_paths, n_rates + 1))
    numeraire_dict_a = np.ones((n_paths, n_rates + 1))

    # Initialize forward rates
    forward_rates_init = np.full(n_rates, start_rate)
    zcbs = zcb2(0, forward_rates_init, tenors)
    numeraire_dict[:, 1] = 1 / zcbs[0]
    numeraire_dict_a[:, 1] = 1 / zcbs[0]

    # Parallel simulation of paths
    for nsim in prange(n_paths):
        # Create local copies of forward rates for this path
        forward_rates = np.zeros((n_rates, n_rates))
        forward_rates[:, 0] = forward_rates_init
        forward_rates_a = np.zeros((n_rates, n_rates))
        forward_rates_a[:, 0] = forward_rates_init

        exercised = False
        exercised_a = False

        for i in range(1, n_rates):
            dt = tenors[i] - tenors[i - 1]
            W = np.random.normal(0, 1)
            Wa = -W  # Antithetic variate
            scaled_gauss = W * np.sqrt(dt)
            scaled_gauss_a = Wa * np.sqrt(dt)

            # Standard path
            spot_measure_brownian = (
                mu_k(i - 1, lam, forward_rates[:, i - 1], tenors) - 0.5 * lam
            ) * dt + scaled_gauss
            forward_rates[i:, i] = forward_rates[i:, i - 1] * np.exp(
                lam * spot_measure_brownian
            )

            # Antithetic path
            spot_measure_brownian_a = (
                mu_k(i - 1, lam, forward_rates_a[:, i - 1], tenors) - 0.5 * lam
            ) * dt + scaled_gauss_a
            forward_rates_a[i:, i] = forward_rates_a[i:, i - 1] * np.exp(
                lam * spot_measure_brownian_a
            )

            zcbs = zcb2(tenors[i], forward_rates[:, i], tenors)
            zcbs_a = zcb2(tenors[i], forward_rates_a[:, i], tenors)

            if payer:
                payoff = payer_swaption_payoff(zcbs, start_rate, tenors)
                payoff_a = payer_swaption_payoff(zcbs_a, start_rate, tenors)
            else:
                payoff = receiver_swaption_payoff(zcbs, start_rate, tenors)
                payoff_a = receiver_swaption_payoff(zcbs_a, start_rate, tenors)

            if tenors[i] >= maturity:
                if not exercised:
                    boundary_idx = i - np.searchsorted(tenors, maturity)
                    if (
                        boundary_idx < len(boundary)
                        and payoff >= boundary[boundary_idx]
                    ):
                        payoffs_dict[
                            nsim, i - len(tenors[: np.searchsorted(tenors, maturity)])
                        ] = payoff
                        exercised = True
                if not exercised_a:
                    boundary_idx = i - np.searchsorted(tenors, maturity)
                    if (
                        boundary_idx < len(boundary)
                        and payoff_a >= boundary[boundary_idx]
                    ):
                        payoffs_dict_a[
                            nsim, i - len(tenors[: np.searchsorted(tenors, maturity)])
                        ] = payoff_a
                        exercised_a = True

            numeraire_dict[nsim, i + 1] = numeraire_dict[nsim, i] / zcbs[0]
            numeraire_dict_a[nsim, i + 1] = numeraire_dict_a[nsim, i] / zcbs_a[0]

    return payoffs_dict, numeraire_dict, payoffs_dict_a, numeraire_dict_a
