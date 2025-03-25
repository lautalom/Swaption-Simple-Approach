"""
Monte Carlo simulation functions for the Libor market model.

This module provides functions for simulating forward rates using Monte Carlo methods
in the Libor market model framework. It includes both standard and antithetic simulation
approaches to reduce variance in the price estimates.

"""

import argparse
import logging
import os
import random
import sys
import time

import numpy as np
from numba import njit

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.payoff import payer_swaption_payoff, receiver_swaption_payoff, zcb2
from core.rates import mu_k

# Add argument parsing for logging level
parser = argparse.ArgumentParser(
    description="Run Monte Carlo simulations with configurable logging."
)
parser.add_argument(
    "--log-level",
    type=str,
    default="INFO",
    help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
)
args = parser.parse_args()

# Configure logging with the specified level
logging.basicConfig(
    level=getattr(logging, args.log_level.upper(), logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("application.log", mode="w"),
    ],
)


def simulate_forward_rates(
    L_0: float,
    lam: float,
    tenors: np.ndarray,
    maturity: float,
    n_paths: int,
    european: bool = False,
    payer: bool = True,
) -> tuple:
    """
    Simulates forward Libor rates using a log-normal model under the spot measure.

    This function generates Monte Carlo paths for forward rates and calculates
    the intrinsic values of the swaption at each exercise date. It is used primarily
    for calculating the exercise boundary in Bermudan swaptions.

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

    Example:
        >>> tenors = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        >>> values, numeraire = simulate_forward_rates(0.06, 0.2, tenors, 2.0, 10000)
    """
    start_time = time.perf_counter()
    n_rates = len(tenors) - 1
    forward_rates = np.zeros((n_rates, n_rates))
    # Initialize column 0 with flat forward rate
    forward_rates[:, 0] = [L_0] * (n_rates)
    zcbs = zcb2(0, forward_rates[:, 0], tenors)
    if european:
        values_dict = np.zeros((n_paths, 1))
    else:
        values_dict = np.zeros(
            (n_paths, len(tenors[np.searchsorted(tenors, maturity) : -1]))
        )
    numeraire_dict = np.ones((n_paths, n_rates + 1))
    numeraire_dict[:, 1] = 1 / zcbs[0]

    for nsim in range(n_paths):
        for i in range(1, n_rates):
            dt = tenors[i] - tenors[i - 1]
            W = random.normalvariate(0, 1)
            scaled_gauss = W * np.sqrt(dt)
            spot_measure_brownian = (
                mu_k(i - 1, lam, forward_rates[:, i - 1], tenors) - 0.5 * lam
            ) * dt + scaled_gauss
            exp = np.exp(lam * spot_measure_brownian)
            forward_rates[i:, i] = forward_rates[i:, i - 1] * exp
            zcbs = zcb2(tenors[i], forward_rates[:, i], tenors)

            if european and tenors[i] == maturity:
                values_dict[nsim, 0] = (
                    payer_swaption_payoff(zcbs, L_0, tenors)
                    if payer
                    else receiver_swaption_payoff(zcbs, L_0, tenors)
                )
            elif tenors[i] >= maturity and not european:
                offset = len(tenors[: np.searchsorted(tenors, maturity)])
                values_dict[nsim, i - offset] = (
                    payer_swaption_payoff(zcbs, L_0, tenors)
                    if payer
                    else receiver_swaption_payoff(zcbs, L_0, tenors)
                )
            numeraire_dict[nsim, i + 1] = numeraire_dict[nsim, i] / zcbs[0]
    elapsed_time = time.perf_counter() - start_time
    logging.debug(
        f"Elapsed time for simulate_forward_rates: {elapsed_time:.2f} seconds"
    )
    return values_dict, numeraire_dict


@njit
def antithetic_simulate_forward_rates(
    start_rate: float,
    lam: float,
    boundary: np.ndarray,
    n_paths: int,
    tenors: np.ndarray,
    maturity: float,
    payer: bool = True,
) -> tuple:
    """
    Simulates forward Libor rates using antithetic sampling to reduce variance.

    This function generates Monte Carlo paths with antithetic variates, which helps
    reduce the variance of the price estimates. For each random draw, both the draw
    and its negative are used, creating pairs of paths with negative correlation.

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

    Example:
        >>> tenors = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        >>> boundary = np.array([0.02, 0.015, 0.01, 0.005, 0.0])
        >>> payoffs, numeraire, payoffs_a, numeraire_a = antithetic_simulate_forward_rates(
        ...     0.06, 0.2, boundary, 10000, tenors, 2.0)
    """
    n_rates = len(tenors) - 1
    forward_rates = np.zeros((n_rates, n_rates))
    forward_rates[:, 0] = start_rate
    forward_rates_a = np.zeros((n_rates, n_rates))
    forward_rates_a[:, 0] = start_rate
    zcbs = zcb2(0, forward_rates[:, 0], tenors)
    # payoffs are all exercisable dates
    payoffs_dict = np.zeros(
        (n_paths, len(tenors[np.searchsorted(tenors, maturity) : -1]))
    )
    payoffs_dict_a = np.zeros(
        (n_paths, len(tenors[np.searchsorted(tenors, maturity) : -1]))
    )
    numeraire_dict = np.ones((n_paths, n_rates + 1))
    numeraire_dict_a = np.ones((n_paths, n_rates + 1))
    numeraire_dict[:, 1] = 1 / zcbs[0]
    numeraire_dict_a[:, 1] = 1 / zcbs[0]

    for nsim in range(n_paths):
        exercised = False
        exercised_a = False
        for i in range(1, n_rates):
            dt = tenors[i] - tenors[i - 1]
            W = np.random.normal(0, 1)
            Wa = -W
            scaled_gauss = W * np.sqrt(dt)
            scaled_gauss_a = Wa * np.sqrt(dt)

            spot_measure_brownian = (
                mu_k(i - 1, lam, forward_rates[:, i - 1], tenors) - 0.5 * lam
            ) * dt + scaled_gauss
            forward_rates[i:, i] = forward_rates[i:, i - 1] * np.exp(
                lam * spot_measure_brownian
            )

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
