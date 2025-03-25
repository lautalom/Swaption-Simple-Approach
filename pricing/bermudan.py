"""
Bermudan swaption pricing functions.

This module provides functions for pricing Bermudan swaptions using a boundary parametrization approach.

Mathematical background:
The price of a Bermudan swaption is calculated using Monte Carlo simulation with
antithetic variates to reduce variance. The exercise strategy is determined by
the exercise boundary calculated in the boundary module.
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
from numba import njit

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation.monte_carlo import antithetic_simulate_forward_rates

# Add argument parsing for logging level
parser = argparse.ArgumentParser(
    description="Run Bermudan pricing with configurable logging."
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


@njit
def np_index(array: np.ndarray, value: float) -> int:
    """
    Find the index of the first value in an array that is greater than or equal to the target value.

    Args:
        array (np.ndarray): The array to search in
        value (float): The target value to find

    Returns:
        int: The index of the first element in the array that is >= value
    """
    return np.where(array >= value)[0][0]


def price_bermudan(
    start_rate: float,
    lamda: float,
    boundary: np.ndarray,
    tenors: np.ndarray,
    maturity: float,
    n_simulations: int,
    payer: bool = True,
) -> tuple:
    """
    Price a Bermudan swaption using the Andersen method with Monte Carlo simulation.

    This function simulates forward rate paths using antithetic variates to reduce variance,
    and applies the pre-calculated exercise boundary to determine the optimal exercise strategy.
    The price is calculated as the expected discounted payoff under the risk-neutral measure.

    Args:
        start_rate (float): Initial forward rate (e.g., 0.06 for 6%)
        lamda (float): Volatility parameter for the forward rate process
        boundary (np.ndarray): Pre-calculated exercise boundary for each exercise date
        tenors (np.ndarray): Time points for the forward rates
        maturity (float): Maturity of the swaption (first exercise date)
        n_simulations (int): Number of Monte Carlo simulation paths
        payer (bool, optional): Whether the swaption is a payer (True) or receiver (False). Defaults to True.

    Returns:
        tuple: A tuple containing:
            - price (float): The price of the Bermudan swaption
            - std_dev (float): The standard deviation of the price estimate

    Example:
        >>> tenors = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        >>> boundary = np.array([0.02, 0.015, 0.01, 0.005, 0.0])
        >>> price, std_dev = price_bermudan(0.06, 0.2, boundary, tenors, 2.0, 10000)
        >>> print(f"Price: {price:.6f}, Std Dev: {std_dev:.6f}")
    """
    start_time = time.perf_counter()

    # Simulate paths with antithetic variates
    payoffs, numeraire, payoffs_a, numeraire_a = antithetic_simulate_forward_rates(
        start_rate, lamda, boundary, n_simulations, tenors, maturity, payer
    )

    @njit
    def compute_price_and_std_dev(payoffs, numeraire, payoffs_a, numeraire_a):
        """
        Compute the price and standard deviation from the simulation results.

        Args:
            payoffs: Payoffs from the normal simulation
            numeraire: Numeraire values from the normal simulation
            payoffs_a: Payoffs from the antithetic simulation
            numeraire_a: Numeraire values from the antithetic simulation

        Returns:
            tuple: (price, std_dev)
        """
        maturity_index = np_index(tenors, maturity)
        discounted_payoffs = np.sum(payoffs / numeraire[:, maturity_index:-1], axis=1)
        discounted_payoffs_a = np.sum(
            payoffs_a / numeraire_a[:, maturity_index:-1], axis=1
        )

        # Calculate the average price
        prima = np.mean(discounted_payoffs)
        prima_a = np.mean(discounted_payoffs_a)
        price = (prima + prima_a) / 2

        # Calculate the standard deviation
        all_discounted_payoffs = np.concatenate(
            (discounted_payoffs, discounted_payoffs_a)
        )
        mean = np.mean(all_discounted_payoffs)
        variance = np.mean((all_discounted_payoffs - mean) ** 2)
        std_dev = np.sqrt(variance) / np.sqrt(n_simulations)

        return price, std_dev

    price, std_dev = compute_price_and_std_dev(
        payoffs, numeraire, payoffs_a, numeraire_a
    )

    elapsed_time = time.perf_counter() - start_time
    logging.debug(f"Elapsed time for price_bermudan: {elapsed_time:.2f} seconds")

    return price, std_dev
