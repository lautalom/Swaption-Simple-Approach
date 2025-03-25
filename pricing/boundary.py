"""
Exercise boundary calculations for Bermudan swaptions.

This module provides functions for calculating exercise boundaries for Bermudan swaptions
using the Andersen method. The exercise boundary is a critical component in pricing
Bermudan swaptions, as it determines the optimal exercise strategy.

Mathematical background:
The exercise boundary represents the threshold value at each exercise date where it becomes
optimal to exercise the swaption rather than continue holding it. The boundary is calculated
by backward induction, starting from last exercise date and moving towards the first exercise date.
"""

import argparse
import logging
import time

import numpy as np

# Add argument parsing for logging level
parser = argparse.ArgumentParser(
    description="Run boundary calculations with configurable logging."
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


def calculate_exercise_boundary(
    values: np.ndarray, numeraire: np.ndarray
) -> np.ndarray:
    """
    Calculate the exercise boundary for the Bermudan swaption using a dynamic programming approach.

    This function implements the core algorithm for determining the optimal exercise boundary
    at each potential exercise date by maximizing the expected payoff.

    Args:
        values (np.ndarray): A numpy array of payoffs for each path and time step
        numeraire (np.ndarray): A numpy array of numeraire values for each path and time step

    Returns:
        np.ndarray: A list of exercise boundaries for each time step in chronological order
    """
    boundaries = [0]
    max_intrinsic = values[:, -1]
    # iterate backwards until the very first exercise date
    for i in range(2, len(values[0]) + 1):
        discount_factor = numeraire[:, -i] / numeraire[:, -i + 1]
        # Get current step's spot prices (curr_boundary)
        curr_boundary = values[:, -i]
        discounted = max_intrinsic[np.newaxis, :] * discount_factor[np.newaxis, :]
        intrinsic_values = np.where(
            curr_boundary >= curr_boundary[:, np.newaxis],
            curr_boundary[np.newaxis, :],
            discounted,
        )
        means = np.round(np.mean(intrinsic_values, axis=1), 15)

        # Find the boundary (maximizing mean intrinsic value)
        index = np.argmax(means)
        max_intrinsic = intrinsic_values[index, :]
        boundaries.append(curr_boundary[index])
    return boundaries


def exercise_boundary2(values: np.ndarray, numeraire: np.ndarray) -> np.ndarray:
    """
    Calculate the exercise boundary for the Bermudan swaption with performance timing.

    This is a wrapper function that calls calculate_exercise_boundary and measures
    the execution time. It returns the boundary in chronological order.

    Args:
        values (np.ndarray): A numpy array of payoffs for each path and time step
        numeraire (np.ndarray): A numpy array of numeraire values for each path and time step

    Returns:
        np.ndarray: A list of exercise boundaries for each time step in chronological order
    """
    start = time.perf_counter()
    boundaries = calculate_exercise_boundary(values, numeraire)

    logging.debug(
        f"Elapsed time for boundary calculation: {time.perf_counter() - start:.2f} seconds"
    )
    return np.flip(boundaries)
