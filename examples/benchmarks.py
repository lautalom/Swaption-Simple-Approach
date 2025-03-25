"""
Benchmark functions for the Libor market model.
Contains functions for benchmarking the performance of the model.
"""

import argparse
import logging
import os
import sys
import time

import numpy as np

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pricing.bermudan import price_bermudan
from pricing.boundary import exercise_boundary2
from simulation.monte_carlo import simulate_forward_rates
from simulation.parallel import simulate_paths_parallel

# Add argument parsing for logging level
parser = argparse.ArgumentParser(
    description="Run benchmarks with configurable logging."
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

# Update INPUTS to include price and standard deviation values from the paper
INPUTS = [
    # Ts, Te, European, Payer, lambda, price, sd
    (1, 4, True, True, 0.20, 121.9, 0.5),
    (2, 4, True, True, 0.20, 111.2, 0.5),
    (3, 4, True, True, 0.20, 66.0, 0.3),
    (1, 4, False, True, 0.20, 157.7, 0.5),
    (1, 4, False, False, 0.20, 156.6, 0.3),
    (2, 5, True, True, 0.20, 162.0, 0.7),
    (3, 5, True, True, 0.20, 128.2, 0.6),
    (4, 5, True, True, 0.20, 71.7, 0.3),
    (2, 5, False, True, 0.20, 187.9, 0.6),
    (2, 5, False, False, 0.20, 186.6, 0.4),
    (5, 10, True, True, 0.15, 252.3, 1.0),
    (6, 10, True, True, 0.15, 214.6, 0.8),
    (7, 10, True, True, 0.15, 168.6, 0.7),
    (8, 10, True, True, 0.15, 116.5, 0.5),
    (9, 10, True, True, 0.15, 59.9, 0.2),
    (5, 10, False, True, 0.15, 282.7, 0.9),
    (5, 10, False, False, 0.15, 279.5, 0.6),
    (10, 20, True, True, 0.10, 309.0, 0.9),
    (12, 20, True, True, 0.10, 253.9, 0.8),
    (14, 20, True, True, 0.10, 193.2, 0.6),
    (16, 20, True, True, 0.10, 129.3, 0.4),
    (18, 20, True, True, 0.10, 64.6, 0.2),
    (10, 20, False, True, 0.10, 347.8, 0.8),
    (10, 20, False, False, 0.10, 339.6, 0.9),
]

# Constants for benchmarking
PRICING_N = 30000  # pricing simulations
BOUNDARY_N = 10000  # early boundary simulations


def write_results_to_file(results, filename):
    """
    Write the results array to a file in a tabular format.

    Args:
        results: List of tuples containing the results
        filename: The name of the file to write the results to
    """
    with open(filename, "w") as f:
        f.write("Start,End,Type,Payer,Lambda,Price,Sd,myPrice,mySD\n")
        for result in results:
            f.write(",".join(str(x) for x in result) + "\n")


def table_1():
    """
    Reproduce Table 1 from the Andersen paper.
    Benchmarks the performance of the model for various inputs.
    """
    results = []
    start_benchmark_time = time.perf_counter()
    for Ts, Te, european, payer, lamda, original_price, original_sd in INPUTS:
        start_time = time.perf_counter()

        # Create tenors using np.arange to match the original implementation
        tenors = np.arange(0, Te + 0.5, 0.5)

        if european:
            logging.info(
                f"Simulating paths for European swaption (Ts={Ts}, Te={Te})..."
            )
            values, numeraire = simulate_forward_rates(
                0.06, lamda, tenors, Ts, PRICING_N, european, payer
            )

            # Find the index of the maturity in tenors
            maturity_index = np.where(tenors >= Ts)[0][0]

            # Calculate discounted payoffs
            discount = numeraire[:, maturity_index].reshape(-1, 1)
            discounted = values / discount
            price = np.mean(discounted)
            std_dev = np.std(discounted, ddof=1) / np.sqrt(PRICING_N)

            elapsed_time = time.perf_counter() - start_time
            logging.debug(
                f"Ts={Ts}, Te={Te}, European={european}, Payer={payer}, lambda={lamda}"
            )
            logging.debug(f"Price: {price:.6f}, Std Dev: {std_dev:.6f}")
            logging.debug(
                f"Elapsed time for simulate_forward_rates: {elapsed_time:.2f} seconds"
            )
        else:
            # For Bermudan swaptions, use the boundary calculation approach
            logging.info(
                f"Simulating paths for Bermudan swaption (Ts={Ts}, Te={Te})..."
            )
            # Simulate paths to calculate exercise boundary
            values, numeraire = simulate_forward_rates(
                0.06, lamda, tenors, Ts, BOUNDARY_N, european, payer
            )

            # Calculate exercise boundary
            boundary = exercise_boundary2(values, numeraire)

            # Price the Bermudan swaption
            price, std_dev = price_bermudan(
                0.06, lamda, boundary, tenors, Ts, PRICING_N, payer
            )

            elapsed_time = time.perf_counter() - start_time
            logging.debug(
                f"Ts={Ts}, Te={Te}, European={european}, Payer={payer}, lambda={lamda}"
            )
            logging.debug(f"Price: {price:.6f}, Std Dev: {std_dev:.6f}")
            logging.debug(
                f"Elapsed time for price_bermudan: {elapsed_time:.2f} seconds"
            )

        price = round(price * 10000, 1)
        std_dev = round(std_dev * 10000, 1)
        results.append(
            (
                Ts,
                Te,
                european,
                payer,
                lamda,
                original_price,
                original_sd,
                price,
                std_dev,
            )
        )

    # Write results to file
    write_results_to_file(results, "table_1_results.csv")
    end_benchmark_time = time.perf_counter()
    logging.debug(
        f"Total elapsed time for benchmark: {end_benchmark_time - start_benchmark_time:.2f} seconds"
    )
    return results


def table_1_with_parallel():
    """
    Reproduce Table 1 with parallel implementations for comparison.
    """
    results = []
    start_benchmark_time = time.perf_counter()
    for Ts, Te, european, payer, lamda, original_price, original_sd in INPUTS:
        start_time = time.perf_counter()

        # Create tenors using np.arange to match the original implementation
        tenors = np.arange(0, Te + 0.5, 0.5)

        if european:
            logging.info(
                f"Simulating paths for European swaption (Ts={Ts}, Te={Te}) using parallel implementation..."
            )
            values, numeraire = simulate_paths_parallel(
                0.06, lamda, tenors, Ts, PRICING_N, european, payer
            )

            maturity_index = np.where(tenors >= Ts)[0][0]
            discount = numeraire[:, maturity_index].reshape(-1, 1)
            discounted = values / discount
            price = np.mean(discounted)
            std_dev = np.std(discounted, ddof=1) / np.sqrt(PRICING_N)

            elapsed_time = time.perf_counter() - start_time
            logging.debug(
                f"Ts={Ts}, Te={Te}, European={european}, Payer={payer}, lambda={lamda}"
            )
            logging.debug(f"Price: {price:.6f}, Std Dev: {std_dev:.6f}")
            logging.debug(
                f"Elapsed time for simulate_forward_rates: {elapsed_time:.2f} seconds"
            )
        else:
            logging.info(
                f"Simulating paths for Bermudan swaption (Ts={Ts}, Te={Te}) using parallel implementation..."
            )
            values, numeraire = simulate_paths_parallel(
                0.06, lamda, tenors, Ts, BOUNDARY_N, european, payer
            )

            boundary = exercise_boundary2(values, numeraire)

            price, std_dev = price_bermudan(
                0.06, lamda, boundary, tenors, Ts, PRICING_N, payer
            )

            elapsed_time = time.perf_counter() - start_time
            logging.debug(
                f"Ts={Ts}, Te={Te}, European={european}, Payer={payer}, lambda={lamda}"
            )
            logging.debug(f"Price: {price:.6f}, Std Dev: {std_dev:.6f}")
            logging.debug(
                f"Elapsed time for price_bermudan: {elapsed_time:.2f} seconds"
            )

        price = round(price * 10000, 1)
        std_dev = round(std_dev * 10000, 1)
        results.append(
            (
                Ts,
                Te,
                european,
                payer,
                lamda,
                original_price,
                original_sd,
                price,
                std_dev,
            )
        )

    write_results_to_file(results, "table_1_parallel_results.csv")
    end_benchmark_time = time.perf_counter()
    logging.debug(
        f"Total elapsed time for benchmark: {end_benchmark_time - start_benchmark_time:.2f} seconds"
    )
    return results


def test_single(Ts=1, Te=4, european=False, payer=True, lamda=0.2):
    """
    Test a single case for debugging purposes.

    Args:
        Ts (int, optional): Start time (swaption maturity). Defaults to 1.
        Te (int, optional): End time. Defaults to 4.
        european (bool, optional): Whether the swaption is European (True) or Bermudan (False). Defaults to False.
        payer (bool, optional): Whether the swaption is a payer (True) or receiver (False). Defaults to True.
        lamda (float, optional): Volatility parameter. Defaults to 0.2.
    """
    start_time = time.perf_counter()

    # Create tenors using np.arange to match the original implementation
    tenors = np.arange(0, Te + 0.5, 0.5)
    print(f"Tenor structure= {tenors}")

    if european:
        print("Simulating paths for European swaption pricing...")
        values, numeraire = simulate_forward_rates(
            0.06, lamda, tenors, Ts, PRICING_N, european, payer
        )

        # Find the index of the maturity in tenors
        maturity_index = np.where(tenors >= Ts)[0][0]

        # Calculate discounted payoffs
        discount = numeraire[:, maturity_index].reshape(-1, 1)
        discounted = values / discount
        price = np.mean(discounted)
        std_dev = np.std(discounted, ddof=1) / np.sqrt(PRICING_N)

        elapsed_time = time.perf_counter() - start_time
        print(f"Ts={Ts}, Te={Te}, European={european}, Payer={payer}, lambda={lamda}")
        print(
            f"Price (bp): {price*10000:.6f}, Std Dev (bp): {std_dev*10000:.6f}, Time: {elapsed_time:.2f} seconds"
        )
    else:
        # For Bermudan swaptions, use the boundary calculation approach
        print("Simulating paths for exercise boundary...")
        values, numeraire = simulate_forward_rates(
            0.06, lamda, tenors, Ts, BOUNDARY_N, european, payer
        )

        # Calculate exercise boundary
        print("Calculating exercise boundary...")
        boundary = exercise_boundary2(values, numeraire)
        print(f"Exercise boundary: {boundary}")

        # Price the Bermudan swaption
        print("Pricing Bermudan swaption...")
        price, std_dev = price_bermudan(
            0.06, lamda, boundary, tenors, Ts, PRICING_N, payer
        )

        elapsed_time = time.perf_counter() - start_time
        print(f"Ts={Ts}, Te={Te}, European={european}, Payer={payer}, lambda={lamda}")
        print(
            f"Price (bp): {price*10000:.6f}, Std Dev (bp): {std_dev*10000:.6f}, Time: {elapsed_time:.2f} seconds"
        )

    return price, std_dev


if __name__ == "__main__":
    table_1_with_parallel()
