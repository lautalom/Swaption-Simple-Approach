"""
Simulation module for Libor market model.

This module provides Monte Carlo simulation functions for the Libor market model,
including both standard sequential implementations and Numba-optimized parallel
implementations for improved performance.

Available functions:
- simulate_forward_rates: Sequential implementation for forward rate simulation
- antithetic_simulate_forward_rates: Sequential implementation with antithetic variates
- simulate_paths_parallel: Parallel implementation for forward rate simulation
- antithetic_simulate_paths_parallel: Parallel implementation with antithetic variates
"""

from .monte_carlo import antithetic_simulate_forward_rates, simulate_forward_rates
from .parallel import antithetic_simulate_paths_parallel, simulate_paths_parallel

__all__ = [
    "simulate_forward_rates",
    "antithetic_simulate_forward_rates",
    "simulate_paths_parallel",
    "antithetic_simulate_paths_parallel",
]
