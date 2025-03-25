"""
Libor market model for pricing Bermudan swaptions.

This module provides a comprehensive implementation of the Libor market model for pricing
Bermudan swaptions. The implementation has been optimized for performance using Numba's
JIT compilation capabilities.

Key components:
- Core mathematical functions for payoff calculations and rate dynamics
- Monte Carlo simulation with both standard and antithetic approaches
- Exercise boundary calculation for optimal stopping
- Bermudan swaption pricing with variance reduction techniques

"""

# Import core functions
from .core.payoff import payer_swaption_payoff, receiver_swaption_payoff, zcb2
from .core.rates import mu_k
from .core.utils import np_index

# Import example functions
from .examples.benchmarks import INPUTS, table_1, test_single
from .pricing.bermudan import price_bermudan

# Import pricing functions
from .pricing.boundary import calculate_exercise_boundary, exercise_boundary2

# Import simulation functions
from .simulation.monte_carlo import (
    antithetic_simulate_forward_rates,
    simulate_forward_rates,
)
from .simulation.parallel import (
    antithetic_simulate_paths_parallel,
    simulate_paths_parallel,
)

# Constants
PRICING_N = 30000  # Number of simulations for pricing
BOUNDARY_N = 10000  # Number of simulations for boundary calculation

__all__ = [
    # Core functions
    "zcb2",
    "payer_swaption_payoff",
    "receiver_swaption_payoff",
    "mu_k",
    "np_index",
    # Simulation functions
    "simulate_forward_rates",
    "antithetic_simulate_forward_rates",
    "simulate_paths_parallel",
    "antithetic_simulate_paths_parallel",
    # Pricing functions
    "exercise_boundary2",
    "calculate_exercise_boundary",
    "price_bermudan",
    # Example functions
    "table_1",
    "test_single",
    "INPUTS",
    # Constants
    "PRICING_N",
    "BOUNDARY_N",
]
