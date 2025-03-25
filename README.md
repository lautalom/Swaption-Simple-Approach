# Libor market model for Bermudan Swaptions

This module provides a comprehensive implementation of the Libor market model for pricing Bermudan swaptions. The implementation is based on the extension by Andersen (1999). The implementation has been modularized and optimized for performance using Numba's JIT compilation and parallelization capabilities.


## Module Structure

The module is organized into the following submodules:

### Core

- `core/payoff.py`: Contains functions for calculating zero-coupon bond prices and swaption payoffs
- `core/rates.py`: Contains functions for calculating factors in interest rates dynamics
- `core/utils.py`: Contains utility functions used across the module

### Simulation

- `simulation/monte_carlo.py`: Contains functions for simulating forward rates using Monte Carlo methods
- `simulation/parallel.py`: Contains Numba-optimized parallel implementations of simulation functions

### Pricing

- `pricing/boundary.py`: Contains functions for calculating exercise boundaries
- `pricing/bermudan.py`: Contains functions for pricing Bermudan swaptions

### Examples

- `examples/benchmarks.py`: Contains benchmark functions and usage examples

## API Overview

The module exposes the following main functions:

- `simulate_forward_rates`: Simulates forward rate paths for boundary calculation
- `calculate_exercise_boundary`: Determines the optimal exercise boundary
- `price_bermudan`: Prices a Bermudan swaption using Monte Carlo simulation

## Requirements

- Poetry for dependency management. Its recommended to install Poetry in a virtual environment.

## Installation

Create a virtual environment and install the dependencies using Poetry:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
# Install Poetry
pip install poetry
# Install the package and its dependencies
poetry install
```

## Usage Examples

```python
# Import the module
import examples.benchmarks as benchmarks

# Run a single benchmark
benchmarks.test_single(
    Ts=1, # Time to maturity in years
    Te=5, # Total time in years
    european=False, # Set to True for European swaptions
    payer=True, # Set to True for payer swaptions
    lamda=0.2, # lambda parameter for the model
)
```

You can also run the benchmarks directly from the command line:

```bash
python examples/benchmarks.py
```

by default it will run all parallel benchmarks and store the results in a csv file (see `write_results_to_file`). You may also run `table_1()` or `test_single()` to see different benchmarks.

## Performance Optimization

Key optimizations include:

1. JIT compilation of core functions and parallel computations using [Numba](https://github.com/numba/numba)
2. Use of NumPy's vectorized operations
3. Antithetic variates for variance reduction

These optimizations significantly improve performance for CPU-bound Monte Carlo simulations.

## Mathematical Background

The Libor market model extension replicated here derives a parameterized exercise boundary for pricing Bermudan swaptions. The model simulates forward rates using a log-euler discretization of the stochastic differential equation dynamic of the forward rates given by:

dF_k(t) = F_k(t) * λ * [μ_k(t)dt + dW(t)]

where:
- F_k(t) is the forward rate at time t between k and k+1
- μ_k(t) is the term under spot measure
- λ is the a bounded function taken from model parameters
- dW(t) is a standard Brownian motion

The exercise boundary is calculated by solving backward in time from the last exercise date of the swaption to its first exercise date (maturity), using an initial run to determine a lower bound of sufficiently deep in-the-money values.
The pricing of the Bermudan swaption is then performed using Monte Carlo simulation, where the calculated boundary is used to determine the optimal exercise strategy.

## References

Andersen, L. B. (1999). A simple approach to the pricing of Bermudan swaptions in the multi-factor Libor market model. Available at SSRN 155208.
