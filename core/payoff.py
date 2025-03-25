"""
Payoff calculations for interest rate derivatives.

This module provides functions for calculating zero-coupon bond prices and swaption payoffs
in the Libor market model framework. These core functions are used throughout the pricing process
for both European and Bermudan swaptions.

Key components:
- Zero-coupon bond pricing using forward rates
- Payer swaption payoff calculations
- Receiver swaption payoff calculations

Mathematical background:
The zero-coupon bond price P(t,T) at time t for maturity T is calculated using
the forward rates according to the relationship:
P(t,T) = ∏[i=t+1 to T] 1/(1 + δᵢ₋₁ * L(t,Tᵢ₋₁))
where δᵢ₋₁ is the accrual factor (Tᵢ - Tᵢ₋₁) and L(t,Tᵢ₋₁) is the forward rate.
"""

import os
import sys

import numpy as np
from numba import njit

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@njit
def zcb2(t: float, forward_rates: np.ndarray, tenors: np.ndarray) -> np.ndarray:
    """
    Calculate zero-coupon bond prices at time t.

    Args:
        t (float): start time measured in years
        forward_rates (np.ndarray): forward rates starting in 0
        tenors (np.ndarray): time points for the forward rates

    Returns:
        np.ndarray: The value of the zero coupon bonds at time t

    Mathematical explanation:
    The zero-coupon bond price P(t,T) at time t for maturity T is calculated using
    the forward rates according to the relationship:
    P(t,T) = ∏[i=t+1 to T] 1/(1 + δᵢ₋₁ * L(t,Tᵢ₋₁))
    where δᵢ₋₁ is the accrual factor (Tᵢ - Tᵢ₋₁) and L(t,Tᵢ₋₁) is the forward rate.
    """
    start_index = np.searchsorted(tenors, t) + 1
    zcb = np.ones(len(tenors))
    for i in range(start_index, len(zcb)):
        zcb[i] = zcb[i - 1] / (1 + (tenors[i] - tenors[i - 1]) * forward_rates[i - 1])
    return zcb[start_index:]


@njit
def payer_swaption_payoff(
    zcb: np.ndarray, fixed_rate: float, tenors: np.ndarray
) -> float:
    """
    Calculate the intrinsic value of the Payer Bermudan swaption at time t.
    Assumes Notional = 1

    Args:
        zcb (np.ndarray): A 1D array of zero-coupon bond prices at time t
        fixed_rate (float): The fixed rate of the swap
        tenors (np.ndarray): A 1D array of time points for the forward rates

    Returns:
        float: The intrinsic value of the swaption at time t (scalar)

    Mathematical explanation:
    The payer swaption payoff is calculated as the maximum of the difference between
    the floating leg and the fixed leg, and zero.
    """
    swaption_payoff = (
        1
        - zcb[-1]
        - fixed_rate * np.sum((tenors[1 : 1 + len(zcb)] - tenors[: len(zcb)]) * zcb)
    )
    return max(swaption_payoff, 0)


@njit
def receiver_swaption_payoff(
    zcb: np.ndarray, fixed_rate: float, tenors: np.ndarray
) -> float:
    """
    Calculate the intrinsic value of the Receiver Bermudan swaption at time t.
    Assumes Notional = 1

    Args:
        zcb (np.ndarray): A list of zero-coupon bond prices at time t
        fixed_rate (float): The fixed rate of the swap
        tenors (np.ndarray): time points for the forward rates

    Returns:
        float: The intrinsic value of the swaption at time t

    Mathematical explanation:
    The receiver swaption payoff is calculated as the maximum of the difference between
    the fixed leg and the floating leg, and zero.
    """
    swaption_payoff = (
        zcb[-1]
        + fixed_rate * np.sum((tenors[1 : 1 + len(zcb)] - tenors[: len(zcb)]) * zcb)
        - 1
    )
    return max(swaption_payoff, 0)
