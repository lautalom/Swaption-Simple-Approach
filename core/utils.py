"""
Utility functions for the Libor market model.

This module provides helper functions used throughout the Libor market model implementation.
These utilities handle common operations like array indexing and other shared functionality.
"""

import numpy as np
from numba import njit


@njit
def np_index(array: np.ndarray, value: float) -> int:
    """
    Find the index of the first value in an array that is greater than or equal to the target value.

    Args:
        array (np.ndarray): The array to search in
        value (float): The target value to find

    Returns:
        int: The index of the first element in the array that is >= value

    Example:
        >>> np_index(np.array([0.0, 0.5, 1.0, 1.5]), 0.7)
        2  # Returns index of 1.0, which is the first value >= 0.7
    """
    return np.where(array >= value)[0][0]
