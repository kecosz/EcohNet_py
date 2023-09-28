from __future__ import annotations

import numpy as np
import scipy


def std(x: np.ndarray):
    """Standardize column-wise array with experimental standard deviation."""
    return scipy.stats.zscore(x, ddof=1)


def SetVariables(
    data: np.ndarray, targetpos: int, delay: int
) -> tuple[np.ndarray, np.ndarray]:
    """Select input and target variables from time series data.
    Target is delayed for prediction.

    Args:
        data : np.ndarray
            Time series data, 2-D array of the shape of (length of the time series, number of variables).
        targetpos : int
            Index of a target variable.
        delay : int
            Number of time step for target to be delayed.
            It is assumed that the target is predicted using input data up to `delay` steps back.

    Returns:
        input : np.ndarray
        target : np.ndarray
    """
    input = data[:-delay]
    target = data[delay:, targetpos]
    return input, target
