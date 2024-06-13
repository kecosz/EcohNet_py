from __future__ import annotations

from typing import cast
from unittest.mock import patch

import numpy as np
from numba import f8, njit
from scipy.stats import scoreatpercentile
from statsmodels.nonparametric.kde import KDEUnivariate


def _select_sigma(x, percentile=25):
    """
    Make kernel density estimation behave the same as wolfram.
    Overloads `statsmodels.nonparametric.bandwidths`.

    Returns the smaller of std(X, ddof=1) or normalized IQR(X) over axis 0.
    """
    normalize = 1.34  # 1.349 in statsmodels
    IQR = (
        scoreatpercentile(x, 75, interpolation_method="lower")
        - scoreatpercentile(x, 25, interpolation_method="lower")
    ) / normalize
    std_dev = np.std(x, axis=0, ddof=1)
    if IQR > 0:
        return np.minimum(std_dev, IQR)
    else:
        return std_dev


@patch("statsmodels.nonparametric.bandwidths._select_sigma", _select_sigma)
def Prop(dist: np.ndarray) -> tuple[float, KDEUnivariate]:
    """Returns median and kernel density estimation for array.

    Args:
        dist (np.ndarray): Array to infer.

    Returns:
        median (float)
        kde (KDEUnivariate)
    """
    median = float(np.median(dist))
    # pdf is obtained by `kde.evaluate(x_grid)`
    # cdf is obtained by `np.interp(x_grid, kde.support, kde.cdf)`
    kde = KDEUnivariate(dist)
    kde.fit(kernel="gau", bw="silverman")
    return median, kde


def Evp(
    pa: tuple[float, KDEUnivariate],
    pb: tuple[float, KDEUnivariate],
) -> tuple[float, float]:
    """Returns relative difference of distriution pa to pb.

    Args:
        pa (tuple[float, KDEUnivariate])
        pb (tuple[float, KDEUnivariate])

    Returns:
        [0] (float): median difference
        [1] (float):
            area of 窶銀逆he region greater than the median of pa in
            the probability distribution of pb
    """
    cdf_at_median_of_pa = np.interp(
        pa[0],
        pb[1].support,
        cast(np.ndarray, pb[1].cdf),
    )
    return pa[0] - pb[0], 1 - cdf_at_median_of_pa  # type: ignore


@njit(f8(f8[::1], f8[::1]), cache=True)
def RMSE(target: np.ndarray, tgest: np.ndarray) -> float:
    """Returns prediction skill ρ.
    - ρ = exp(-D)
    - D is the distance between the original time series and the predicted time series, removing earlier data.

    Args:
        target (np.ndarray): predicted time series, 1D array.
        tgest (np.ndarray): original time series, 1D array.

    Returns:
        float: prediction skill
    """
    remaining_length = int(0.8 * len(tgest))
    diff = np.subtract(target, tgest)[-remaining_length:]
    rms = np.sqrt(np.mean((diff) ** 2))  # root mean square
    rho = np.exp(-rms)
    return rho


def BRMSE(target: np.ndarray, tgest: np.ndarray):
    """Returns prediction skill ρ.
    Remove too many transients (threshold is 10%).
    """
    cr = sum(target == min(target)) / len(target)
    remaining_length = int(0.8 * len(target))
    if cr > 0.1:
        mtg = min(target)
        arr = np.array([target, tgest]).T[-remaining_length:]
        diff = np.array([0 if t <= mtg and e < t else e - t for t, e in arr])
        rms = np.sqrt(np.mean((diff**2)))
        return np.exp(-rms)
    else:
        diff = np.subtract(tgest, target)[-remaining_length:]
        rms = np.sqrt(np.mean((diff**2)))
        return np.exp(-rms)


def ES(p, q):
    return (p[0] - q[0]) / (
        (((p[2] - 1) * p[1] + (q[2] - 1) * q[1]) / (p[2] + q[2] - 2)) ** 0.5
    )
