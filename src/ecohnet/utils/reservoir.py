from __future__ import annotations

from typing import Callable

import numba
import numpy as np
from numba import f8, i8, njit
from numba.core.types import Tuple

from .constant import Const
from .math import RMSE
from .random import set_seed

from .cy_reservoir import initializeNN as cy_initializeNN


def calc_bagg(input: np.ndarray, target: np.ndarray, seed: int) -> float:
    """Calculate prediction skill.

    Args:
        input (np.ndarray): Explanatory variable, 2-D array.
        target (np.ndarray): Objective variable, 1-D array.
        seed (int): Random seed.

    Returns:
        float: Prediction skill.
    """
    set_seed(seed)
    tgxe = Reservoirepd(
        input,
        target,
        nu_initializeNN(input.shape[1]),
    )
    return RMSE(target, tgxe)


def calc_baggs(inputs: np.ndarray, target: np.ndarray, seed: int) -> np.ndarray:
    """Calculate prediction skill for multi input sets.

    Args:
        inputs (np.ndarray): Multiple set of explanatory variable, 3-D array.
        target (np.ndarray): Objective variable, 1-D array.
        seed (int): Random seed.

    Returns:
        np.ndarray: Prediction skills for each input.
    """
    return np.array(
        [calc_bagg(input, target, seed + i) for i, input in enumerate(inputs)]
    )


def calc_bagg_and_tgtest(
    input: np.ndarray, target: np.ndarray, seed: int
) -> tuple[float, np.ndarray]:
    """Calculate prediction skill with all target data (without dropout).

    Args:
        input (np.ndarray): Explanatory variable, 2-D array.
        target (np.ndarray): Objective variable, 1-D array.
        seed (int): Random seed.

    Returns:
        float: Prediction skill.
        np.ndarray: Predicted time-series.
    """
    set_seed(seed)
    tgtest = Reservoirep(input, target, nu_initializeNN(len(input[0])))
    return RMSE(target, tgtest), tgtest


@njit(
    Tuple(
        (i8, i8, f8[:, ::1], f8[:, ::1], f8, f8),
    )(i8),
    cache=True,
)
def nu_initializeNN(
    nin: int,
) -> tuple[int, int, np.ndarray, np.ndarray, float, float]:
    """Return a new instance of ESN(echo state network).

    Parameter
    ---------
    nin : int
        Number of input variables.

    Returns
    -------
    nin : int
        Number of input variables.
    nx : int
        Number of recurrent nodes.
    win : np.ndarray.
        Input weight matrix, 1-D array of the shape (nx, nin).
    ww : np.ndarray
        Reservoir weight matrix, 2-D array of the shape (nx, nx).
    ilambda : float
        Inverse of forgetting rate in RLS.
    delta : float
        Regularize coefficient in RLS.
    """
    with numba.objmode(nx=i8, radius=f8, lamb=f8, delta=f8):
        nx, radius, lamb, delta = Const.q1, Const.p1, Const.p2, Const.p3

    scwin = 0.1
    win = (-1 + np.random.random((nx, nin)) * 2) * scwin
    # make ww, weight of resorvoir network
    conn = 0.1
    ww0 = np.empty((nx, nx))
    while True:
        for i in range(nx):
            for j in range(nx):
                if np.random.random() < conn:
                    ww0[i, j] = -1 + np.random.random() * 2
                else:
                    ww0[i, j] = 0

        crit = np.sum((np.abs((np.sum(ww0, axis=0) * np.sum(ww0, axis=1))) < 1e-8))
        if crit == 0:
            break
    rho0 = np.max(np.abs(np.linalg.eigvals(ww0.astype(np.complex64))))
    gamma = radius * (1 / rho0)
    ww = gamma * ww0
    # make others
    ilambda = 1 / lamb
    return nin, nx, win, ww, ilambda, delta


def initializeNN(nin: int) -> tuple[int, int, np.ndarray, np.ndarray, float, float]:
    """Return a new instance of ESN(echo state network).

    Parameter
    ---------
    nin : int
        Number of input variables.

    Returns
    -------
    nin : int
        Number of input variables.
    nx : int
        Number of recurrent nodes.
    win : np.ndarray.
        Input weight matrix, 1-D array of the shape (nx, nin).
    ww : np.ndarray
        Reservoir weight matrix, 2-D array of the shape (nx, nx).
    ilambda : float
        Inverse of forgetting rate in RLS.
    delta : float
        Regularize coefficient in RLS.
    """
    nx = Const.q1
    # make win
    scwin = 0.1
    win = (-1 + np.random.random((nx, nin)) * 2) * scwin
    # make ww, weight of resorvoir network
    conn = 0.1
    make_ww0: Callable = np.vectorize(
        lambda _: -1 + np.random.random() * 2 if np.random.random() < conn else 0,
        otypes=[np.float64],
    )
    while True:
        ww0 = make_ww0(np.empty((nx, nx)))
        crit = sum(np.abs((sum(ww0) * sum(ww0.T))) < 1e-8)
        if crit == 0:
            break
    rho0 = max(abs(np.linalg.eigvals(ww0)))
    gamma = Const.p1 * (1 / rho0)
    ww = gamma * ww0
    # make others
    ilambda = 1 / Const.p2
    delta = Const.p3
    return nin, nx, win, ww, ilambda, delta


def Reservoirep(inn: np.ndarray, out: np.ndarray, parameters: tuple):
    """Predict multivariate time series with ESN-RLS
    (echo state network updated by recursive least square method).

    Parameters
    ----------
    inn : np.ndarray
        Input time series, 2-D array of the shape (length of time series, number of variables).
    out : np.ndarray
        Target variable, 1-D array of the length of time series.
        output[t] is predicted from input[:t+1], including input[t] (output is already delayed).
    parameters : tuple[Any]
        input: np.ndarray,
        output: np.ndarray,
        nin: int,
        nx: int,
        win: np.ndarray,
        ww: np.ndarray,
        ilambda: float,
        delta: float,

    Returns
    -------
    np.ndarray :
        predicted target variable, 1-D array of the length of the time series.
    """
    return comp(inn, out, *parameters, Const.p4)


@numba.njit(
    [
        f8[::1](
            f8[:, ::1],
            f8[::1],
            i8,
            i8,
            f8[:, ::1],
            f8[:, ::1],
            f8,
            f8,
            i8,
        ),
    ],
    cache=True,
)
def comp(
    input: np.ndarray,
    output: np.ndarray,
    nin: int,
    nx: int,
    win: np.ndarray,
    ww: np.ndarray,
    ilambda: float,
    delta: float,
    n_update: int,
) -> np.ndarray:
    """Auxiliary functions for the `Reservoirep` function."""
    nt = input.shape[0]
    wou = np.zeros(nx)
    xi = np.zeros(nx)
    pn = np.eye(nx) / delta
    yprds = np.empty(len(input))

    for i in range(nt):
        l1 = np.tanh(win @ input[i] + ww @ xi)
        yprds[i] = wou @ l1
        for _ in range(n_update):
            ouprd = wou @ l1
            vn = output[i] - ouprd
            tmp = pn @ xi
            gn = ilambda * tmp / (1 + ilambda * (xi @ tmp))
            pn = ilambda * (pn - np.outer(gn, xi) * pn)
            # state update
            xi = l1
            wou = wou + vn * gn
    return yprds


def Reservoirepd(inn: np.ndarray, out: np.ndarray, parameters: tuple):
    """Predict multivariate time series with ESN-RLS
    (echo state network updated by recursive least square method).
    The difference from the `Reservoiep` function is the addition of a dropout layer.

    Parameters
    ----------
    inn : np.ndarray
        Input time series, 2-D array of the shape (length of time series, number of variables).
    out : np.ndarray
        Target variable, 1-D array of the length of time series.
        output[t] is predicted from input[:t+1], including input[t] (target is already delayed).
    parameters : tuple[Any]
        input: np.ndarray,
        output: np.ndarray,
        nin: int,
        nx: int,
        win: np.ndarray,
        ww: np.ndarray,
        ilambda: float,
        delta: float,

    Returns
    -------
    np.ndarray :
        Predicted target variable, 1-D array of the length of the time series.
    """
    ret = nu_compd(inn, out, *parameters, Const.p4)
    return ret


def compd(
    input: np.ndarray,
    output: np.ndarray,
    nin: int,
    nx: int,
    win: np.ndarray,
    ww: np.ndarray,
    ilambda: float,
    delta: float,
    n_update: int,
) -> np.ndarray:
    """Auxiliary functions for the `Reservoirepd` function."""
    wou = np.zeros(nx)
    xi = np.zeros(nx)
    pn = np.eye(nx) / delta
    yprds = np.empty(len(input))

    for i, (inx, oux) in enumerate(zip(input, output)):
        inx = inx * np.random.randint(0, 2)
        l1 = np.tanh(win @ inx + ww @ xi)
        yprds[i] = wou @ l1
        for _ in range(n_update):
            ouprd = wou @ l1
            vn = output[i] - ouprd
            tmp = pn @ xi
            gn = ilambda * tmp / (1 + ilambda * (xi @ tmp))
            pn = ilambda * (pn - np.outer(gn, xi) * pn)
            # state update
            xi = l1
            wou = wou + vn * gn
    return yprds


@numba.njit(
    [
        f8[::1](
            f8[:, ::1],
            f8[::1],
            i8,
            i8,
            f8[:, ::1],
            f8[:, ::1],
            f8,
            f8,
            i8,
        ),
    ],
    cache=True,
)
def nu_compd(
    input: np.ndarray,
    output: np.ndarray,
    nin: int,
    nx: int,
    win: np.ndarray,
    ww: np.ndarray,
    ilambda: float,
    delta: float,
    n_update: int,
) -> np.ndarray:
    """Auxiliary functions for the `Reservoirepd` function."""
    nt = len(input)
    dropout = np.random.randint(0, 2, size=nt)
    wou = np.zeros(nx)
    xi = np.zeros(nx)
    pn = np.eye(nx) / delta
    yprds = np.empty(nt)

    for i in range(nt):
        l1 = np.tanh(win @ (input[i] * dropout[i]) + ww @ xi)
        yprds[i] = wou @ l1
        for _ in range(n_update):
            ouprd = wou @ l1
            vn = output[i] - ouprd
            tmp = pn @ xi
            gn = ilambda * tmp / (1 + ilambda * (xi @ tmp))
            pn = ilambda * (pn - np.outer(gn, xi) * pn)
            # state update
            xi = l1
            wou = wou + vn * gn
    return yprds
