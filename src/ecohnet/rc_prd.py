from __future__ import annotations

from multiprocessing import Pool, cpu_count
from typing import Optional

import numpy as np
from statsmodels.nonparametric.kde import KDEUnivariate
from tqdm.autonotebook import tqdm

from .utils.constant import Const
from .utils.math import Prop
from .utils.preprocess import SetVariables
from .utils.random import randuint32
from .utils.reservoir import calc_bagg_and_tgtest


def RCprd(
    data: np.ndarray,
    predyns: list[
        tuple[
            tuple[float, KDEUnivariate],
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]
    ],
    num_worker: Optional[int] = None,
) -> list[tuple[tuple[float, KDEUnivariate], list[np.ndarray], np.ndarray]]:
    """Predict time-series from the variable set that maximizes prediction skill.

    Args:
        data (np.ndarray)
        predyns (list[ tuple[ tuple[float, KDEUnivariate], np.ndarray, np.ndarray, np.ndarray, ] ]): Output of RCcore function.
        num_worker (Optional[int], optional): number of node for multiprocessing. Defaults to None.

    Returns:
        list[
            tuple[tuple[float, KDEUnivariate], list[np.ndarray], np.ndarray]
        ]:
            Generated data for each target variable.
            [0] (tuple[float, KDEUnivariate]): Property of prediction skill (cf function Prop).
            [1] (list[np.ndarray]): Quantiles of predicted time-series.
            [2] (np.ndarray): Original time-series of target variable.
    """
    num_worker = cpu_count() - 1 if num_worker is None else num_worker
    iactives = [predyn[3] for predyn in predyns]
    itmax = Const.q2
    ni = len(data[0])
    strps: list[tuple[tuple[float, KDEUnivariate], list[np.ndarray], np.ndarray]] = []
    for w in tqdm(range(ni), desc="RCprd"):
        targetpos = w
        iactive = iactives[w]
        input, target = SetVariables(data, targetpos, 1)
        in_ = np.ascontiguousarray(input[:, iactive == 1])
        with Pool(num_worker) as p:
            seed = randuint32()
            baggs_and_tgtests = p.starmap(
                calc_bagg_and_tgtest,
                ((in_, target, seed + i) for i in range(itmax)),
            )
        baggs = np.array([e[0] for e in baggs_and_tgtests])
        tgtests = [e[1] for e in baggs_and_tgtests]
        quantiles = [
            np.quantile(tgtest_step, [0.025, 0.5, 0.975], method="inverted_cdf")
            for tgtest_step in np.transpose(tgtests)
        ]
        strps.append(
            (
                Prop(baggs),
                quantiles,
                target,
            )
        )

    return strps


def RCprdn(
    data,
    predyns: list[
        tuple[
            tuple[float, KDEUnivariate],
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]
    ],
) -> list[tuple[tuple[float, KDEUnivariate], list[np.ndarray], np.ndarray]]:
    """Not parallel version."""
    iactives = [predyn[3] for predyn in predyns]
    itmax = Const.q2
    ni = len(data[0])
    strps: list[tuple[tuple[float, KDEUnivariate], list[np.ndarray], np.ndarray]] = []
    for w in tqdm(range(ni), desc="RCprdn"):
        targetpos = w
        iactive = iactives[w]
        input, target = SetVariables(data, targetpos, 1)
        in_ = np.ascontiguousarray(input[:, iactive == 1])
        seed = randuint32()
        baggs_and_tgtests = [
            calc_bagg_and_tgtest(in_, target, seed + i) for i in range(itmax)
        ]
        baggs = np.array([e[0] for e in baggs_and_tgtests])
        tgtests = [e[1] for e in baggs_and_tgtests]
        quantiles = [
            np.quantile(tgtest_step, [0.025, 0.5, 0.975], method="inverted_cdf")
            for tgtest_step in np.transpose(tgtests)
        ]
        strps.append(
            (
                Prop(baggs),
                quantiles,
                target,
            )
        )
    return strps
