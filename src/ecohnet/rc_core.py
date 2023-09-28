from __future__ import annotations

import time
from multiprocessing import Pool, cpu_count
from typing import Optional

import numpy as np
from statsmodels.nonparametric.kde import KDEUnivariate
from tqdm.autonotebook import tqdm

from .rc_baselineprop import RCbaselineprop, RCbaselinepropn
from .utils.constant import Const
from .utils.math import Evp, Prop
from .utils.preprocess import SetVariables
from .utils.random import randuint32
from .utils.reservoir import calc_bagg, calc_baggs


def RCcore(
    data: np.ndarray,
    targetpos: int,
    num_worker: Optional[int] = None,
    progress_bar: Optional[tqdm] = None,
) -> tuple[tuple[float, KDEUnivariate], np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the set of the indices of optimal variables that maximizes the prediction.
    Forecast time series and calculate prediction skills with progressive selection of variable.

    Args:
        data (np.ndarray): time-series data, 2-D array (num_timestamp, num_input).
        targetpos (int): index of predicted variable
        num_worker (int| None): number of multiprocessing node. If None (default), set to cpu_count - 1.
        progress_bar (tqdm | None): object to show progress. Defaults to None.

    Returns:
        pprev (tuple[float, KDEUnivariate]):
            pprev[0] (float): maximum prediction  skill.
            pprev[1] (KDEUnivariate): distribution of prediction skills for maximum.

        aij (np.ndarray): Prediction skills when each variable is added to input.

        1-pij (np.ndarray):
            The area of the region greater than the prediction skill of each added variable,
            In the distribution of prediction skills for the input set one step before.

        iactive (np.ndarray): Flags of the set of optinal variables that maximizes the prediction.
    """
    if num_worker is None:
        num_worker = cpu_count() - 1
    start_time = time.time()
    itmax = Const.q2
    input, target = SetVariables(data, targetpos, 1)
    ni = len(data[0])
    nl = len(data)
    baseprop = RCbaselineprop(target)
    # initialize active variables
    iactive: np.ndarray = np.eye(ni)[targetpos]
    links = sum(iactive)
    input_single = input[:, iactive == 1]
    with Pool(num_worker) as p:
        seed = randuint32()
        bagg0 = np.array(
            p.starmap(
                calc_bagg,
                ((input_single, target, seed + i) for i in range(itmax)),
            )
        )

    pprev = Prop(bagg0)
    eprev = Evp(pprev, baseprop)
    aij = eprev[0] * iactive
    pij = (1 - eprev[1]) * iactive
    while links < ni:
        links = sum(iactive) + 1
        # logging
        if progress_bar is not None:
            progress_bar.set_postfix(
                {"links": links, "elapsed": f"{time.time() - start_time:.2f} s"}
            )
        # make input subsets
        new_indices = [i for i, is_active in enumerate(iactive) if is_active == 0]
        new_iactives = [iactive + np.eye(ni)[i] for i in new_indices]
        new_inputs = np.array(
            [input[:, new_iactive == 1] for new_iactive in new_iactives]
        )
        with Pool(num_worker) as p:
            seed = randuint32()
            baggs = np.transpose(
                p.starmap(
                    calc_baggs,
                    ((new_inputs, target, seed + i) for i in range(itmax)),
                )
            )

        pbags = [Prop(bagg) for bagg in baggs]
        ebags = [Evp(pbag, pprev) for pbag in pbags]
        ebest = sorted(ebags)[-1]
        ppb = [ebag[0] for ebag in ebags].index(ebest[0])
        pbest = pbags[ppb]
        newiact = new_iactives[ppb]
        ppba = list(newiact - iactive).index(1)
        if ebest[0] > 0 and ebags[ppb][1] < 0.48:
            iactive = newiact
            aij[ppba] = ebest[0]
            pij[ppba] = 1 - ebest[1]
            pprev = pbest
            eprev = ebest
        else:
            break

    return pprev, aij, 1 - pij, iactive


def RCcoren(
    data: np.ndarray,
    targetpos: int,
    progress_bar: Optional[tqdm] = None,
):
    """
    Not parallel version
    """
    start_time = time.time()
    itmax = Const.q2
    input, target = SetVariables(data, targetpos, 1)
    ni = len(data[0])
    nl = len(data)
    baseprop = RCbaselinepropn(target)
    # initialize active variables
    iactive: np.ndarray = np.eye(ni)[targetpos]
    links = sum(iactive)
    in_ = input[:, iactive == 1]
    seed = randuint32()
    bagg0 = np.array([calc_bagg(in_, target, seed + i) for i in range(itmax)])

    pprev = Prop(bagg0)
    eprev = Evp(pprev, baseprop)
    aij = eprev[0] * iactive
    pij = (1 - eprev[1]) * iactive
    while links < ni:
        links = sum(iactive) + 1
        # logging
        if progress_bar is not None:
            progress_bar.set_postfix(
                {"links": links, "elapsed": f"{time.time() - start_time:.2f} s"}
            )
        # make input subsets
        new_indices = [i for i, is_active in enumerate(iactive) if is_active == 0]
        new_iactives = [iactive + np.eye(ni)[i] for i in new_indices]
        new_ins = np.array([input[:, new_iactive == 1] for new_iactive in new_iactives])
        seed = randuint32()
        baggs = np.transpose(
            [calc_baggs(new_ins, target, seed + i) for i in range(itmax)]
        )

        pbags = [Prop(bagg) for bagg in baggs]
        ebags = [Evp(pbag, pprev) for pbag in pbags]
        ebest = sorted(ebags)[-1]
        ppb = [ebag[0] for ebag in ebags].index(ebest[0])
        pbest = pbags[ppb]
        newiact = new_iactives[ppb]
        ppba = list(newiact - iactive).index(1)
        if ebest[0] > 0 and ebags[ppb][1] < 0.48:
            iactive = newiact
            aij[ppba] = ebest[0]
            pij[ppba] = 1 - ebest[1]
            pprev = pbest
            eprev = ebest
        else:
            break

    return pprev, aij, 1 - pij, iactive
