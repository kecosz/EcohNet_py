from __future__ import annotations

from multiprocessing import Pool, cpu_count

import numpy as np
from statsmodels.nonparametric.kde import KDEUnivariate
from tqdm.autonotebook import tqdm

from .utils.constant import Const
from .utils.math import Evp, Prop
from .utils.preprocess import SetVariables
from .utils.random import randuint32
from .utils.reservoir import calc_baggs


def RCmat(
    data: np.ndarray,
    predyns: list[
        tuple[
            tuple[float, KDEUnivariate],
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]
    ],
    num_worker: int = cpu_count() - 1,
) -> np.ndarray:
    """Calculate the prediction skill matrix.

    Args:
        data (np.ndarray)
        predyns (list[ tuple[
            tuple[float, KDEUnivariate], np.ndarray, np.ndarray, np.ndarray]
        ]):
            Output of RCcore function.
        num_worker (int, optional): number of node for multiprocessing. Defaults to cpu_count()-1.

    Returns:
        np.ndarray:
            [0]: Prediction skill matrix. Element [i][j] is the prediction skill of `j` for `i`.
            [1]: Matrix.
                Element [i][j] is the area of the region less than prediction skill of `j` for `i`
                in the distribution of the maximum prediction skill of `i`.
    """
    props = [predyn[0] for predyn in predyns]
    iactives: list[np.ndarray] = [predyn[3] for predyn in predyns]
    itmax = Const.q2
    ni = len(data[0])
    nl = len(data)
    strps = []
    for w in tqdm(range(ni), desc="RCmat"):
        targetpos = w
        prop0 = props[w]
        iactive = iactives[w]
        input, target = SetVariables(data, targetpos, 1)
        active_indices = [i for i, is_active in enumerate(iactive) if is_active == 1]
        iacs = [iactive - np.eye(ni)[i] for i in active_indices]
        iacz = [np.eye(ni)[i] for i in active_indices]
        newins = np.array(
            [
                input[:, iac == 1]
                if (iac == 1).any()
                else np.random.random((len(input), 1)) * 0.4 + 0.8
                for iac in iacs
            ]
        )

        with Pool(num_worker) as p:
            seed = randuint32()
            baggs = np.transpose(
                p.starmap(
                    calc_baggs, ((newins, target, seed + i) for i in range(itmax))
                )
            )
        bagprops = [Prop(bagg) for bagg in baggs]
        evps = [Evp(bagprop, prop0) for bagprop in bagprops]
        strps.append(
            (
                sum([-evp[0] * iac for evp, iac in zip(evps, iacz)]),
                1 - sum([evp[1] * iac for evp, iac in zip(evps, iacz)]),
            )
        )

    return np.transpose(strps, (1, 0, 2))


def RCmatn(data, predyns):
    """Not parallel version"""
    props = [predyn[0] for predyn in predyns]
    iactives: list[np.ndarray] = [predyn[3] for predyn in predyns]
    itmax = Const.q2
    ni = len(data[0])
    nl = len(data)
    strps = []
    for w in tqdm(range(ni), desc="RCmatn"):
        targetpos = w
        prop0 = props[w]
        iactive = iactives[w]
        input, target = SetVariables(data, targetpos, 1)
        active_indices = [i for i, is_active in enumerate(iactive) if is_active == 1]
        iacs = [iactive - np.eye(ni)[i] for i in active_indices]
        iacz = [np.eye(ni)[i] for i in active_indices]
        newins = []
        for iac in iacs:
            random_in = np.random.random((len(input), 1)) * 0.4 + 0.8
            if (iac == 1).any():
                newins.append(input[:, iac == 1])
            else:
                newins.append(random_in)
        newins = np.array(newins)

        seed = randuint32()
        baggs = np.transpose(
            [calc_baggs(newins, target, seed + i) for i in range(itmax)]
        )
        bagprops = [Prop(bagg) for bagg in baggs]
        evps = [Evp(bagprop, prop0) for bagprop in bagprops]
        strps.append(
            (
                sum([-evp[0] * iac for evp, iac in zip(evps, iacz)]),
                1 - sum([evp[1] * iac for evp, iac in zip(evps, iacz)]),
            )
        )

    return np.transpose(strps, (1, 0, 2))
