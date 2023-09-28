from __future__ import annotations

from multiprocessing import Pool, cpu_count

import numpy as np
from statsmodels.nonparametric.kde import KDEUnivariate

from .utils.constant import Const
from .utils.math import Prop
from .utils.random import randuint32
from .utils.reservoir import calc_bagg


def RCbaselineprop(target: np.ndarray) -> tuple[float, KDEUnivariate]:
    """calculate baseline predection skill for a random variable"""

    def gen_args(itmax: int):
        seed = randuint32()
        for i in range(itmax):
            dummy_in = np.random.random((target.shape[0], 1)) * 0.4 + 0.8
            yield dummy_in, target, seed + i

    itmax = Const.q2
    with Pool(cpu_count() - 1) as p:
        bagg = np.array(p.starmap(calc_bagg, gen_args(itmax)))
    return Prop(bagg)


def RCbaselinepropn(target: np.ndarray) -> tuple[float, KDEUnivariate]:
    """not prallel version"""

    def gen_args(itmax: int):
        seed = randuint32()
        for i in range(itmax):
            dummy_in = np.random.random((target.shape[0], 1)) * 0.4 + 0.8
            yield dummy_in, target, seed + i

    itmax = Const.q2
    bagg = np.array(
        [
            calc_bagg(dummy_in, target, seed)
            for dummy_in, target, seed in gen_args(itmax)
        ]
    )
    return Prop(bagg)
