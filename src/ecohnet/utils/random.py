import numpy as np
from numba import njit


def set_seed(seed: int) -> None:
    """set random seed"""
    seed = seed % (2**32 - 1)
    set_seed_numba(seed)
    set_seed_numpy(seed)


@njit
def set_seed_numba(seed: int) -> None:
    np.random.seed(seed)


def set_seed_numpy(seed: int) -> None:
    np.random.seed(seed)


@njit
def randuint32() -> int:
    return np.random.randint(0, 2**32)
