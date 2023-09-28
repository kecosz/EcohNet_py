from __future__ import annotations

from .utils.constant import PP, REP, Const


def RCsetup(
    pp: tuple[float, float, float, int] = PP,
    rep: int = REP,
) -> None:
    """Setup constants.

    Parameters
    ----------
    pp : tuple
        pp[0] : float
            Spectral radius of reccurent matrix.
        pp[1] : float
            Foggetting rate in RLS.
        pp[2] : float
            Regularize coefficient in RLS.
        pp[3] : int
            Number of RLS update per one time step.

    rep : int
        Number of ESN population.
    """
    Const.p1, Const.p2, Const.p3, Const.p4 = pp
    Const.q2 = rep
