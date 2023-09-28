from tqdm.autonotebook import tqdm

from .rc_core import RCcore, RCcoren
from .rc_mat import RCmat, RCmatn
from .rc_prd import RCprd, RCprdn
from .rc_setup import RCsetup
from .utils.random import set_seed


def RCall(data, rw, rep=10000, seed=42):
    set_seed(seed)
    rcc = []
    with tqdm(range(len(data[0])), desc="RCcore") as pbar:
        for tgti in pbar:
            RCsetup(rw, rep=rep)
            rcc.append(RCcore(data, tgti, progress_bar=pbar))
    rcprd = RCprd(data, rcc)
    rcmat = RCmat(data, rcc)
    return rcc, rcprd, rcmat, rw


def RCalln(data, rw, rep=10000, seed=42):
    "not parallel version"
    set_seed(seed)
    rcc = []
    with tqdm(range(len(data[0])), desc="RCcoren") as pbar:
        for tgti in pbar:
            RCsetup(rw, rep=rep)
            rcc.append(RCcoren(data, tgti, progress_bar=pbar))
    rcprd = RCprdn(data, rcc)
    rcmat = RCmatn(data, rcc)
    return rcc, rcprd, rcmat, rw
