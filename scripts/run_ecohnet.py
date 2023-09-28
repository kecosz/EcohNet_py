import pprint
import time
from pathlib import Path

import dill
import pandas as pd
import datetime

from ecohnet import RCall
from ecohnet.utils.preprocess import std
import os

REP = 10000

if not os.path.exists('out'):
    os.mkdir('out')

OUT_DIR = Path("out")
DATA_DIR = Path("data")
DATA_FILE = "rdata_011322_2.csv"
DATE = datetime.date.today().strftime("%Y%m%d")
OUT_FILE_NAME = f"{DATA_FILE}_{REP}_{DATE}.dill"

if __name__ == "__main__":
    data_path = DATA_DIR / DATA_FILE

    assert data_path.exists()
    assert data_path.is_file()

    rdata = pd.read_csv(data_path, header=0)
    print(rdata.head(4))
    pprint.pprint([f"{i} {col}" for i, col in enumerate(rdata.columns)])

    datk0 = rdata

    datk = datk0.to_numpy() ** 0.5

    out_fpath = OUT_DIR / OUT_FILE_NAME

    assert not out_fpath.exists()

    tu = time.time()
    rcc, rcprd, rcmat, rw = RCall(std(datk), (0.95, 0.95, 0.001, 8), seed=42, rep=REP)
    rcall = (rcc, rcprd, rcmat, rw, datk)

    print(f"elapsed time of RCall: {time.time() - tu}")

    with open(out_fpath, "wb") as f:
        dill.dump(rcall, f)
