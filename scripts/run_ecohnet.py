import pprint
import time
from pathlib import Path

import dill
import pandas as pd
import datetime

from ecohnet import RCall
from ecohnet.utils.preprocess import std

REP = 10000
OUT_DIR = Path("out")
DATA_DIR = Path("data")
DATE = datetime.date.today().strftime("%Y%m%d")
OUT_FILE_NAME = f"kasmircall_{REP}_{DATE}.dill"

if __name__ == "__main__":
    data_path = DATA_DIR / "rdata_011322_2.csv"

    assert data_path.exists()
    assert data_path.is_file()

    rdata = pd.read_csv(data_path, header=0)
    print(rdata.head(4))
    rdata = rdata.rename(
        columns={
            "Anabena": "Nostocales",
            "Planktothrix": "Oscillatoriales",
            "Actinocyclus": "Thalassiosiraceae",
        }
    )
    pprint.pprint([f"{i} {col}" for i, col in enumerate(rdata.columns)])

    datk0 = rdata.iloc[137:, 3:]
    datk0.head(4)

    datk = datk0.to_numpy() ** 0.5

    out_fpath = OUT_DIR / OUT_FILE_NAME

    assert not out_fpath.exists()

    tu = time.time()
    rcc, rcprd, rcmat, rw = RCall(std(datk), (0.95, 0.95, 0.001, 8), seed=42, rep=REP)
    kasmircall = (rcc, rcprd, rcmat, rw, datk)

    print(f"elapsed time of RCall: {time.time() - tu}")

    with open(out_fpath, "wb") as f:
        dill.dump(kasmircall, f)
