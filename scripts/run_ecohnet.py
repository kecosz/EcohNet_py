import pprint
import time
from pathlib import Path

import dill
import pandas as pd
import datetime

from ecohnet import RCall
from ecohnet.utils.preprocess import std
import os
import sys

###############
# Define options and defaults
options = {'-rep': True, '-seed': True}  # Options that expect a value
args = {'rep': '10000', 'seed': '1'}  # Default values

# Parse command-line arguments
for key in options.keys():
    if key in sys.argv:
        idx = sys.argv.index(key)
        if options[key]:  # If the option expects a value
            value = sys.argv[idx + 1]
            if value.startswith('-'):
                raise ValueError(f'option {key} must have a value.')
            args[key[1:]] = value
            del sys.argv[idx:idx + 2]
        else:  # If the option does not expect a value
            args[key[1:]] = True
            del sys.argv[idx]

# Convert rep and seed to integers
REP = int(args['rep'])
SEED = int(args['seed'])
###############

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
    rcc, rcprd, rcmat, rw = RCall(std(datk), (0.95, 0.95, 0.001, 8), seed=SEED, rep=REP)
    rcall = (rcc, rcprd, rcmat, rw, datk)

    print(f"elapsed time of RCall: {time.time() - tu}")

    with open(out_fpath, "wb") as f:
        dill.dump(rcall, f)
