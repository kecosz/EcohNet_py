[![DOI](https://zenodo.org/badge/697528491.svg)](https://zenodo.org/badge/latestdoi/697528491)

# Python version of EcohNet
## What
EcohNet is a method for computing predictability-based relationships among variables in a multivariate time series.
EcohNet is implemented by a type of neural network called echo state network and the progressive selection of its 
input variables to identify the smallest set of predictors that minimize the prediction error for a given target 
variable, and then evaluate the unique contribution of each variable. Based on the concept of Granger causality, 
the network obtained by EcohNet can be interpreted as representative of the causal relationships inherent in a 
given time series.


Reference: https://www.pnas.org/doi/10.1073/pnas.2204405119


## How to run
### Prerequisits
- Python 3.8
- pipenv
- numpy

### Run

#### On Google Colab
See `ecohnet.ipynb`.

#### On your environment
```` sh
pipenv install
pipenv shell
$ python --version
> Python 3.8.*
python sctipts/run_ecohnet.py
````

Initially, the code executes echonet on `rdata_011322_2.csv` in `data` folder. To run it for your own data, the following steps are required:
1) Place your data (csv file) in `data` folder
2) Edit `run_ecohnet.py` as follows:
```python
DATA_FILE = "YOURDATA.csv"
````
See `ecohnet_script.ipynb` for an example of execution and visualization.

## Folder structure
- `src/ecohnet`.
    - Folder containing the main implementation.

- `data`
    - Folder containing the observation data.

- `out`
    - Folder where the experimental results are stored.

- `ecohnet.ipynb`.
    - The notebook to run the experiment and visualize it. It also contains the implementation needed for visualization.

- `external/wolfram/LakeColors.txt`
    - Exported mathematica colormaps. Used for visualization.

- `scripts/`
    - `run_ecohnet.py`
        - Script to run experiments on csv data from the console. Faster than running on a notebook.
