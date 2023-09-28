from os.path import join
from setuptools import setup


setup(
    name="EcohNet",
    version="0.12",
    description="Python implementation of EcohNet: timeseries-based causal inference using echo state network",
    author="ayabe fumihiko, kenta suzuki",
    author_email="ayabe.fumihiko@plus-zero.co.jp, kenta.suzuki.zk@riken.jp",
    url="https://github.com/kecosz/EcohNet_py",
    packages=["ecohnet"],
    package_dir={"": "src"},
    install_requires=[
        "dill",
        "tqdm",
        "numpy",
        "numba",
        "cython",
        "scipy",
        "statsmodels",
        "matplotlib",
        "seaborn",
        "graphviz"
    ],
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
