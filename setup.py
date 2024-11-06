from os.path import join

import numpy as np
from Cython.Distutils import Extension, build_ext
from setuptools import setup

ext = Extension(
    "ecohnet.utils.cy_reservoir",
    sources=["src/ecohnet/utils/cy_reservoir.pyx"],
    include_dirs=[".", np.get_include()],
    library_dirs=[
        join(np.get_include(), "..", "..", "random", "lib"),
        join(np.get_include(), "..", "lib"),
    ],
    libraries=["npyrandom", "npymath"],
    cython_directives={"language_level": "3"},
)

setup(
    name="EcohNet",
    version="0.15",
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
        "statsmodels==0.13.5",
        "matplotlib",
        "seaborn",
        "graphviz"
    ],
    zip_safe=False,
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
