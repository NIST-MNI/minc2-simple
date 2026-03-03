#!/usr/bin/env python

import os
import sys

from setuptools import setup, find_packages

_HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(_HERE)

print(find_packages())

_readme = os.path.join(_HERE, "README.txt")
long_description = ""
if os.path.isfile(_readme):
    with open(_readme, "rt") as f:
        long_description = f.read()

setup(
    name = "minc2_simple",
    version="0.2.31",
    description="MINC2 Simple interface using CFFI",
    long_description=long_description,
    url="https://github.com/vfonov/minc2_simple",
    author="Vladimir S. FONOV",
    author_email="vladimir.fonov@gmail.com",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Programming Language :: Python :: Implementation :: CPython",
        "License :: OSI Approved :: BSD License",
    ],
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=["cffi>=1.0.0"],
    cffi_modules=[
        "minc2_simple/minc2_simple_build.py:ffibuilder",
    ],
    scripts=(['example/python/xfmavg_scipy.py']),
    ###
    include_package_data=True
)
