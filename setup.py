import os
import re
import sys
import platform
import subprocess

from setuptools import setup

__version__ = '0.0.8'

setup(
    name='henon_map',
    version=__version__,
    author='Carlo Emilio Montanari',
    author_email='carlidel95@gmail.com',
    description='Henon map implemented in Numba',
    packages=["henon_map"],
    install_requires=['numba', 'numpy', 'matplotlib'],
    setup_requires=['numba', 'numpy', 'matplotlib'],
    license='MIT',
)
