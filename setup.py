#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fashion_code import __version__
from setuptools import find_packages, setup

setup(
    description='Code for Kaggle challenge',
    include_package_data=True,
    name='iMaterialist Challenge (Fashion) App',
    packages=find_packages(),
    version=__version__,
    install_requires=[
          'keras',
    ],
)
