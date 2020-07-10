# -*- coding: utf-8 -*-
"""
@Time   : 2020/6/30 2:54 下午
@Author : sunjiawei
@E-mail : j.w.sun1992@gmail.com
"""
# run `python setup.py build_ext --inplace` to build cython extension
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(["subgraphs.pyx", "text_sim_utils.pyx", "utils.pyx"])
)
