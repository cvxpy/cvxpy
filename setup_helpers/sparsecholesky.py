import platform
from setuptools import Extension
import sys
import subprocess
from pybind11.setup_helpers import Pybind11Extension


def not_on_windows(s: str) -> str:
    return s if platform.system().lower() != "windows" else ""


SWIG = False

if SWIG:
    # directories below are interpreted from the perspective of cvxpy's setup.py file.
    sparsecholesky = Extension(
        '_sparsecholesky_swig',
        sources=['cvxpy/utilities/cpp/sparsecholesky_swig/sparsecholesky_swig.cpp',
                 'cvxpy/utilities/cpp/sparsecholesky_swig/sparsecholesky_swig_wrap.cxx'],
        include_dirs=['cvxpy/utilities/cpp/sparsecholesky_swig',
                      'cvxpy/cvxcore/include'],
        extra_compile_args=[
            '-O3',
            '-std=c++11',
            '-Wall',
            '-pedantic',
            not_on_windows('-Wextra'),
            not_on_windows('-Wno-unused-parameter'),
        ],
        extra_link_args=['-O3'],
    )
else:
    # pybind11
    sparsecholesky = Pybind11Extension("_sparsecholesky_swig",
          ["cvxpy/utilities/cpp/sparsecholesky_pb/main.cpp"],
          define_macros=[('VERSION_INFO', "0.0.1")],
           extra_compile_args=[
               '-O3',
               '-std=c++11',
               '-Wall',
               '-pedantic',
               not_on_windows('-Wextra'),
               not_on_windows('-Wno-unused-parameter'),
           ],
           extra_link_args=['-O3'],
    )
