"""
Copyright 2023, the CVXPY authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import platform

from pybind11.setup_helpers import Pybind11Extension
from setuptools import Extension


def not_on_windows(s: str) -> str:
    return s if platform.system().lower() != "windows" else ""


compiler_args = [
        '-O3',
        '-std=c++11',
        '-Wall',
        '-pedantic',
        not_on_windows('-Wextra'),
        not_on_windows('-Wno-unused-parameter'),
]

# Optionally specify openmp flags when installing, eg
#
# CFLAGS="-fopenmp" LDFLAGS="-lgomp" python setup.py install
#
# TODO wheels should be compiled with openmp ...
cvxcore = Extension(
    '_cvxcore',
    sources=['cvxpy/cvxcore/src/cvxcore.cpp',
             'cvxpy/cvxcore/src/LinOpOperations.cpp',
             'cvxpy/cvxcore/src/Utils.cpp',
             'cvxpy/cvxcore/python/cvxcore_wrap.cxx'],
    include_dirs=['cvxpy/cvxcore/src/',
                  'cvxpy/cvxcore/python/',
                  'cvxpy/cvxcore/include/'],
    extra_compile_args=compiler_args,
    extra_link_args=['-O3'],
)

sparsecholesky = Pybind11Extension(
    "_cvxpy_sparsecholesky",
    sources=[
        "cvxpy/utilities/cpp/sparsecholesky/main.cpp"
    ],
    define_macros=[('VERSION_INFO', "0.0.1")],
    extra_compile_args=compiler_args,
    extra_link_args=['-O3'],
)
