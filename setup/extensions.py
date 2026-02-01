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
import glob
import os
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

# Diff engine C extension for NLP support (optional)
# Source: https://github.com/dance858/DNLP-Differentiation-Engine
# Only built if the submodule is initialized (git submodule update --init)
# Python bindings are in cvxpy/.../diff_engine/_bindings/ (separate from C library)
diffengine = None
_diffengine_bindings = 'cvxpy/reductions/solvers/nlp_solvers/diff_engine/_bindings/bindings.c'
_diffengine_include = 'diff_engine_core/include/'
if os.path.exists(_diffengine_bindings) and os.path.exists(_diffengine_include):
    diff_engine_sources = [
        s for s in glob.glob('diff_engine_core/src/**/*.c', recursive=True)
        if 'dnlp_diff_engine' not in s  # Exclude standalone Python package
    ] + [_diffengine_bindings]

    # Define _POSIX_C_SOURCE on Linux for clock_gettime and struct timespec
    diffengine_defines = []
    if platform.system().lower() == 'linux':
        diffengine_defines.append(('_POSIX_C_SOURCE', '200809L'))

    diffengine = Extension(
        '_diffengine',
        sources=diff_engine_sources,
        include_dirs=[
            'diff_engine_core/include/',
            'diff_engine_core/src/',
            'cvxpy/reductions/solvers/nlp_solvers/diff_engine/_bindings/',
        ],
        define_macros=diffengine_defines,
        extra_compile_args=[
            '-O3',
            '-std=c99',
            '-Wall',
            not_on_windows('-Wextra'),
            '-DDIFF_ENGINE_VERSION="0.0.1"',
        ],
        extra_link_args=['-lm'] if platform.system().lower() != 'windows' else [],
    )
