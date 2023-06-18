import platform

from setuptools import Extension


def not_on_windows(s: str) -> str:
    return s if platform.system().lower() != "windows" else ""


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