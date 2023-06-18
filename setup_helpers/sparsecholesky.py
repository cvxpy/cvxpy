import platform
from setuptools import Extension


def not_on_windows(s: str) -> str:
    return s if platform.system().lower() != "windows" else ""


"""
qdldl = Extension('qdldl',
                  sources=glob(os.path.join('cpp', '*.cpp')),
                  include_dirs=[os.path.join('c'),
                                os.path.join('c', 'qdldl', 'include'),
                                get_pybind_include(),
                                get_pybind_include(user=False)],
                  language='c++',
                  extra_compile_args=compile_args + ['-std=c++11'],
                  extra_objects=[qdldl_lib])
"""

sparsechol = Extension(
    '_sparsecholesky',
    sources=['cvxpy/utilities/cpp/sparsechol/sparsecholesky.cpp',
             'cvxpy/utilities/cpp/sparsechol/sparsecholesky_wrap.cxx'],
    include_dirs=['cvxpy/utilities/cpp/sparsechol',
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
