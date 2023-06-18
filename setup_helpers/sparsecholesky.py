import platform
from setuptools import Extension
import sys
import subprocess


def not_on_windows(s: str) -> str:
    return s if platform.system().lower() != "windows" else ""


SWIG = True

if SWIG:
    sparsechol = Extension(
        '_sparsecholesky',
        sources=['cvxpy/utilities/cpp/sparsecholesky/sparsecholesky.cpp',
                 'cvxpy/utilities/cpp/sparsecholesky/sparsecholesky_wrap.cxx'],
        include_dirs=['cvxpy/utilities/cpp/sparsecholesky',
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
    class get_pybind_include(object):
        """Helper class to determine the pybind11 include path

        The purpose of this class is to postpone importing pybind11
        until it is actually installed, so that the ``get_include()``
        method can be invoked. """

        def __init__(self, user=False):
            try:
                import pybind11
                pybind11
            except ImportError:
                if subprocess.call([sys.executable, '-m', 'pip', 'install', 'pybind11']):
                    raise RuntimeError('pybind11 install failed.')
            self.user = user

        def __str__(self):
            import pybind11
            return pybind11.get_include(self.user)

