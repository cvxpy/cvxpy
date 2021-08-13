import distutils.sysconfig
import distutils.version
import os
import platform
import sys

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


# inject numpy headers
class build_ext_cvxpy(build_ext):
    def finalize_options(self) -> None:
        build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        # `__builtins__` can be a dict
        # see https://docs.python.org/2/reference/executionmodel.html
        if isinstance(__builtins__, dict):
            __builtins__['__NUMPY_SETUP__'] = False
        else:
            __builtins__.__NUMPY_SETUP__ = False

        import numpy
        self.include_dirs.append(numpy.get_include())


def is_platform_mac() -> bool:
    return sys.platform == 'darwin'


def not_on_windows(s: str) -> str:
    return s if platform.system().lower() != "windows" else ""


# For mac, ensure extensions are built for macos 10.9 when compiling on a
# 10.9 system or above, overriding distutils behaviour which is to target
# the version that python was built for. This may be overridden by setting
# MACOSX_DEPLOYMENT_TARGET before calling setup.py. This behavior is
# motivated by Apple dropping support for libstdc++.
if is_platform_mac():
    if 'MACOSX_DEPLOYMENT_TARGET' not in os.environ:
        current_system = distutils.version.LooseVersion(platform.mac_ver()[0])
        python_target = distutils.version.LooseVersion(
            distutils.sysconfig.get_config_var('MACOSX_DEPLOYMENT_TARGET'))
        if python_target < '10.9' and current_system >= '10.9':
            os.environ['MACOSX_DEPLOYMENT_TARGET'] = '10.9'

# Optionally specify openmp flags when installing, eg
#
# CFLAGS="-fopenmp" LDFLAGS="-lgomp" python setup.py install
#
# TODO wheels should be compiled with openmp ...
canon = Extension(
    '_cvxcore',
    sources=['cvxpy/cvxcore/src/cvxcore.cpp',
             'cvxpy/cvxcore/src/LinOpOperations.cpp',
             'cvxpy/cvxcore/src/Utils.cpp',
             'cvxpy/cvxcore/python/cvxcore_wrap.cpp'],
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


setup(
    name='cvxpy',
    version='1.1.15',
    author='Steven Diamond, Eric Chu, Stephen Boyd',
    author_email='stevend2@stanford.edu, akshayka@cs.stanford.edu, '
                 'echu508@stanford.edu, boyd@stanford.edu',
    cmdclass={'build_ext': build_ext_cvxpy},
    ext_modules=[canon],
    packages=find_packages(exclude=["cvxpy.performance_tests"]),
    url='http://github.com/cvxgrp/cvxpy/',
    license='Apache License, Version 2.0',
    zip_safe=False,
    description='A domain-specific language for modeling convex optimization '
                'problems in Python.',
    package_data={
        'cvxpy': ['py.typed'],
    },
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    python_requires='>=3.6',
    install_requires=["osqp >= 0.4.1",
                      "ecos >= 2",
                      "scs >= 1.1.6",
                      "numpy >= 1.15",
                      "scipy >= 1.1.0"],
    setup_requires=["numpy >= 1.15"],
)
