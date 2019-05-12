import distutils.sysconfig
import distutils.version
import os
import platform
import sys

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


# inject numpy headers
class build_ext_cvxpy(build_ext):
    def finalize_options(self):
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


def is_platform_mac():
    return sys.platform == 'darwin'


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

canon = Extension(
    '_cvxcore',
    sources=['cvxpy/cvxcore/src/cvxcore.cpp',
             'cvxpy/cvxcore/src/LinOpOperations.cpp',
             'cvxpy/cvxcore/src/Utils.cpp',
             'cvxpy/cvxcore/python/cvxcore_wrap.cpp'],
    include_dirs=['cvxpy/cvxcore/src/',
                  'cvxpy/cvxcore/python/',
                  'cvxpy/cvxcore/include/Eigen'],
)


setup(
    name='cvxpy',
    version='1.0.22',
    author='Steven Diamond, Eric Chu, Stephen Boyd',
    author_email='stevend2@stanford.edu, echu508@stanford.edu, boyd@stanford.edu',
    cmdclass={'build_ext': build_ext_cvxpy},
    ext_modules=[canon],
    packages=find_packages(exclude=["cvxpy.performance_tests"]),
    url='http://github.com/cvxgrp/cvxpy/',
    license='Apache License, Version 2.0',
    zip_safe=False,
    description='A domain-specific language for modeling convex optimization problems in Python.',
    install_requires=["osqp >= 0.4.1",
                      "ecos >= 2",
                      "scs >= 1.1.3",
                      "multiprocess",
                      "six",
                      "numpy >= 1.15",
                      "scipy >= 1.1.0"],
    setup_requires=["numpy >= 1.15"],
    extras_require={
        'glpk': [
            (
                'cvxopt >= 1.2.0; sys_platform == "darwin" or '
                'sys_platform == "linux" or os_name == "nt" and '
                'python_version >= "3.5"'
            ),
        ],
    },
    use_2to3=True,
)
