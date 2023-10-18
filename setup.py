import builtins
import distutils.version
import os
import platform
import sys
import sysconfig

import setup.extensions as setup_extensions
import setup.versioning as setup_versioning

# BEFORE importing setuptools, remove MANIFEST. Otherwise it may not be
# properly updated when the contents of directories change (true for distutils,
# not sure about setuptools).
if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')

from setuptools import find_packages, setup
from setuptools.command.build_ext import build_ext

# The comment below is from the SciPy code which we adapted for cvxpy.
#
#   This is a bit hackish: we are setting a global variable so that the main
#   cvxpy __init__ can detect if it is being loaded by the setup routine, to
#   avoid attempting to load components that aren't built yet.  While ugly, it's
#   a lot more robust than what was previously being used.
builtins.__CVXPY_SETUP__ = True


# inject numpy headers
class build_ext_cvxpy(build_ext):
    def finalize_options(self) -> None:
        build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        builtins.__NUMPY_SETUP__ = False

        import numpy
        self.include_dirs.append(numpy.get_include())


# For mac, ensure extensions are built for macos 10.9 when compiling on a
# 10.9 system or above, overriding distutils behaviour which is to target
# the version that python was built for. This may be overridden by setting
# MACOSX_DEPLOYMENT_TARGET before calling setup.py. This behavior is
# motivated by Apple dropping support for libstdc++.
if sys.platform == 'darwin':
    if 'MACOSX_DEPLOYMENT_TARGET' not in os.environ:
        current_system = distutils.version.LooseVersion(platform.mac_ver()[0])
        python_target = distutils.version.LooseVersion(
            sysconfig.get_config_var('MACOSX_DEPLOYMENT_TARGET'))
        if python_target < '10.9' and current_system >= '10.9':
            os.environ['MACOSX_DEPLOYMENT_TARGET'] = '10.9'

setup_versioning.write_version_py()
VERSION = setup_versioning.VERSION

setup(
    name='cvxpy',
    version=str(VERSION),
    author='Steven Diamond, Eric Chu, Stephen Boyd',
    author_email='stevend2@stanford.edu, akshayka@cs.stanford.edu, '
                 'echu508@stanford.edu, boyd@stanford.edu',
    cmdclass={'build_ext': build_ext_cvxpy},
    ext_modules=[setup_extensions.cvxcore, setup_extensions.sparsecholesky],
    packages=find_packages(exclude=["doc*",
                                    "examples*",
                                    "cvxpy.performance_tests*"]),
    url='https://github.com/cvxpy/cvxpy',
    license='Apache License, Version 2.0',
    zip_safe=False,
    description='A domain-specific language for modeling convex optimization '
                'problems in Python.',
    package_data={
        'cvxpy': ['py.typed'],
    },
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    python_requires='>=3.8',
    install_requires=[
        "osqp >= 0.6.2",
        "ecos >= 2",
        "clarabel >= 0.5.0",
        "scs >= 3.0",
        "numpy >= 1.15",
        "scipy >= 1.1.0",
        "pybind11"
    ],
)
