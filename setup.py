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
    version='1.0.11',
    author='Steven Diamond, Eric Chu, Stephen Boyd',
    author_email='stevend2@stanford.edu, echu508@stanford.edu, boyd@stanford.edu',
    cmdclass={'build_ext': build_ext_cvxpy},
    ext_modules=[canon],
    packages=find_packages(exclude=["cvxpy.performance_tests"]),
    url='http://github.com/cvxgrp/cvxpy/',
    license='Apache License, Version 2.0',
    zip_safe=False,
    description='A domain-specific language for modeling convex optimization problems in Python.',
    install_requires=["osqp",
                      "ecos >= 2",
                      "scs >= 1.1.3",
                      "multiprocess",
                      "fastcache",
                      "six",
                      "numpy >= 1.14",
                      "scipy >= 0.19"],
    setup_requires=["numpy >= 1.14"],
    use_2to3=True,
)
