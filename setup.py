from setuptools import setup, Extension


class get_numpy_include(object):
    """Returns Numpy's include path with lazy import.
    """
    def __str__(self):
        import numpy
        return numpy.get_include()


canon = Extension(
    '_CVXcanon',
    sources=['cvxpy/CVXcanon/src/CVXcanon.cpp',
             'cvxpy/CVXcanon/src/LinOpOperations.cpp',
             'cvxpy/CVXcanon/src/Utils.cpp',
             'cvxpy/CVXcanon/python/CVXcanon_wrap.cpp'],
    include_dirs=['cvxpy/CVXcanon/src/',
                  'cvxpy/CVXcanon/python/',
                  'cvxpy/CVXcanon/include/Eigen',
                  get_numpy_include()],
)


setup(
    name='cvxpy',
    version='0.4.8',
    author='Steven Diamond, Eric Chu, Stephen Boyd',
    author_email='stevend2@stanford.edu, echu508@stanford.edu, boyd@stanford.edu',
    ext_modules=[canon],
    packages=['cvxpy',
              'cvxpy.atoms',
              'cvxpy.atoms.affine',
              'cvxpy.atoms.elementwise',
              'cvxpy.CVXcanon',
              'cvxpy.constraints',
              'cvxpy.expressions',
              'cvxpy.expressions.constants',
              'cvxpy.interface',
              'cvxpy.interface.numpy_interface',
              'cvxpy.lin_ops',
              'cvxpy.problems',
              'cvxpy.problems.problem_data',
              'cvxpy.problems.solvers',
              'cvxpy.reductions',
              'cvxpy.reductions.dcp2cone',
              'cvxpy.reductions.dcp2cone.atom_canonicalizers',
              'cvxpy.tests',
              'cvxpy.transforms',
              'cvxpy.utilities'],
    package_dir={'cvxpy': 'cvxpy'},
    url='http://github.com/cvxgrp/cvxpy/',
    license='GPLv3',
    zip_safe=False,
    description='A domain-specific language for modeling convex optimization problems in Python.',
    install_requires=["ecos >= 2",
                      "scs >= 1.1.3",
                      "multiprocess",
                      "fastcache",
                      "six",
                      "toolz",
                      "numpy >= 1.9",
                      "scipy >= 0.15"],
    use_2to3=True,
)
