from setuptools import setup, Extension


class get_numpy_include(object):
    """Returns Numpy's include path with lazy import.
    """
    def __str__(self):
        import numpy
        return numpy.get_include()


canon = Extension(
    '_cvxcore',
    sources=['cvxpy/cvxcore/src/cvxcore.cpp',
             'cvxpy/cvxcore/src/LinOpOperations.cpp',
             'cvxpy/cvxcore/src/Utils.cpp',
             'cvxpy/cvxcore/python/cvxcore_wrap.cpp'],
    include_dirs=['cvxpy/cvxcore/src/',
                  'cvxpy/cvxcore/python/',
                  'cvxpy/cvxcore/include/Eigen',
                  get_numpy_include()],
)


setup(
    name='cvxpy',
    version='1.0.3',
    author='Steven Diamond, Eric Chu, Stephen Boyd',
    author_email='stevend2@stanford.edu, echu508@stanford.edu, boyd@stanford.edu',
    ext_modules=[canon],
    packages=['cvxpy',
              'cvxpy.atoms',
              'cvxpy.atoms.affine',
              'cvxpy.atoms.elementwise',
              'cvxpy.cvxcore',
              'cvxpy.cvxcore.python',
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
              'cvxpy.reductions.complex2real',
              'cvxpy.reductions.complex2real.atom_canonicalizers',
              'cvxpy.reductions.dcp2cone',
              'cvxpy.reductions.dcp2cone.atom_canonicalizers',
              'cvxpy.reductions.eliminate_pwl',
              'cvxpy.reductions.eliminate_pwl.atom_canonicalizers',
              'cvxpy.reductions.qp2quad_form',
              'cvxpy.reductions.qp2quad_form.atom_canonicalizers',
              'cvxpy.reductions.eliminate_pwl.atom_canonicalizers',
              'cvxpy.reductions.solvers',
              'cvxpy.reductions.solvers.conic_solvers',
              'cvxpy.reductions.solvers.qp_solvers',
              'cvxpy.reductions.solvers.lp_solvers',
              'cvxpy.tests',
              'cvxpy.transforms',
              'cvxpy.utilities',
              'cvxpy.cvxcore.python'],
    package_dir={'cvxpy': 'cvxpy'},
    url='http://github.com/cvxgrp/cvxpy/',
    license='GPLv3',
    zip_safe=False,
    description='A domain-specific language for modeling convex optimization problems in Python.',
    install_requires=["osqp",
                      "ecos >= 2",
                      "scs >= 1.1.3",
                      "multiprocess",
                      "fastcache",
                      "six",
                      "toolz",
                      "numpy >= 1.13",
                      "scipy >= 0.19"],
    use_2to3=True,
)
