from setuptools import setup

from cvxpy import __version__

setup(
    name='cvxpy',
    version=__version__,
    author='Steven Diamond, Eric Chu, Stephen Boyd',
    author_email='stevend2@stanford.edu, echu508@stanford.edu, boyd@stanford.edu',
    packages=['cvxpy',
              'cvxpy.atoms',
              'cvxpy.atoms.affine',
              'cvxpy.atoms.elementwise',
              'cvxpy.constraints',
              'cvxpy.expressions',
              'cvxpy.expressions.constants',
              'cvxpy.expressions.variables',
              'cvxpy.interface',
              'cvxpy.interface.numpy_interface',
              'cvxpy.interface.cvxopt_interface',
              'cvxpy.problems',
              'cvxpy.tests',
              'cvxpy.utilities'],
    package_dir={'cvxpy': 'cvxpy'},
        url='http://github.com/cvxgrp/cvxpy/',
    license='...',
    description='A domain-specific language for modeling convex optimization problems in Python.',
    install_requires=["cvxopt >= 1.1.6",
                        "ecos >= 1.0"],
    extras_require={'doc': ['sphinx']},
)
