"""
Copyright 2013 Steven Diamond

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""

from setuptools import setup

setup(
    name='cvxpy',
    version='0.4.8',
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
                      "scipy >= 0.15",
                      "CVXcanon >= 0.0.22",
                      "mathprogbasepy >= 0.1.1"],
    use_2to3=True,
)
