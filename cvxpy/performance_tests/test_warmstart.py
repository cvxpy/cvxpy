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

from __future__ import print_function
import unittest
import time
from cvxpy import *

"""
THIS FILE IS DEPRECATED AND MAY BE REMOVED WITHOUT WARNING!
DO NOT CALL THESE FUNCTIONS IN YOUR CODE!
"""

class TestWarmstart(unittest.TestCase):

    def test_warmstart(self):
        """Testing warmstart LASSO with SCS.
        """
        import numpy

        # Problem data.
        n = 15
        m = 10
        numpy.random.seed(1)
        A = numpy.random.randn(n, m)
        b = numpy.random.randn(n)
        # gamma must be positive due to DCP rules.
        gamma = Parameter(nonneg=True)

        # Construct the problem.
        x = Variable(m)
        error = sum_squares(A*x - b)
        obj = Minimize(error + gamma*norm(x, 1))
        prob = Problem(obj)

        # Construct a trade-off curve of ||Ax-b||^2 vs. ||x||_1
        sq_penalty = []
        l1_penalty = []
        x_values = []
        gamma_vals = numpy.logspace(-4, 6, 10)

        start = time.time()
        for val in gamma_vals:
            gamma.value = val
            prob.solve(solver=SCS, warm_start=True, use_indirect=True)
            # Use expr.value to get the numerical value of
            # an expression in the problem.
            sq_penalty.append(error.value)
            l1_penalty.append(norm(x, 1).value)
            x_values.append(x.value)
        end = time.time()
        print("time elapsed=", end - start)
