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

THIS FILE IS DEPRECATED AND MAY BE REMOVED WITHOUT WARNING!
DO NOT CALL THESE FUNCTIONS IN YOUR CODE!
"""

import unittest
import time
import cvxpy as cvx


class TestParamCache(unittest.TestCase):

    def test_param_timings(self):
        """Test that it is faster to solve a parameterized
        problem after the first solve.
        """
        N = 1000
        x = cvx.Variable(N)
        total = 0
        constraints = []
        for i in range(N):
            total += x[i]
            constraints += [x[i] == i]

        prob = cvx.Problem(cvx.Minimize(total), constraints)
        time0 = time.time()
        result = prob.solve()
        time1 = time.time() - time0
        self.assertAlmostEqual(result, N*(N-1)/2.0, places=4)

        time0 = time.time()
        result = prob.solve()
        time2 = time.time() - time0
        self.assertAlmostEqual(result, N*(N-1)/2.0, places=4)
        # assert time2 < time1
