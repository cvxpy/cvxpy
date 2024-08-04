"""
Copyright 2013 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


THIS FILE IS DEPRECATED AND MAY BE REMOVED WITHOUT WARNING!
DO NOT CALL THESE FUNCTIONS IN YOUR CODE!
"""

import time
import unittest

import cvxpy as cp


class TestWarmstart(unittest.TestCase):

    def test_warmstart(self) -> None:
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
        gamma = cp.Parameter(nonneg=True)

        # Construct the problem.
        x = cp.Variable(m)
        error = cp.sum_squares(A@x - b)
        obj = cp.Minimize(error + gamma*cp.norm(x, 1))
        prob = cp.Problem(obj)

        # Construct a trade-off curve of ||Ax-b||^2 vs. ||x||_1
        sq_penalty = []
        l1_penalty = []
        x_values = []
        gamma_vals = numpy.logspace(-4, 6, 10)

        start = time.time()
        for val in gamma_vals:
            gamma.value = val
            prob.solve(solver=cp.SCS, warm_start=True, use_indirect=True)
            # Use expr.value to get the numerical value of
            # an expression in the problem.
            sq_penalty.append(error.value)
            l1_penalty.append(cp.norm(x, 1).value)
            x_values.append(x.value)
        end = time.time()
        print("time elapsed=", end - start)
