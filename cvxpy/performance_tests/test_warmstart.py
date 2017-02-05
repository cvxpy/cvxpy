"""
Copyright 2017 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import print_function
import unittest
import time
from cvxpy import *


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
        b = numpy.random.randn(n, 1)
        # gamma must be positive due to DCP rules.
        gamma = Parameter(sign="positive")

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
