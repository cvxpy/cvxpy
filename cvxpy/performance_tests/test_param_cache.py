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
        assert time2 < time1
