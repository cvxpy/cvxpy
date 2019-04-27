"""
Copyright, the CVXPY authors

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
import cvxpy as cp
from cvxpy.reductions.solvers import bisection
from cvxpy.tests import base_test


class TestDqcp2Dcp(base_test.BaseTest):
    def test_basic_with_interval(self):
        x = cp.Variable()
        expr = cp.ceil(x)

        self.assertTrue(expr.is_dqcp())
        self.assertTrue(expr.is_quasiconvex())
        self.assertTrue(expr.is_quasiconcave())
        self.assertFalse(expr.is_convex())
        self.assertFalse(expr.is_concave())
        self.assertFalse(expr.is_dcp())
        self.assertFalse(expr.is_dgp())

        problem = cp.Problem(cp.Minimize(expr), [x >= 12, x <= 17])
        self.assertTrue(problem.is_dqcp())
        self.assertFalse(problem.is_dcp())
        self.assertFalse(problem.is_dgp())

        red = cp.Dqcp2Dcp(problem)
        reduced = red.reduce()
        self.assertTrue(reduced.is_dcp())
        self.assertEqual(len(reduced.parameters()), 1)
        soln = bisection.bisect(red.bisection_data, low=12, high=17)
        self.assertAlmostEqual(soln.opt_val, 12.0, places=3)

        problem.unpack(soln)
        self.assertEqual(soln.opt_val, problem.value)
        self.assertAlmostEqual(x.value, 12.0, places=3)

    def test_basic_without_interval(self):
        x = cp.Variable()
        expr = cp.ceil(x)

        self.assertTrue(expr.is_dqcp())
        self.assertTrue(expr.is_quasiconvex())
        self.assertTrue(expr.is_quasiconcave())
        self.assertFalse(expr.is_convex())
        self.assertFalse(expr.is_concave())
        self.assertFalse(expr.is_dcp())
        self.assertFalse(expr.is_dgp())

        problem = cp.Problem(cp.Minimize(expr), [x >= 12, x <= 17])
        self.assertTrue(problem.is_dqcp())
        self.assertFalse(problem.is_dcp())
        self.assertFalse(problem.is_dgp())

        red = cp.Dqcp2Dcp(problem)
        reduced = red.reduce()
        self.assertTrue(reduced.is_dcp())
        self.assertEqual(len(reduced.parameters()), 1)
        soln = bisection.bisect(red.bisection_data)
        self.assertAlmostEqual(soln.opt_val, 12.0, places=3)

        problem.unpack(soln)
        self.assertEqual(soln.opt_val, problem.value)
        self.assertAlmostEqual(x.value, 12.0, places=3)

    def test_basic_solve(self):
        x = cp.Variable()
        expr = cp.ceil(x)

        self.assertTrue(expr.is_dqcp())
        self.assertTrue(expr.is_quasiconvex())
        self.assertTrue(expr.is_quasiconcave())
        self.assertFalse(expr.is_convex())
        self.assertFalse(expr.is_concave())
        self.assertFalse(expr.is_dcp())
        self.assertFalse(expr.is_dgp())

        problem = cp.Problem(cp.Minimize(expr), [x >= 12, x <= 17])
        self.assertTrue(problem.is_dqcp())
        self.assertFalse(problem.is_dcp())
        self.assertFalse(problem.is_dgp())
        problem.solve(qcp=True, low=12, high=17)
        self.assertAlmostEqual(problem.value, 12.0, places=3)
        self.assertAlmostEqual(x.value, 12.0, places=3)

        problem._clear_solution()
        problem.solve(qcp=True)
        self.assertAlmostEqual(problem.value, 12.0, places=3)
        self.assertAlmostEqual(x.value, 12.0, places=3)

        problem._clear_solution()
        problem.solve(qcp=True, high=17)
        self.assertAlmostEqual(problem.value, 12.0, places=3)
        self.assertAlmostEqual(x.value, 12.0, places=3)

        problem._clear_solution()
        problem.solve(qcp=True, low=12)
        self.assertAlmostEqual(problem.value, 12.0, places=3)
        self.assertAlmostEqual(x.value, 12.0, places=3)

        problem._clear_solution()
        problem.solve(qcp=True, low=0, high=100)
        self.assertAlmostEqual(problem.value, 12.0, places=3)
        self.assertAlmostEqual(x.value, 12.0, places=3)
