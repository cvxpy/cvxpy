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

import numpy as np


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

    def test_basic_maximization_with_interval(self):
        x = cp.Variable()
        expr = cp.ceil(x)

        self.assertTrue(expr.is_dqcp())
        self.assertTrue(expr.is_quasiconvex())
        self.assertTrue(expr.is_quasiconcave())
        self.assertFalse(expr.is_convex())
        self.assertFalse(expr.is_concave())
        self.assertFalse(expr.is_dcp())
        self.assertFalse(expr.is_dgp())

        problem = cp.Problem(cp.Maximize(expr), [x >= 12, x <= 17])
        self.assertTrue(problem.is_dqcp())
        self.assertFalse(problem.is_dcp())
        self.assertFalse(problem.is_dgp())

        problem.solve(qcp=True)
        self.assertAlmostEqual(x.value, 17.0, places=3)

    def test_basic_maximum(self):
        x, y = cp.Variable(2)
        expr = cp.maximum(cp.ceil(x), cp.ceil(y))

        problem = cp.Problem(cp.Minimize(expr), [x >= 12, x <= 17, y >= 17.4])
        self.assertTrue(problem.is_dqcp())
        problem.solve(qcp=True)
        self.assertEqual(problem.objective.value, 18.0)
        self.assertLess(x.value, 17.1)
        self.assertGreater(x.value, 11.9)
        self.assertGreater(y.value, 17.3)

    def test_basic_minimum(self):
        x, y = cp.Variable(2)
        expr = cp.minimum(cp.ceil(x), cp.ceil(y))

        problem = cp.Problem(cp.Maximize(expr), [x >= 11.9, x <= 15.8, y >= 17.4])
        self.assertTrue(problem.is_dqcp())
        problem.solve(qcp=True)
        self.assertEqual(problem.objective.value, 16.0)
        self.assertLess(x.value, 16.0)
        self.assertGreater(x.value, 14.9)
        self.assertGreater(y.value, 17.3)

    def test_basic_composition(self):
        x, y = cp.Variable(2)
        expr = cp.maximum(cp.ceil(cp.ceil(x)), cp.ceil(cp.ceil(y)))

        problem = cp.Problem(cp.Minimize(expr), [x >= 12, x <= 17, y >= 17.4])
        self.assertTrue(problem.is_dqcp())
        problem.solve(qcp=True)
        self.assertEqual(problem.objective.value, 18.0)
        self.assertLess(x.value, 17.1)
        self.assertGreater(x.value, 11.9)
        self.assertGreater(y.value, 17.3)

        # This problem should have the same solution.
        expr = cp.maximum(cp.floor(cp.ceil(x)), cp.floor(cp.ceil(y)))
        problem = cp.Problem(cp.Minimize(expr), [x >= 12, x <= 17, y >= 17.4])
        self.assertTrue(problem.is_dqcp())
        problem.solve(qcp=True)
        self.assertEqual(problem.objective.value, 18.0)
        self.assertLess(x.value, 17.1)
        self.assertGreater(x.value, 11.9)
        self.assertGreater(y.value, 17.3)

    def test_basic_floor(self):
        x = cp.Variable()
        expr = cp.floor(x)

        self.assertTrue(expr.is_dqcp())
        self.assertTrue(expr.is_quasiconvex())
        self.assertTrue(expr.is_quasiconcave())
        self.assertFalse(expr.is_convex())
        self.assertFalse(expr.is_concave())
        self.assertFalse(expr.is_dcp())
        self.assertFalse(expr.is_dgp())

        problem = cp.Problem(cp.Minimize(expr), [x >= 11.8, x <= 17])
        self.assertTrue(problem.is_dqcp())
        self.assertFalse(problem.is_dcp())
        self.assertFalse(problem.is_dgp())

        problem.solve(qcp=True)
        self.assertEqual(problem.objective.value, 11.0)
        self.assertGreater(x.value, 11.7)

    def test_basic_multiply_nonneg(self):
        x, y = cp.Variable(2, nonneg=True)
        expr = x * y
        self.assertTrue(expr.is_dqcp())
        self.assertTrue(expr.is_quasiconcave())
        self.assertFalse(expr.is_quasiconvex())

        self.assertFalse(expr.is_dcp())

        problem = cp.Problem(cp.Maximize(expr), [x <= 12, y <= 6])
        self.assertTrue(problem.is_dqcp())
        self.assertFalse(problem.is_dcp())
        self.assertFalse(problem.is_dgp())

        problem.solve(qcp=True, solver=cp.SCS)
        self.assertAlmostEqual(problem.objective.value, 72, places=1)
        self.assertAlmostEqual(x.value, 12, places=1)
        self.assertAlmostEqual(y.value, 6, places=1)

    def test_basic_multiply_nonpos(self):
        x, y = cp.Variable(2, nonpos=True)
        expr = x * y
        self.assertTrue(expr.is_dqcp())
        self.assertTrue(expr.is_quasiconcave())
        self.assertFalse(expr.is_quasiconvex())

        self.assertFalse(expr.is_dcp())

        problem = cp.Problem(cp.Maximize(expr), [x >= -12, y >= -6])
        self.assertTrue(problem.is_dqcp())
        self.assertFalse(problem.is_dcp())
        self.assertFalse(problem.is_dgp())

        problem.solve(qcp=True, solver=cp.SCS)
        self.assertAlmostEqual(problem.objective.value, 72, places=1)
        self.assertAlmostEqual(x.value, -12, places=1)
        self.assertAlmostEqual(y.value, -6, places=1)

    def test_basic_multiply_qcvx(self):
        x = cp.Variable(nonneg=True)
        y = cp.Variable(nonpos=True)
        expr = x * y
        self.assertTrue(expr.is_dqcp())
        self.assertTrue(expr.is_quasiconvex())
        self.assertFalse(expr.is_quasiconcave())

        self.assertFalse(expr.is_dcp())

        problem = cp.Problem(cp.Minimize(expr), [x <= 7, y >= -6])
        self.assertTrue(problem.is_dqcp())
        self.assertFalse(problem.is_dcp())
        self.assertFalse(problem.is_dgp())

        problem.solve(qcp=True, solver=cp.SCS)
        self.assertAlmostEqual(problem.objective.value, -42, places=1)
        self.assertAlmostEqual(x.value, 7, places=1)
        self.assertAlmostEqual(y.value, -6, places=1)

        x = cp.Variable(nonneg=True)
        y = cp.Variable(nonpos=True)
        expr = y * x
        self.assertTrue(expr.is_dqcp())
        self.assertTrue(expr.is_quasiconvex())
        self.assertFalse(expr.is_quasiconcave())

        self.assertFalse(expr.is_dcp())

        problem = cp.Problem(cp.Minimize(expr), [x <= 7, y >= -6])
        self.assertTrue(problem.is_dqcp())
        self.assertFalse(problem.is_dcp())
        self.assertFalse(problem.is_dgp())

        problem.solve(qcp=True, solver=cp.SCS)
        self.assertAlmostEqual(problem.objective.value, -42, places=1)
        self.assertAlmostEqual(x.value, 7, places=1)
        self.assertAlmostEqual(y.value, -6, places=1)

    def test_concave_multiply(self):
        x, y = cp.Variable(2, nonneg=True)
        expr = cp.sqrt(x) * cp.sqrt(y)
        self.assertTrue(expr.is_dqcp())
        self.assertTrue(expr.is_quasiconcave())
        self.assertFalse(expr.is_quasiconvex())

        problem = cp.Problem(cp.Maximize(expr), [x <= 4, y <= 9])
        problem.solve(qcp=True, solver=cp.SCS)
        self.assertAlmostEqual(problem.objective.value, 6, places=1)
        self.assertAlmostEqual(x.value, 4, places=1)
        self.assertAlmostEqual(y.value, 9, places=1)

        x, y = cp.Variable(2, nonneg=True)
        expr = (cp.sqrt(x) + 2.0) * (cp.sqrt(y) + 4.0)
        self.assertTrue(expr.is_dqcp())
        self.assertTrue(expr.is_quasiconcave())
        self.assertFalse(expr.is_quasiconvex())

        problem = cp.Problem(cp.Maximize(expr), [x <= 4, y <= 9])
        problem.solve(qcp=True, solver=cp.SCS)
        # (2 + 2) * (3 + 4) = 28
        self.assertAlmostEqual(problem.objective.value, 28, places=1)
        self.assertAlmostEqual(x.value, 4, places=1)
        self.assertAlmostEqual(y.value, 9, places=1)

    def test_basic_ratio(self):
        x = cp.Variable()
        y = cp.Variable(nonneg=True)
        expr = x / y
        self.assertTrue(expr.is_dqcp())
        self.assertTrue(expr.is_quasiconcave())
        self.assertTrue(expr.is_quasiconvex())

        problem = cp.Problem(cp.Minimize(expr), [x == 12, y <= 6])
        self.assertTrue(problem.is_dqcp())

        problem.solve(qcp=True)
        self.assertAlmostEqual(problem.objective.value, 2.0, places=1)
        self.assertAlmostEqual(x.value, 12, places=1)
        self.assertAlmostEqual(y.value, 6, places=1)

        x = cp.Variable()
        y = cp.Variable(nonpos=True)
        expr = x / y
        self.assertTrue(expr.is_dqcp())
        self.assertTrue(expr.is_quasiconcave())
        self.assertTrue(expr.is_quasiconvex())

        problem = cp.Problem(cp.Maximize(expr), [x == 12, y >= -6])
        self.assertTrue(problem.is_dqcp())

        problem.solve(qcp=True)
        self.assertAlmostEqual(problem.objective.value, -2.0, places=1)
        self.assertAlmostEqual(x.value, 12, places=1)
        self.assertAlmostEqual(y.value, -6, places=1)

    def test_lin_frac(self):
        x = cp.Variable((2,), nonneg=True)
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.arange(2)
        C = 2 * A
        d = np.arange(2)
        lin_frac = (A @ x + b) / (C @ x + d)
        self.assertTrue(lin_frac.is_dqcp())
        self.assertTrue(lin_frac.is_quasiconvex())
        self.assertTrue(lin_frac.is_quasiconcave())

        problem = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 0, lin_frac <= 1])
        self.assertTrue(problem.is_dqcp())
        problem.solve(qcp=True)
        self.assertAlmostEqual(problem.objective.value, 0, places=1)
        np.testing.assert_almost_equal(x.value, 0)

    def test_concave_frac(self):
        x = cp.Variable(nonneg=True)
        concave_frac = cp.sqrt(x) / cp.exp(x)
        self.assertTrue(concave_frac.is_dqcp())
        self.assertTrue(concave_frac.is_quasiconcave())
        self.assertFalse(concave_frac.is_quasiconvex())

        problem = cp.Problem(cp.Maximize(concave_frac))
        self.assertTrue(problem.is_dqcp())
        problem.solve(qcp=True)
        self.assertAlmostEqual(problem.objective.value, 0.428, places=1)
        self.assertAlmostEqual(x.value, 0.5, places=1)

    def test_length(self):
        x = cp.Variable(5)
        expr = cp.length(x)
        self.assertTrue(expr.is_dqcp())
        self.assertTrue(expr.is_quasiconvex())
        self.assertFalse(expr.is_quasiconcave())

        problem = cp.Problem(cp.Minimize(expr), [x[0] == 2.0, x[1] == 1.0])
        problem.solve(qcp=True, high=2.1)
        self.assertEqual(problem.objective.value, 2)
        np.testing.assert_almost_equal(x.value, np.array([2, 1, 0, 0, 0]))
