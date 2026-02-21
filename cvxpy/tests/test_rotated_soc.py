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
import numpy as np

import cvxpy as cp
from cvxpy.tests.base_test import BaseTest


class TestRSOC(BaseTest):

    def test_rsoc_basic(self) -> None:
        n = 3
        x = cp.Variable(n)
        y = cp.Variable()
        z = cp.Variable()

        constraints = [
            cp.RSOC(x, y, z),
            y == 1,
            z == 1,
        ]

        prob = cp.Problem(cp.Minimize(cp.norm(x)), constraints)
        prob.solve(solver=cp.CLARABEL)

        self.assertEqual(prob.status, cp.OPTIMAL)
        self.assertItemsAlmostEqual(x.value, np.zeros(n))

    def test_rsoc_equivalence_to_quad_over_lin(self) -> None:
        n = 3
        x = cp.Variable(n)
        y = cp.Variable(nonneg=True)
        z = cp.Variable()

        constraints1 = [cp.RSOC(x, y, z), y == 1]
        constraints2 = [cp.quad_over_lin(x, y) <= z, y == 1]

        prob1 = cp.Problem(cp.Minimize(z), constraints1)
        prob2 = cp.Problem(cp.Minimize(z), constraints2)

        val1 = prob1.solve(solver=cp.CLARABEL)
        val2 = prob2.solve(solver=cp.CLARABEL)

        self.assertAlmostEqual(val1, val2, places=5)

    def test_rsoc_residual_feasible(self) -> None:
        """Residual is ~0 at a feasible (optimal) point."""
        n = 2
        x = cp.Variable(n)
        y = cp.Variable()
        z = cp.Variable()

        con = cp.RSOC(x, y, z)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x)),
                          [con, y + z == 2, y >= 0.5])
        prob.solve(solver=cp.CLARABEL)

        self.assertEqual(prob.status, cp.OPTIMAL)
        self.assertIsNotNone(con.residual)
        self.assertAlmostEqual(float(con.residual), 0.0, places=4)

    def test_rsoc_residual_infeasible_point(self) -> None:
        """Residual is positive when the point violates the RSOC."""
        n = 2
        x = cp.Variable(n)
        y = cp.Variable()
        z = cp.Variable()

        con = cp.RSOC(x, y, z)

        # Set values manually: y*z = 0.25, ||x||^2 = 2 → violated
        x.save_value(np.array([1.0, 1.0]))
        y.save_value(0.5)
        z.save_value(0.5)

        self.assertGreater(float(con.residual), 0.0)

    def test_rsoc_residual_none_when_unset(self) -> None:
        """Residual is None when variable values are not set."""
        x = cp.Variable(2)
        y = cp.Variable()
        z = cp.Variable()

        con = cp.RSOC(x, y, z)
        self.assertIsNone(con.residual)

    def test_rsoc_dual_variable(self) -> None:
        """Dual variables are recovered and satisfy the RSOC dual cone."""
        n = 2
        x = cp.Variable(n)
        y = cp.Variable()
        z = cp.Variable()

        con = cp.RSOC(x, y, z)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x)),
                          [con, y == 1, z == 1])
        prob.solve(solver=cp.CLARABEL)

        self.assertEqual(prob.status, cp.OPTIMAL)

        dual_x, dual_y, dual_z = con.dual_value

        self.assertIsNotNone(dual_x)
        self.assertIsNotNone(dual_y)
        self.assertIsNotNone(dual_z)

        # Dual must satisfy the RSOC dual cone:
        # 4 * dual_y * dual_z >= ||dual_x||^2
        lhs = 4.0 * float(dual_y) * float(dual_z)
        rhs = float(np.dot(np.atleast_1d(dual_x).flatten(),
                           np.atleast_1d(dual_x).flatten()))
        self.assertGreaterEqual(lhs, rhs - 1e-4)

    def test_rsoc_dual_residual(self) -> None:
        """dual_residual reports the violation of the dual cone condition."""
        n = 2
        x = cp.Variable(n)
        y = cp.Variable()
        z = cp.Variable()

        con = cp.RSOC(x, y, z)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x)),
                          [con, y == 1, z == 2])
        prob.solve(solver=cp.CLARABEL)

        self.assertEqual(prob.status, cp.OPTIMAL)
        # At an optimal solution the dual cone residual should be ~0
        self.assertAlmostEqual(float(con.dual_residual), 0.0, places=4)

    def test_rsoc_dpp(self) -> None:
        """RSOC constraint is DPP-compliant when arguments are affine."""
        n = 2
        x = cp.Variable(n)
        y = cp.Parameter(value=1.0)
        z = cp.Parameter(value=2.0)

        prob = cp.Problem(cp.Minimize(cp.sum_squares(x)),
                          [cp.RSOC(x, y, z)])
        self.assertTrue(prob.is_dpp())
        prob.solve(solver=cp.CLARABEL)
        self.assertEqual(prob.status, cp.OPTIMAL)

        # Re-solve with changed parameter values
        y.value = 2.0
        z.value = 3.0
        prob.solve(solver=cp.CLARABEL)
        self.assertEqual(prob.status, cp.OPTIMAL)
