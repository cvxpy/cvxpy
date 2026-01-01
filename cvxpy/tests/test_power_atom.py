"""
Copyright 2025 CVXPY developers

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


class TestPowerAtom(BaseTest):
    """Unit tests for power atom."""

    def _get_cone_counts(self, prob, solver):
        """Helper to get SOC and power cone counts from problem data."""
        data, _, _ = prob.get_problem_data(solver)
        dims = data['dims']
        return len(dims.soc), len(dims.p3d)

    def test_explicit_approx_true_forces_soc(self) -> None:
        """Test that _approx=True forces SOC even with power-cone-capable solver."""
        x = cp.Variable(3)
        prob = cp.Problem(
            cp.Minimize(x[0] + x[1] - x[2]),
            [cp.power(x, 3.3, _approx=True) <= np.ones(3)]
        )
        soc_count, p3d_count = self._get_cone_counts(prob, cp.CLARABEL)
        # Should use SOC because user explicitly requested approximation
        self.assertGreater(soc_count, 0, "_approx=True should force SOC cones")
        self.assertEqual(p3d_count, 0, "_approx=True should not use power cones")

    def test_explicit_approx_false_forces_power_cones(self) -> None:
        """Test that _approx=False forces power cones."""
        x = cp.Variable(3)
        prob = cp.Problem(
            cp.Minimize(x[0] + x[1] - x[2]),
            [cp.power(x, 3.3, _approx=False) <= np.ones(3)]
        )
        soc_count, p3d_count = self._get_cone_counts(prob, cp.CLARABEL)
        # Should use power cones
        self.assertGreater(p3d_count, 0, "_approx=False should use power cones")
        self.assertEqual(soc_count, 0, "_approx=False should not use SOC cones")

    def test_power_approx(self) -> None:
        """Test power atom with approximation."""
        x = cp.Variable(3)
        constr = [cp.power(x, 3.3, _approx=True) <= np.ones(3)]
        prob = cp.Problem(cp.Minimize(x[0] + x[1] - x[2]), constr)
        prob.solve(solver=cp.SCS)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        self.assertAlmostEqual(prob.value, -1.0, places=3)
        expected_x = np.array([0.0, 0.0, 1.0])
        self.assertItemsAlmostEqual(x.value, expected_x, places=3)

    def test_power_no_approx(self) -> None:
        """Test power atom without approximation."""
        x = cp.Variable(3)
        constr = [cp.power(x, 3.3, _approx=False) <= np.ones(3)]
        prob = cp.Problem(cp.Minimize(x[0] + x[1] - x[2]), constr)
        prob.solve(solver=cp.SCS)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        self.assertAlmostEqual(prob.value, -1.0, places=3)
        expected_x = np.array([0.0, 0.0, 1.0])
        self.assertItemsAlmostEqual(x.value, expected_x, places=3)

    def test_power_with_and_without_approx_low(self) -> None:
        """Compare answers with and without approximation on the same problem."""
        x = cp.Variable(3)
        constr = [
            cp.power(x, -1.5, _approx=True) <= np.ones(3),
        ]
        prob = cp.Problem(cp.Minimize(x[0] + x[1] - x[2] + (x[1] + x[2])**2), constr)
        prob.solve(solver=cp.CLARABEL)
        x_approx = x.value

        constr = [
            cp.power(x, -1.5, _approx=False) <= np.ones(3),
        ]
        prob = cp.Problem(cp.Minimize(x[0] + x[1] - x[2] + (x[1] + x[2])**2), constr)
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        self.assertItemsAlmostEqual(x.value, x_approx, places=3)

    def test_power_with_and_without_approx_mid(self) -> None:
        """Compare answers with and without approximation on the same problem."""
        x = cp.Variable(3)
        constr = [
            cp.power(x, 0.8, _approx=True) >= np.ones(3),
        ]
        prob = cp.Problem(cp.Minimize(x[0] + x[1] - x[2] + (x[1] + x[2])**2), constr)
        prob.solve(solver=cp.CLARABEL)
        x_approx = x.value

        constr = [
            cp.power(x, 0.8, _approx=False) >= np.ones(3),
        ]
        prob = cp.Problem(cp.Minimize(x[0] + x[1] - x[2] + (x[1] + x[2])**2), constr)
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        self.assertItemsAlmostEqual(x.value, x_approx, places=3)

    def test_power_with_and_without_approx_high(self) -> None:
        """Compare answers with and without approximation on the same problem."""
        x = cp.Variable(3)
        constr = [
            cp.power(x, 4.5, _approx=True) <= np.ones(3),
        ]
        prob = cp.Problem(cp.Minimize(x[0] + x[1] - x[2] + (x[1] + x[2])**2), constr)
        prob.solve(solver=cp.CLARABEL)
        x_approx = x.value

        constr = [
            cp.power(x, 4.5, _approx=False) <= np.ones(3),
        ]
        prob = cp.Problem(cp.Minimize(x[0] + x[1] - x[2] + (x[1] + x[2])**2), constr)
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        self.assertItemsAlmostEqual(x.value, x_approx, places=3)

    def test_power_with_and_without_approx_even(self) -> None:
        """Compare answers with and without approximation on the same problem."""
        x = cp.Variable(3)
        constr = [
            cp.power(x, 8, _approx=True) <= np.ones(3),
        ]
        prob = cp.Problem(cp.Minimize(x[0] + x[1] - x[2] + (x[1] + x[2])**2), constr)
        prob.solve(solver=cp.CLARABEL)
        x_approx = x.value

        constr = [
            cp.power(x, 8, _approx=False) <= np.ones(3),
        ]
        prob = cp.Problem(cp.Minimize(x[0] + x[1] - x[2] + (x[1] + x[2])**2), constr)
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        self.assertItemsAlmostEqual(x.value, x_approx, places=3)

    def test_power_no_approx_unsupported_solver(self) -> None:
        """
        Test fallback behavior: _approx=False with a solver that doesn't
        support power cones should fall back to SOC.
        This test is skipped if ECOS is not installed.
        """
        if cp.ECOS not in cp.installed_solvers():
            self.skipTest("ECOS not installed.")
        x = cp.Variable(3)
        prob = cp.Problem(
            cp.Minimize(x[0] + x[1] - x[2]),
            [cp.power(x, 3.3, _approx=False) <= np.ones(3)]
        )
        # ECOS doesn't support power cones, so should fall back to SOC
        soc_count, p3d_count = self._get_cone_counts(prob, cp.ECOS)
        self.assertGreater(soc_count, 0, "Should fall back to SOC cones")
        self.assertEqual(p3d_count, 0, "Should not use power cones with ECOS")
