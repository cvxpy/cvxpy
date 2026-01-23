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

import warnings

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

    def test_explicitapprox_true_forces_soc(self) -> None:
        """Test that approx=True forces SOC even with power-cone-capable solver."""
        x = cp.Variable(3)
        prob = cp.Problem(
            cp.Minimize(x[0] + x[1] - x[2]),
            [cp.power(x, 3.3, approx=True) <= np.ones(3)]
        )
        soc_count, p3d_count = self._get_cone_counts(prob, cp.CLARABEL)
        # Should use SOC because user explicitly requested approximation
        self.assertGreater(soc_count, 0, "approx=True should force SOC cones")
        self.assertEqual(p3d_count, 0, "approx=True should not use power cones")

    def test_explicitapprox_false_forces_power_cones(self) -> None:
        """Test that approx=False forces power cones."""
        x = cp.Variable(3)
        prob = cp.Problem(
            cp.Minimize(x[0] + x[1] - x[2]),
            [cp.power(x, 3.3, approx=False) <= np.ones(3)]
        )
        soc_count, p3d_count = self._get_cone_counts(prob, cp.CLARABEL)
        # Should use power cones
        self.assertGreater(p3d_count, 0, "approx=False should use power cones")
        self.assertEqual(soc_count, 0, "approx=False should not use SOC cones")

    def test_powerapprox(self) -> None:
        """Test power atom with approximation."""
        x = cp.Variable(3)
        constr = [cp.power(x, 3.3, approx=True) <= np.ones(3)]
        prob = cp.Problem(cp.Minimize(x[0] + x[1] - x[2]), constr)
        prob.solve(solver=cp.SCS)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        self.assertAlmostEqual(prob.value, -1.0, places=3)
        expected_x = np.array([0.0, 0.0, 1.0])
        self.assertItemsAlmostEqual(x.value, expected_x, places=3)

    def test_power_noapprox(self) -> None:
        """Test power atom without approximation."""
        x = cp.Variable(3)
        constr = [cp.power(x, 3.3, approx=False) <= np.ones(3)]
        prob = cp.Problem(cp.Minimize(x[0] + x[1] - x[2]), constr)
        prob.solve(solver=cp.SCS)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        self.assertAlmostEqual(prob.value, -1.0, places=3)
        expected_x = np.array([0.0, 0.0, 1.0])
        self.assertItemsAlmostEqual(x.value, expected_x, places=3)

    def test_power_with_and_withoutapprox_low(self) -> None:
        """Compare answers with and without approximation on the same problem."""
        x = cp.Variable(3)
        constr = [
            cp.power(x, -1.5, approx=True) <= np.ones(3),
        ]
        prob = cp.Problem(cp.Minimize(x[0] + x[1] - x[2] + (x[1] + x[2])**2), constr)
        prob.solve(solver=cp.CLARABEL)
        x_approx = x.value

        constr = [
            cp.power(x, -1.5, approx=False) <= np.ones(3),
        ]
        prob = cp.Problem(cp.Minimize(x[0] + x[1] - x[2] + (x[1] + x[2])**2), constr)
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        self.assertItemsAlmostEqual(x.value, x_approx, places=3)

    def test_power_with_and_withoutapprox_mid(self) -> None:
        """Compare answers with and without approximation on the same problem."""
        x = cp.Variable(3)
        constr = [
            cp.power(x, 0.8, approx=True) >= np.ones(3),
        ]
        prob = cp.Problem(cp.Minimize(x[0] + x[1] - x[2] + (x[1] + x[2])**2), constr)
        prob.solve(solver=cp.CLARABEL)
        x_approx = x.value

        constr = [
            cp.power(x, 0.8, approx=False) >= np.ones(3),
        ]
        prob = cp.Problem(cp.Minimize(x[0] + x[1] - x[2] + (x[1] + x[2])**2), constr)
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        self.assertItemsAlmostEqual(x.value, x_approx, places=3)

    def test_power_with_and_withoutapprox_high(self) -> None:
        """Compare answers with and without approximation on the same problem."""
        x = cp.Variable(3)
        constr = [
            cp.power(x, 4.5, approx=True) <= np.ones(3),
        ]
        prob = cp.Problem(cp.Minimize(x[0] + x[1] - x[2] + (x[1] + x[2])**2), constr)
        prob.solve(solver=cp.CLARABEL)
        x_approx = x.value

        constr = [
            cp.power(x, 4.5, approx=False) <= np.ones(3),
        ]
        prob = cp.Problem(cp.Minimize(x[0] + x[1] - x[2] + (x[1] + x[2])**2), constr)
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        self.assertItemsAlmostEqual(x.value, x_approx, places=3)

    def test_power_with_and_withoutapprox_even(self) -> None:
        """Compare answers with and without approximation on the same problem."""
        x = cp.Variable(3)
        constr = [
            cp.power(x, 8, approx=True) <= np.ones(3),
        ]
        prob = cp.Problem(cp.Minimize(x[0] + x[1] - x[2] + (x[1] + x[2])**2), constr)
        prob.solve(solver=cp.CLARABEL)
        x_approx = x.value

        constr = [
            cp.power(x, 8, approx=False) <= np.ones(3),
        ]
        prob = cp.Problem(cp.Minimize(x[0] + x[1] - x[2] + (x[1] + x[2])**2), constr)
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        self.assertItemsAlmostEqual(x.value, x_approx, places=3)

    def test_power_noapprox_unsupported_solver(self) -> None:
        """
        Test fallback behavior: approx=False with a solver that doesn't
        support power cones should fall back to SOC.
        This test is skipped if ECOS is not installed.
        """
        if cp.ECOS not in cp.installed_solvers():
            self.skipTest("ECOS not installed.")
        x = cp.Variable(3)
        prob = cp.Problem(
            cp.Minimize(x[0] + x[1] - x[2]),
            [cp.power(x, 3.3, approx=False) <= np.ones(3)]
        )
        # ECOS doesn't support power cones, so should fall back to SOC
        soc_count, p3d_count = self._get_cone_counts(prob, cp.ECOS)
        self.assertGreater(soc_count, 0, "Should fall back to SOC cones")
        self.assertEqual(p3d_count, 0, "Should not use power cones with ECOS")

    def test_approx_warning_triggered_many_soc(self) -> None:
        """Warning should be triggered when many SOC constraints are needed."""
        x = cp.Variable(3)
        prob = cp.Problem(
            cp.Minimize(x[0] + x[1] - x[2]),
            [cp.power(x, 3.3, approx=True) <= np.ones(3)]
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            prob.solve(solver=cp.CLARABEL)
            # Should have at least one warning about power approximation
            power_warnings = [
                warning for warning in w
                if "Power atom" in str(warning.message)
                and "SOC constraints" in str(warning.message)
            ]
            self.assertGreater(len(power_warnings), 0,
                               "Should warn about SOC approximation")

    def test_approx_warning_not_triggered_with_approx_false(self) -> None:
        """Warning should NOT be triggered when using approx=False."""
        x = cp.Variable(3)
        prob = cp.Problem(
            cp.Minimize(x[0] + x[1] - x[2]),
            [cp.power(x, 3.3, approx=False) <= np.ones(3)]
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            prob.solve(solver=cp.CLARABEL)
            power_warnings = [
                warning for warning in w
                if "Power atom" in str(warning.message)
                and "SOC constraints" in str(warning.message)
            ]
            self.assertEqual(len(power_warnings), 0,
                             "Should not warn when using power cones")

    def test_approx_warning_not_triggered_unsupported_solver(self) -> None:
        """Warning should NOT be triggered when solver doesn't support power cones."""
        if cp.ECOS not in cp.installed_solvers():
            self.skipTest("ECOS not installed.")
        x = cp.Variable(3)
        prob = cp.Problem(
            cp.Minimize(x[0] + x[1] - x[2]),
            [cp.power(x, 3.3, approx=True) <= np.ones(3)]
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            prob.solve(solver=cp.ECOS)
            power_warnings = [
                warning for warning in w
                if "Power atom" in str(warning.message)
                and "SOC constraints" in str(warning.message)
            ]
            self.assertEqual(len(power_warnings), 0,
                             "Should not warn when solver doesn't support power cones")

    def test_approx_warning_not_triggered_few_soc(self) -> None:
        """Warning should NOT be triggered when few SOC constraints are needed."""
        x = cp.Variable(3)
        # x^2 uses only 3 SOC constraints, which is <= 4
        prob = cp.Problem(
            cp.Minimize(x[0] + x[1] - x[2]),
            [cp.power(x, 2, approx=True) <= np.ones(3)]
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            prob.solve(solver=cp.CLARABEL)
            power_warnings = [
                warning for warning in w
                if "Power atom" in str(warning.message)
                and "SOC constraints" in str(warning.message)
            ]
            self.assertEqual(len(power_warnings), 0,
                             "Should not warn when few SOC constraints are needed")


class TestGeoMeanApprox(BaseTest):
    """Unit tests for geo_mean approx parameter."""

    def _get_cone_counts(self, prob, solver):
        """Helper to get SOC and power cone counts from problem data."""
        data, _, _ = prob.get_problem_data(solver)
        dims = data['dims']
        # pnd is n-dimensional power cones
        pnd_count = len(dims.pnd) if hasattr(dims, 'pnd') else 0
        return len(dims.soc), pnd_count

    def test_geo_mean_approx_true_uses_soc(self) -> None:
        """Test that geo_mean with approx=True uses SOC constraints."""
        x = cp.Variable(3, pos=True)
        prob = cp.Problem(cp.Maximize(cp.geo_mean(x, approx=True)), [cp.sum(x) <= 3])
        soc_count, pnd_count = self._get_cone_counts(prob, cp.CLARABEL)
        self.assertGreater(soc_count, 0, "approx=True should use SOC cones")
        self.assertEqual(pnd_count, 0, "approx=True should not use power cones")

    def test_geo_mean_approx_false_uses_power_cones(self) -> None:
        """Test that geo_mean with approx=False uses power cones."""
        x = cp.Variable(3, pos=True)
        prob = cp.Problem(cp.Maximize(cp.geo_mean(x, approx=False)), [cp.sum(x) <= 3])
        soc_count, pnd_count = self._get_cone_counts(prob, cp.CLARABEL)
        self.assertGreater(pnd_count, 0, "approx=False should use power cones")
        self.assertEqual(soc_count, 0, "approx=False should not use SOC cones")

    def test_geo_mean_approx_true_solves(self) -> None:
        """Test that geo_mean with approx=True solves correctly."""
        x = cp.Variable(3, pos=True)
        prob = cp.Problem(cp.Maximize(cp.geo_mean(x, approx=True)), [cp.sum(x) <= 3])
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        # Optimal is x = [1, 1, 1], geo_mean = 1
        self.assertAlmostEqual(prob.value, 1.0, places=3)
        self.assertItemsAlmostEqual(x.value, [1.0, 1.0, 1.0], places=3)

    def test_geo_mean_approx_false_solves(self) -> None:
        """Test that geo_mean with approx=False solves correctly."""
        x = cp.Variable(3, pos=True)
        prob = cp.Problem(cp.Maximize(cp.geo_mean(x, approx=False)), [cp.sum(x) <= 3])
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        # Optimal is x = [1, 1, 1], geo_mean = 1
        self.assertAlmostEqual(prob.value, 1.0, places=3)
        self.assertItemsAlmostEqual(x.value, [1.0, 1.0, 1.0], places=3)

    def test_geo_mean_with_and_without_approx(self) -> None:
        """Compare answers with and without approximation."""
        x = cp.Variable(4, pos=True)
        constr = [cp.sum(x) <= 4, x[0] >= 0.5]
        prob = cp.Problem(cp.Maximize(cp.geo_mean(x, approx=True)), constr)
        prob.solve(solver=cp.CLARABEL)
        x_approx = x.value.copy()
        val_approx = prob.value

        prob = cp.Problem(cp.Maximize(cp.geo_mean(x, approx=False)), constr)
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        self.assertAlmostEqual(prob.value, val_approx, places=3)
        self.assertItemsAlmostEqual(x.value, x_approx, places=3)

    def test_geo_mean_weighted_approx(self) -> None:
        """Test weighted geo_mean with approx parameter."""
        x = cp.Variable(3, pos=True)
        weights = [1, 2, 1]
        prob = cp.Problem(
            cp.Maximize(cp.geo_mean(x, weights, approx=True)),
            [cp.sum(x) <= 4]
        )
        prob.solve(solver=cp.CLARABEL)
        val_approx = prob.value
        x_approx = x.value.copy()

        prob = cp.Problem(
            cp.Maximize(cp.geo_mean(x, weights, approx=False)),
            [cp.sum(x) <= 4]
        )
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        self.assertAlmostEqual(prob.value, val_approx, places=3)
        self.assertItemsAlmostEqual(x.value, x_approx, places=3)

    def test_geo_mean_approx_warning_triggered(self) -> None:
        """Warning should be triggered for geo_mean with many SOC constraints."""
        x = cp.Variable(5, pos=True)
        # 5 elements will require more SOC constraints
        prob = cp.Problem(cp.Maximize(cp.geo_mean(x, approx=True)), [cp.sum(x) <= 5])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            prob.solve(solver=cp.CLARABEL)
            geo_mean_warnings = [
                warning for warning in w
                if "geo_mean" in str(warning.message)
                and "SOC constraints" in str(warning.message)
            ]
            self.assertGreater(len(geo_mean_warnings), 0,
                               "Should warn about SOC approximation")

    def test_geo_mean_approx_warning_not_triggered_with_approx_false(self) -> None:
        """Warning should NOT be triggered when using approx=False."""
        x = cp.Variable(5, pos=True)
        prob = cp.Problem(cp.Maximize(cp.geo_mean(x, approx=False)), [cp.sum(x) <= 5])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            prob.solve(solver=cp.CLARABEL)
            geo_mean_warnings = [
                warning for warning in w
                if "geo_mean" in str(warning.message)
                and "SOC constraints" in str(warning.message)
            ]
            self.assertEqual(len(geo_mean_warnings), 0,
                             "Should not warn when using power cones")


class TestPnormApprox(BaseTest):
    """Unit tests for pnorm approx parameter."""

    def _get_cone_counts(self, prob, solver):
        """Helper to get SOC and power cone counts from problem data."""
        data, _, _ = prob.get_problem_data(solver)
        dims = data['dims']
        return len(dims.soc), len(dims.p3d)

    def test_pnorm_approx_true_uses_soc(self) -> None:
        """Test that pnorm with approx=True uses SOC constraints."""
        x = cp.Variable(3)
        prob = cp.Problem(cp.Minimize(cp.pnorm(x, 3, approx=True)), [cp.sum(x) >= 3, x >= 0])
        soc_count, p3d_count = self._get_cone_counts(prob, cp.CLARABEL)
        self.assertGreater(soc_count, 0, "approx=True should use SOC cones")
        self.assertEqual(p3d_count, 0, "approx=True should not use power cones")

    def test_pnorm_approx_false_uses_power_cones(self) -> None:
        """Test that pnorm with approx=False uses power cones."""
        x = cp.Variable(3)
        prob = cp.Problem(cp.Minimize(cp.pnorm(x, 3, approx=False)), [cp.sum(x) >= 3, x >= 0])
        soc_count, p3d_count = self._get_cone_counts(prob, cp.CLARABEL)
        self.assertGreater(p3d_count, 0, "approx=False should use power cones")

    def test_pnorm_approx_true_solves(self) -> None:
        """Test that pnorm with approx=True solves correctly."""
        x = cp.Variable(3)
        prob = cp.Problem(cp.Minimize(cp.pnorm(x, 3, approx=True)), [cp.sum(x) >= 3, x >= 0])
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        # Optimal is x = [1, 1, 1], pnorm_3 = 3^(1/3)
        expected_norm = 3 ** (1/3)
        self.assertAlmostEqual(prob.value, expected_norm, places=3)
        self.assertItemsAlmostEqual(x.value, [1.0, 1.0, 1.0], places=3)

    def test_pnorm_approx_false_solves(self) -> None:
        """Test that pnorm with approx=False solves correctly."""
        x = cp.Variable(3)
        prob = cp.Problem(cp.Minimize(cp.pnorm(x, 3, approx=False)), [cp.sum(x) >= 3, x >= 0])
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        # Optimal is x = [1, 1, 1], pnorm_3 = 3^(1/3)
        expected_norm = 3 ** (1/3)
        self.assertAlmostEqual(prob.value, expected_norm, places=3)
        self.assertItemsAlmostEqual(x.value, [1.0, 1.0, 1.0], places=3)

    def test_pnorm_with_and_without_approx(self) -> None:
        """Compare answers with and without approximation."""
        x = cp.Variable(3)
        constr = [cp.sum(x) >= 3, x >= 0, x[0] <= 2]
        prob = cp.Problem(cp.Minimize(cp.pnorm(x, 3, approx=True)), constr)
        prob.solve(solver=cp.CLARABEL)
        x_approx = x.value.copy()
        val_approx = prob.value

        prob = cp.Problem(cp.Minimize(cp.pnorm(x, 3, approx=False)), constr)
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        self.assertAlmostEqual(prob.value, val_approx, places=3)
        self.assertItemsAlmostEqual(x.value, x_approx, places=3)

    def test_pnorm_fractional_p_with_and_without_approx(self) -> None:
        """Compare answers with and without approximation for fractional p."""
        x = cp.Variable(3)
        constr = [cp.sum(x) >= 3, x >= 0, x[0] <= 2]
        prob = cp.Problem(cp.Minimize(cp.pnorm(x, 2.5, approx=True)), constr)
        prob.solve(solver=cp.CLARABEL)
        x_approx = x.value.copy()
        val_approx = prob.value

        prob = cp.Problem(cp.Minimize(cp.pnorm(x, 2.5, approx=False)), constr)
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        self.assertAlmostEqual(prob.value, val_approx, places=3)
        self.assertItemsAlmostEqual(x.value, x_approx, places=3)

    def test_pnorm_concave_with_and_without_approx(self) -> None:
        """Compare answers with and without approximation for concave pnorm (p < 1)."""
        x = cp.Variable(3, pos=True)
        constr = [cp.sum(x) <= 3, x >= 0.1]
        prob = cp.Problem(cp.Maximize(cp.pnorm(x, 0.5, approx=True)), constr)
        prob.solve(solver=cp.CLARABEL)
        x_approx = x.value.copy()
        val_approx = prob.value

        prob = cp.Problem(cp.Maximize(cp.pnorm(x, 0.5, approx=False)), constr)
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        self.assertAlmostEqual(prob.value, val_approx, places=3)
        self.assertItemsAlmostEqual(x.value, x_approx, places=3)

    def test_pnorm_approx_warning_triggered(self) -> None:
        """Warning should be triggered for pnorm with many SOC constraints."""
        x = cp.Variable(3)
        # p=3.3 will require more SOC constraints
        prob = cp.Problem(cp.Minimize(cp.pnorm(x, 3.3, approx=True)), [cp.sum(x) >= 3, x >= 0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            prob.solve(solver=cp.CLARABEL)
            pnorm_warnings = [
                warning for warning in w
                if "pnorm" in str(warning.message)
                and "SOC constraints" in str(warning.message)
            ]
            self.assertGreater(len(pnorm_warnings), 0,
                               "Should warn about SOC approximation")

    def test_pnorm_approx_warning_not_triggered_with_approx_false(self) -> None:
        """Warning should NOT be triggered when using approx=False."""
        x = cp.Variable(3)
        prob = cp.Problem(cp.Minimize(cp.pnorm(x, 3.3, approx=False)), [cp.sum(x) >= 3, x >= 0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            prob.solve(solver=cp.CLARABEL)
            pnorm_warnings = [
                warning for warning in w
                if "pnorm" in str(warning.message)
                and "SOC constraints" in str(warning.message)
            ]
            self.assertEqual(len(pnorm_warnings), 0,
                             "Should not warn when using power cones")
