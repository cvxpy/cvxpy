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
from cvxpy.atoms.elementwise.power import Power, PowerApprox
from cvxpy.atoms.geo_mean import GeoMean, GeoMeanApprox
from cvxpy.tests.base_test import BaseTest


def _get_cone_counts(prob, solver):
    """Helper to get SOC and power cone counts from problem data."""
    data, _, _ = prob.get_problem_data(solver)
    dims = data['dims']
    p3d = len(dims.p3d)
    pnd = len(dims.pnd) if hasattr(dims, 'pnd') else 0
    return len(dims.soc), p3d, pnd


class TestPowerAtom(BaseTest):
    """Unit tests for power atom approx parameter."""

    def test_dunder_pow_returns_approx(self) -> None:
        """x**p uses PowerApprox so it canonicalizes via SOC."""
        x = cp.Variable(3)
        expr = x ** 2
        self.assertIsInstance(expr, PowerApprox)
        self.assertIsInstance(expr, Power)

        expr2 = x ** 0.5
        self.assertIsInstance(expr2, PowerApprox)

        expr3 = x ** 3
        self.assertIsInstance(expr3, PowerApprox)

    def test_approx_controls_cone_type(self) -> None:
        """approx=True uses SOC; approx=False uses power cones."""
        x = cp.Variable(3)
        obj = cp.Minimize(x[0] + x[1] - x[2])

        prob = cp.Problem(obj, [cp.power(x, 3.3, approx=True) <= np.ones(3)])
        soc, p3d, _ = _get_cone_counts(prob, cp.CLARABEL)
        self.assertGreater(soc, 0)
        self.assertEqual(p3d, 0)

        prob = cp.Problem(obj, [cp.power(x, 3.3, approx=False) <= np.ones(3)])
        soc, p3d, _ = _get_cone_counts(prob, cp.CLARABEL)
        self.assertEqual(soc, 0)
        self.assertGreater(p3d, 0)

    def test_approx_and_exact_agree(self) -> None:
        """Approx and exact power cones give the same answer for all p ranges."""
        # p < 0 (low), 0 < p < 1 (mid), p > 1 non-integer (high), p > 1 even integer
        cases = [
            (-1.5, "<="),
            (0.8, ">="),
            (4.5, "<="),
            (8, "<="),
        ]
        for p, direction in cases:
            with self.subTest(p=p):
                x = cp.Variable(3)
                if direction == "<=":
                    constr = [cp.power(x, p, approx=True) <= np.ones(3)]
                else:
                    constr = [cp.power(x, p, approx=True) >= np.ones(3)]
                obj = cp.Minimize(x[0] + x[1] - x[2] + (x[1] + x[2])**2)
                prob = cp.Problem(obj, constr)
                prob.solve(solver=cp.CLARABEL)
                x_approx = x.value.copy()

                if direction == "<=":
                    constr = [cp.power(x, p, approx=False) <= np.ones(3)]
                else:
                    constr = [cp.power(x, p, approx=False) >= np.ones(3)]
                obj = cp.Minimize(x[0] + x[1] - x[2] + (x[1] + x[2])**2)
                prob = cp.Problem(obj, constr)
                prob.solve(solver=cp.CLARABEL)
                self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
                self.assertItemsAlmostEqual(x.value, x_approx, places=3)

    def test_approx_false_errors_without_power_cone_support(self) -> None:
        """approx=False raises SolverError when solver lacks power cone support."""
        if cp.ECOS not in cp.installed_solvers():
            self.skipTest("ECOS not installed.")
        x = cp.Variable(3)
        prob = cp.Problem(
            cp.Minimize(x[0] + x[1] - x[2]),
            [cp.power(x, 3.3, approx=False) <= np.ones(3)]
        )
        with self.assertRaises(cp.error.SolverError):
            prob.solve(solver=cp.ECOS)

    def test_approx_warning(self) -> None:
        """Warning fires for approx=True with many SOCs, not for approx=False."""
        x = cp.Variable(3)
        constr_approx = [cp.power(x, 3.3, approx=True) <= np.ones(3)]
        constr_exact = [cp.power(x, 3.3, approx=False) <= np.ones(3)]
        obj = cp.Minimize(x[0] + x[1] - x[2])

        # approx=True on a power-cone-capable solver should warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cp.Problem(obj, constr_approx).solve(solver=cp.CLARABEL)
            self.assertTrue(
                any("Power atom" in str(wi.message) for wi in w),
                "Should warn about SOC approximation")

        # approx=False should not warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cp.Problem(obj, constr_exact).solve(solver=cp.CLARABEL)
            self.assertFalse(
                any("Power atom" in str(wi.message) for wi in w),
                "Should not warn when using power cones")


class TestGeoMeanApprox(BaseTest):
    """Unit tests for geo_mean approx parameter."""

    def test_approx_controls_cone_type(self) -> None:
        """approx=True uses SOC; approx=False uses power cones."""
        x = cp.Variable(3, pos=True)
        obj = cp.Maximize(cp.geo_mean(x, approx=True))
        prob = cp.Problem(obj, [cp.sum(x) <= 3])
        soc, _, pnd = _get_cone_counts(prob, cp.CLARABEL)
        self.assertGreater(soc, 0)
        self.assertEqual(pnd, 0)

        obj = cp.Maximize(cp.geo_mean(x, approx=False))
        prob = cp.Problem(obj, [cp.sum(x) <= 3])
        soc, _, pnd = _get_cone_counts(prob, cp.CLARABEL)
        self.assertEqual(soc, 0)
        self.assertGreater(pnd, 0)

    def test_approx_and_exact_agree(self) -> None:
        """Approx and exact give the same answer, including weighted case."""
        cases = [
            (None, 3),   # uniform weights, 3 vars
            ([1, 2, 1], 3),  # non-uniform weights
            (None, 4),   # uniform weights, 4 vars
        ]
        for weights, n in cases:
            with self.subTest(weights=weights, n=n):
                x = cp.Variable(n, pos=True)
                constr = [cp.sum(x) <= n, x[0] >= 0.5]

                prob = cp.Problem(
                    cp.Maximize(cp.geo_mean(x, weights, approx=True)), constr)
                prob.solve(solver=cp.CLARABEL)
                val_approx = prob.value
                x_approx = x.value.copy()

                prob = cp.Problem(
                    cp.Maximize(cp.geo_mean(x, weights, approx=False)), constr)
                prob.solve(solver=cp.CLARABEL)
                self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
                self.assertAlmostEqual(prob.value, val_approx, places=3)
                self.assertItemsAlmostEqual(x.value, x_approx, places=3)

    def test_approx_warning(self) -> None:
        """Warning fires for approx=True with many SOCs, not for approx=False."""
        x = cp.Variable(5, pos=True)
        constr = [cp.sum(x) <= 5]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cp.Problem(cp.Maximize(cp.geo_mean(x, approx=True)), constr).solve(
                solver=cp.CLARABEL)
            self.assertTrue(
                any("geo_mean" in str(wi.message) for wi in w),
                "Should warn about SOC approximation")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cp.Problem(cp.Maximize(cp.geo_mean(x, approx=False)), constr).solve(
                solver=cp.CLARABEL)
            self.assertFalse(
                any("geo_mean" in str(wi.message) for wi in w),
                "Should not warn when using power cones")


class TestPnormApprox(BaseTest):
    """Unit tests for pnorm approx parameter."""

    def test_approx_controls_cone_type(self) -> None:
        """approx=True uses SOC; approx=False uses power cones."""
        x = cp.Variable(3)
        constr = [cp.sum(x) >= 3, x >= 0]

        prob = cp.Problem(cp.Minimize(cp.pnorm(x, 3, approx=True)), constr)
        soc, p3d, _ = _get_cone_counts(prob, cp.CLARABEL)
        self.assertGreater(soc, 0)
        self.assertEqual(p3d, 0)

        prob = cp.Problem(cp.Minimize(cp.pnorm(x, 3, approx=False)), constr)
        soc, p3d, _ = _get_cone_counts(prob, cp.CLARABEL)
        self.assertGreater(p3d, 0)

    def test_approx_and_exact_agree(self) -> None:
        """Approx and exact give the same answer for convex and concave pnorm."""
        cases = [
            (3, cp.Minimize, ">=", 3),     # convex, integer p
            (2.5, cp.Minimize, ">=", 3),   # convex, fractional p
            (0.5, cp.Maximize, "<=", 3),   # concave p
        ]
        for p, sense, direction, rhs in cases:
            with self.subTest(p=p):
                x = cp.Variable(3, pos=(p < 1))
                if direction == ">=":
                    constr = [cp.sum(x) >= rhs, x >= 0, x[0] <= 2]
                else:
                    constr = [cp.sum(x) <= rhs, x >= 0.1]

                prob = cp.Problem(sense(cp.pnorm(x, p, approx=True)), constr)
                prob.solve(solver=cp.CLARABEL)
                val_approx = prob.value
                x_approx = x.value.copy()

                prob = cp.Problem(sense(cp.pnorm(x, p, approx=False)), constr)
                prob.solve(solver=cp.CLARABEL)
                self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
                self.assertAlmostEqual(prob.value, val_approx, places=3)
                self.assertItemsAlmostEqual(x.value, x_approx, places=3)

    def test_approx_warning(self) -> None:
        """Warning fires for approx=True with many SOCs, not for approx=False."""
        x = cp.Variable(3)
        constr = [cp.sum(x) >= 3, x >= 0]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cp.Problem(cp.Minimize(cp.pnorm(x, 3.3, approx=True)), constr).solve(
                solver=cp.CLARABEL)
            self.assertTrue(
                any("pnorm" in str(wi.message) for wi in w),
                "Should warn about SOC approximation")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cp.Problem(cp.Minimize(cp.pnorm(x, 3.3, approx=False)), constr).solve(
                solver=cp.CLARABEL)
            self.assertFalse(
                any("pnorm" in str(wi.message) for wi in w),
                "Should not warn when using power cones")


class TestGeoMeanSingleWeight(BaseTest):
    """Tests for geo_mean with a single non-zero weight."""

    def test_single_weight_is_affine(self) -> None:
        """geo_mean with w=(0, 0, 1) should be affine."""
        x = cp.Variable(3)
        g = cp.geo_mean(x, [0, 0, 1])
        self.assertTrue(g.is_convex())
        self.assertTrue(g.is_concave())
        self.assertTrue(g.is_affine())

    def test_single_weight_value(self) -> None:
        """geo_mean with w=(0, 0, 1) should equal x[2]."""
        x = cp.Variable(3)
        prob = cp.Problem(cp.Maximize(cp.geo_mean(x, [0, 0, 1])),
                          [x <= [1, 2, 3], x >= 0])
        prob.solve(solver=cp.SCS)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        self.assertAlmostEqual(prob.value, 3.0, places=3)

    def test_single_weight_nonneg_domain(self) -> None:
        """geo_mean with single weight still requires x >= 0."""
        x = cp.Variable(3)
        g = cp.geo_mean(x, [0, 0, 1])
        domain = g.domain
        # There should be a nonnegativity constraint on the selected element.
        self.assertGreater(len(domain), 0)

    def test_single_weight_nonneg_enforced(self) -> None:
        """Solver should enforce x >= 0 for the single-weight element."""
        x = cp.Variable(1)
        # Minimizing geo_mean(x, [1]) with no lower bound: domain x >= 0
        # means optimal is x = 0.
        prob = cp.Problem(cp.Minimize(x[0]),
                          [cp.geo_mean(x, [1]) >= 0, x <= 5])
        prob.solve(solver=cp.SCS)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        # x should be 0 (not negative), enforced by geo_mean domain.
        self.assertAlmostEqual(x.value[0], 0.0, places=3)

    def test_single_weight_no_cones(self) -> None:
        """Single non-zero weight should not produce any cone constraints."""
        x = cp.Variable(3, pos=True)
        for approx in [True, False]:
            with self.subTest(approx=approx):
                prob = cp.Problem(
                    cp.Maximize(cp.geo_mean(x, [0, 0, 1], approx=approx)),
                    [cp.sum(x) <= 3])
                soc, p3d, pnd = _get_cone_counts(prob, cp.CLARABEL)
                self.assertEqual(soc, 0,
                                 f"approx={approx}: should not use SOC cones")
                self.assertEqual(p3d, 0,
                                 f"approx={approx}: should not use PowCone3D")
                self.assertEqual(pnd, 0,
                                 f"approx={approx}: should not use PowConeND")


class TestInvProdApprox(BaseTest):
    """Unit tests for inv_prod approx parameter."""

    def test_approx_controls_cone_type(self) -> None:
        """approx=True uses SOC; approx=False uses power cones."""
        x = cp.Variable(3, pos=True)
        constr = [cp.sum(x) >= 3]

        prob = cp.Problem(cp.Minimize(cp.inv_prod(x, approx=True)), constr)
        soc, p3d, pnd = _get_cone_counts(prob, cp.CLARABEL)
        self.assertGreater(soc, 0)
        self.assertEqual(p3d, 0)
        self.assertEqual(pnd, 0)

        prob = cp.Problem(cp.Minimize(cp.inv_prod(x, approx=False)), constr)
        soc, p3d, pnd = _get_cone_counts(prob, cp.CLARABEL)
        self.assertEqual(soc, 0)
        # Exact inv_prod uses power cones (3D and/or ND).
        self.assertGreater(p3d + pnd, 0)

    def test_approx_produces_correct_types(self) -> None:
        """approx flag propagates to inner geo_mean and power atoms."""
        x = cp.Variable(3, pos=True)

        expr_approx = cp.inv_prod(x, approx=True)
        self.assertIsInstance(expr_approx, PowerApprox)
        # The inner geo_mean arg should be GeoMeanApprox.
        inner = expr_approx.args[0]
        self.assertIsInstance(inner, GeoMeanApprox)

        expr_exact = cp.inv_prod(x, approx=False)
        self.assertIsInstance(expr_exact, Power)
        self.assertNotIsInstance(expr_exact, PowerApprox)
        inner = expr_exact.args[0]
        self.assertIsInstance(inner, GeoMean)
        self.assertNotIsInstance(inner, GeoMeanApprox)

    def test_approx_and_exact_agree(self) -> None:
        """Approx and exact inv_prod give the same answer."""
        x = cp.Variable(3, pos=True)
        constr = [cp.sum(x) >= 3, x <= 5]

        prob = cp.Problem(cp.Minimize(cp.inv_prod(x, approx=True)), constr)
        prob.solve(solver=cp.CLARABEL)
        val_approx = prob.value

        prob = cp.Problem(cp.Minimize(cp.inv_prod(x, approx=False)), constr)
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        self.assertAlmostEqual(prob.value, val_approx, places=3)
