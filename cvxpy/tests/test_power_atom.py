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
        """approx=False raises ValueError when solver lacks power cone support."""
        if cp.ECOS not in cp.installed_solvers():
            self.skipTest("ECOS not installed.")
        x = cp.Variable(3)
        prob = cp.Problem(
            cp.Minimize(x[0] + x[1] - x[2]),
            [cp.power(x, 3.3, approx=False) <= np.ones(3)]
        )
        with self.assertRaises(ValueError, msg="approx=False"):
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
