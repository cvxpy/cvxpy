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

import unittest
import warnings

import numpy as np

import cvxpy as cp
from cvxpy.atoms.elementwise.power import Power, PowerApprox
from cvxpy.atoms.geo_mean import GeoMean
from cvxpy.constraints.power import PowCone3D, PowConeND
from cvxpy.constraints.second_order import SOC
from cvxpy.reductions.cone2cone.approx import (
    APPROX_CONE_CONVERSIONS,
    ApproxCone2Cone,
)
from cvxpy.reductions.cone2cone.exact import (
    EXACT_CONE_CONVERSIONS,
    ExactCone2Cone,
)
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
        """approx flag controls cone type based on solver support.

        With new semantics:
        - approx=True: use power cones if supported, fall back to SOC otherwise
        - approx=False: require power cones (don't allow SOC approximation)

        When solver supports power cones, both use PowCone3D.
        """
        x = cp.Variable(3)
        obj = cp.Minimize(x[0] + x[1] - x[2])

        # CLARABEL supports power cones, so both should use PowCone3D
        prob = cp.Problem(obj, [cp.power(x, 3.3, approx=True) <= np.ones(3)])
        soc, p3d, _ = _get_cone_counts(prob, cp.CLARABEL)
        self.assertGreater(p3d, 0)  # Uses power cones when available

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
        """Warning fires when SOC approximation is used (solver lacks power cones).

        With new semantics:
        - When solver supports power cones: no SOC approximation, no warning
        - When solver doesn't support power cones: SOC approximation, may warn
        """
        x = cp.Variable(3)
        constr_approx = [cp.power(x, 3.3, approx=True) <= np.ones(3)]
        constr_exact = [cp.power(x, 3.3, approx=False) <= np.ones(3)]
        obj = cp.Minimize(x[0] + x[1] - x[2])

        # approx=True on a power-cone-capable solver: uses power cones, no warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cp.Problem(obj, constr_approx).solve(solver=cp.CLARABEL)
            self.assertFalse(
                any("SOC constraints" in str(wi.message) for wi in w),
                "Should not warn when using power cones (solver supports them)")

        # approx=False should not warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cp.Problem(obj, constr_exact).solve(solver=cp.CLARABEL)
            self.assertFalse(
                any("SOC constraints" in str(wi.message) for wi in w),
                "Should not warn when using power cones")


class TestGeoMeanApprox(BaseTest):
    """Unit tests for geo_mean approx parameter."""

    def test_approx_controls_cone_type(self) -> None:
        """approx flag controls cone type based on solver support.

        With new semantics: when solver supports power cones, both use PowConeND.
        """
        x = cp.Variable(3, pos=True)

        # CLARABEL supports power cones, so both should use PowConeND
        obj = cp.Maximize(cp.geo_mean(x, approx=True))
        prob = cp.Problem(obj, [cp.sum(x) <= 3])
        soc, _, pnd = _get_cone_counts(prob, cp.CLARABEL)
        self.assertGreater(pnd, 0)  # Uses power cones when available

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
        """Warning fires when SOC approximation is used (solver lacks power cones).

        With new semantics: no warning when solver supports power cones.
        """
        x = cp.Variable(5, pos=True)
        constr = [cp.sum(x) <= 5]

        # approx=True on a power-cone-capable solver: uses power cones, no warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cp.Problem(cp.Maximize(cp.geo_mean(x, approx=True)), constr).solve(
                solver=cp.CLARABEL)
            self.assertFalse(
                any("SOC constraints" in str(wi.message) for wi in w),
                "Should not warn when using power cones (solver supports them)")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cp.Problem(cp.Maximize(cp.geo_mean(x, approx=False)), constr).solve(
                solver=cp.CLARABEL)
            self.assertFalse(
                any("SOC constraints" in str(wi.message) for wi in w),
                "Should not warn when using power cones")


class TestPnormApprox(BaseTest):
    """Unit tests for pnorm approx parameter."""

    def test_approx_controls_cone_type(self) -> None:
        """approx flag controls cone type based on solver support.

        With new semantics: when solver supports power cones, both use PowCone3D.
        """
        x = cp.Variable(3)
        constr = [cp.sum(x) >= 3, x >= 0]

        # CLARABEL supports power cones, so both should use PowCone3D
        prob = cp.Problem(cp.Minimize(cp.pnorm(x, 3, approx=True)), constr)
        soc, p3d, _ = _get_cone_counts(prob, cp.CLARABEL)
        self.assertGreater(p3d, 0)  # Uses power cones when available

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
        """Warning fires when SOC approximation is used (solver lacks power cones).

        With new semantics: no warning when solver supports power cones.
        """
        x = cp.Variable(3)
        constr = [cp.sum(x) >= 3, x >= 0]

        # approx=True on a power-cone-capable solver: uses power cones, no warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cp.Problem(cp.Minimize(cp.pnorm(x, 3.3, approx=True)), constr).solve(
                solver=cp.CLARABEL)
            self.assertFalse(
                any("SOC constraints" in str(wi.message) for wi in w),
                "Should not warn when using power cones (solver supports them)")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cp.Problem(cp.Minimize(cp.pnorm(x, 3.3, approx=False)), constr).solve(
                solver=cp.CLARABEL)
            self.assertFalse(
                any("SOC constraints" in str(wi.message) for wi in w),
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
        """approx flag controls cone type based on solver support.

        With new semantics: when solver supports power cones, both use power cones.
        """
        x = cp.Variable(3, pos=True)
        constr = [cp.sum(x) >= 3]

        # CLARABEL supports power cones, so both should use power cones
        prob = cp.Problem(cp.Minimize(cp.inv_prod(x, approx=True)), constr)
        soc, p3d, pnd = _get_cone_counts(prob, cp.CLARABEL)
        self.assertGreater(p3d + pnd, 0)  # Uses power cones when available

        prob = cp.Problem(cp.Minimize(cp.inv_prod(x, approx=False)), constr)
        soc, p3d, pnd = _get_cone_counts(prob, cp.CLARABEL)
        self.assertEqual(soc, 0)
        # Exact inv_prod uses power cones (3D and/or ND).
        self.assertGreater(p3d + pnd, 0)

    def test_approx_produces_correct_types(self) -> None:
        """approx flag propagates to inner geo_mean and power atoms."""
        x = cp.Variable(3, pos=True)

        expr_approx = cp.inv_prod(x, approx=True)
        self.assertIsInstance(expr_approx, Power)
        self.assertTrue(expr_approx.allow_approx)
        # The inner geo_mean arg should have allow_approx=True.
        inner = expr_approx.args[0]
        self.assertIsInstance(inner, GeoMean)
        self.assertTrue(inner.allow_approx)

        expr_exact = cp.inv_prod(x, approx=False)
        self.assertIsInstance(expr_exact, Power)
        self.assertFalse(expr_exact.allow_approx)
        inner = expr_exact.args[0]
        self.assertIsInstance(inner, GeoMean)
        self.assertFalse(inner.allow_approx)

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


class TestUnifiedCone2Cone(BaseTest):
    """Tests for the unified ExactCone2Cone / ApproxCone2Cone framework."""

    def test_exact_cone_conversions_map(self) -> None:
        """EXACT_CONE_CONVERSIONS contains PowConeND and SOC."""
        from cvxpy.constraints.psd import PSD
        self.assertIn(PowConeND, EXACT_CONE_CONVERSIONS)
        self.assertEqual(EXACT_CONE_CONVERSIONS[PowConeND], {PowCone3D})
        self.assertIn(SOC, EXACT_CONE_CONVERSIONS)
        self.assertEqual(EXACT_CONE_CONVERSIONS[SOC], {PSD})
        # PowCone3D should NOT be in exact conversions
        self.assertNotIn(PowCone3D, EXACT_CONE_CONVERSIONS)

    def test_approx_cone_conversions_map(self) -> None:
        """APPROX_CONE_CONVERSIONS contains PowCone3D, RelEntrConeQuad, OpRelEntrConeQuad."""
        self.assertIn(PowCone3D, APPROX_CONE_CONVERSIONS)
        self.assertEqual(APPROX_CONE_CONVERSIONS[PowCone3D], {SOC})

    def test_exact_cone2cone_target_cones_filtering(self) -> None:
        """ExactCone2Cone(target_cones={PowConeND}) only converts PowConeND."""
        reduction = ExactCone2Cone(target_cones={PowConeND})
        self.assertIn(PowConeND, reduction.canon_methods)
        self.assertNotIn(SOC, reduction.canon_methods)

    def test_exact_cone2cone_target_cones_soc(self) -> None:
        """ExactCone2Cone(target_cones={SOC}) only converts SOC."""
        reduction = ExactCone2Cone(target_cones={SOC})
        self.assertIn(SOC, reduction.canon_methods)
        self.assertNotIn(PowConeND, reduction.canon_methods)

    def test_approx_cone2cone_target_cones_filtering(self) -> None:
        """ApproxCone2Cone(target_cones={PowCone3D}) only converts PowCone3D."""
        from cvxpy.constraints.exponential import RelEntrConeQuad
        reduction = ApproxCone2Cone(target_cones={PowCone3D})
        self.assertIn(PowCone3D, reduction.canon_methods)
        self.assertNotIn(RelEntrConeQuad, reduction.canon_methods)

    @unittest.skipUnless(
        cp.CVXOPT in cp.installed_solvers(),
        'CVXOPT solver is not installed.'
    )
    def test_soc_to_psd_via_exact_cone2cone(self) -> None:
        """SOC constraint solved via CVXOPT (PSD-only) uses ExactCone2Cone."""
        x = cp.Variable(3)
        prob = cp.Problem(
            cp.Minimize(cp.sum(x)),
            [cp.norm(x, 2) <= 1]
        )
        prob.solve(solver=cp.CVXOPT)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        self.assertAlmostEqual(prob.value, -np.sqrt(3), places=3)

    @unittest.skipUnless(
        cp.CVXOPT in cp.installed_solvers(),
        'CVXOPT solver is not installed.'
    )
    def test_soc_to_psd_packed(self) -> None:
        """Packed SOC constraints via CVXOPT (PSD-only) work correctly."""
        x = cp.Variable(3)
        t = cp.Variable(2)
        # Two SOC constraints packed together
        prob = cp.Problem(
            cp.Minimize(t[0] + t[1]),
            [cp.SOC(t, cp.vstack([x[:2], x[1:]]).T, axis=0),
             cp.sum(x) == 1, x >= 0]
        )
        prob.solve(solver=cp.CVXOPT)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])

    def test_explicit_powcone3d_approximated_when_unsupported(self) -> None:
        """Explicit PowCone3D constraint is approximated via ApproxCone2Cone
        when the solver doesn't support PowCone3D."""
        x = cp.Variable(pos=True)
        y = cp.Variable(pos=True)
        z = cp.Variable()
        prob = cp.Problem(
            cp.Maximize(z),
            [PowCone3D(x, y, z, 0.5),
             x + y <= 2]
        )
        # CVXOPT does not support PowCone3D, so it should be approximated
        if cp.CVXOPT in cp.installed_solvers():
            prob.solve(solver=cp.CVXOPT)
            self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
            # x^0.5 * y^0.5 >= |z|  with x+y<=2, max z
            # at x=y=1, z=1
            self.assertAlmostEqual(z.value, 1.0, places=2)

    def test_approx_cone2cone_dual_recovery(self) -> None:
        """ApproxCone2Cone recovers dual variables for approximated constraints."""
        from cvxpy.reductions.solution import Solution
        x = cp.Variable(pos=True)
        y = cp.Variable(pos=True)
        z = cp.Variable()
        pow_con = PowCone3D(x, y, z, 0.5)
        prob = cp.Problem(cp.Maximize(z), [pow_con, x + y <= 2])

        # Apply ApproxCone2Cone reduction
        reduction = ApproxCone2Cone(problem=prob, target_cones={PowCone3D})
        reduced_prob, inverse_data = reduction.apply(prob)

        # Verify the constraint ID mapping is set up
        self.assertIn(pow_con.id, inverse_data.cons_id_map)
        canon_id = inverse_data.cons_id_map[pow_con.id]

        # Create a mock solution with dual variables for the canonical constraint
        mock_dual_value = np.array([1.0])
        mock_solution = Solution(
            status=cp.OPTIMAL,
            opt_val=1.0,
            primal_vars={},
            dual_vars={canon_id: mock_dual_value},
            attr={}
        )

        # Invert the solution
        inverted = reduction.invert(mock_solution, inverse_data)

        # Verify dual value is recovered for the original constraint
        self.assertIn(pow_con.id, inverted.dual_vars)
        self.assertEqual(inverted.dual_vars[pow_con.id], mock_dual_value)

