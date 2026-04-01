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
from cvxpy.reductions.chain import Chain
from cvxpy.reductions.cone2cone.dualize import Dualize
from cvxpy.reductions.cvx_attr2constr import CvxAttr2Constr
from cvxpy.reductions.dcp2cone.dcp2cone import Dcp2Cone
from cvxpy.reductions.solution import Solution
from cvxpy.tests.base_test import BaseTest


class TestDualizeReduction(BaseTest):
    """Tests for the expression-level Dualize reduction."""

    @staticmethod
    def _canonicalize(prob):
        """Apply Dcp2Cone + CvxAttr2Constr to get a conic problem."""
        chain = Chain(None, [Dcp2Cone(), CvxAttr2Constr()])
        return chain.apply(prob), chain

    @staticmethod
    def _solve_via_dualize(prob, solver=cp.CLARABEL):
        """Solve a problem by dualizing, solving the dual, and inverting."""
        (canon, pre_inv), pre_chain = TestDualizeReduction._canonicalize(prob)

        dualize = Dualize()
        assert dualize.accepts(canon), "Dualize does not accept canonicalized problem"
        dual, dual_inv = dualize.apply(canon)

        dual.solve(solver=solver)

        dual_sol = Solution(
            dual.status, dual.value,
            {v.id: v.value for v in dual.variables() if v.value is not None},
            {c.id: c.dual_value for c in dual.constraints
             if c.dual_value is not None},
            {},
        )

        primal_sol = dualize.invert(dual_sol, dual_inv)
        orig_sol = pre_chain.invert(primal_sol, pre_inv)
        prob.unpack(orig_sol)

    # ------------------------------------------------------------------ #
    #  LP tests                                                           #
    # ------------------------------------------------------------------ #

    def test_lp_nonneg(self) -> None:
        x = cp.Variable(3)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 1])
        prob.solve(solver=cp.CLARABEL)
        ref_val = prob.value
        ref_x = x.value.copy()

        x.value = None
        self._solve_via_dualize(prob)
        self.assertAlmostEqual(prob.value, ref_val, places=4)
        np.testing.assert_allclose(x.value, ref_x, atol=1e-4)

    def test_lp_equality(self) -> None:
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(x[0] + 2 * x[1]),
                          [x[0] + x[1] == 1, x >= 0])
        prob.solve(solver=cp.CLARABEL)
        ref_val = prob.value
        ref_x = x.value.copy()

        x.value = None
        self._solve_via_dualize(prob)
        self.assertAlmostEqual(prob.value, ref_val, places=4)
        np.testing.assert_allclose(x.value, ref_x, atol=1e-4)

    def test_lp_mixed_constraints(self) -> None:
        x = cp.Variable(3)
        prob = cp.Problem(
            cp.Minimize(x[0] - x[1] + 2 * x[2]),
            [x >= 0, cp.sum(x) == 5, x[0] <= 3],
        )
        prob.solve(solver=cp.CLARABEL)
        ref_val = prob.value
        ref_x = x.value.copy()

        x.value = None
        self._solve_via_dualize(prob)
        self.assertAlmostEqual(prob.value, ref_val, places=4)
        np.testing.assert_allclose(x.value, ref_x, atol=1e-4)

    def test_lp_infeasible(self) -> None:
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.sum(x)),
                          [x >= 1, x <= -1])
        prob.solve(solver=cp.CLARABEL)
        ref_status = prob.status

        self._solve_via_dualize(prob)
        self.assertEqual(prob.status, ref_status)

    def test_lp_unbounded(self) -> None:
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x <= 10])
        prob.solve(solver=cp.CLARABEL)
        ref_status = prob.status

        self._solve_via_dualize(prob)
        self.assertEqual(prob.status, ref_status)

    # ------------------------------------------------------------------ #
    #  SOCP tests                                                         #
    # ------------------------------------------------------------------ #

    def test_socp(self) -> None:
        x = cp.Variable(3)
        prob = cp.Problem(cp.Minimize(cp.sum(x)),
                          [cp.norm(x, 2) <= 1])
        prob.solve(solver=cp.CLARABEL)
        ref_val = prob.value
        ref_x = x.value.copy()

        x.value = None
        self._solve_via_dualize(prob)
        self.assertAlmostEqual(prob.value, ref_val, places=4)
        np.testing.assert_allclose(x.value, ref_x, atol=1e-4)

    def test_socp_with_linear(self) -> None:
        x = cp.Variable(2)
        prob = cp.Problem(
            cp.Minimize(x[0] + x[1]),
            [cp.norm(x, 2) <= 2, x >= 0],
        )
        prob.solve(solver=cp.CLARABEL)
        ref_val = prob.value
        ref_x = x.value.copy()

        x.value = None
        self._solve_via_dualize(prob)
        self.assertAlmostEqual(prob.value, ref_val, places=4)
        np.testing.assert_allclose(x.value, ref_x, atol=1e-4)

    # ------------------------------------------------------------------ #
    #  Exponential cone tests                                             #
    # ------------------------------------------------------------------ #

    def test_exp_cone(self) -> None:
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(-cp.sum(x)),
                          [cp.exp(x[0]) + cp.exp(x[1]) <= 2, x >= -1])
        prob.solve(solver=cp.CLARABEL)
        ref_val = prob.value
        ref_x = x.value.copy()

        x.value = None
        self._solve_via_dualize(prob)
        self.assertAlmostEqual(prob.value, ref_val, places=3)
        np.testing.assert_allclose(x.value, ref_x, atol=1e-3)

    # ------------------------------------------------------------------ #
    #  SDP tests                                                          #
    # ------------------------------------------------------------------ #

    def test_sdp(self) -> None:
        X = cp.Variable((2, 2), symmetric=True)
        prob = cp.Problem(cp.Minimize(cp.trace(X)),
                          [X >> np.eye(2)])
        prob.solve(solver=cp.CLARABEL)
        ref_val = prob.value

        X.value = None
        self._solve_via_dualize(prob)
        self.assertAlmostEqual(prob.value, ref_val, places=3)

    # ------------------------------------------------------------------ #
    #  Accepts / rejects                                                  #
    # ------------------------------------------------------------------ #

    def test_rejects_maximize(self) -> None:
        x = cp.Variable()
        prob = cp.Problem(cp.Maximize(x), [x <= 1])
        dualize = Dualize()
        # After Dcp2Cone the objective is still Maximize → rejected
        self.assertFalse(dualize.accepts(prob))

    def test_rejects_integer(self) -> None:
        x = cp.Variable(2, integer=True)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 0])
        dualize = Dualize()
        self.assertFalse(dualize.accepts(prob))

    def test_rejects_parameters(self) -> None:
        x = cp.Variable(2)
        p = cp.Parameter()
        p.value = 1.0
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= p])
        # After canonicalization, parameters are still present for DPP
        (canon, _), _ = self._canonicalize(prob)
        dualize = Dualize()
        self.assertFalse(dualize.accepts(canon))

    # ------------------------------------------------------------------ #
    #  Strong duality (opt values match)                                  #
    # ------------------------------------------------------------------ #

    def test_strong_duality_values(self) -> None:
        """Verify that the dual optimal value equals the primal."""
        x = cp.Variable(3)
        prob = cp.Problem(
            cp.Minimize(x[0] + 2 * x[1] + 3 * x[2]),
            [x >= 0, cp.sum(x) == 10, x[2] <= 5],
        )
        prob.solve(solver=cp.CLARABEL)
        primal_val = prob.value

        (canon, pre_inv), _ = self._canonicalize(prob)
        dualize = Dualize()
        dual, dual_inv = dualize.apply(canon)
        dual.solve(solver=cp.CLARABEL)

        # Dual-as-min optimal = -p*
        self.assertAlmostEqual(-dual.value, primal_val, places=5)

    # ------------------------------------------------------------------ #
    #  No constraints edge case                                           #
    # ------------------------------------------------------------------ #

    def test_no_constraints(self) -> None:
        """With no constraints the reduction is a no-op."""
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x))
        (canon, _), _ = self._canonicalize(prob)
        dualize = Dualize()
        dual, inv = dualize.apply(canon)
        self.assertIsNone(inv)
