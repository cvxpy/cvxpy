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

from unittest import mock

import numpy as np
import scipy.sparse as sp

import cvxpy as cp
import cvxpy.reductions.solvers.conic_solvers.conic_solver as cs_mod
from cvxpy.reductions.solvers.conic_solvers.conic_solver import (
    IdentityOperator,
    NegativeIdentityOperator,
    diagonal_restruct_signs,
)
from cvxpy.tests.base_test import BaseTest

SOLVER = cp.CLARABEL


def _shapes():
    """Problems covering diagonal and non-diagonal restructuring."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((3, 4))

    x = cp.Variable(4)
    yield "nonneg only", cp.Problem(cp.Minimize(cp.sum(x)), [A @ x >= 1]), True

    y = cp.Variable(4)
    yield "zero only", cp.Problem(cp.Minimize(cp.sum(y)), [A @ y == 1]), True

    z = cp.Variable(4)
    yield "zero and nonneg", cp.Problem(
        cp.Minimize(cp.sum(z)), [cp.sum(z) == 1, z >= -2, A @ z <= 3]), True

    S = cp.Variable((3, 3), symmetric=True)
    yield "psd and zero", cp.Problem(
        cp.Minimize(cp.trace(S)), [S >> np.eye(3), cp.trace(S) == 5]), True

    w = cp.Variable(4)
    yield "soc", cp.Problem(
        cp.Minimize(cp.sum(w)), [cp.norm2(w) <= 1, cp.sum(w) == 0.5]), False

    v = cp.Variable(3)
    yield "exp cone", cp.Problem(
        cp.Maximize(cp.sum(cp.log(v))), [cp.sum(v) <= 3, v >= 0.1]), False


class TestDiagonalRestructSigns(BaseTest):
    """``diagonal_restruct_signs`` decides whether restructuring is a row sign
    flip (Zero/NonNeg/PSD only) or a genuine row permutation (SOC, and the
    exponential/power cones, which interleave their arguments' rows)."""

    def test_identity_blocks_give_positive_signs(self) -> None:
        signs = diagonal_restruct_signs([IdentityOperator(3), IdentityOperator(2)])
        self.assertItemsAlmostEqual(signs, np.ones(5))

    def test_zero_cone_blocks_give_negative_signs(self) -> None:
        signs = diagonal_restruct_signs(
            [NegativeIdentityOperator(2), IdentityOperator(3)])
        self.assertItemsAlmostEqual(signs, [-1, -1, 1, 1, 1])

    def test_interleaving_block_is_not_diagonal(self) -> None:
        self.assertIsNone(
            diagonal_restruct_signs([IdentityOperator(2), sp.eye_array(3)]))

    def test_no_constraints_is_not_diagonal(self) -> None:
        self.assertIsNone(diagonal_restruct_signs([]))


class TestRestructuringFastPath(BaseTest):
    """The fast path must be indistinguishable from the general path."""

    def test_fast_path_fires_only_for_diagonal_cones(self) -> None:
        for name, prob, expect_diagonal in _shapes():
            with self.subTest(name):
                with mock.patch.object(
                    cs_mod, "as_block_diag_linear_operator",
                    side_effect=cs_mod.as_block_diag_linear_operator
                ) as spy:
                    prob.get_problem_data(SOLVER)
                # The block-diagonal operator is only built on the general path.
                self.assertEqual(spy.called, not expect_diagonal)

    def test_stuffed_data_matches_general_path(self) -> None:
        for name, prob, _ in _shapes():
            fast, _, _ = prob.get_problem_data(SOLVER)
            fast = dict(fast)
            # Force the general path and compare.
            with mock.patch.object(cs_mod, "diagonal_restruct_signs",
                                   return_value=None):
                prob._cache.invalidate()
                slow, _, _ = prob.get_problem_data(SOLVER)
            for key in ("A", "b", "c"):
                if key not in slow:
                    continue
                with self.subTest(f"{name}/{key}"):
                    a, b = fast[key], slow[key]
                    if sp.issparse(a):
                        self.assertEqual(a.shape, b.shape)
                        self.assertItemsAlmostEqual(
                            a.toarray(), b.toarray(), places=12)
                    else:
                        self.assertItemsAlmostEqual(a, b, places=12)

    def test_solutions_match_general_path(self) -> None:
        for name, prob, _ in _shapes():
            with self.subTest(name):
                prob.solve(solver=SOLVER)
                fast_val = prob.value
                with mock.patch.object(cs_mod, "diagonal_restruct_signs",
                                       return_value=None):
                    prob._cache.invalidate()
                    prob.solve(solver=SOLVER)
                self.assertAlmostEqual(fast_val, prob.value, places=6)
