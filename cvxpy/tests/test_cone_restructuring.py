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

import pathlib
from unittest import mock

import numpy as np
import scipy.sparse as sp

import cvxpy as cp
import cvxpy.reductions.solvers.conic_solvers.conic_solver as cs_mod
from cvxpy.reductions.cone_format import ConeFormat
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


class TestConeFormatReduction(BaseTest):
    """Formatting is a chain step, not a per-interface convention."""

    def test_conic_chain_formats_before_the_solver(self) -> None:
        x = cp.Variable(3)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x - 1)),
                          [cp.norm(x, 2) <= 2])
        prob.solve(solver=SOLVER)
        reductions = prob._cache.solving_chain.reductions
        self.assertIsInstance(reductions[-2], ConeFormat)
        # The solver receives an already-formatted program; it no longer
        # carries a guard to re-derive the layout itself.
        self.assertTrue(prob._cache.param_prog.formatted)

    def test_every_conic_interface_relies_on_the_reduction(self) -> None:
        """No conic interface may re-derive the cone layout: `formatted` is a
        precondition of ConicSolver.apply, established once by ConeFormat."""
        import cvxpy.reductions.solvers.conic_solvers as conic_pkg
        src_dir = pathlib.Path(conic_pkg.__file__).parent
        offenders = [f.name for f in sorted(src_dir.glob('*.py'))
                     if 'if not problem.formatted:' in f.read_text()]
        self.assertEqual(offenders, [])

    def test_solver_apply_no_longer_formats(self) -> None:
        """The reduction is load-bearing: handed an unformatted program, the
        solver interface uses its rows as they are rather than restructuring
        them."""
        # Several cones in one constraint: R interleaves each t with its own
        # X rows, so it is a genuine permutation rather than the identity.
        X = cp.Variable((3, 2))
        prob = cp.Problem(cp.Minimize(cp.sum(X)), [cp.norm(X, 2, axis=0) <= 2])
        chain = prob._construct_chain(solver=SOLVER)
        solver = chain.reductions[-1]
        chain.reductions = [r for r in chain.reductions
                            if not isinstance(r, (ConeFormat, type(solver)))]
        unformatted = chain.apply(prob)[0]
        self.assertFalse(unformatted.formatted)

        raw = solver.apply(unformatted)[0][cp.settings.A].toarray()
        formatted = ConeFormat(solver).apply(unformatted)[0]
        laid_out = solver.apply(formatted)[0][cp.settings.A].toarray()
        self.assertFalse(np.allclose(raw, laid_out))
        # `raw` is the stuffed order passed straight through: the same rows,
        # merely permuted, which is what the interface no longer corrects.
        self.assertItemsAlmostEqual(np.sort(raw, axis=0), np.sort(laid_out, axis=0))

    def test_qp_chain_has_no_cone_format(self) -> None:
        x = cp.Variable(3)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x - 1)), [x >= 0])
        prob.solve(solver=cp.OSQP)
        self.assertFalse(any(isinstance(r, ConeFormat)
                             for r in prob._cache.solving_chain.reductions))

    def test_format_for_matches_format_constraints(self) -> None:
        for name, prob, _ in _shapes():
            with self.subTest(name):
                chain = prob._construct_chain(solver=SOLVER)
                solver = chain.reductions[-1]
                chain.reductions = [r for r in chain.reductions
                                    if not isinstance(r, (ConeFormat, type(solver)))]
                stuffed = chain.apply(prob)[0]
                self.assertFalse(stuffed.formatted)
                expected = solver.format_constraints(stuffed, solver.EXP_CONE_ORDER)
                got = ConeFormat(solver).apply(stuffed)[0]
                self.assertTrue(got.formatted)
                self.assertItemsAlmostEqual(got.A.toarray(), expected.A.toarray(),
                                            places=12)
