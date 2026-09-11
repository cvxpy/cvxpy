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
from cvxpy.constraints import SOC, ExpCone, Zero
from cvxpy.reductions.cone_format import ConeFormat
from cvxpy.reductions.solvers.conic_solvers.conic_solver import (
    ConicSolver,
    restruct_permutation,
)
from cvxpy.reductions.solvers.solver import Solver
from cvxpy.tests.base_test import BaseTest

SOLVER = cp.CLARABEL


def _shapes():
    """Problems whose restructuring is a pure sign flip, and problems whose
    restructuring genuinely permutes rows. The flag is the latter."""
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

    w = cp.Variable((3, 2))
    # Several cones in one constraint: a single SOC is stored as [t; X], which
    # is already cone order, so only multiple cones actually interleave.
    yield "soc", cp.Problem(
        cp.Minimize(cp.sum(w)), [cp.norm(w, 2, axis=0) <= 1, cp.sum(w) == 0.5]), False

    v = cp.Variable(3)
    yield "exp cone", cp.Problem(
        cp.Maximize(cp.sum(cp.log(v))), [cp.sum(v) <= 3, v >= 0.1]), False


def _materialize(cons, exp_cone_order):
    """R as an explicit sparse matrix, for use as a test oracle only.

    Production code never builds R -- that is the point of
    ``restruct_permutation`` -- so the matrix form lives here.
    """
    new_row, sign = restruct_permutation(cons, exp_cone_order)
    m = new_row.shape[0]
    return sp.csc_array((sign, (new_row, np.arange(m))), shape=(m, m))


def _stuff(prob):
    """The stuffed, *unformatted* program for a problem."""
    chain = prob._construct_chain(solver=SOLVER)
    chain.reductions = [r for r in chain.reductions
                        if not isinstance(r, (Solver, ConeFormat))]
    stuffed = chain.apply(prob)[0]
    assert not stuffed.formatted
    return stuffed


class TestRestructPermutation(BaseTest):
    """R is a signed permutation for every cone family, so restructuring is a
    row remap plus a sign flip rather than a matrix product."""

    def test_is_a_signed_permutation(self) -> None:
        for name, prob, _ in _shapes():
            with self.subTest(name):
                prog = _stuff(prob)
                new_row, sign = restruct_permutation(prog.constraints, [0, 1, 2])
                # Every row lands somewhere, and nowhere twice.
                self.assertItemsAlmostEqual(np.sort(new_row), np.arange(new_row.size))
                self.assertItemsAlmostEqual(np.abs(sign), np.ones(sign.size))

    def test_only_zero_cones_are_negated(self) -> None:
        for name, prob, _ in _shapes():
            with self.subTest(name):
                prog = _stuff(prob)
                _, sign = restruct_permutation(prog.constraints, [0, 1, 2])
                expected = np.concatenate([
                    np.full(c.size, -1.0 if type(c) == Zero else 1.0)
                    for c in prog.constraints])
                self.assertItemsAlmostEqual(sign, expected)

    def test_diagonal_cones_do_not_move(self) -> None:
        """Zero/NonNeg/PSD contribute +-I, so only the sign differs."""
        for name, prob, identity_perm in _shapes():
            with self.subTest(name):
                prog = _stuff(prob)
                new_row, _ = restruct_permutation(prog.constraints, [0, 1, 2])
                is_identity = np.array_equal(new_row, np.arange(new_row.size))
                self.assertEqual(is_identity, identity_perm)

    def test_soc_interleaves_each_t_with_its_own_cone(self) -> None:
        # t has 2 entries, X is 3x2 (column-major), so cone j is
        # (t[j], X[:, j]) and lands on rows 4j .. 4j+3.
        cons = [SOC(cp.Variable(2), cp.Variable((3, 2)))]
        new_row, sign = restruct_permutation(cons, [0, 1, 2])
        self.assertItemsAlmostEqual(new_row, [0, 4, 1, 2, 3, 5, 6, 7])
        self.assertItemsAlmostEqual(sign, np.ones(8))

    def test_exp_cone_follows_the_solver_order(self) -> None:
        v = cp.Variable(2)
        self.assertItemsAlmostEqual(
            restruct_permutation([ExpCone(v, v, v)], [0, 1, 2])[0],
            [0, 3, 1, 4, 2, 5])
        # A solver using the reverse convention gets its arguments swapped.
        self.assertItemsAlmostEqual(
            restruct_permutation([ExpCone(v, v, v)], [2, 1, 0])[0],
            [2, 5, 1, 4, 0, 3])

    def test_no_constraints(self) -> None:
        self.assertIsNone(restruct_permutation([], [0, 1, 2]))


class TestRestructuringEquivalence(BaseTest):
    """Remapping the parameter tensor must equal applying R to the concrete
    ``[A | b]`` -- the check that the index arithmetic is right."""

    def test_tensor_remap_matches_matrix_product(self) -> None:
        for name, prob, _ in _shapes():
            for order in ([0, 1, 2], [2, 1, 0]):
                with self.subTest(f"{name}/{order}"):
                    prog = _stuff(prob)
                    _, _, A, b = prog.apply_parameters()
                    R = _materialize(prog.constraints, order)
                    formatted = ConicSolver.format_constraints(prog, order)
                    _, _, A_new, b_new = formatted.apply_parameters()
                    self.assertEqual(A_new.shape, A.shape)
                    self.assertItemsAlmostEqual(
                        A_new.toarray(), (R @ A).toarray(), places=12)
                    self.assertItemsAlmostEqual(b_new, R @ b, places=12)

    def test_remap_keeps_the_index_dtype(self) -> None:
        """The remap computes in int64 but must store indices in the tensor's
        own dtype: scipy requires indices and indptr to agree, and the permuted
        index has the same bound as the one it replaces.

        The CPP backend emits int32 indices for some problems, so build that
        case explicitly rather than relying on a problem that happens to.
        """
        for name, prob, _ in _shapes():
            with self.subTest(name):
                prog = _stuff(prob)
                narrow = prog.A.tocsc(copy=True)
                narrow.indices = narrow.indices.astype(np.int32)
                narrow.indptr = narrow.indptr.astype(np.int32)
                prog.A = narrow

                A = ConicSolver.format_constraints(prog, [0, 1, 2]).A.tocsc()
                self.assertEqual(A.indices.dtype, np.int32)
                self.assertEqual(A.indices.dtype, A.indptr.dtype)
                # eliminate_zeros raises outright on a mismatch, which is how
                # this surfaced downstream rather than here.
                A.copy().eliminate_zeros()

    def test_identity_restructuring_does_no_work(self) -> None:
        """NonNeg-only cones give R = I, so there is nothing to copy."""
        x = cp.Variable(4)
        rng = np.random.default_rng(0)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [rng.standard_normal((3, 4)) @ x >= 1])
        prog = _stuff(prob)
        self.assertIs(ConicSolver.format_constraints(prog, [0, 1, 2]).A, prog.A)

        # A Zero cone flips signs, so that one does have to copy.
        y = cp.Variable(4)
        signed = _stuff(cp.Problem(cp.Minimize(cp.sum(y)), [cp.sum(y) == 1]))
        self.assertIsNot(ConicSolver.format_constraints(signed, [0, 1, 2]).A, signed.A)

    def test_format_constraints_does_not_mutate_its_input(self) -> None:
        for name, prob, _ in _shapes():
            with self.subTest(name):
                prog = _stuff(prob)
                before = prog.A.copy().toarray()
                ConicSolver.format_constraints(prog, [0, 1, 2])
                self.assertItemsAlmostEqual(prog.A.toarray(), before, places=12)

    def test_solutions_are_unchanged(self) -> None:
        for name, prob, _ in _shapes():
            with self.subTest(name):
                prob.solve(solver=SOLVER)
                reference = prob.value
                prob._cache.invalidate()
                prob.solve(solver=SOLVER, canon_backend=cp.settings.SCIPY_CANON_BACKEND)
                self.assertAlmostEqual(reference, prob.value, places=6)


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
        self.assertEqual(raw.shape, laid_out.shape)
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

    def test_format_for_dispatches_on_the_program(self) -> None:
        """ConeFormat delegates to the program, which is the seam a program
        owning a re-extractable form overrides."""
        for name, prob, _ in _shapes():
            with self.subTest(name):
                chain = prob._construct_chain(solver=SOLVER)
                solver = chain.reductions[-1]
                stuffed = _stuff(prob)
                with mock.patch.object(
                        type(stuffed), 'format_for',
                        side_effect=stuffed.format_for) as spy:
                    got = ConeFormat(solver).apply(stuffed)[0]
                spy.assert_called_once_with(solver)
                self.assertTrue(got.formatted)
                # ...and an already-formatted program is left alone.
                self.assertIs(ConeFormat(solver).apply(got)[0], got)
