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
from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np
import pytest
import scipy.sparse as sp

import cvxpy as cp
import cvxpy.settings as s
from cvxpy.atoms.quad_form import SymbolicQuadForm
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.reductions.dcp2cone.diffengine_cone_program import DiffengineConeProgram
from cvxpy.reductions.solvers.defines import INSTALLED_MI_SOLVERS
from cvxpy.tests.base_test import BaseTest

try:
    from sparsediffpy import _sparsediffengine as _engine
except ImportError:  # pragma: no cover
    _engine = None

# On an older sparsediffpy wheel every test here would AttributeError
# mid-solve, so skip the module loudly instead.
REQUIRED_BINDINGS = ("make_left_kron", "make_right_kron", "make_power", "make_quad_form")
MISSING = (["sparsediffpy not installed"] if _engine is None else
           [name for name in REQUIRED_BINDINGS if not hasattr(_engine, name)])
pytestmark = pytest.mark.skipif(
    bool(MISSING),
    reason="DIFFENGINE backend requires sparsediffpy >= 0.6.0 "
           f"(missing: {', '.join(MISSING)})",
)

SOLVER = cp.CLARABEL
DIFFENGINE = s.DIFFENGINE_CANON_BACKEND


class TestDiffengineConverter(BaseTest):
    """Converter correctness and fail-loud behavior: unsupported constructs
    must raise immediately instead of silently miscompiling."""

    def test_unsupported_atom_raises(self) -> None:
        """convert_expr names the offending atom."""
        from cvxpy.reductions.solvers.nlp_solvers.diff_engine.converters import (
            convert_expr,
        )
        x = cp.Variable(4)
        with self.assertRaisesRegex(NotImplementedError, "cumsum"):
            convert_expr(cp.cumsum(x), {x.id: None}, 4, {})

    def test_constant_nonlinear_subtree_is_evaluated(self) -> None:
        """A nonlinear atom over plain constants is evaluated numerically;
        the engine cannot differentiate through it."""
        x = cp.Variable(2)
        const_term = cp.quad_over_lin(np.array([1.0, 1.0]), 1.0)  # == 2.0
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), [cp.sum(x) >= const_term])
        prob.solve(solver=SOLVER, canon_backend=DIFFENGINE)
        self.assertEqual(prob.status, cp.OPTIMAL)
        self.assertItemsAlmostEqual(x.value, np.array([1.0, 1.0]), places=4)

    def test_symbolic_quad_form_block_indices_raises(self) -> None:
        from cvxpy.reductions.solvers.nlp_solvers.diff_engine.converters import (
            convert_symbolic_quad_form,
        )
        x = cp.Variable(4)
        sqf = SymbolicQuadForm(x, sp.eye_array(4), cp.sum_squares(x),
                               block_indices=[np.array([0, 1]), np.array([2, 3])])
        with self.assertRaisesRegex(NotImplementedError, "block_indices"):
            convert_symbolic_quad_form(sqf, {}, 4, {})

    def test_symbolic_quad_form_unsupported_orig_raises(self) -> None:
        from cvxpy.reductions.solvers.nlp_solvers.diff_engine.converters import (
            convert_symbolic_quad_form,
        )
        x = cp.Variable(4)
        sqf = SymbolicQuadForm(x, sp.eye_array(4), cp.norm1(x))
        with self.assertRaisesRegex(NotImplementedError, "norm1"):
            convert_symbolic_quad_form(sqf, {}, 4, {})

    def test_zero_divisor_raises(self) -> None:
        x = cp.Variable(3)
        divisor = np.array([1.0, 0.0, 2.0])
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x / divisor - 1.0)))
        with self.assertRaisesRegex(ValueError, "[Dd]ivision by zero"):
            prob.solve(solver=SOLVER, canon_backend=DIFFENGINE)

    def test_gt_2d_expression_raises_clearly(self) -> None:
        from cvxpy.reductions.solvers.nlp_solvers.diff_engine.helpers import (
            normalize_shape,
        )
        with self.assertRaisesRegex(NotImplementedError, ">2-D"):
            normalize_shape((2, 3, 4))

    def test_required_bindings_present(self) -> None:
        for name in REQUIRED_BINDINGS:
            self.assertTrue(hasattr(_engine, name))

    def test_kron_var_left_const_right(self) -> None:
        """kron(X, C) exercises make_right_kron (with a structural zero in C)."""
        rng = np.random.default_rng(0)
        C = np.array([[1.0, 2.0], [0.0, 3.0]])
        X0 = rng.standard_normal((2, 2))
        X = cp.Variable((2, 2))
        target = np.kron(X0, C)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(cp.kron(X, C) - target)))
        prob.solve(solver=SOLVER, canon_backend=DIFFENGINE)
        self.assertEqual(prob.status, cp.OPTIMAL)
        self.assertAlmostEqual(prob.value, 0.0)
        self.assertItemsAlmostEqual(X.value, X0, places=3)

    def test_kron_parametric_operand_resolves(self) -> None:
        """A bare-Parameter kron operand stays symbolic and must be
        re-evaluated between solves, for both orientations."""
        rng = np.random.default_rng(0)
        target = rng.standard_normal((4, 4))
        for param_side in ("left", "right"):
            P = cp.Parameter((2, 2))
            X = cp.Variable((2, 2))
            expr = cp.kron(P, X) if param_side == "left" else cp.kron(X, P)
            prob = cp.Problem(cp.Minimize(cp.sum_squares(expr - target)))

            for seed in (1, 2):
                P_val = np.random.default_rng(seed).standard_normal((2, 2))
                P.value = P_val
                # kron(P, X) is not DPP: pass ignore_dpp to opt out of the
                # non-DPP warning while selecting the backend explicitly.
                prob.solve(solver=SOLVER, ignore_dpp=True,
                           canon_backend=DIFFENGINE)
                self.assertEqual(prob.status, cp.OPTIMAL)

                X_base = cp.Variable((2, 2))
                expr_base = (cp.kron(P_val, X_base) if param_side == "left"
                             else cp.kron(X_base, P_val))
                base = cp.Problem(cp.Minimize(cp.sum_squares(expr_base - target)))
                base.solve(solver=SOLVER)
                self.assertAlmostEqual(prob.value, base.value, places=3)
                self.assertItemsAlmostEqual(X.value, X_base.value, places=3)

    def test_diag_offset_both_directions(self) -> None:
        """diag(x, k) and diag(X, k) for off-main diagonals."""
        n = 4
        A = np.arange(n * n, dtype=float).reshape((n, n))
        for k in (-2, -1, 1, 3):
            x = cp.Variable(n - abs(k))
            target = np.diag(np.diag(A, k), k)
            prob = cp.Problem(cp.Minimize(cp.sum_squares(cp.diag(x, k) - target)))
            prob.solve(solver=SOLVER, canon_backend=DIFFENGINE)
            self.assertItemsAlmostEqual(x.value, np.diag(A, k), places=4)

            X = cp.Variable((n, n))
            prob = cp.Problem(cp.Minimize(cp.sum_squares(X)),
                              [cp.diag(X, k) == np.diag(A, k)])
            prob.solve(solver=SOLVER, canon_backend=DIFFENGINE)
            self.assertItemsAlmostEqual(np.diag(X.value, k), np.diag(A, k), places=4)

    def test_matmul_chain_const_tail_matches_default(self) -> None:
        """Matmul-chain reassociation must be value-preserving, not just
        solvable: each case solves on both paths and must match."""
        rng = np.random.default_rng(0)
        A = rng.standard_normal((3, 4))
        C_sq = rng.standard_normal((4, 4))
        C2 = rng.standard_normal((3, 2))
        c_vec = rng.standard_normal(3)
        c_vec4 = rng.standard_normal(4)
        X = cp.Variable((4, 3))
        T_mat = rng.standard_normal((3, 2))
        T_vec = rng.standard_normal(4)

        cases = [
            A @ X @ C2,                    # (A @ X) @ C2: matmul recursion
            (X + C_sq @ X) @ c_vec,        # AddExpression with a vector tail
            (-X) @ c_vec,                  # NegExpression push-through
            X.T @ C_sq @ c_vec4,           # (E @ C) @ c: constants fold together
        ]
        targets = [T_mat, T_vec, T_vec, rng.standard_normal(3)]
        for expr, target in zip(cases, targets):
            objective = cp.Minimize(cp.sum_squares(expr - target) + cp.sum_squares(X))
            prob_de = cp.Problem(objective)
            prob_de.solve(solver=SOLVER, canon_backend=DIFFENGINE)
            self.assertEqual(prob_de.status, cp.OPTIMAL)
            X_de = np.array(X.value)

            prob_base = cp.Problem(objective)
            prob_base.solve(solver=SOLVER)
            self.assertEqual(prob_base.status, cp.OPTIMAL)
            self.assertAlmostEqual(prob_de.value, prob_base.value, places=4)
            self.assertItemsAlmostEqual(X_de, X.value, places=3)

    def test_quad_objective_data_matches_cpp(self) -> None:
        """The extractor's Hessian path must produce the same stuffed
        (P, q, A) as the default CPP backend."""
        P0 = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
        q0 = np.array([1.0, -2.0, 0.5])
        x = cp.Variable(3)
        prob = cp.Problem(cp.Minimize(cp.quad_form(x, P0) + q0 @ x), [x >= -1])

        data_de, _, _ = prob.get_problem_data(cp.OSQP, canon_backend=DIFFENGINE)
        data_cpp, _, _ = prob.get_problem_data(cp.OSQP)
        for key in data_cpp:
            expected = data_cpp[key]
            if sp.issparse(expected):
                self.assertItemsAlmostEqual(
                    data_de[key].toarray(), expected.toarray(), places=10)
            elif isinstance(expected, np.ndarray):
                self.assertItemsAlmostEqual(data_de[key], expected, places=10)


class TestDiffengineConeProgram(BaseTest):
    """DiffengineConeProgram re-solve semantics."""

    def test_apply_parameters_explicit_dict(self) -> None:
        """The id_to_param_value branch must behave exactly like setting
        Parameter.value."""
        p = cp.Parameter(2)
        p.value = np.array([1.0, 2.0])
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x - p)), [x >= 0])

        chain = prob._construct_chain(solver=SOLVER, canon_backend=DIFFENGINE)
        program = prob
        for reduction in chain.reductions[:-1]:
            program, _ = reduction.apply(program)
        self.assertIsInstance(program, DiffengineConeProgram)

        new_val = np.array([5.0, 7.0])
        from_dict = program.apply_parameters(
            id_to_param_value={p.id: new_val}, quad_obj=True)
        p.value = new_val
        from_value = program.apply_parameters(quad_obj=True)
        for got, want in zip(from_dict, from_value):
            if sp.issparse(want):
                self.assertItemsAlmostEqual(got.toarray(), want.toarray(), places=10)
            else:
                self.assertItemsAlmostEqual(got, want, places=10)

    def test_scaled_coefficient_dpp_cached_resolve(self) -> None:
        """(p * A) @ x is DPP-affine, so it rides the cached DPP path on the
        explicit backend; the composite coefficient (promote inside the
        engine's side subtree) must serve fresh values on every re-solve.
        The optimal point (value 0.5/p) moves with p, so staleness cannot
        hide behind a parameter-invariant argmin."""
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        x = cp.Variable(2)
        p = cp.Parameter(nonneg=True)
        prob = cp.Problem(cp.Minimize(cp.sum(x)),
                          [(p * A) @ x >= 1, x >= 0])
        self.assertTrue(prob.is_dpp())
        for val in (1.0, 100.0, 1.0):
            p.value = val
            prob.solve(solver=SOLVER, canon_backend=DIFFENGINE)
            self.assertEqual(prob.status, cp.OPTIMAL)
            self.assertAlmostEqual(prob.value, 0.5 / val, places=4)

    def test_parametric_soc_restruct_resolve_and_duals(self) -> None:
        """The SOC restructuring matrix must be re-applied to freshly
        extracted (A, b) on re-solves; duals must match the default path."""
        c = np.array([1.0, 2.0])
        p = cp.Parameter(2)
        x = cp.Variable(2)
        constraints = [cp.norm(x - p) <= 2.0, x >= -5]
        prob = cp.Problem(cp.Minimize(c @ x), constraints)

        for val in (np.array([1.0, 1.0]), np.array([-2.0, 3.0])):
            p.value = val
            prob.solve(solver=SOLVER, canon_backend=DIFFENGINE)
            self.assertEqual(prob.status, cp.OPTIMAL)

            x_base = cp.Variable(2)
            base_cons = [cp.norm(x_base - val) <= 2.0, x_base >= -5]
            base = cp.Problem(cp.Minimize(c @ x_base), base_cons)
            base.solve(solver=SOLVER)
            self.assertAlmostEqual(prob.value, base.value, places=4)
            self.assertItemsAlmostEqual(x.value, x_base.value, places=4)
            for con_de, con_base in zip(constraints, base_cons):
                self.assertItemsAlmostEqual(
                    con_de.dual_value, con_base.dual_value, places=4)

    def test_parametric_psd_restruct_resolve(self) -> None:
        """PSD constraints exercise the symmetric restructuring across
        parametric re-solves."""
        P = cp.Parameter((2, 2), symmetric=True)
        X = cp.Variable((2, 2), symmetric=True)
        prob = cp.Problem(cp.Minimize(cp.trace(P @ X)),
                          [X >> np.eye(2), cp.trace(X) <= 5])

        for seed in (0, 3):
            rng = np.random.default_rng(seed)
            M = rng.standard_normal((2, 2))
            P_val = M @ M.T + np.eye(2)
            P.value = P_val
            prob.solve(solver=SOLVER, canon_backend=DIFFENGINE)
            self.assertEqual(prob.status, cp.OPTIMAL)

            X_base = cp.Variable((2, 2), symmetric=True)
            base = cp.Problem(cp.Minimize(cp.trace(P_val @ X_base)),
                              [X_base >> np.eye(2), cp.trace(X_base) <= 5])
            base.solve(solver=SOLVER)
            self.assertAlmostEqual(prob.value, base.value, places=3)

    @unittest.skipUnless(INSTALLED_MI_SOLVERS, "no mixed-integer solver installed")
    def test_mixed_integer(self) -> None:
        x = cp.Variable(3, integer=True)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 0.5, x <= 3.7])
        prob.solve(canon_backend=DIFFENGINE)
        self.assertEqual(prob.status, cp.OPTIMAL)
        self.assertItemsAlmostEqual(x.value, np.ones(3), places=4)


class TestDiffengineSelection(BaseTest):
    """canon_backend='DIFFENGINE' is user-selectable, explicitly or via
    CVXPY_DEFAULT_CANON_BACKEND."""

    def _stuffing_backend(self, chain) -> str:
        stuffing = [r for r in chain.reductions
                    if isinstance(r, ConeMatrixStuffing)][0]
        return stuffing.canon_backend

    def test_explicit_diffengine_param_free(self) -> None:
        x = cp.Variable(3)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x - 1)), [x >= 0])
        prob.solve(solver=SOLVER, canon_backend=DIFFENGINE)
        self.assertEqual(self._stuffing_backend(prob._cache.solving_chain),
                         DIFFENGINE)
        self.assertIsInstance(prob._cache.param_prog, DiffengineConeProgram)
        self.assertAlmostEqual(prob.value, 0.0)
        self.assertItemsAlmostEqual(x.value, np.ones(3), places=4)

    def test_explicit_diffengine_dpp_parametric_resolve(self) -> None:
        """On the DPP path the DiffengineConeProgram is cached across solves;
        parameter updates must flow through the cached extractor."""
        A = cp.Parameter((2, 2))
        b = cp.Parameter(2)
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)), [x >= -10])
        self.assertTrue(prob.is_dpp())

        A.value = np.eye(2)
        b.value = np.array([1.0, 2.0])
        prob.solve(solver=SOLVER, canon_backend=DIFFENGINE)
        self.assertIsInstance(prob._cache.param_prog, DiffengineConeProgram)
        cached_prog = prob._cache.param_prog
        self.assertItemsAlmostEqual(x.value, np.array([1.0, 2.0]), places=4)

        A.value = 2.0 * np.eye(2)
        b.value = np.array([2.0, -4.0])
        prob.solve(solver=SOLVER, canon_backend=DIFFENGINE)
        self.assertIs(prob._cache.param_prog, cached_prog)  # cached prog reused
        self.assertItemsAlmostEqual(x.value, np.array([1.0, -2.0]), places=4)

        # Solve again with the SAME values: the cached program must keep
        # serving the current parameter values.
        prob.solve(solver=SOLVER, canon_backend=DIFFENGINE)
        self.assertItemsAlmostEqual(x.value, np.array([1.0, -2.0]), places=4)

    def test_env_var_selects_diffengine_on_dpp_path(self) -> None:
        """CVXPY_DEFAULT_CANON_BACKEND=DIFFENGINE routes DPP solves through
        the diff engine, with program caching."""
        with mock.patch.dict(
                os.environ,
                {"CVXPY_DEFAULT_CANON_BACKEND": DIFFENGINE}):
            p = cp.Parameter(2)
            x = cp.Variable(2)
            prob = cp.Problem(cp.Minimize(cp.sum_squares(x - p)))
            self.assertTrue(prob.is_dpp())

            p.value = np.array([1.0, 2.0])
            prob.solve(solver=SOLVER)
            self.assertEqual(self._stuffing_backend(prob._cache.solving_chain),
                             DIFFENGINE)
            self.assertIsInstance(prob._cache.param_prog, DiffengineConeProgram)
            self.assertItemsAlmostEqual(x.value, p.value, places=4)

            p.value = np.array([-1.0, 3.0])
            prob.solve(solver=SOLVER)
            self.assertItemsAlmostEqual(x.value, p.value, places=4)

            # An explicit user backend still wins over the env default.
            prob2 = cp.Problem(cp.Minimize(cp.sum_squares(x - 1)))
            _, chain, _ = prob2.get_problem_data(
                SOLVER, canon_backend=s.SCIPY_CANON_BACKEND)
            self.assertNotEqual(self._stuffing_backend(chain), DIFFENGINE)

    def test_nd_problem_explicit_diffengine_raises(self) -> None:
        x = cp.Variable((2, 2, 2))
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x)))
        with self.assertRaisesRegex(ValueError, "dimension greater than 2"):
            prob.solve(solver=SOLVER, canon_backend=DIFFENGINE)
