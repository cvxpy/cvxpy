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

import unittest

import numpy as np
import pytest
import scipy.sparse as sp

import cvxpy as cp
import cvxpy.settings as s
from cvxpy.atoms.quad_form import SymbolicQuadForm
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import (
    ConeMatrixStuffing,
    ParamConeProg,
)
from cvxpy.reductions.solvers.defines import INSTALLED_MI_SOLVERS
from cvxpy.reductions.solvers.nlp_solvers.diff_engine.converters import (
    convert_expr,
    convert_symbolic_quad_form,
)
from cvxpy.reductions.solvers.nlp_solvers.diff_engine.helpers import normalize_shape
from cvxpy.reductions.solvers.nlp_solvers.diff_engine.registry import convert_kron
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
    reason="DIFFENGINE backend requires sparsediffpy >= 0.6.1 "
           f"(missing: {', '.join(MISSING)})",
)

SOLVER = cp.CLARABEL
DIFFENGINE = s.DIFFENGINE_CANON_BACKEND


class TestDiffengineConverter(BaseTest):
    """Converter correctness and fail-loud behavior: unsupported constructs
    must raise immediately instead of silently miscompiling."""

    def test_unsupported_atom_raises(self) -> None:
        """convert_expr names the offending atom."""
        x = cp.Variable(4)
        with self.assertRaisesRegex(NotImplementedError, "cumsum"):
            convert_expr(cp.cumsum(x), {x.id: None}, 4, {})

    def test_symbolic_quad_form_block_indices_raises(self) -> None:
        x = cp.Variable(4)
        sqf = SymbolicQuadForm(x, sp.eye_array(4), cp.sum_squares(x),
                               block_indices=[np.array([0, 1]), np.array([2, 3])])
        with self.assertRaisesRegex(NotImplementedError, "block_indices"):
            convert_symbolic_quad_form(sqf, {}, 4, {})

    def test_symbolic_quad_form_unsupported_orig_raises(self) -> None:
        x = cp.Variable(4)
        sqf = SymbolicQuadForm(x, sp.eye_array(4), cp.norm1(x))
        with self.assertRaisesRegex(NotImplementedError, "norm1"):
            convert_symbolic_quad_form(sqf, {}, 4, {})

    def test_kron_parametric_operand_raises(self) -> None:
        """kron's variable-free operand must be a plain constant."""
        P = cp.Parameter((2, 2))
        P.value = np.eye(2)
        X = cp.Variable((2, 2))
        with self.assertRaisesRegex(NotImplementedError, "parametric operand"):
            convert_kron(cp.kron(P, X), [None, None])

    def test_gt_2d_expression_raises_clearly(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, ">2-D"):
            normalize_shape((2, 3, 4))


class TestDiffengineBackend(BaseTest):
    """End-to-end behavior of canon_backend='DIFFENGINE': same stuffed data
    and solutions as the default tensor backends, for parameter-free
    problems; loud rejection otherwise."""

    def test_constant_nonlinear_subtree_is_evaluated(self) -> None:
        """A nonlinear atom over plain constants is evaluated numerically;
        the engine cannot differentiate through it."""
        x = cp.Variable(2)
        const_term = cp.quad_over_lin(np.array([1.0, 1.0]), 1.0)  # == 2.0
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), [cp.sum(x) >= const_term])
        prob.solve(solver=SOLVER, canon_backend=DIFFENGINE)
        self.assertEqual(prob.status, cp.OPTIMAL)
        self.assertItemsAlmostEqual(x.value, np.array([1.0, 1.0]), places=4)

    def test_zero_divisor_raises(self) -> None:
        x = cp.Variable(3)
        divisor = np.array([1.0, 0.0, 2.0])
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x / divisor - 1.0)))
        with self.assertRaisesRegex(ValueError, "[Dd]ivision by zero"):
            prob.solve(solver=SOLVER, canon_backend=DIFFENGINE)

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

    def test_matmul_chain_1d_collapse_not_reassociated(self) -> None:
        """A chain that contracts through a 1-D operand must be converted as
        written: (a @ b) @ c != a @ (b @ c) across numpy's 1-D collapse."""
        rng = np.random.default_rng(0)
        X = cp.Variable((2, 3))
        v = np.array([1.0, 2.0, 3.0])
        w = np.array([1.0, 1.0])
        prob = cp.Problem(cp.Minimize(cp.sum_squares(X)), [(X @ v) @ w >= 1])
        prob.solve(solver=SOLVER, canon_backend=DIFFENGINE)
        self.assertEqual(prob.status, cp.OPTIMAL)
        ref = cp.Problem(cp.Minimize(cp.sum_squares(X)), [(X @ v) @ w >= 1])
        ref.solve(solver=SOLVER)
        self.assertAlmostEqual(prob.value, ref.value, places=5)

        A = rng.standard_normal((3, 4))
        y = cp.Variable(4)
        c = rng.standard_normal(3)
        prob2 = cp.Problem(cp.Minimize((A @ y) @ c + cp.sum_squares(y)))
        prob2.solve(solver=SOLVER, canon_backend=DIFFENGINE)
        self.assertEqual(prob2.status, cp.OPTIMAL)
        ref2 = cp.Problem(cp.Minimize((A @ y) @ c + cp.sum_squares(y)))
        ref2.solve(solver=SOLVER)
        self.assertAlmostEqual(prob2.value, ref2.value, places=5)

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

    def test_lp_and_soc_data_matches_cpp(self) -> None:
        """LP and SOC stuffed data must match the default backend through the
        single-column tensor encode."""
        rng = np.random.default_rng(0)
        A0 = rng.standard_normal((4, 6))
        c0 = rng.standard_normal(6)
        x = cp.Variable(6)
        prob = cp.Problem(cp.Minimize(c0 @ x),
                          [A0 @ x <= 1, cp.norm2(x - 0.5) <= 2, cp.sum(x) == 0.5])
        data_de, _, _ = prob.get_problem_data(SOLVER, canon_backend=DIFFENGINE)
        data_cpp, _, _ = prob.get_problem_data(SOLVER)
        for key in data_cpp:
            expected = data_cpp[key]
            if sp.issparse(expected):
                self.assertItemsAlmostEqual(
                    data_de[key].toarray(), expected.toarray(), places=10)
            elif isinstance(expected, np.ndarray):
                self.assertItemsAlmostEqual(data_de[key], expected, places=10)

    @staticmethod
    def _quad_form_problem(P0):
        x = cp.Variable(P0.shape[0])
        return cp.Problem(cp.Minimize(cp.quad_form(x, P0)), [cp.sum(x) == 1])

    def test_constant_P_density_routing(self) -> None:
        """Constant quad_form P is routed by content, not container: a
        dense-stored but mostly-zero P (H * np.eye(n)) yields a stuffed P with
        only the true pattern, a genuinely dense P keeps its full pattern (and
        matches CPP), and explicitly stored zeros in a sparse P are dropped."""
        # Dense-stored diagonal, density 1/n < SPARSE_DENSITY_THRESHOLD.
        n = 30
        data, _, _ = self._quad_form_problem(2.5 * np.eye(n)).get_problem_data(
            SOLVER, canon_backend=DIFFENGINE)
        self.assertEqual(data["P"].nnz, n)

        # Genuinely dense P: full pattern intact, values match CPP.
        m = 10
        rng = np.random.default_rng(0)
        A = rng.standard_normal((m, m))
        P0 = A @ A.T + m * np.eye(m)
        data, _, _ = self._quad_form_problem(P0).get_problem_data(
            SOLVER, canon_backend=DIFFENGINE)
        data_cpp, _, _ = self._quad_form_problem(P0).get_problem_data(SOLVER)
        self.assertEqual(data["P"].nnz, data_cpp["P"].nnz)
        self.assertItemsAlmostEqual(
            data["P"].toarray(), data_cpp["P"].toarray(), places=10)

        # Sparse P with explicitly stored zeros (scipy block_diag of dense
        # blocks stores them): the stuffed pattern is not inflated.
        blocks = [np.diag([1.0, 2.0]) for _ in range(5)]  # 2 stored zeros each
        P1 = sp.block_diag(blocks, format="csr")
        self.assertEqual(P1.nnz, 20)
        data, _, _ = self._quad_form_problem(P1).get_problem_data(
            SOLVER, canon_backend=DIFFENGINE)
        self.assertEqual(data["P"].nnz, 10)

    def test_soc_solve_matches_default_with_duals(self) -> None:
        """SOC constraints exercise the cone restructuring at format time;
        primal, value, and duals must match the default path."""
        c = np.array([1.0, 2.0])
        center = np.array([1.0, 1.0])
        x = cp.Variable(2)
        constraints = [cp.norm(x - center) <= 2.0, x >= -5]
        prob = cp.Problem(cp.Minimize(c @ x), constraints)
        prob.solve(solver=SOLVER, canon_backend=DIFFENGINE)
        self.assertEqual(prob.status, cp.OPTIMAL)

        x_base = cp.Variable(2)
        base_cons = [cp.norm(x_base - center) <= 2.0, x_base >= -5]
        base = cp.Problem(cp.Minimize(c @ x_base), base_cons)
        base.solve(solver=SOLVER)
        self.assertAlmostEqual(prob.value, base.value, places=4)
        self.assertItemsAlmostEqual(x.value, x_base.value, places=4)
        for con_de, con_base in zip(constraints, base_cons):
            self.assertItemsAlmostEqual(
                con_de.dual_value, con_base.dual_value, places=4)

    def test_psd_solve_matches_default(self) -> None:
        """PSD constraints exercise the symmetric restructuring."""
        rng = np.random.default_rng(0)
        M = rng.standard_normal((2, 2))
        P_val = M @ M.T + np.eye(2)
        X = cp.Variable((2, 2), symmetric=True)
        prob = cp.Problem(cp.Minimize(cp.trace(P_val @ X)),
                          [X >> np.eye(2), cp.trace(X) <= 5])
        prob.solve(solver=SOLVER, canon_backend=DIFFENGINE)
        self.assertEqual(prob.status, cp.OPTIMAL)

        X_base = cp.Variable((2, 2), symmetric=True)
        base = cp.Problem(cp.Minimize(cp.trace(P_val @ X_base)),
                          [X_base >> np.eye(2), cp.trace(X_base) <= 5])
        base.solve(solver=SOLVER)
        self.assertAlmostEqual(prob.value, base.value, places=3)

    def test_parametric_problem_rejected(self) -> None:
        """Explicit DIFFENGINE on a DPP-parametric problem must raise and
        point the user at ignore_dpp."""
        p = cp.Parameter()
        p.value = 1.0
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.square(x - p)))
        with self.assertRaisesRegex(ValueError, "ignore_dpp"):
            prob.get_problem_data(SOLVER, canon_backend=DIFFENGINE)

    def test_ignore_dpp_parametric_solves_and_tracks_values(self) -> None:
        """With ignore_dpp=True, EvalParams bakes the parameters and the
        backend compiles the resulting parameter-free problem each solve."""
        p = cp.Parameter()
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.square(x - p)))
        for val in (1.0, -3.0):
            p.value = val
            prob.solve(solver=SOLVER, canon_backend=DIFFENGINE, ignore_dpp=True)
            self.assertEqual(prob.status, cp.OPTIMAL)
            self.assertAlmostEqual(x.value, val, places=5)

    def test_parametric_bounds_rejected_both_routes(self) -> None:
        """EvalParams does not bake variable bounds, so they leak past
        ignore_dpp by two routes, both of which must fail loudly naming the
        cause: as bounds attributes (bounds-capable solver), or lowered by
        CvxAttr2Constr into constraints that still carry live Parameters
        (non-bounds solver)."""
        lb = cp.Parameter(2)
        lb.value = np.array([0.5, 0.5])
        x = cp.Variable(2, bounds=[lb, 10])
        prob = cp.Problem(cp.Minimize(cp.sum(x)))
        with self.assertRaisesRegex(NotImplementedError, "parametric variable bounds"):
            prob.get_problem_data(cp.SCIPY, canon_backend=DIFFENGINE,
                                  ignore_dpp=True)
        with self.assertRaisesRegex(ValueError, "parametric variable bounds"):
            prob.get_problem_data(SOLVER, canon_backend=DIFFENGINE,
                                  ignore_dpp=True)

    @unittest.skipUnless(INSTALLED_MI_SOLVERS, "no mixed-integer solver installed")
    def test_mixed_integer(self) -> None:
        x = cp.Variable(3, integer=True)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 0.5, x <= 3.7])
        prob.solve(canon_backend=DIFFENGINE)
        self.assertEqual(prob.status, cp.OPTIMAL)
        self.assertItemsAlmostEqual(x.value, np.ones(3), places=4)


class TestDiffengineSelection(BaseTest):
    """canon_backend='DIFFENGINE' is explicit opt-in and produces a stock
    parameter-free ParamConeProg."""

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
        self.assertIsInstance(prob._cache.param_prog, ParamConeProg)
        self.assertEqual(prob._cache.param_prog.parameters, [])
        self.assertAlmostEqual(prob.value, 0.0)
        self.assertItemsAlmostEqual(x.value, np.ones(3), places=4)

    def test_nd_problem_explicit_diffengine_raises(self) -> None:
        x = cp.Variable((2, 2, 2))
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x)))
        with self.assertRaisesRegex(ValueError, "dimension greater than 2"):
            prob.get_problem_data(SOLVER, canon_backend=DIFFENGINE)
