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

Behavioral tests for the DIFFENGINE canon backend (the ``ignore_dpp`` /
non-DPP path and explicit ``canon_backend="DIFFENGINE"`` selection):
fail-loud behavior on unsupported constructs, converter correctness for the
branches only exercised end-to-end elsewhere (parametric divisors, kron
orientations, matmul-chain reassociation, quadratic-objective extraction),
and ``DiffengineConeProgram`` re-solve semantics (explicit parameter dicts,
cone restructuring, mixed-integer problems).
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

import cvxpy as cp
import cvxpy.settings as s
from cvxpy.atoms.quad_form import SymbolicQuadForm
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.reductions.dcp2cone.diffengine_cone_program import DiffengineConeProgram
from cvxpy.reductions.solvers.defines import INSTALLED_MI_SOLVERS

try:
    from sparsediffpy import _sparsediffengine as _engine
except ImportError:  # pragma: no cover
    _engine = None

# The DIFFENGINE code in this branch needs bindings introduced in
# sparsediffpy 0.6.0; on an older wheel every test here would AttributeError
# mid-solve, so skip the module loudly instead.
_REQUIRED_BINDINGS = ("make_left_kron", "make_right_kron", "make_power", "make_quad_form")
_MISSING = (["sparsediffpy not installed"] if _engine is None else
            [name for name in _REQUIRED_BINDINGS if not hasattr(_engine, name)])
pytestmark = pytest.mark.skipif(
    bool(_MISSING),
    reason="DIFFENGINE backend requires sparsediffpy >= 0.6.0 "
           f"(missing: {', '.join(_MISSING)})",
)

SOLVER = cp.CLARABEL


def _stuffing_backend(chain):
    return next(r.canon_backend for r in chain.reductions
                if isinstance(r, ConeMatrixStuffing))


def _stuffed_program(prob, **chain_kwargs):
    """Apply the chain up to (and including) ConeMatrixStuffing."""
    chain = prob._construct_chain(solver=SOLVER, **chain_kwargs)
    program = prob
    for reduction in chain.reductions[:-1]:
        program, _ = reduction.apply(program)
    return program


def _solve_both_paths(objective, constraints=()):
    """Solve the same problem on the DIFFENGINE and default paths.

    Fresh Problem objects share the same variable objects, so after each solve
    the variables hold that path's values; returns the two optimal values and
    the variable values from each path.
    """
    prob_de = cp.Problem(objective, list(constraints))
    prob_de.solve(solver=SOLVER, ignore_dpp=True)
    assert prob_de.status == cp.OPTIMAL
    de_vars = [np.array(v.value) for v in prob_de.variables()]

    prob_base = cp.Problem(objective, list(constraints))
    prob_base.solve(solver=SOLVER)
    assert prob_base.status == cp.OPTIMAL
    base_vars = [np.array(v.value) for v in prob_base.variables()]
    return prob_de.value, prob_base.value, de_vars, base_vars


class TestFailLoud:
    """The diffengine path has no baking fallback: unsupported constructs
    must raise immediately instead of silently miscompiling."""

    def test_unsupported_atom_raises(self):
        """convert_expr names the offending atom. (End to end, Dcp2Cone
        canonicalizes exotic atoms into supported primitives before the
        converter sees them, so this is pinned at the converter level.)"""
        from cvxpy.reductions.solvers.nlp_solvers.diff_engine.converters import (
            convert_expr,
        )
        expr = cp.cumsum(cp.Constant(np.arange(4.0)))
        with pytest.raises(NotImplementedError, match="cumsum"):
            convert_expr(expr, {}, 0, {})

    def test_symbolic_quad_form_block_indices_raises(self):
        from cvxpy.reductions.solvers.nlp_solvers.diff_engine.converters import (
            convert_symbolic_quad_form,
        )
        x = cp.Variable(4)
        sqf = SymbolicQuadForm(x, sp.eye_array(4), cp.sum_squares(x),
                               block_indices=[np.array([0, 1]), np.array([2, 3])])
        with pytest.raises(NotImplementedError, match="block_indices"):
            convert_symbolic_quad_form(sqf, {}, 4, {})

    def test_symbolic_quad_form_unsupported_orig_raises(self):
        from cvxpy.reductions.solvers.nlp_solvers.diff_engine.converters import (
            convert_symbolic_quad_form,
        )
        x = cp.Variable(4)
        sqf = SymbolicQuadForm(x, sp.eye_array(4), cp.norm1(x))
        with pytest.raises(NotImplementedError, match="norm1"):
            convert_symbolic_quad_form(sqf, {}, 4, {})

    def test_zero_divisor_raises(self):
        x = cp.Variable(3)
        divisor = np.array([1.0, 0.0, 2.0])
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x / divisor - 1.0)))
        with pytest.raises(ValueError, match="[Dd]ivision by zero"):
            prob.solve(solver=SOLVER, ignore_dpp=True)

    def test_gt_2d_expression_raises_clearly(self):
        from cvxpy.reductions.solvers.nlp_solvers.diff_engine.helpers import (
            normalize_shape,
        )
        with pytest.raises(NotImplementedError, match=">2-D"):
            normalize_shape((2, 3, 4))

    def test_required_bindings_present(self):
        """Doubles as the loud signal that the installed sparsediffpy build
        provides the 0.6.0 bindings this branch depends on."""
        for name in _REQUIRED_BINDINGS:
            assert hasattr(_engine, name)


class TestConverterCoverage:
    """Converter branches that only pass incidentally in end-to-end suites."""

    def test_parametric_scalar_divisor_resolves(self):
        p = cp.Parameter(pos=True)
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.square(x / p - 1.0)))
        for val in (2.0, 5.0):
            p.value = val
            prob.solve(solver=SOLVER, ignore_dpp=True)
            assert prob.status == cp.OPTIMAL
            np.testing.assert_allclose(x.value, val, rtol=1e-5)

    def test_parametric_vector_divisor_resolves(self):
        p = cp.Parameter(3, pos=True)
        x = cp.Variable(3)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x / p - 1.0)))
        for val in (np.array([1.0, 2.0, 4.0]), np.array([3.0, 0.5, 1.5])):
            p.value = val
            prob.solve(solver=SOLVER, ignore_dpp=True)
            assert prob.status == cp.OPTIMAL
            np.testing.assert_allclose(x.value, val, rtol=1e-5)

    def test_kron_var_left_const_right(self):
        """kron(X, C) exercises make_right_kron; existing tests only cover
        the constant-left orientation."""
        rng = np.random.default_rng(0)
        C = np.array([[1.0, 2.0], [0.0, 3.0]])  # a structural zero, too
        X0 = rng.standard_normal((2, 2))
        X = cp.Variable((2, 2))
        target = np.kron(X0, C)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(cp.kron(X, C) - target)))
        prob.solve(solver=SOLVER, ignore_dpp=True)
        assert prob.status == cp.OPTIMAL
        np.testing.assert_allclose(prob.value, 0.0, atol=1e-6)
        np.testing.assert_allclose(X.value, X0, atol=1e-4)

    @pytest.mark.parametrize("param_side", ["left", "right"])
    def test_kron_parametric_operand_resolves(self, param_side):
        """A bare-Parameter kron operand stays symbolic (EvalParams does not
        fold it) and takes the all-blocks-active branch; the engine must
        re-evaluate it between solves."""
        rng = np.random.default_rng(0)
        P = cp.Parameter((2, 2))
        X = cp.Variable((2, 2))
        expr = cp.kron(P, X) if param_side == "left" else cp.kron(X, P)
        target = rng.standard_normal((4, 4))
        prob = cp.Problem(cp.Minimize(cp.sum_squares(expr - target)))

        for seed in (1, 2):
            P_val = np.random.default_rng(seed).standard_normal((2, 2))
            P.value = P_val
            prob.solve(solver=SOLVER, ignore_dpp=True)
            assert prob.status == cp.OPTIMAL

            X_base = cp.Variable((2, 2))
            expr_base = (cp.kron(P_val, X_base) if param_side == "left"
                         else cp.kron(X_base, P_val))
            base = cp.Problem(cp.Minimize(cp.sum_squares(expr_base - target)))
            base.solve(solver=SOLVER)
            np.testing.assert_allclose(prob.value, base.value, rtol=1e-4)
            np.testing.assert_allclose(X.value, X_base.value, atol=1e-4)

    def test_matmul_chain_const_tail_matches_default(self):
        """_normalize_matmul rewrites these trees (constant tail pushed toward
        the leaves); the rewrite must be value-preserving, not just solvable."""
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
            v_de, v_base, de_vars, base_vars = _solve_both_paths(objective)
            np.testing.assert_allclose(v_de, v_base, rtol=1e-5, atol=1e-7)
            for got, want in zip(de_vars, base_vars):
                np.testing.assert_allclose(got, want, atol=1e-4)

    def test_quad_objective_data_matches_cpp(self):
        """The extractor's Hessian path (lower-tri -> full symmetric mirror)
        must produce the same stuffed (P, q, A) as the default CPP backend,
        including off-diagonal coupling."""
        P0 = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
        q0 = np.array([1.0, -2.0, 0.5])
        x = cp.Variable(3)
        prob = cp.Problem(cp.Minimize(cp.quad_form(x, P0) + q0 @ x), [x >= -1])

        data_de, _, _ = prob.get_problem_data(cp.OSQP, ignore_dpp=True)
        data_cpp, _, _ = prob.get_problem_data(cp.OSQP)
        for key in data_cpp:
            expected = data_cpp[key]
            if sp.issparse(expected):
                np.testing.assert_allclose(
                    data_de[key].toarray(), expected.toarray(), atol=1e-12,
                    err_msg=f"mismatch in stuffed '{key}'")
            elif isinstance(expected, np.ndarray):
                np.testing.assert_allclose(
                    data_de[key], expected, atol=1e-12,
                    err_msg=f"mismatch in stuffed '{key}'")


class TestDiffengineConeProgramBehavior:

    def test_apply_parameters_explicit_dict(self):
        """The id_to_param_value branch (used by problem-data machinery) must
        behave exactly like setting Parameter.value."""
        p = cp.Parameter(2)
        p.value = np.array([1.0, 2.0])
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x - p)), [x >= 0])
        program = _stuffed_program(prob, ignore_dpp=True)
        assert isinstance(program, DiffengineConeProgram)

        new_val = np.array([5.0, 7.0])
        from_dict = program.apply_parameters(
            id_to_param_value={p.id: new_val}, quad_obj=True)
        p.value = new_val
        from_value = program.apply_parameters(quad_obj=True)
        for got, want in zip(from_dict, from_value):
            if sp.issparse(want):
                np.testing.assert_allclose(got.toarray(), want.toarray(), atol=1e-12)
            else:
                np.testing.assert_allclose(got, want, atol=1e-12)

    def test_parametric_soc_restruct_resolve_and_duals(self):
        """SOC constraints trigger a non-trivial restruct matrix; it must be
        re-applied to the freshly extracted (A, b) on parametric re-solves,
        and duals must match the default path."""
        c = np.array([1.0, 2.0])
        p = cp.Parameter(2)
        x = cp.Variable(2)
        constraints = [cp.norm(x - p) <= 2.0, x >= -5]
        prob = cp.Problem(cp.Minimize(c @ x), constraints)

        for val in (np.array([1.0, 1.0]), np.array([-2.0, 3.0])):
            p.value = val
            prob.solve(solver=SOLVER, ignore_dpp=True)
            assert prob.status == cp.OPTIMAL

            x_base = cp.Variable(2)
            base_cons = [cp.norm(x_base - val) <= 2.0, x_base >= -5]
            base = cp.Problem(cp.Minimize(c @ x_base), base_cons)
            base.solve(solver=SOLVER)
            np.testing.assert_allclose(prob.value, base.value, rtol=1e-5)
            np.testing.assert_allclose(x.value, x_base.value, atol=1e-5)
            for con_de, con_base in zip(constraints, base_cons):
                np.testing.assert_allclose(
                    con_de.dual_value, con_base.dual_value, atol=1e-5)

    def test_parametric_psd_restruct_resolve(self):
        """PSD constraints exercise the symmetric restructuring; re-solves with
        a new parametric objective must match a baked baseline."""
        P = cp.Parameter((2, 2), symmetric=True)
        X = cp.Variable((2, 2), symmetric=True)
        prob = cp.Problem(cp.Minimize(cp.trace(P @ X)),
                          [X >> np.eye(2), cp.trace(X) <= 5])

        for seed in (0, 3):
            rng = np.random.default_rng(seed)
            M = rng.standard_normal((2, 2))
            P_val = M @ M.T + np.eye(2)
            P.value = P_val
            prob.solve(solver=SOLVER, ignore_dpp=True)
            assert prob.status == cp.OPTIMAL

            X_base = cp.Variable((2, 2), symmetric=True)
            base = cp.Problem(cp.Minimize(cp.trace(P_val @ X_base)),
                              [X_base >> np.eye(2), cp.trace(X_base) <= 5])
            base.solve(solver=SOLVER)
            np.testing.assert_allclose(prob.value, base.value, rtol=1e-4)

    @pytest.mark.skipif(len(INSTALLED_MI_SOLVERS) == 0,
                        reason="no mixed-integer solver installed")
    def test_mixed_integer_ignore_dpp(self):
        """is_mixed_integer / extract_mip_idx on the DiffengineConeProgram."""
        x = cp.Variable(3, integer=True)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 0.5, x <= 3.7])
        prob.solve(ignore_dpp=True)
        assert prob.status == cp.OPTIMAL
        np.testing.assert_allclose(x.value, np.ones(3), atol=1e-6)


class TestExplicitSelectionAndPrecedence:
    """canon_backend='DIFFENGINE' is user-selectable; the ignore_dpp / non-DPP
    path force-selects it over any user-supplied backend."""

    def test_explicit_diffengine_param_free(self):
        x = cp.Variable(3)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x - 1)), [x >= 0])
        prob.solve(solver=SOLVER, canon_backend=s.DIFFENGINE_CANON_BACKEND)
        assert _stuffing_backend(prob._cache.solving_chain) == s.DIFFENGINE_CANON_BACKEND
        assert isinstance(prob._cache.param_prog, DiffengineConeProgram)
        np.testing.assert_allclose(prob.value, 0.0, atol=1e-6)
        np.testing.assert_allclose(x.value, np.ones(3), atol=1e-5)

    def test_explicit_diffengine_dpp_parametric_resolve(self):
        """On the DPP path there is no EvalParams, so the chain (and the
        DiffengineConeProgram) is cached across solves; parameter updates must
        flow through the cached extractor."""
        A = cp.Parameter((2, 2))
        b = cp.Parameter(2)
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)), [x >= -10])
        assert prob.is_dpp()

        A.value = np.eye(2)
        b.value = np.array([1.0, 2.0])
        prob.solve(solver=SOLVER, canon_backend=s.DIFFENGINE_CANON_BACKEND)
        assert isinstance(prob._cache.param_prog, DiffengineConeProgram)
        cached_prog = prob._cache.param_prog
        np.testing.assert_allclose(x.value, np.array([1.0, 2.0]), atol=1e-5)

        A.value = 2.0 * np.eye(2)
        b.value = np.array([2.0, -4.0])
        prob.solve(solver=SOLVER, canon_backend=s.DIFFENGINE_CANON_BACKEND)
        assert prob._cache.param_prog is cached_prog  # cached program reused
        np.testing.assert_allclose(x.value, np.array([1.0, -2.0]), atol=1e-5)

    def test_ignore_dpp_overrides_user_backend(self):
        p = cp.Parameter()
        p.value = 2.0
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.square(x - p)))
        _, chain, _ = prob.get_problem_data(SOLVER, ignore_dpp=True,
                                            canon_backend=s.CPP_CANON_BACKEND)
        assert _stuffing_backend(chain) == s.DIFFENGINE_CANON_BACKEND
        prob.solve(solver=SOLVER, ignore_dpp=True,
                   canon_backend=s.CPP_CANON_BACKEND)
        np.testing.assert_allclose(x.value, 2.0, atol=1e-6)

    def test_env_var_does_not_override_ignore_dpp(self, monkeypatch):
        monkeypatch.setenv("CVXPY_DEFAULT_CANON_BACKEND", s.SCIPY_CANON_BACKEND)
        p = cp.Parameter()
        p.value = 3.0
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.square(x - p)))
        _, chain, _ = prob.get_problem_data(SOLVER, ignore_dpp=True)
        assert _stuffing_backend(chain) == s.DIFFENGINE_CANON_BACKEND
        prob.solve(solver=SOLVER, ignore_dpp=True)
        np.testing.assert_allclose(x.value, 3.0, atol=1e-6)
