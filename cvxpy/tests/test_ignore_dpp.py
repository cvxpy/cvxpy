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
from unittest import mock

import numpy as np
import pytest
import scipy.sparse as sp

import cvxpy as cp
import cvxpy.settings as s
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import (
    ConeMatrixStuffing,
    ParamConeProg,
)
from cvxpy.reductions.eval_params import EvalParams
from cvxpy.reductions.fold_callback_params import CallbackParamFold
from cvxpy.reductions.solvers.defines import INSTALLED_MI_SOLVERS
from cvxpy.reductions.solvers.nlp_solvers.diff_engine.parametric_program import (
    DiffengineParamConeProg,
)
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests.test_diffengine_backend import MISSING

pytestmark = pytest.mark.skipif(
    bool(MISSING),
    reason="DIFFENGINE backend requires sparsediffpy >= 0.6.1 "
           f"(missing: {', '.join(MISSING)})",
)

SOLVER = cp.CLARABEL
DIFFENGINE = s.DIFFENGINE_CANON_BACKEND


def _stuffing_backend(chain) -> str | None:
    stuffing = [r for r in chain.reductions
                if isinstance(r, ConeMatrixStuffing)][0]
    return stuffing.canon_backend


class TestIgnoreDppSelection(BaseTest):
    """The ignore_dpp / non-DPP branch defaults to the DIFFENGINE backend when
    the stuffed problem is parameter-free, and falls back silently otherwise."""

    def test_ignore_dpp_defaults_to_diffengine(self) -> None:
        x = cp.Variable(3)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x - 1)), [x >= 0])
        prob.solve(solver=SOLVER, ignore_dpp=True)
        self.assertEqual(_stuffing_backend(prob._cache.solving_chain), DIFFENGINE)
        # Parameter-free programs are cacheable (nothing to re-evaluate).
        self.assertIsInstance(prob._cache.param_prog, ParamConeProg)
        self.assertItemsAlmostEqual(x.value, np.ones(3), places=4)

    def test_parametric_ignore_dpp_stays_symbolic(self) -> None:
        """Parametric problems keep their parameters symbolic: the chain
        folds non-affine parametric constants (no EvalParams), and values
        track the parameter across re-solves."""
        p = cp.Parameter()
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.square(x - p * p)))  # not DPP
        for val in (1.0, -3.0):
            p.value = val
            prob.solve(solver=SOLVER, ignore_dpp=True)
            self.assertEqual(_stuffing_backend(prob._cache.solving_chain),
                             DIFFENGINE)
            chain_types = [type(r) for r in prob._cache.solving_chain.reductions]
            self.assertIn(CallbackParamFold, chain_types)
            self.assertNotIn(EvalParams, chain_types)
            self.assertAlmostEqual(x.value, val * val, places=5)

    def test_non_dpp_defaults_to_diffengine(self) -> None:
        """The non-DPP branch (without an explicit ignore_dpp) selects the
        backend the same way."""
        p = cp.Parameter()
        p.value = 2.0
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.square(x - p * p)))
        with pytest.warns(UserWarning, match="not DPP"):
            prob.solve(solver=SOLVER)
        self.assertEqual(_stuffing_backend(prob._cache.solving_chain), DIFFENGINE)
        self.assertAlmostEqual(x.value, 4.0, places=5)

    def test_symbolic_data_matches_dpp_tensor_path(self) -> None:
        """Stuffed data on the symbolic DIFFENGINE route equals the DPP
        tensor path at the same parameter values."""
        p = cp.Parameter(2)
        p.value = np.array([1.0, -2.0])
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x - p)),
                          [cp.norm2(x) <= 3, cp.sum(x) == 0.5])
        self.assertTrue(prob.is_dpp())
        data_de, _, _ = prob.get_problem_data(SOLVER, ignore_dpp=True)
        data_cpp, _, _ = prob.get_problem_data(
            SOLVER, canon_backend=s.CPP_CANON_BACKEND)
        for key in data_cpp:
            expected = data_cpp[key]
            if sp.issparse(expected):
                self.assertItemsAlmostEqual(
                    data_de[key].toarray(), expected.toarray(), places=10)
            elif isinstance(expected, np.ndarray):
                self.assertItemsAlmostEqual(data_de[key], expected, places=10)

    def test_explicit_tensor_backend_with_params_raises(self) -> None:
        """An explicit tensor backend cannot serve the symbolic parametric
        path."""
        p = cp.Parameter()
        p.value = 2.0
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.square(x - p * p)))
        with self.assertRaisesRegex(ValueError, "keeps parameters symbolic"):
            prob.get_problem_data(SOLVER, ignore_dpp=True,
                                  canon_backend=s.CPP_CANON_BACKEND)

    def test_nd_problem_falls_back(self) -> None:
        """N-D expressions fall back to the tensor backends silently."""
        x = cp.Variable((2, 2, 2))
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x)))
        prob.solve(solver=SOLVER, ignore_dpp=True)
        self.assertNotEqual(_stuffing_backend(prob._cache.solving_chain),
                            DIFFENGINE)
        self.assertAlmostEqual(prob.value, 0.0, places=6)

    def test_parametric_bounds_fall_back(self) -> None:
        """Parametric variable bounds survive EvalParams, so default selection
        must fall back to the tensor backends instead of tripping the
        backend's guards."""
        lb = cp.Parameter(2)
        lb.value = np.array([0.5, 0.25])
        x = cp.Variable(2, bounds=[lb, 10])
        prob = cp.Problem(cp.Minimize(cp.sum(x)))
        prob.solve(solver=SOLVER, ignore_dpp=True)
        self.assertNotEqual(_stuffing_backend(prob._cache.solving_chain),
                            DIFFENGINE)
        self.assertItemsAlmostEqual(x.value, lb.value, places=4)

    def test_parametric_pow_cone_alpha(self) -> None:
        """PowCone3D's alpha lives outside the constraint args and survives
        EvalParams; it must flow to the solver identically on the default
        DIFFENGINE route."""
        alpha = cp.Parameter()
        alpha.value = 0.4
        x = cp.Variable(pos=True)
        y = cp.Variable(pos=True)
        z = cp.Variable()
        cons = [cp.constraints.PowCone3D(x, y, z, alpha), x <= 2, y <= 3]
        prob = cp.Problem(cp.Maximize(z), cons)
        prob.solve(solver=SOLVER, ignore_dpp=True)
        self.assertEqual(_stuffing_backend(prob._cache.solving_chain), DIFFENGINE)

        x2 = cp.Variable(pos=True)
        y2 = cp.Variable(pos=True)
        z2 = cp.Variable()
        base = cp.Problem(cp.Maximize(z2),
                          [cp.constraints.PowCone3D(x2, y2, z2, 0.4),
                           x2 <= 2, y2 <= 3])
        base.solve(solver=SOLVER, canon_backend=s.CPP_CANON_BACKEND,
                   ignore_dpp=True)
        self.assertAlmostEqual(prob.value, base.value, places=4)

    def test_explicit_backend_still_honored(self) -> None:
        """An explicit canon_backend on the ignore_dpp path wins over the
        default."""
        x = cp.Variable(3)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x - 1)))
        _, chain, _ = prob.get_problem_data(
            SOLVER, ignore_dpp=True, canon_backend=s.SCIPY_CANON_BACKEND)
        self.assertEqual(_stuffing_backend(chain), s.SCIPY_CANON_BACKEND)

    def test_dpp_path_unaffected(self) -> None:
        """DPP solves without ignore_dpp keep the tensor pipeline."""
        p = cp.Parameter()
        p.value = 1.0
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x - p)))
        prob.solve(solver=SOLVER)
        self.assertNotEqual(_stuffing_backend(prob._cache.solving_chain),
                            DIFFENGINE)

    def test_perspective_solves_with_pinned_internal_chain(self) -> None:
        """The perspective canonicalizer's internal ignore_dpp chain is pinned
        to CPP; perspective problems keep solving correctly."""
        x = cp.Variable()
        t = cp.Variable(pos=True)
        persp = cp.perspective(cp.square(x), s=t)
        prob = cp.Problem(cp.Minimize(persp + t), [x >= 2, t <= 4])
        prob.solve(solver=SOLVER)
        self.assertEqual(prob.status, cp.OPTIMAL)
        # x^2/t + t minimized at t = min(x=2 -> 2), value 2^2/2+2 = 4 at t=2
        self.assertAlmostEqual(prob.value, 4.0, places=3)


class TestIgnoreDppBehavior(BaseTest):
    """ignore_dpp / non-DPP solves keep parameters symbolic on the DIFFENGINE
    backend; values must refresh across solves and unsound canonicalizations
    must not happen."""

    def test_parametric_divisor_resolves(self) -> None:
        """A parametric divisor is re-evaluated by the engine on each solve."""
        p = cp.Parameter(pos=True)
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.square(x / p - 1.0)))
        for val in (2.0, 5.0):
            p.value = val
            prob.solve(solver=SOLVER, ignore_dpp=True)
            self.assertEqual(prob.status, cp.OPTIMAL)
            self.assertAlmostEqual(x.value, val, places=4)

        pv = cp.Parameter(3, pos=True)
        xv = cp.Variable(3)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(xv / pv - 1.0)))
        for val in (np.array([1.0, 2.0, 4.0]), np.array([3.0, 0.5, 1.5])):
            pv.value = val
            prob.solve(solver=SOLVER, ignore_dpp=True)
            self.assertEqual(prob.status, cp.OPTIMAL)
            self.assertItemsAlmostEqual(xv.value, val, places=4)

    def test_param_constant_in_concave_position_sound(self) -> None:
        """A parametric-constant composite on the 'wrong' side of an
        inequality must not be epigraph-relaxed: x <= power(t, 2) would
        relax to x <= s, s >= t**2 (vacuous). It folds to a CallbackParam
        leaf before canonicalization and refreshes on each solve."""
        t = cp.Parameter()
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(-x), [x <= cp.power(t, 2)])
        for val in (2.0, 3.0):
            t.value = val
            prob.solve(solver=SOLVER, ignore_dpp=True)
            self.assertEqual(prob.status, cp.OPTIMAL)
            self.assertAlmostEqual(x.value, val ** 2, places=4)

    def test_unsupported_variable_free_atom_evaluates(self) -> None:
        """An atom with no symbolic converter over parameters only (floor,
        as emitted by DQCP bisection) folds to a CallbackParam leaf whose
        value refreshes per solve."""
        p = cp.Parameter()
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.square(x - cp.floor(p))))
        for val in (2.7, -1.2):
            p.value = val
            prob.solve(solver=SOLVER, ignore_dpp=True)
            self.assertAlmostEqual(x.value, np.floor(val), places=4)

    def test_symbolic_quad_matrix_refreshes(self) -> None:
        """quad_over_lin(x, p) keeps its quadratic matrix symbolic (I/p fed
        to the engine as a composite param_source); re-solves must serve
        fresh values."""
        x = cp.Variable()
        p = cp.Parameter()
        prob = cp.Problem(cp.Minimize(cp.quad_over_lin(x, p) + x))
        for val in (1.0, 1000.0, 1.0):
            p.value = val
            prob.solve(solver=SOLVER, ignore_dpp=True)
            self.assertAlmostEqual(x.value, -val / 2.0, places=3)

    def test_soc_restruct_resolve_and_duals(self) -> None:
        """The pre-applied SOC restructuring matrix must act on freshly
        extracted (A, b) on re-solves; duals must match the default path."""
        c = np.array([1.0, 2.0])
        p = cp.Parameter(2)
        x = cp.Variable(2)
        constraints = [cp.norm(x - p) <= 2.0, x >= -5]
        prob = cp.Problem(cp.Minimize(c @ x), constraints)

        for val in (np.array([1.0, 1.0]), np.array([-2.0, 3.0])):
            p.value = val
            prob.solve(solver=SOLVER, ignore_dpp=True)
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

    def test_psd_restruct_resolve(self) -> None:
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
            prob.solve(solver=SOLVER, ignore_dpp=True)
            self.assertEqual(prob.status, cp.OPTIMAL)

            X_base = cp.Variable((2, 2), symmetric=True)
            base = cp.Problem(cp.Minimize(cp.trace(P_val @ X_base)),
                              [X_base >> np.eye(2), cp.trace(X_base) <= 5])
            base.solve(solver=SOLVER)
            self.assertAlmostEqual(prob.value, base.value, places=3)

    def test_derivative_contract_rejected(self) -> None:
        """The symbolic program refuses the DPP-tensor differentiation
        contract loudly instead of returning silently wrong derivatives."""
        p = cp.Parameter()
        p.value = 1.0
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.square(x - p)))
        data, _, _ = prob.get_problem_data(SOLVER, ignore_dpp=True)
        param_prog = data[s.PARAM_PROB]
        with self.assertRaisesRegex(NotImplementedError, "differentiation"):
            param_prog.apply_parameters({p.id: 0.5}, zero_offset=True)
        with self.assertRaisesRegex(NotImplementedError, "differentiation"):
            param_prog.split_adjoint({})
        with self.assertRaisesRegex(NotImplementedError, "differentiation"):
            param_prog.apply_param_jac(None, None, None, None)

    @unittest.skipUnless(INSTALLED_MI_SOLVERS, "no mixed-integer solver installed")
    def test_mixed_integer_ignore_dpp(self) -> None:
        x = cp.Variable(3, integer=True)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 0.5, x <= 3.7])
        prob.solve(ignore_dpp=True)
        self.assertEqual(prob.status, cp.OPTIMAL)
        self.assertItemsAlmostEqual(x.value, np.ones(3), places=4)


class TestResolveCaching(BaseTest):
    """The compiled DiffengineParamConeProg is cached across ignore_dpp /
    non-DPP re-solves; values must refresh through the cache, and the one
    value-consuming canonicalization must disable it."""

    def test_extraction_runs_once_per_solve(self) -> None:
        """The solver's apply_parameters must reuse the matrices stuffing
        just extracted; a cached re-solve with changed values must re-extract
        exactly once without rebuilding the compiled problem."""
        from cvxpy.reductions.solvers.nlp_solvers.diff_engine.extractor import (
            DiffEngineExtractor,
        )

        p = cp.Parameter(2)
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x - p)), [x >= -10])

        real_extract = DiffEngineExtractor.extract
        real_build = DiffEngineExtractor.build
        with mock.patch.object(DiffEngineExtractor, "extract",
                               autospec=True, side_effect=real_extract) as spy, \
             mock.patch.object(DiffEngineExtractor, "build",
                               autospec=True, side_effect=real_build) as build_spy:
            p.value = np.array([1.0, 2.0])
            prob.solve(solver=SOLVER, ignore_dpp=True)
            self.assertEqual(spy.call_count, 1)  # stuffing only
            self.assertItemsAlmostEqual(x.value, p.value, places=4)

            # The program is cached across solves; changed parameter values
            # trigger exactly one re-extraction in apply_parameters, and the
            # compiled problem is NOT rebuilt.
            p.value = np.array([-3.0, 4.0])
            prob.solve(solver=SOLVER, ignore_dpp=True)
            self.assertEqual(spy.call_count, 2)
            self.assertEqual(build_spy.call_count, 1)
            self.assertItemsAlmostEqual(x.value, p.value, places=4)

    def test_parametric_constraint_quad_form_not_cached(self) -> None:
        """quad_form(x, P) in a constraint factorizes P.value during
        canonicalization (the one value-consuming canon), so the program is
        never cached -- and re-solves must still track fresh values."""
        n = 2
        P = cp.Parameter((n, n), PSD=True)
        y = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.sum(y)), [cp.quad_form(y, P) <= 1])

        P.value = np.eye(n)
        prob.solve(solver=SOLVER, ignore_dpp=True)
        self.assertIsNone(prob._cache.param_prog)
        val1 = prob.value

        P.value = 4 * np.eye(n)
        prob.solve(solver=SOLVER, ignore_dpp=True)
        self.assertIsNone(prob._cache.param_prog)
        self.assertAlmostEqual(prob.value, val1 / 2.0, places=4)

    def test_cache_hit_through_restructured_cones(self) -> None:
        """Cache-hit re-solves must re-apply the stored restructuring matrix
        to freshly extracted (A, b): SOC + PSD problem with changing values."""
        p = cp.Parameter(2)
        x = cp.Variable(2)
        X = cp.Variable((2, 2), symmetric=True)
        prob = cp.Problem(
            cp.Minimize(cp.sum(x) + cp.trace(X)),
            [cp.norm(x - p) <= 2.0, X >> cp.diag(p)])
        cached = None
        for seed in (1, 2, 1):
            val = np.random.default_rng(seed).standard_normal(2)
            p.value = val
            prob.solve(solver=SOLVER, ignore_dpp=True)
            self.assertEqual(prob.status, cp.OPTIMAL)
            if cached is None:
                cached = prob._cache.param_prog
                self.assertIsInstance(cached, DiffengineParamConeProg)
            else:
                self.assertIs(prob._cache.param_prog, cached)

            x_b = cp.Variable(2)
            X_b = cp.Variable((2, 2), symmetric=True)
            base = cp.Problem(
                cp.Minimize(cp.sum(x_b) + cp.trace(X_b)),
                [cp.norm(x_b - val) <= 2.0, X_b >> np.diag(val)])
            base.solve(solver=SOLVER)
            self.assertAlmostEqual(prob.value, base.value, places=4)

    def test_scaled_matrix_coefficient_refreshes_through_cache(self) -> None:
        """(p * A) @ x: broadcasting promotes p over A, so the coefficient
        reaches the engine as a composite side subtree whose values must
        rebuild on re-solves. The constraint boundary (hence the optimal
        point, value 0.5/p) moves with p, so a stale coefficient produces a
        wrong optimum, not just a wrong offset."""
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        x = cp.Variable(2)
        p = cp.Parameter(nonneg=True)
        prob = cp.Problem(cp.Minimize(cp.sum(x)),
                          [(p * A) @ x >= 1, x >= 0])
        cached = None
        for val in (1.0, 100.0, 1.0):
            p.value = val
            prob.solve(solver=SOLVER, ignore_dpp=True)
            self.assertAlmostEqual(prob.value, 0.5 / val, places=4)
            if cached is None:
                cached = prob._cache.param_prog
                self.assertIsInstance(cached, DiffengineParamConeProg)
            else:
                self.assertIs(prob._cache.param_prog, cached)

    def test_scaled_quad_matrix_refreshes_through_cache(self) -> None:
        """quad_form(x, g * Sigma): the scaled constant matrix stays symbolic
        as the engine's quadratic matrix. The unconstrained optimum
        x = (g Sigma)^{-1} 1 (value -5/(6g)) moves with g, so a stale Q
        produces a wrong optimum on cached re-solves."""
        Sigma = np.array([[2.0, 0.0], [0.0, 3.0]])
        x = cp.Variable(2)
        g = cp.Parameter(nonneg=True)
        prob = cp.Problem(cp.Minimize(
            cp.quad_form(x, g * Sigma, assume_PSD=True) - 2.0 * cp.sum(x)))
        cached = None
        for val in (1.0, 50.0, 1.0):
            g.value = val
            prob.solve(solver=SOLVER, ignore_dpp=True)
            self.assertAlmostEqual(prob.value, -5.0 / (6.0 * val), places=4)
            self.assertItemsAlmostEqual(
                x.value, [1.0 / (2 * val), 1.0 / (3 * val)], places=4)
            if cached is None:
                cached = prob._cache.param_prog
                self.assertIsInstance(cached, DiffengineParamConeProg)
            else:
                self.assertIs(prob._cache.param_prog, cached)

    def test_scaled_param_coefficient_refreshes_through_cache(self) -> None:
        """A scaled parameter multiplying a variable ((2*p) * x) reaches the
        engine as a coefficient; the multiplicative node inside it caches
        parameter data, so the cached program must serve fresh values on
        re-solves."""
        x = cp.Variable()
        y = cp.Variable()
        p = cp.Parameter()
        prob = cp.Problem(cp.Minimize(y),
                          [y >= (2 * p) * x, x == 1])
        cached = None
        for val in (1.0, 100.0, 1.0):
            p.value = val
            prob.solve(solver=SOLVER, ignore_dpp=True)
            self.assertAlmostEqual(y.value, 2 * val, places=4)
            if cached is None:
                cached = prob._cache.param_prog
                self.assertIsInstance(cached, DiffengineParamConeProg)
            else:
                self.assertIs(prob._cache.param_prog, cached)


class TestIgnoreDppCacheHygiene(BaseTest):
    """Toggling ignore_dpp between solves must fully invalidate the cache:
    the chain, the cached parametric program, and the solver warm-start cache
    all belong to one cache key."""

    def test_toggle_ignore_dpp_switches_chain_and_stays_correct(self) -> None:
        """ignore_dpp -> default -> default (new params) -> ignore_dpp (new
        params): each step must use the right backend and match a fresh
        baseline, so any stale cache shows up numerically."""
        def stuffing_backend(prob):
            return [r for r in prob._cache.solving_chain.reductions
                    if isinstance(r, ConeMatrixStuffing)][0].canon_backend

        def baseline(A_val, b_val):
            x = cp.Variable(A_val.shape[1])
            base = cp.Problem(
                cp.Minimize(cp.sum_squares(A_val @ x - b_val)), [x >= -10])
            base.solve(solver=SOLVER)
            return base.value

        rng = np.random.default_rng(0)
        m, n = 8, 5  # overdetermined so the optimum is strictly positive
        A = cp.Parameter((m, n))
        b = cp.Parameter(m)
        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)), [x >= -10])

        # 1. ignore_dpp first: DIFFENGINE chain with a cached parametric
        # program (parameters stay symbolic, so caching is safe).
        A.value = rng.standard_normal((m, n))
        b.value = rng.standard_normal(m)
        prob.solve(solver=SOLVER, ignore_dpp=True)
        solver_cache_de = prob._solver_cache
        self.assertEqual(stuffing_backend(prob), s.DIFFENGINE_CANON_BACKEND)
        de_prog = prob._cache.param_prog
        self.assertIsInstance(de_prog, DiffengineParamConeProg)
        self.assertAlmostEqual(prob.value, baseline(A.value, b.value), places=4)

        # 2. default solve, same parameter values: key change must rebuild the
        # chain (DPP path, not DIFFENGINE) and reset the solver cache.
        prob.solve(solver=SOLVER)
        self.assertNotEqual(stuffing_backend(prob), s.DIFFENGINE_CANON_BACKEND)
        self.assertIsNotNone(prob._cache.param_prog)  # DPP program cached
        self.assertIsNot(prob._solver_cache, solver_cache_de)
        self.assertAlmostEqual(prob.value, baseline(A.value, b.value), places=4)

        # 3. new parameter values, default solve again: the DPP fast path
        # (cached param_prog) must serve fresh values, not stale ones.
        dpp_prog = prob._cache.param_prog
        A.value = rng.standard_normal((m, n))
        b.value = rng.standard_normal(m)
        prob.solve(solver=SOLVER)
        self.assertIs(prob._cache.param_prog, dpp_prog)  # fast path reused
        self.assertAlmostEqual(prob.value, baseline(A.value, b.value), places=4)

        # 4. back to ignore_dpp with new values: the key toggle must have
        # invalidated step 1's program (a fresh one is built and cached).
        A.value = rng.standard_normal((m, n))
        b.value = rng.standard_normal(m)
        prob.solve(solver=SOLVER, ignore_dpp=True)
        self.assertEqual(stuffing_backend(prob), s.DIFFENGINE_CANON_BACKEND)
        self.assertIsInstance(prob._cache.param_prog, DiffengineParamConeProg)
        self.assertIsNot(prob._cache.param_prog, de_prog)
        self.assertAlmostEqual(prob.value, baseline(A.value, b.value), places=4)

    def test_param_free_program_cached_and_reused(self) -> None:
        """A parameter-free ignore_dpp solve caches its stock program and the
        fast path reuses it."""
        y = cp.Variable(3)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(y - 1)), [y >= 0])

        prob.solve(solver=SOLVER, ignore_dpp=True)
        cached = prob._cache.param_prog
        self.assertIsInstance(cached, ParamConeProg)

        prob.solve(solver=SOLVER, ignore_dpp=True)
        self.assertIs(prob._cache.param_prog, cached)  # fast path reused
        self.assertAlmostEqual(prob.value, 0.0)
        self.assertItemsAlmostEqual(y.value, np.ones(3), places=4)

    def test_shared_parameter_across_problems(self) -> None:
        """Two problems sharing a Parameter, one on each path, must not
        contaminate each other's caches or values."""
        p = cp.Parameter()
        x = cp.Variable()
        y = cp.Variable()
        prob_de = cp.Problem(cp.Minimize(cp.square(x - p)))
        prob_dpp = cp.Problem(cp.Minimize(cp.square(y - 2 * p)))

        p.value = 1.0
        prob_de.solve(solver=SOLVER, ignore_dpp=True)
        prob_dpp.solve(solver=SOLVER)
        self.assertAlmostEqual(x.value, 1.0)
        self.assertAlmostEqual(y.value, 2.0)

        p.value = -3.0
        prob_dpp.solve(solver=SOLVER)
        prob_de.solve(solver=SOLVER, ignore_dpp=True)
        self.assertAlmostEqual(x.value, -3.0)
        self.assertAlmostEqual(y.value, -6.0)
