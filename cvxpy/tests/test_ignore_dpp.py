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
import warnings
from unittest import mock

import numpy as np
import scipy.sparse as sp

import cvxpy as cp
import cvxpy.settings as s
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import (
    ConeMatrixStuffing,
    ParamConeProg,
)
from cvxpy.reductions.eval_params import EvalParams
from cvxpy.reductions.fold_callback_params import CallbackParamFold
from cvxpy.reductions.solvers.defines import INSTALLED_MI_SOLVERS, INSTALLED_SOLVERS
from cvxpy.reductions.solvers.nlp_solvers.diff_engine.cone_program import (
    DiffengineConeProg,
    _Matrices,
    _RowLayout,
)
from cvxpy.tests.base_test import BaseTest

SOLVER = cp.CLARABEL
DIFFENGINE = s.DIFFENGINE_CANON_BACKEND


def _stuffing_backend(chain) -> str | None:
    stuffing = [r for r in chain.reductions
                if isinstance(r, ConeMatrixStuffing)][0]
    return stuffing.canon_backend


class TestIgnoreDppSelection(BaseTest):
    """The ignore_dpp / non-DPP branch defaults to the DIFFENGINE backend and
    keeps the parameters symbolic, falling back to EvalParams plus the tensor
    backends only where DIFFENGINE is not eligible."""

    def test_default_backend(self) -> None:
        """Which backend the ignore_dpp / non-DPP branch picks by default,
        and that an ineligible problem still solves on a tensor backend."""
        x, X3 = cp.Variable(3), cp.Variable((2, 2, 2))
        p = cp.Parameter()
        p.value = 2.0
        lb = cp.Parameter(2)
        lb.value = np.array([0.5, 0.25])
        xb = cp.Variable(2, bounds=[lb, 10])
        param_free = cp.Problem(cp.Minimize(cp.sum_squares(x - 1)), [x >= 0])
        cases = [
            # (name, problem, solve kwargs, expected backend or None for "not DIFFENGINE")
            ("ignore_dpp", param_free, {"ignore_dpp": True}, DIFFENGINE),
            ("non-DPP", cp.Problem(cp.Minimize(cp.square(x[0] - p * p))), {}, DIFFENGINE),
            ("explicit backend", param_free,
             {"ignore_dpp": True, "canon_backend": s.SCIPY_CANON_BACKEND},
             s.SCIPY_CANON_BACKEND),
            ("DPP", cp.Problem(cp.Minimize(cp.sum_squares(x[0] - p))), {}, None),
            ("N-D", cp.Problem(cp.Minimize(cp.sum_squares(X3))), {"ignore_dpp": True}, None),
            ("parametric bounds", cp.Problem(cp.Minimize(cp.sum(xb))),
             {"ignore_dpp": True}, None),
        ]
        for name, prob, kwargs, expected in cases:
            with self.subTest(name), warnings.catch_warnings():
                warnings.simplefilter("ignore")  # non-DPP warning
                prob.solve(solver=SOLVER, **kwargs)
                self.assertEqual(prob.status, cp.OPTIMAL)
                backend = _stuffing_backend(prob._cache.solving_chain)
                if expected is None:
                    self.assertNotEqual(backend, DIFFENGINE)
                else:
                    self.assertEqual(backend, expected)

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

    def test_direct_cone_solver_falls_back(self) -> None:
        """ExtractDirectCones detects its cones in the parameter tensors,
        which the symbolic program does not have; solvers advertising
        DIR_CONE_KINDS therefore stay on the tensor path."""
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        x = cp.Variable(2)
        p = cp.Parameter(nonneg=True)
        p.value = 1.0
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [(p * A) @ x >= 1, x >= 0])
        solver_cls = type(prob._construct_chain(solver=SOLVER).solver)
        with mock.patch.object(solver_cls, "DIR_CONE_KINDS",
                               frozenset({"nonneg", "soc"})):
            chain = prob._construct_chain(solver=SOLVER, ignore_dpp=True)
        self.assertNotEqual(_stuffing_backend(chain), DIFFENGINE)
        self.assertTrue(any(isinstance(r, EvalParams) for r in chain.reductions))

    @unittest.skipUnless(s.DIFFCP in INSTALLED_SOLVERS, 'diffcp is not installed')
    def test_tensor_consuming_solver_falls_back(self) -> None:
        """diffcp reads the parameter tensors directly (keep_zeros), which a
        re-extracted program cannot honor, so it stays on the tensor path."""
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        x = cp.Variable(2)
        p = cp.Parameter(nonneg=True)
        p.value = 2.0
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [(p * A) @ x >= 1, x >= 0])
        chain = prob._construct_chain(solver=s.DIFFCP, ignore_dpp=True)
        self.assertNotEqual(_stuffing_backend(chain), DIFFENGINE)
        prob.solve(solver=s.DIFFCP, ignore_dpp=True)
        ref = cp.Problem(cp.Minimize(cp.sum(x)),
                         [(p.value * A) @ x >= 1, x >= 0])
        ref.solve(solver=SOLVER)
        self.assertAlmostEqual(prob.value, ref.value, places=4)

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

    def test_parametric_quad_form_falls_back(self) -> None:
        """The cone quad_form canon factorizes P.value, so a parametric P (in
        a constraint, or scaled in the objective) is not eligible for the
        symbolic route: it falls back, and re-solves track fresh values."""
        y = cp.Variable(2)
        P = cp.Parameter((2, 2), PSD=True)
        g = cp.Parameter(nonneg=True)
        Sigma = np.diag([2.0, 3.0])
        cases = {
            # min sum(y) s.t. y'(vI)y <= 1 has value -sqrt(2/v).
            "constraint": (cp.Problem(cp.Minimize(cp.sum(y)), [cp.quad_form(y, P) <= 1]),
                           P, lambda v: v * np.eye(2), lambda v: -np.sqrt(2.0 / v)),
            # Unconstrained optimum y = (g Sigma)^{-1} 1, value -5/(6g).
            "objective": (cp.Problem(cp.Minimize(
                              cp.quad_form(y, g * Sigma, assume_PSD=True) - 2 * cp.sum(y))),
                          g, lambda v: v, lambda v: -5.0 / (6.0 * v)),
        }
        for name, (prob, param, value, expected) in cases.items():
            with self.subTest(name):
                for val in (1.0, 4.0, 1.0):
                    param.value = value(val)
                    prob.solve(solver=SOLVER, ignore_dpp=True)
                    self.assertNotEqual(
                        _stuffing_backend(prob._cache.solving_chain), DIFFENGINE)
                    self.assertAlmostEqual(prob.value, expected(val), places=4)

    def test_perspective_solves(self) -> None:
        """A parameter-free perspective is eligible; it keeps solving
        correctly through the canonicalizer's own auxiliary chain."""
        x = cp.Variable()
        t = cp.Variable(pos=True)
        persp = cp.perspective(cp.square(x), s=t)
        prob = cp.Problem(cp.Minimize(persp + t), [x >= 2, t <= 4])
        prob.solve(solver=SOLVER)
        self.assertEqual(prob.status, cp.OPTIMAL)
        # x = 2 at its bound, then min over t <= 4 of 4/t + t is 4 at t = 2.
        self.assertAlmostEqual(prob.value, 4.0, places=3)

    def test_parametric_perspective_falls_back(self) -> None:
        """perspective_canon compiles expr.f through its own chain, freezing
        the parameter values in force at canonicalization time, so a
        perspective with parameters in f is not eligible for the symbolic
        route. f is not in args, so perspective._supports_diffengine is the
        only place those parameters can be seen -- with or without a second,
        visible parameter that would otherwise make the problem eligible."""
        def build(p_val, extra_param):
            p = cp.Parameter()
            x = cp.Variable()
            t = cp.Variable(pos=True)
            obj = cp.perspective(cp.square(x - p), s=t) + t
            if extra_param:
                # Without this the problem looks parameter-free and takes the
                # EvalParams fallback anyway; with it, only the atom's own
                # eligibility check keeps it off the symbolic route.
                q = cp.Parameter()
                q.value = 0.5
                obj = obj + q * q * x
            p.value = p_val
            return cp.Problem(cp.Minimize(obj), [x >= 2, t <= 4]), p

        for extra_param in (False, True):
            with self.subTest(extra_param=extra_param):
                baseline, _ = build(1.5, extra_param)
                baseline.solve(solver=SOLVER, ignore_dpp=True)

                prob, p = build(0.0, extra_param)
                prob.solve(solver=SOLVER, ignore_dpp=True)
                self.assertNotEqual(
                    _stuffing_backend(prob._cache.solving_chain), DIFFENGINE)
                self.assertIsNone(prob._cache.param_prog)
                p.value = 1.5
                prob.solve(solver=SOLVER, ignore_dpp=True)
                self.assertAlmostEqual(prob.value, baseline.value, places=4)


class TestDiffengineCapability(BaseTest):
    """Atoms with no converter must keep the tensor backends when the
    backend is auto-selected, and still fail loudly when it is requested."""

    @staticmethod
    def _concatenate():
        p = cp.Parameter()
        p.value = 2.0
        x = cp.Variable(4)
        return cp.Problem(
            cp.Minimize(cp.sum(cp.concatenate([x, x])) + p * p * cp.sum(x)),
            [x >= 1]), 24.0

    @staticmethod
    def _offset_diag():
        p = cp.Parameter()
        p.value = 2.0
        y = cp.Variable(3)
        return cp.Problem(
            cp.Minimize(cp.sum(cp.diag(y, 1)) + p * p * cp.sum(y)),
            [y >= 1]), 15.0

    @staticmethod
    def _axis_quad_over_lin():
        # Canonicalizes to a SymbolicQuadForm with block_indices.
        p = cp.Parameter()
        p.value = 2.0
        X = cp.Variable((3, 2))
        return cp.Problem(
            cp.Minimize(cp.sum(cp.sum_squares(X, axis=0)) + p * p * cp.sum(X)),
            [X >= 1]), 30.0

    def test_unsupported_atom_falls_back(self) -> None:
        for name in ("_concatenate", "_offset_diag", "_axis_quad_over_lin"):
            for ignore_dpp in (False, True):
                with self.subTest(atom=name, ignore_dpp=ignore_dpp):
                    prob, expected = getattr(self, name)()
                    prob.solve(solver=SOLVER, ignore_dpp=ignore_dpp)
                    self.assertNotEqual(
                        _stuffing_backend(prob._cache.solving_chain), DIFFENGINE)
                    self.assertAlmostEqual(prob.value, expected, places=4)

    def test_unsupported_atom_explicit_request_raises(self) -> None:
        for name in ("_concatenate", "_offset_diag", "_axis_quad_over_lin"):
            with self.subTest(atom=name):
                prob, _ = getattr(self, name)()
                with self.assertRaises(NotImplementedError):
                    prob.solve(solver=SOLVER, canon_backend=DIFFENGINE)


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
            prob.solve(solver=SOLVER, ignore_dpp=True, canon_backend=DIFFENGINE)
            self.assertEqual(prob.status, cp.OPTIMAL)
            self.assertAlmostEqual(x.value, val, places=4)

        pv = cp.Parameter(3, pos=True)
        xv = cp.Variable(3)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(xv / pv - 1.0)))
        for val in (np.array([1.0, 2.0, 4.0]), np.array([3.0, 0.5, 1.5])):
            pv.value = val
            prob.solve(solver=SOLVER, ignore_dpp=True, canon_backend=DIFFENGINE)
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
            prob.solve(solver=SOLVER, ignore_dpp=True, canon_backend=DIFFENGINE)
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
            prob.solve(solver=SOLVER, ignore_dpp=True, canon_backend=DIFFENGINE)
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
            prob.solve(solver=SOLVER, ignore_dpp=True, canon_backend=DIFFENGINE)
            self.assertAlmostEqual(x.value, -val / 2.0, places=3)

    def test_derivative_contract_rejected(self) -> None:
        """The symbolic program refuses the DPP-tensor differentiation
        contract loudly instead of returning silently wrong derivatives."""
        p = cp.Parameter()
        p.value = 1.0
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.square(x - p)))
        data, _, _ = prob.get_problem_data(SOLVER, ignore_dpp=True, canon_backend=DIFFENGINE)
        param_prog = data[s.PARAM_PROB]
        with self.assertRaisesRegex(NotImplementedError, "differentiation"):
            param_prog.apply_parameters({p.id: 0.5}, zero_offset=True)
        with self.assertRaisesRegex(NotImplementedError, "differentiation"):
            param_prog.apply_param_jac(None, None, None, None)

    def test_split_solution_round_trips(self) -> None:
        """split_solution/split_adjoint are about the stuffed variable, not
        the parameter tensors, so they are inherited unchanged."""
        p = cp.Parameter()
        p.value = 2.0
        x = cp.Variable(2)
        y = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x - p * p) + cp.square(y)))
        prob.solve(solver=SOLVER, ignore_dpp=True, canon_backend=DIFFENGINE)
        prog = prob._cache.param_prog
        self.assertIsInstance(prog, DiffengineConeProg)

        stacked = np.arange(1.0, prog.x.size + 1.0)
        split = prog.split_solution(stacked)
        self.assertItemsAlmostEqual(prog.split_adjoint(split), stacked)

    @unittest.skipUnless(INSTALLED_MI_SOLVERS, "no mixed-integer solver installed")
    def test_mixed_integer_ignore_dpp(self) -> None:
        x = cp.Variable(3, integer=True)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 0.5, x <= 3.7])
        prob.solve(ignore_dpp=True, canon_backend=DIFFENGINE)
        self.assertEqual(prob.status, cp.OPTIMAL)
        self.assertItemsAlmostEqual(x.value, np.ones(3), places=4)


class TestResolveCaching(BaseTest):
    """The compiled DiffengineConeProg is cached across ignore_dpp /
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
            prob.solve(solver=SOLVER, ignore_dpp=True, canon_backend=DIFFENGINE)
            self.assertEqual(spy.call_count, 1)  # stuffing only
            self.assertItemsAlmostEqual(x.value, p.value, places=4)

            # The program is cached across solves; changed parameter values
            # trigger exactly one re-extraction in apply_parameters, and the
            # compiled problem is NOT rebuilt.
            p.value = np.array([-3.0, 4.0])
            prob.solve(solver=SOLVER, ignore_dpp=True, canon_backend=DIFFENGINE)
            self.assertEqual(spy.call_count, 2)
            self.assertEqual(build_spy.call_count, 1)
            self.assertItemsAlmostEqual(x.value, p.value, places=4)

    def test_row_layout_survives_resolves(self) -> None:
        """The stored row layout must act on freshly extracted (A, b) on
        cache-hit re-solves, with values and duals matching the tensor path.

        Multi-cone SOC (which permutes rows), an equality (which flips signs)
        and a PSD cone: one SOC alone would give the identity layout.
        """
        def build(pv, Y, X):
            P = cp.reshape(cp.hstack([pv, pv]), (2, 2), order='F')
            D = cp.diag(pv) if isinstance(pv, cp.Parameter) else np.diag(pv)
            return [cp.norm(Y - P, 2, axis=0) <= 2.0,
                    cp.sum(Y[:, 0]) == 1.0,
                    X >> D]

        p = cp.Parameter(2)
        Y, X = cp.Variable((2, 2)), cp.Variable((2, 2), symmetric=True)
        cons = build(p, Y, X)
        prob = cp.Problem(cp.Minimize(cp.sum(Y) + cp.trace(X)), cons)
        cached = None
        for seed in (1, 2, 1):
            p.value = np.random.default_rng(seed).standard_normal(2)
            prob.solve(solver=SOLVER, ignore_dpp=True, canon_backend=DIFFENGINE)
            self.assertIsInstance(prob._cache.param_prog, DiffengineConeProg)
            cached = cached or prob._cache.param_prog
            self.assertIs(prob._cache.param_prog, cached)

            Y_b, X_b = cp.Variable((2, 2)), cp.Variable((2, 2), symmetric=True)
            base_cons = build(p.value, Y_b, X_b)
            base = cp.Problem(cp.Minimize(cp.sum(Y_b) + cp.trace(X_b)), base_cons)
            base.solve(solver=SOLVER)
            self.assertAlmostEqual(prob.value, base.value, places=4)
            for con, con_b in zip(cons[:2], base_cons[:2]):
                self.assertItemsAlmostEqual(con.dual_value, con_b.dual_value, places=3)

    def test_scaled_coefficient_refreshes_through_cache(self) -> None:
        """A coefficient built from a parameter ((p * A) @ x, (2 * p) * x)
        reaches the engine as a composite subtree; the cached program must
        serve fresh values. The optimum moves with p, so a stale coefficient
        gives a wrong answer, not just a wrong offset."""
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        p = cp.Parameter(nonneg=True)
        x, y, z = cp.Variable(2), cp.Variable(), cp.Variable()
        cases = {
            "matrix": (cp.Problem(cp.Minimize(cp.sum(x)), [(p * A) @ x >= 1, x >= 0]),
                       lambda v: 0.5 / v),
            "scalar": (cp.Problem(cp.Minimize(y), [y >= (2 * p) * z, z == 1]),
                       lambda v: 2 * v),
        }
        for name, (prob, expected) in cases.items():
            with self.subTest(name):
                cached = None
                for val in (1.0, 100.0, 1.0):
                    p.value = val
                    prob.solve(solver=SOLVER, ignore_dpp=True, canon_backend=DIFFENGINE)
                    self.assertAlmostEqual(prob.value, expected(val), places=4)
                    cached = cached or prob._cache.param_prog
                    self.assertIs(prob._cache.param_prog, cached)


class TestIgnoreDppCacheHygiene(BaseTest):
    """Toggling ignore_dpp between solves must fully invalidate the cache:
    the chain, the cached parametric program, and the solver warm-start cache
    all belong to one cache key."""

    def test_toggle_ignore_dpp_switches_chain_and_stays_correct(self) -> None:
        """ignore_dpp -> default -> default (new params) -> ignore_dpp (new
        params): each step must use the right backend and match a fresh
        baseline, so any stale cache shows up numerically."""
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
        prob.solve(solver=SOLVER, ignore_dpp=True, canon_backend=DIFFENGINE)
        solver_cache_de = prob._solver_cache
        self.assertEqual(_stuffing_backend(prob._cache.solving_chain), DIFFENGINE)
        de_prog = prob._cache.param_prog
        self.assertIsInstance(de_prog, DiffengineConeProg)
        self.assertAlmostEqual(prob.value, baseline(A.value, b.value), places=4)

        # 2. default solve, same parameter values: key change must rebuild the
        # chain (DPP path, not DIFFENGINE) and reset the solver cache.
        prob.solve(solver=SOLVER)
        self.assertNotEqual(_stuffing_backend(prob._cache.solving_chain), DIFFENGINE)
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
        prob.solve(solver=SOLVER, ignore_dpp=True, canon_backend=DIFFENGINE)
        self.assertEqual(_stuffing_backend(prob._cache.solving_chain), DIFFENGINE)
        self.assertIsInstance(prob._cache.param_prog, DiffengineConeProg)
        self.assertIsNot(prob._cache.param_prog, de_prog)
        self.assertAlmostEqual(prob.value, baseline(A.value, b.value), places=4)

    def test_program_cached_and_reused(self) -> None:
        """The symbolic program is cached and the fast path reuses the same
        object. A parameter-free ignore_dpp solve still gets EvalParams and so
        is not cached, exactly as before this route existed."""
        p = cp.Parameter()
        p.value = 1.0
        y = cp.Variable(3)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(y - p * p)), [y >= 0])

        prob.solve(solver=SOLVER, ignore_dpp=True, canon_backend=DIFFENGINE)
        cached = prob._cache.param_prog
        self.assertIsInstance(cached, DiffengineConeProg)

        prob.solve(solver=SOLVER, ignore_dpp=True, canon_backend=DIFFENGINE)
        self.assertIs(prob._cache.param_prog, cached)  # fast path reused
        self.assertItemsAlmostEqual(y.value, np.ones(3), places=4)

        free = cp.Problem(cp.Minimize(cp.sum_squares(y - 1)), [y >= 0])
        free.solve(solver=SOLVER, ignore_dpp=True, canon_backend=DIFFENGINE)
        self.assertIsNone(free._cache.param_prog)

    def test_shared_parameter_across_problems(self) -> None:
        """Two problems sharing a Parameter, one on each path, must not
        contaminate each other's caches or values."""
        p = cp.Parameter()
        x = cp.Variable()
        y = cp.Variable()
        prob_de = cp.Problem(cp.Minimize(cp.square(x - p)))
        prob_dpp = cp.Problem(cp.Minimize(cp.square(y - 2 * p)))

        p.value = 1.0
        prob_de.solve(solver=SOLVER, ignore_dpp=True, canon_backend=DIFFENGINE)
        prob_dpp.solve(solver=SOLVER)
        self.assertAlmostEqual(x.value, 1.0)
        self.assertAlmostEqual(y.value, 2.0)

        p.value = -3.0
        prob_dpp.solve(solver=SOLVER)
        prob_de.solve(solver=SOLVER, ignore_dpp=True, canon_backend=DIFFENGINE)
        self.assertAlmostEqual(x.value, -3.0)
        self.assertAlmostEqual(y.value, -6.0)


class TestDiffengineProgram(BaseTest):
    """The symbolic program carries concrete matrices and no parameter
    tensors, so it is a sibling of ParamConeProg rather than a subclass."""

    def test_program_exposes_no_tensors(self) -> None:
        rng = np.random.default_rng(0)
        m, n = 8, 4
        A = cp.Parameter((m, n))
        b = cp.Parameter(m)
        x = cp.Variable(n)
        # A @ A.T @ A is nonlinear in A, so the problem is not DPP.
        resid = A @ A.T @ A @ x - b
        for quad_obj in (False, True):
            obj = cp.sum_squares(resid) if quad_obj else cp.norm1(resid)
            prob = cp.Problem(cp.Minimize(obj), [x >= -10])
            A.value = rng.standard_normal((m, n))
            b.value = rng.standard_normal(m)
            prob.solve(solver=SOLVER, ignore_dpp=True, canon_backend=DIFFENGINE)
            prog = prob._cache.param_prog
            self.assertIsInstance(prog, DiffengineConeProg)
            self.assertNotIsInstance(prog, ParamConeProg)
            for tensor in ('q', 'A', 'P', 'reduced_A', 'reduced_P',
                           'lb_tensor', 'ub_tensor'):
                self.assertFalse(hasattr(prog, tensor), tensor)
            # has_quad_obj answers from the concrete matrices, which is the
            # only thing ConicSolver.apply asks of the objective term.
            self.assertEqual(prog.has_quad_obj, quad_obj)


class TestRowLayout(BaseTest):
    """The cone row layout is applied to each extraction as a gather of
    A.data, planned once per sparsity pattern."""

    @staticmethod
    def _layout(rng, m):
        new_row = rng.permutation(m)
        sign = rng.choice([-1.0, 1.0], size=m)
        return new_row, sign

    @staticmethod
    def _reference(A, b, new_row, sign):
        R = sp.csc_array((sign, (new_row, np.arange(new_row.size))),
                         shape=(new_row.size,) * 2)
        return (R @ A).toarray(), R @ b

    def _check(self, layout, A, b, new_row, sign):
        out = layout.apply(_Matrices(np.zeros(A.shape[1]), 0.0, A, b, None))
        A_ref, b_ref = self._reference(A, b, new_row, sign)
        self.assertIsInstance(out.A, sp.csc_array)
        np.testing.assert_array_equal(out.A.toarray(), A_ref)
        np.testing.assert_array_equal(out.b, b_ref)
        # The result is canonical CSC, as solvers expect.
        A_sorted = out.A.copy()
        A_sorted.has_sorted_indices = False
        A_sorted.sort_indices()
        np.testing.assert_array_equal(A_sorted.indices, out.A.indices)

    def test_matches_permutation_matrix(self) -> None:
        rng = np.random.default_rng(0)
        m, n = 40, 15
        new_row, sign = self._layout(rng, m)
        layout = _RowLayout(new_row, sign)
        A = sp.random_array((m, n), density=0.2, format='csc', rng=rng)
        b = rng.standard_normal(m)
        self._check(layout, A, b, new_row, sign)
        plan = layout._plan
        # New values on the same pattern reuse the plan.
        A2 = A.copy()
        A2.data = rng.standard_normal(A2.nnz)
        self._check(layout, A2, rng.standard_normal(m), new_row, sign)
        self.assertIs(layout._plan, plan)
        # A different pattern is planned again.
        A3 = sp.random_array((m, n), density=0.3, format='csc', rng=rng)
        self._check(layout, A3, b, new_row, sign)
        self.assertIsNot(layout._plan, plan)

    def test_unsorted_indices(self) -> None:
        rng = np.random.default_rng(1)
        m, n = 30, 10
        new_row, sign = self._layout(rng, m)
        A = sp.random_array((m, n), density=0.3, format='csc', rng=rng)
        # Reverse the values within each column.
        indices, data = A.indices.copy(), A.data.copy()
        for j in range(n):
            lo, hi = A.indptr[j], A.indptr[j + 1]
            indices[lo:hi] = indices[lo:hi][::-1]
            data[lo:hi] = data[lo:hi][::-1]
        A_unsorted = sp.csc_array((data, indices, A.indptr), shape=A.shape)
        self._check(_RowLayout(new_row, sign), A_unsorted,
                    rng.standard_normal(m), new_row, sign)

    def test_empty_columns_and_rows(self) -> None:
        rng = np.random.default_rng(2)
        m, n = 12, 6
        new_row, sign = self._layout(rng, m)
        A = sp.csc_array(([1.0, -2.0, 3.0], ([0, 5, 11], [1, 1, 4])), shape=(m, n))
        self._check(_RowLayout(new_row, sign), A, rng.standard_normal(m),
                    new_row, sign)
