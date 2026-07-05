"""Tests for canonicalization backend selection logic.

The backend selection priority is:
1. User-specified backend (via canon_backend parameter)
2. COO for DPP problems with total parameter size >= DPP_PARAM_THRESHOLD (1000)
3. CPP if supported
4. SCIPY as fallback when CPP doesn't work
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import cvxpy as cp
import cvxpy.settings as s
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.reductions.solvers.solving_chain_utils import get_canon_backend


class TestBackendSelectionDPP:
    """Tests for DPP-based backend selection."""

    def test_large_dpp_selects_coo(self):
        """DPP problem with >= 1000 params should select COO backend."""
        A = cp.Parameter((100, 20))  # 2000 params
        x = cp.Variable(20)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x)))

        assert prob.is_dpp()
        assert sum(p.size for p in prob.parameters()) >= s.DPP_PARAM_THRESHOLD

        chain = prob._construct_chain(solver=cp.CLARABEL)
        stuffing = [r for r in chain.reductions if isinstance(r, ConeMatrixStuffing)][0]
        assert stuffing.canon_backend == s.COO_CANON_BACKEND

    def test_small_dpp_selects_cpp(self):
        """DPP problem with < 1000 params should use default (CPP)."""
        A = cp.Parameter((10, 10))  # 100 params
        x = cp.Variable(10)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x)))

        assert prob.is_dpp()
        assert sum(p.size for p in prob.parameters()) < s.DPP_PARAM_THRESHOLD

        chain = prob._construct_chain(solver=cp.CLARABEL)
        stuffing = [r for r in chain.reductions if isinstance(r, ConeMatrixStuffing)][0]
        # None means use default (CPP)
        assert stuffing.canon_backend is None

    def test_exactly_threshold_selects_coo(self):
        """DPP problem with exactly DPP_PARAM_THRESHOLD params should select COO."""
        # Create parameter with exactly 1000 elements
        A = cp.Parameter((50, 20))  # 1000 params
        x = cp.Variable(20)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x)))

        assert prob.is_dpp()
        assert sum(p.size for p in prob.parameters()) == s.DPP_PARAM_THRESHOLD

        chain = prob._construct_chain(solver=cp.CLARABEL)
        stuffing = [r for r in chain.reductions if isinstance(r, ConeMatrixStuffing)][0]
        assert stuffing.canon_backend == s.COO_CANON_BACKEND

    def test_non_dpp_with_large_params_uses_diffengine(self):
        """Non-DPP problem with many params routes to DIFFENGINE, NOT COO."""
        A = cp.Parameter((100, 20))  # 2000 params
        B = cp.Parameter((20, 20))  # 400 more params
        x = cp.Variable(20)
        # A @ B @ x is not DPP (parameter * parameter in A @ B)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ B @ x)))

        assert not prob.is_dpp()
        assert sum(p.size for p in prob.parameters()) >= s.DPP_PARAM_THRESHOLD

        chain = prob._construct_chain(solver=cp.CLARABEL)
        stuffing = [r for r in chain.reductions if isinstance(r, ConeMatrixStuffing)][0]
        # Non-DPP uses the DIFFENGINE backend (and never COO).
        assert stuffing.canon_backend == s.DIFFENGINE_CANON_BACKEND

    def test_ignore_dpp_skips_coo_selection(self):
        """With ignore_dpp=True, large DPP routes to DIFFENGINE, NOT COO."""
        A = cp.Parameter((100, 20))  # 2000 params
        x = cp.Variable(20)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x)))

        assert prob.is_dpp()

        chain = prob._construct_chain(solver=cp.CLARABEL, ignore_dpp=True)
        stuffing = [r for r in chain.reductions if isinstance(r, ConeMatrixStuffing)][0]
        # ignore_dpp uses the DIFFENGINE backend, not DPP-based COO selection.
        assert stuffing.canon_backend == s.DIFFENGINE_CANON_BACKEND


class TestBackendSelectionUserOverride:
    """Tests for user-specified backend override."""

    def test_user_scipy_overrides_dpp_coo(self):
        """User-specified SCIPY should override DPP-based COO selection."""
        A = cp.Parameter((100, 20))  # 2000 params
        x = cp.Variable(20)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x)))

        assert prob.is_dpp()

        chain = prob._construct_chain(solver=cp.CLARABEL, canon_backend=s.SCIPY_CANON_BACKEND)
        stuffing = [r for r in chain.reductions if isinstance(r, ConeMatrixStuffing)][0]
        assert stuffing.canon_backend == s.SCIPY_CANON_BACKEND

    def test_user_cpp_overrides_dpp_coo(self):
        """User-specified CPP should override DPP-based COO selection."""
        A = cp.Parameter((100, 20))  # 2000 params
        x = cp.Variable(20)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x)))

        assert prob.is_dpp()

        chain = prob._construct_chain(solver=cp.CLARABEL, canon_backend=s.CPP_CANON_BACKEND)
        stuffing = [r for r in chain.reductions if isinstance(r, ConeMatrixStuffing)][0]
        assert stuffing.canon_backend == s.CPP_CANON_BACKEND

    def test_user_coo_for_small_problem(self):
        """User can explicitly request COO even for small problems."""
        x = cp.Variable(10)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x)))

        chain = prob._construct_chain(solver=cp.CLARABEL, canon_backend=s.COO_CANON_BACKEND)
        stuffing = [r for r in chain.reductions if isinstance(r, ConeMatrixStuffing)][0]
        assert stuffing.canon_backend == s.COO_CANON_BACKEND


class TestBackendSelectionFallback:
    """Tests for SCIPY fallback when CPP doesn't work."""

    def test_unsupported_cpp_atom_fallback_to_scipy(self):
        """Problems with atoms that don't support CPP should fallback to SCIPY."""
        x = cp.Variable((2, 3))
        # broadcast_to doesn't support CPP
        expr = cp.broadcast_to(x, (4, 2, 3))
        prob = cp.Problem(cp.Minimize(cp.sum(expr)), [x >= 0])

        assert not prob._supports_cpp()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            backend = get_canon_backend(prob, None)

        assert backend == s.SCIPY_CANON_BACKEND
        assert len(w) == 1
        assert "SCIPY" in str(w[0].message)

    def test_ndim_gt_2_fallback_to_scipy(self):
        """Problems with >2D expressions should fallback to SCIPY."""
        x = cp.Variable((2, 3, 4))  # 3D variable
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 0])

        assert prob._max_ndim() > 2

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            backend = get_canon_backend(prob, None)

        assert backend == s.SCIPY_CANON_BACKEND
        assert len(w) == 1
        assert "SCIPY" in str(w[0].message)

    def test_ndim_gt_2_user_override_respected(self):
        """User-specified COO for >2D problem should be respected."""
        x = cp.Variable((2, 3, 4))  # 3D variable
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 0])

        backend = get_canon_backend(prob, s.COO_CANON_BACKEND)
        assert backend == s.COO_CANON_BACKEND

    def test_ndim_gt_2_cpp_raises_error(self):
        """User-specified CPP for >2D problem should raise error."""
        x = cp.Variable((2, 3, 4))  # 3D variable
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 0])

        with pytest.raises(ValueError, match="dimension greater than 2"):
            get_canon_backend(prob, s.CPP_CANON_BACKEND)


class TestBackendSelectionSolve:
    """Integration tests that actually solve problems."""

    def test_large_dpp_solves_with_coo(self):
        """Large DPP problem should solve successfully with auto-selected COO."""
        np.random.seed(42)
        A = cp.Parameter((100, 20))
        A.value = np.random.randn(100, 20)
        b = np.random.randn(100)
        x = cp.Variable(20)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)))

        prob.solve(solver=cp.CLARABEL)
        assert prob.status == cp.OPTIMAL

    def test_small_dpp_solves_with_cpp(self):
        """Small DPP problem should solve successfully with default CPP."""
        np.random.seed(42)
        A = cp.Parameter((10, 10))
        A.value = np.random.randn(10, 10)
        b = np.random.randn(10)
        x = cp.Variable(10)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)))

        prob.solve(solver=cp.CLARABEL)
        assert prob.status == cp.OPTIMAL

    def test_3d_problem_solves_with_scipy(self):
        """3D problem should solve successfully with auto-selected SCIPY."""
        x = cp.Variable((2, 3, 4))
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 0, cp.sum(x) >= 1])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore the SCIPY fallback warning
            prob.solve(solver=cp.CLARABEL)

        assert prob.status == cp.OPTIMAL


def _stuffing_backend(prob):
    """canon_backend of the ConeMatrixStuffing in the problem's cached chain."""
    return next(r.canon_backend for r in prob._cache.solving_chain.reductions
                if isinstance(r, ConeMatrixStuffing))


class TestIgnoreDppCacheHygiene:
    """Toggling ignore_dpp between solves must fully invalidate the cache:
    the chain, the cached parametric program, and the solver warm-start cache
    all belong to one (solver, gp, ignore_dpp, use_quad_obj) key."""

    @staticmethod
    def _least_squares_baseline(A_val, b_val):
        n = A_val.shape[1]
        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(A_val @ x - b_val)), [x >= -10])
        prob.solve(solver=cp.CLARABEL)
        return prob.value

    def test_toggle_ignore_dpp_switches_chain_and_stays_correct(self):
        """ignore_dpp -> default -> default (new params) -> ignore_dpp (new
        params): each step must use the right backend and match a fresh
        baseline, so any stale cache shows up numerically."""
        rng = np.random.default_rng(0)
        m, n = 8, 5  # overdetermined so the optimum is strictly positive
        A = cp.Parameter((m, n))
        b = cp.Parameter(m)
        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)), [x >= -10])

        # 1. ignore_dpp first: DIFFENGINE chain; parametric problems are not
        # cached on this path (EvalParams keeps the chain uncached so folded
        # variable-free constants refresh).
        A.value = rng.standard_normal((m, n))
        b.value = rng.standard_normal(m)
        prob.solve(solver=cp.CLARABEL, ignore_dpp=True)
        solver_cache_de = prob._solver_cache
        assert _stuffing_backend(prob) == s.DIFFENGINE_CANON_BACKEND
        assert prob._cache.param_prog is None
        np.testing.assert_allclose(
            prob.value, self._least_squares_baseline(A.value, b.value), rtol=1e-5)

        # 2. default solve, same parameter values: key change must rebuild the
        # chain (DPP path, not DIFFENGINE) and reset the solver cache.
        prob.solve(solver=cp.CLARABEL)
        assert _stuffing_backend(prob) != s.DIFFENGINE_CANON_BACKEND
        assert prob._cache.param_prog is not None  # DPP program cached
        assert prob._solver_cache is not solver_cache_de
        np.testing.assert_allclose(
            prob.value, self._least_squares_baseline(A.value, b.value), rtol=1e-5)

        # 3. new parameter values, default solve again: the DPP fast path
        # (cached param_prog) must serve fresh values, not stale ones.
        dpp_prog = prob._cache.param_prog
        A.value = rng.standard_normal((m, n))
        b.value = rng.standard_normal(m)
        prob.solve(solver=cp.CLARABEL)
        assert prob._cache.param_prog is dpp_prog  # fast path reused
        np.testing.assert_allclose(
            prob.value, self._least_squares_baseline(A.value, b.value), rtol=1e-5)

        # 4. back to ignore_dpp with new values: invalidated again.
        A.value = rng.standard_normal((m, n))
        b.value = rng.standard_normal(m)
        prob.solve(solver=cp.CLARABEL, ignore_dpp=True)
        assert _stuffing_backend(prob) == s.DIFFENGINE_CANON_BACKEND
        assert prob._cache.param_prog is None
        np.testing.assert_allclose(
            prob.value, self._least_squares_baseline(A.value, b.value), rtol=1e-5)

    def test_param_free_diffengine_program_cached_and_dropped_on_toggle(self):
        """A parameter-free ignore_dpp solve caches its DiffengineConeProgram
        (no EvalParams in the chain) and reuses it; switching to a default
        solve must drop it."""
        from cvxpy.reductions.dcp2cone.diffengine_cone_program import (
            DiffengineConeProgram,
        )

        y = cp.Variable(3)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(y - 1)), [y >= 0])

        prob.solve(solver=cp.CLARABEL, ignore_dpp=True)
        cached = prob._cache.param_prog
        assert isinstance(cached, DiffengineConeProgram)

        prob.solve(solver=cp.CLARABEL, ignore_dpp=True)
        assert prob._cache.param_prog is cached  # fast path reused
        np.testing.assert_allclose(prob.value, 0.0, atol=1e-6)
        np.testing.assert_allclose(y.value, np.ones(3), atol=1e-5)

        prob.solve(solver=cp.CLARABEL)
        assert not isinstance(prob._cache.param_prog, DiffengineConeProgram)
        np.testing.assert_allclose(prob.value, 0.0, atol=1e-6)

    def test_variable_free_fold_refreshes_between_solves(self):
        """The objective offset from a folded variable-free composite (norm(p))
        must track the current parameter value on re-solves; this is why the
        chain stays uncached (EvalParams) for parametric ignore_dpp problems."""
        p = cp.Parameter(3)
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x - 1) + cp.norm(p)))

        p.value = np.array([3.0, 0.0, 4.0])
        prob.solve(solver=cp.CLARABEL, ignore_dpp=True)
        assert prob._cache.param_prog is None
        np.testing.assert_allclose(prob.value, 5.0, atol=1e-6)

        p.value = np.array([0.0, 0.0, 0.5])
        prob.solve(solver=cp.CLARABEL, ignore_dpp=True)
        np.testing.assert_allclose(prob.value, 0.5, atol=1e-6)

    def test_duals_match_default_path(self):
        """Dual variables must map back through the DIFFENGINE inverse data
        exactly like the default path."""
        rng = np.random.default_rng(0)
        m, n = 4, 6
        A_val = rng.standard_normal((m, n))
        b_val = rng.standard_normal(m)
        c_val = rng.random(n) + 0.5

        def build():
            x = cp.Variable(n)
            return cp.Problem(cp.Minimize(c_val @ x + cp.sum_squares(x)),
                              [A_val @ x == b_val, x >= -5])

        de, base = build(), build()
        de.solve(solver=cp.CLARABEL, ignore_dpp=True)
        base.solve(solver=cp.CLARABEL)

        np.testing.assert_allclose(de.value, base.value, rtol=1e-5)
        for c_de, c_base in zip(de.constraints, base.constraints):
            np.testing.assert_allclose(
                c_de.dual_value, c_base.dual_value, atol=1e-5)

    def test_shared_parameter_across_problems(self):
        """Two problems sharing a Parameter, one on each path, must not
        contaminate each other's caches or values."""
        p = cp.Parameter()
        x = cp.Variable()
        y = cp.Variable()
        prob_de = cp.Problem(cp.Minimize(cp.square(x - p)))
        prob_dpp = cp.Problem(cp.Minimize(cp.square(y - 2 * p)))

        p.value = 1.0
        prob_de.solve(solver=cp.CLARABEL, ignore_dpp=True)
        prob_dpp.solve(solver=cp.CLARABEL)
        np.testing.assert_allclose(x.value, 1.0, atol=1e-6)
        np.testing.assert_allclose(y.value, 2.0, atol=1e-6)

        p.value = -3.0
        prob_dpp.solve(solver=cp.CLARABEL)
        prob_de.solve(solver=cp.CLARABEL, ignore_dpp=True)
        np.testing.assert_allclose(x.value, -3.0, atol=1e-6)
        np.testing.assert_allclose(y.value, -6.0, atol=1e-6)
