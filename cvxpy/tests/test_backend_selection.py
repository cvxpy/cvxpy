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

    def test_non_dpp_with_large_params_uses_cpp(self):
        """Non-DPP problem with many params should NOT select COO."""
        A = cp.Parameter((100, 20))  # 2000 params
        B = cp.Parameter((20, 20))  # 400 more params
        x = cp.Variable(20)
        # A @ B @ x is not DPP (parameter * parameter in A @ B)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ B @ x)))

        assert not prob.is_dpp()
        assert sum(p.size for p in prob.parameters()) >= s.DPP_PARAM_THRESHOLD

        chain = prob._construct_chain(solver=cp.CLARABEL)
        stuffing = [r for r in chain.reductions if isinstance(r, ConeMatrixStuffing)][0]
        # Non-DPP doesn't trigger COO
        assert stuffing.canon_backend is None

    def test_ignore_dpp_skips_coo_selection(self):
        """With ignore_dpp=True, large DPP should NOT select COO."""
        A = cp.Parameter((100, 20))  # 2000 params
        x = cp.Variable(20)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x)))

        assert prob.is_dpp()

        chain = prob._construct_chain(solver=cp.CLARABEL, ignore_dpp=True)
        stuffing = [r for r in chain.reductions if isinstance(r, ConeMatrixStuffing)][0]
        # ignore_dpp means we don't use DPP-based selection
        assert stuffing.canon_backend is None


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
