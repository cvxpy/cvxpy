"""
Copyright 2013 Steven Diamond

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

import warnings

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose, assert_equal
from scipy.linalg import ldl

import cvxpy as cp
from cvxpy.atoms.quad_form import decomp_quad
from cvxpy.settings import EIGVAL_TOL
from cvxpy.tests.base_test import BaseTest


class TestDecompQuad(BaseTest):
    """Tests for the decomp_quad helper function."""

    @staticmethod
    def _check(P) -> None:
        """Assert P = scale * (M1 @ M1.T - M2 @ M2.T) within tolerance."""
        scale, M1, M2 = decomp_quad(P)
        P_dense = P.toarray() if sp.issparse(P) else np.asarray(P)
        pos = scale * (M1 @ M1.conj().T) if M1.size else np.zeros_like(P_dense)
        neg = scale * (M2 @ M2.conj().T) if M2.size else np.zeros_like(P_dense)
        assert_allclose(P_dense, pos - neg, atol=1e-10)

    def test_psd_nsd(self) -> None:
        rng = np.random.default_rng(0)
        for n in (2, 5, 20):
            A = rng.standard_normal((n, n))
            self._check(A @ A.T)           # PSD, full rank
            self._check(-(A @ A.T))        # NSD
        A = rng.standard_normal((10, 7))
        self._check(A @ A.T)               # PSD, rank-deficient

    def test_indefinite(self) -> None:
        rng = np.random.default_rng(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for n in (3, 10, 30):
                A = rng.standard_normal((n, n))
                P = A + A.T
                # Verify LDL uses nontrivial permutation
                _, _, perm = ldl(P)
                assert not np.array_equal(perm, np.arange(n))
                self._check(P)

    def test_2x2_blocks(self) -> None:
        """Zero diagonal forces Bunch-Kaufman 2x2 pivots."""
        rng = np.random.default_rng(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for n in (6, 10, 30):
                A = rng.standard_normal((n, n))
                P = A + A.T
                np.fill_diagonal(P, 0.0)
                _, d, _ = ldl(P)
                assert np.any(np.diag(d, -1) != 0)
                self._check(P)

    def test_special(self) -> None:
        scale, M1, M2 = decomp_quad(np.zeros((4, 4)))
        assert_equal(scale, 0.0)
        assert_equal(M1.size, 0)
        assert_equal(M2.size, 0)
        self._check(np.diag([3.0, 0.0, 5.0, 0.0]))

    def test_complex_hermitian(self) -> None:
        rng = np.random.default_rng(0)
        A = rng.standard_normal((5, 5)) + 1j * rng.standard_normal((5, 5))
        self._check(A @ A.conj().T)        # PSD
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._check(A + A.conj().T)    # indefinite

    def test_sparse(self) -> None:
        self._check(sp.eye_array(5, format="csc") * 3.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._check(sp.diags_array([1., -1, 2, -2, 3], format="csc"))

    def test_sparse_nsd(self) -> None:
        """Regression test: sparse NSD must use row permutation L[p,:], not column L[:,p].

        For NSD matrices, decomp_quad calls sparse_cholesky(-P) which returns (L, p) with
        L[p,:] @ L[p,:].T == -P.  The fix is M2 = L[p,:] (row perm), not L[:,p] (col perm).
        A tridiagonal matrix produces a non-trivial AMD ordering that exposes the bug.
        """
        n = 6
        off = np.ones(n - 1)
        # Negative discrete Laplacian: NSD, tridiagonal, AMD gives non-identity permutation.
        P_nsd = sp.diags([-4 * np.ones(n), off, off], [0, -1, 1], format="csc")
        self._check(P_nsd)


class TestNonOptimal(BaseTest):
    def test_singular_quad_form(self) -> None:
        """Test quad form with a singular matrix.
        """
        # Solve a quadratic program.
        np.random.seed(1234)
        for n in (3, 4, 5):
            for i in range(5):

                # construct a random 1d finite distribution
                v = np.exp(np.random.randn(n))
                v = v / np.sum(v)

                # construct a random positive definite matrix
                A = np.random.randn(n, n)
                Q = np.dot(A, A.T)

                # Project onto the orthogonal complement of v.
                # This turns Q into a singular matrix with a known nullspace.
                E = np.identity(n) - np.outer(v, v) / np.inner(v, v)
                Q = np.dot(E, np.dot(Q, E.T))
                observed_rank = np.linalg.matrix_rank(Q)
                desired_rank = n-1
                assert_equal(observed_rank, desired_rank)

                for action in 'minimize', 'maximize':

                    # Look for the extremum of the quadratic form
                    # under the simplex constraint.
                    x = cp.Variable(n)
                    if action == 'minimize':
                        q = cp.quad_form(x, Q)
                        objective = cp.Minimize(q)
                    elif action == 'maximize':
                        q = cp.quad_form(x, -Q)
                        objective = cp.Maximize(q)
                    constraints = [0 <= x, cp.sum(x) == 1]
                    p = cp.Problem(objective, constraints)
                    p.solve(solver=cp.OSQP)

                    # check that cvxpy found the right answer
                    xopt = x.value.flatten()
                    yopt = np.dot(xopt, np.dot(Q, xopt))
                    assert_allclose(yopt, 0, atol=1e-3)
                    assert_allclose(xopt, v, atol=1e-3)

    def test_sparse_quad_form(self) -> None:
        """Test quad form with a sparse matrix.
        """
        Q = sp.eye_array(2)
        x = cp.Variable(2)
        cost = cp.quad_form(x, Q)
        prob = cp.Problem(cp.Minimize(cost), [x == [1, 2]])
        self.assertAlmostEqual(prob.solve(solver=cp.OSQP), 5)

        # Here are our QP factors
        A = cp.Constant(sp.eye_array(4))
        c = np.ones(4).reshape((1, 4))

        # Here is our optimization variable
        x = cp.Variable(4)

        # And the QP problem setup
        function = cp.quad_form(x, A) - cp.matmul(c, x)
        objective = cp.Minimize(function)
        problem = cp.Problem(objective)

        problem.solve(solver=cp.OSQP)
        self.assertEqual(len(function.value), 1)

    def test_param_quad_form(self) -> None:
        """Test quad form with a parameter.
        """
        P = cp.Parameter((2, 2), PSD=True)
        Q = np.eye(2)
        x = cp.Variable(2)
        cost = cp.quad_form(x, P)
        P.value = Q
        prob = cp.Problem(cp.Minimize(cost), [x == [1, 2]])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.assertAlmostEqual(prob.solve(solver=cp.SCS), 5)

    def test_non_symmetric(self) -> None:
        """Test when P is constant and not symmetric.
        """
        P = np.array([[2, 2], [3, 4]])
        x = cp.Variable(2)
        with self.assertRaises(Exception) as cm:
            cp.quad_form(x, P)
        self.assertTrue("Quadratic form matrices must be symmetric/Hermitian."
                        in str(cm.exception))

    def test_non_psd(self) -> None:
        """Test error when P is symmetric but not definite.
        """
        P = np.array([[1, 0], [0, -1]])
        x = cp.Variable(2)
        # Forming quad_form is okay
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cost = cp.quad_form(x, P)
        prob = cp.Problem(cp.Minimize(cost), [x == [1, 2]])
        with self.assertRaises(Exception) as cm:
            prob.solve(solver=cp.SCS)
        self.assertTrue("Problem does not follow DCP rules."
                        in str(cm.exception))

    def test_psd_exactly_tolerance(self) -> None:
        """Test that PSD check when eigenvalue is exactly -EIGVAL_TOL
        """
        P = np.array([[-0.999*EIGVAL_TOL, 0], [0, 10]])
        x = cp.Variable(2)
        # Forming quad_form is okay
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cost = cp.quad_form(x, P)
            prob = cp.Problem(cp.Minimize(cost), [x == [1, 2]])
            prob.solve(solver=cp.SCS)

    def test_nsd_exactly_tolerance(self) -> None:
        """Test that NSD check when eigenvalue is exactly EIGVAL_TOL
        """
        P = np.array([[0.999*EIGVAL_TOL, 0], [0, -10]])
        x = cp.Variable(2)
        # Forming quad_form is okay
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cost = cp.quad_form(x, P)
            prob = cp.Problem(cp.Maximize(cost), [x == [1, 2]])
            prob.solve(solver=cp.SCS)

    def test_obj_eval(self) -> None:
        """Test case where objective evaluation differs from result.
        """
        x = cp.Variable((2, 1))
        A = np.array([[1.0]])
        B = np.array([[1.0, 1.0]]).T
        obj0 = -B.T @ x
        obj1 = cp.quad_form(B.T @ x, A)
        prob = cp.Problem(cp.Minimize(obj0 + obj1))
        prob.solve(solver=cp.SCS)
        self.assertAlmostEqual(prob.value, prob.objective.value)

    def test_zero_term(self) -> None:
        """Test a quad form multiplied by zero.
        """
        data_norm = np.random.random(5)
        M = np.random.random(5*2).reshape((5, 2))
        c = cp.Variable(M.shape[1])
        lopt = 0
        laplacian_matrix = np.ones((2, 2))
        design_matrix = cp.Constant(M)
        objective = cp.Minimize(
            cp.sum_squares(design_matrix @ c - data_norm) +
            lopt * cp.quad_form(c, laplacian_matrix)
        )
        constraints = [(M[0] @ c) == 1]  # (K * c) >= -0.1]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS)

    def test_zero_matrix(self) -> None:
        """Test quad_form with P = 0.
        """
        x = cp.Variable(3)
        A = np.eye(3)
        b = np.ones(3,)
        c = -np.ones(3,)
        P = np.zeros((3, 3))
        expr = (1/2) * cp.quad_form(x, P) + c.T @ x
        prob = cp.Problem(cp.Minimize(expr),
                          [A @ x <= b])
        prob.solve(solver=cp.SCS)

    def test_assume_psd(self) -> None:
        """Test assume_PSD argument.
        """
        x = cp.Variable(3)
        A = np.eye(3)
        expr = cp.quad_form(x, A, assume_PSD=True)
        assert expr.is_convex()

        A = -np.eye(3)
        expr = cp.quad_form(x, A, assume_PSD=True)
        assert expr.is_convex()

        prob = cp.Problem(cp.Minimize(expr))
        # Transform to a SolverError.
        with pytest.raises(cp.SolverError):
            prob.solve(solver=cp.OSQP)

    def test_indefinite_assume_psd_raises(self) -> None:
        """quad_form_canon must raise ValueError for indefinite P with assume_PSD=True.

        In CVXPY 1.8.1 and earlier, the M1 (positive-eigenvalue) term was silently dropped.
        For P=diag(2,-1) and x=[1,2], the true x^T P x = -2, but the old
        code returned -4 with OPTIMAL status.
        """
        P = np.array([[2.0, 0.0], [0.0, -1.0]])  # indefinite: eigenvalues 2 and -1
        x = cp.Variable(2)
        expr = cp.quad_form(x, P, assume_PSD=True)
        prob = cp.Problem(cp.Minimize(expr), [x == [1.0, 2.0]])
        # use_quad_obj=False forces the conic canonicalization path where
        # quad_form_canon is called (vs. the QP path used by SCS >= 3.0).
        with pytest.raises(ValueError) as exc_info:
            prob.solve(solver=cp.SCS, use_quad_obj=False)
        assert "indefinite" in str(exc_info.value)
        assert "PSD" in str(exc_info.value)
