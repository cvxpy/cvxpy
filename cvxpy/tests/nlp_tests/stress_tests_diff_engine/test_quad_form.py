import numpy as np
import pytest
import scipy.sparse as sp

import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestQuadFormDifferentFormats:
    
    def test_quad_form_dense_sparse_sparse(self):
        # Generate a random non-trivial quadratic program.
        m = 15
        n = 10
        p = 5
        np.random.seed(1)
        P = np.random.randn(n, n)
        P = P.T @ P
        q = np.random.randn(n)
        G = np.random.randn(m, n)
        h = G @ np.random.randn(n, 1)
        A = np.random.randn(p, n)
        b = np.random.randn(p, 1)
        x = cp.Variable((n, 1))

        constraints = [G @ x <= h,
                       A @ x == b]

        # dense problem
        x.value = None
        prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P) + q.T @ x),
                        constraints)
        prob.solve(nlp=True, verbose=False)
        dense_val = x.value

        # CSR problem
        x.value = None
        P_csr = sp.csr_matrix(P)
        prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P_csr) + q.T @ x),
                        constraints)
        prob.solve(nlp=True, verbose=False)
        csr_val = x.value

        # CSC problem
        x.value = None
        P_csc = sp.csc_matrix(P)
        prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P_csc) + q.T @ x),
                        constraints)
        prob.solve(nlp=True, verbose=False)
        csc_val = x.value

        assert np.allclose(dense_val, csr_val)
        assert np.allclose(dense_val, csc_val)

    def test_quad_form_dense_sparse_sparse_different_x(self):
        # Generate a random non-trivial quadratic program.
        m = 15
        n = 10
        p = 5
        np.random.seed(1)
        P = np.random.randn(n, n)
        P = P.T @ P
        q = np.random.randn(n)
        G = np.random.randn(m, n)
        h = G @ np.random.randn(n)
        A = np.random.randn(p, n)
        b = np.random.randn(p)
        x = cp.Variable(n)

        constraints = [G @ x <= h,
                       A @ x == b]

        # dense problem
        x.value = None
        prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P) + q.T @ x),
                        constraints)
        prob.solve(nlp=True, verbose=False)
        dense_val = x.value

        # CSR problem
        x.value = None
        P_csr = sp.csr_matrix(P)
        prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P_csr) + q.T @ x),
                        constraints)
        prob.solve(nlp=True, verbose=False)
        csr_val = x.value

        # CSC problem
        x.value = None
        P_csc = sp.csc_matrix(P)
        prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P_csc) + q.T @ x),
                        constraints)
        prob.solve(nlp=True, verbose=False)
        csc_val = x.value

        assert np.allclose(dense_val, csr_val)
        assert np.allclose(dense_val, csc_val)