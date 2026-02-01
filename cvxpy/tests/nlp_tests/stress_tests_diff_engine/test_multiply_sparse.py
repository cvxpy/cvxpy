import numpy as np
import pytest
import scipy.sparse as sp

import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
from cvxpy.reductions.solvers.nlp_solvers.nlp_solver import DerivativeChecker


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestMultiplyDifferentFormats:
    
    def test_dense_sparse_sparse(self):
        np.random.seed(0)
        n = 5
       
        # dense
        x = cp.Variable((n, n), bounds=[-2, 2])
        A = np.random.rand(n, n) - 0.5
        obj = cp.Minimize(cp.sum(cp.multiply(A, x)))
        prob = cp.Problem(obj)
        x.value = np.random.rand(n, n)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()
        prob.solve(nlp=True, verbose=False)
        assert np.allclose(x.value[(A > 0)], -2)
        assert np.allclose(x.value[(A < 0)], 2)

        # CSR
        x = cp.Variable((n, n), bounds=[-2, 2])
        A = sp.csr_matrix(A)
        obj = cp.Minimize(cp.sum(cp.multiply(A, x)))
        prob = cp.Problem(obj)
        x.value = np.random.rand(n, n)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()
        prob.solve(nlp=True, verbose=False)
        assert np.allclose(x.value[(A > 0).todense()], -2)
        assert np.allclose(x.value[(A < 0).todense()], 2)

        # CSC
        x = cp.Variable((n, n), bounds=[-2, 2])
        A = sp.csc_matrix(A)
        obj = cp.Minimize(cp.sum(cp.multiply(A, x)))
        prob = cp.Problem(obj)
        x.value = np.random.rand(n, n)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()
        prob.solve(nlp=True, verbose=False)
        assert np.allclose(x.value[(A > 0).todense()], -2)
        assert np.allclose(x.value[(A < 0).todense()], 2)
            