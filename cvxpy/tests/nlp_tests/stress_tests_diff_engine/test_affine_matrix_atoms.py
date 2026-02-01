
import numpy as np
import pytest

import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
from cvxpy.reductions.solvers.nlp_solvers.nlp_solver import DerivativeChecker


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestAffineMatrixAtomsDiffEngine:
    # Stress tests for affine matrix atoms in the diff engine.		
    
    def test_one_trace(self):
        np.random.seed(0)
        X = cp.Variable((10, 10))
        A = np.random.rand(10, 10)
        obj = cp.Minimize(cp.Trace(cp.log(A@ X)))
        constr = [X >= 0.5, X <= 1]
        prob = cp.Problem(obj, constr)
        prob.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_two_trace(self):
        np.random.seed(0)
        Y = cp.Variable((15, 5), bounds=[0.5, 1])
        A = np.random.rand(5, 15)
        obj = cp.Minimize(cp.Trace(A @ Y))
        constr =[]
        prob = cp.Problem(obj, constr)
        prob.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_three_trace(self):
        np.random.seed(0)
        X = cp.Variable((20, 20), bounds=[0.5, 1])
        Y = cp.Variable((20, 20), bounds=[0, 1])
        A = np.random.rand(20, 20)
        obj = cp.Minimize(cp.Trace(cp.log(A @ X) + X @ Y))
        prob = cp.Problem(obj)
        prob.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_one_transpose(self):
        np.random.seed(0)
        n = 10
        k = 3 
        A = np.random.rand(n, k)
        X = cp.Variable((n, k), bounds = [1, 5])
        obj = cp.sum(A @ cp.transpose(cp.log(X)))
        prob = cp.Problem(cp.Minimize(obj))
        prob.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()
    
    def test_two_transpose(self):
        np.random.seed(0)
        n = 10
        A = np.random.rand(n, n)
        X = cp.Variable((n, n), bounds = [0.5, 5])
        obj = cp.sum(A @ (cp.log(X).T + cp.exp(X)))
        constraints = [cp.sum((A @ X).T) == np.sum(A @ np.ones((n, n)))]
        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()
    
    def test_three_transpose(self):
        np.random.seed(0)
        n = 10
        A = np.random.rand(n, n)
        X = cp.Variable((n, n), bounds = [0.5, 5])
        obj = cp.sum(A @ (cp.log(X).T + cp.exp(X).T))
        constraints = [cp.sum((A @ X).T.T) == np.sum(A @ np.ones((n, n)))]
        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_four_transpose(self):
        np.random.seed(0)
        n, k = 10, 5
        A = np.random.randn(n, n)
        A = A + A.T
        V = cp.Variable((n, k))
        constraints = [V.T @ V == np.eye(k)]

        # Get eigenvectors for proper initialization (eigh returns sorted ascending)
        eigvals, eigvecs = np.linalg.eigh(A)

        # find k smallest eigenvalues - initialize with k smallest eigenvectors
        obj = cp.Minimize(cp.Trace(V.T @ A @ V))
        prob = cp.Problem(obj, constraints)
        V.value = eigvecs[:, :k]  # smallest k eigenvectors
        prob.solve(solver=cp.IPOPT, nlp=True, least_square_init_duals='no')
        checker = DerivativeChecker(prob)
        checker.run_and_assert()
        assert np.allclose(prob.value, np.sum(eigvals[:k]))

        # find k largest eigenvalues - initialize with k largest eigenvectors
        obj = cp.Maximize(cp.Trace(V.T @ A @ V))
        prob = cp.Problem(obj, constraints)
        V.value = eigvecs[:, -k:]  # largest k eigenvectors
        prob.solve(solver=cp.IPOPT, nlp=True, least_square_init_duals='no')
        checker = DerivativeChecker(prob)
        checker.run_and_assert()
        assert np.allclose(prob.value, np.sum(eigvals[-k:]))

    def test_one_diag_vec(self):
        np.random.seed(0)
        n = 5
        x = cp.Variable(n, bounds=[0.5, 2])
        A = np.random.rand(n, n)
        # diag(x) creates diagonal matrix from vector x
        obj = cp.Minimize(cp.sum(A @ cp.diag(cp.log(x))))
        prob = cp.Problem(obj)
        prob.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_two_diag_vec(self):
        np.random.seed(0)
        n = 8
        x = cp.Variable(n, bounds=[1, 3])
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        # Trace of product with diagonal matrix
        obj = cp.Minimize(cp.Trace(A @ cp.diag(cp.exp(x)) @ B))
        prob = cp.Problem(obj)
        prob.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_three_diag_vec(self):
        np.random.seed(0)
        n = 6
        x = cp.Variable(n, bounds=[0.5, 2])
        y = cp.Variable(n, bounds=[0.5, 2])
        A = np.random.rand(n, n)
        # Two diagonal matrices in expression
        obj = cp.Minimize(cp.sum(cp.diag(x) @ A @ cp.diag(y)))
        prob = cp.Problem(obj)
        prob.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

