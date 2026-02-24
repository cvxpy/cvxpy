import numpy as np
import pytest

import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
from cvxpy.tests.nlp_tests.derivative_checker import DerivativeChecker


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestSumIPOPT:
    """Test solving sum problems with IPOPT."""

    def test_sum_without_axis(self):
        x = cp.Variable((2, 1))
        obj = cp.Minimize((cp.sum(x) - 3)**2)
        constr = [x <= 1]
        prob = cp.Problem(obj, constr)
        prob.solve(solver=cp.IPOPT, nlp=True)
        assert np.allclose(x.value, [[1.0], [1.0]])

        checker = DerivativeChecker(prob)
        checker.run_and_assert()
        

    def test_sum_with_axis(self):
        """Test sum with axis parameter."""
        X = cp.Variable((2, 3))
        obj = cp.Minimize(cp.sum((cp.sum(X, axis=1) - 4)**2))
        constr = [X >= 0, X <= 1]
        prob = cp.Problem(obj, constr)
        prob.solve(solver=cp.IPOPT, nlp=True)
        expected = np.full((2, 3), 1)
        assert np.allclose(X.value, expected)

        checker = DerivativeChecker(prob)
        checker.run_and_assert()
    
    def test_two_sum_with_axis(self):
        """Test sum with axis parameter."""
        np.random.seed(0)
        X = cp.Variable((2, 3))
        A = np.random.rand(4, 2)
        obj = cp.Minimize(cp.prod(cp.sum(A @ X, axis=1)))
        constr = [X >= 0, X <= 1]
        prob = cp.Problem(obj, constr)
        prob.solve(solver=cp.IPOPT, nlp=True)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_sum_with_other_axis(self):
        """Test sum with axis parameter."""
        X = cp.Variable((2, 3))
        obj = cp.Minimize(cp.sum((cp.sum(X, axis=0) - 4)**2))
        constr = [X >= 0, X <= 1]
        prob = cp.Problem(obj, constr)
        prob.solve(solver=cp.IPOPT, nlp=True)
        expected = np.full((2, 3), 1)
        assert np.allclose(X.value, expected)

        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_two_sum_with_other_axis(self):
        """Test sum with axis parameter."""
        np.random.seed(0)
        X = cp.Variable((2, 3))
        A = np.random.rand(4, 2)
        obj = cp.Minimize(cp.prod(cp.sum(A @ X, axis=0)))
        constr = [X >= 0, X <= 1]
        prob = cp.Problem(obj, constr)
        prob.solve(solver=cp.IPOPT, nlp=True)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_sum_matrix_arg(self):
        np.random.seed(0)
        n, m, k = 40, 20, 4
        A = np.random.rand(n, k) @ np.random.rand(k, m) 
        T = cp.Variable((n, m), name='T')
        obj = cp.sum(cp.multiply(A, T))
        constraints = [T >= 1, T <= 2]
        problem = cp.Problem(cp.Minimize(obj), constraints)
        problem.solve(solver=cp.IPOPT, nlp=True, verbose=True, derivative_test='none')
        assert(np.allclose(T.value, 1))
        assert problem.status == cp.OPTIMAL

        checker = DerivativeChecker(problem)
        checker.run_and_assert()