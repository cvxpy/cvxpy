import numpy as np
import pytest

import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestMatmul():

    def test_simple_matmul_graph_form(self):
        np.random.seed(0)
        m, n, p = 5, 7, 11
        X = cp.Variable((m, n), bounds=[-1, 1], name='X')
        Y = cp.Variable((n, p), bounds=[-2, 2], name='Y')
        t = cp.Variable(name='t')
        X.value = np.random.rand(m, n)
        Y.value = np.random.rand(n, p)
        constraints = [t == cp.sum(cp.matmul(X, Y))]
        problem = cp.Problem(cp.Minimize(t), constraints)

        problem.solve(solver=cp.IPOPT, nlp=True, hessian_approximation='exact',
                    derivative_test='none', verbose=False)
        assert(problem.status == cp.OPTIMAL)
        
    def test_simple_matmul_not_graph_form(self):
        np.random.seed(0)
        m, n, p = 5, 7, 11
        X = cp.Variable((m, n), bounds=[-1, 1], name='X')
        Y = cp.Variable((n, p), bounds=[-2, 2], name='Y')
        X.value = np.random.rand(m, n)
        Y.value = np.random.rand(n, p)
        obj = cp.sum(cp.matmul(X, Y))
        problem = cp.Problem(cp.Minimize(obj))

        problem.solve(solver=cp.IPOPT, nlp=True, hessian_approximation='exact',
                    derivative_test='none', verbose=False)
        assert(problem.status == cp.OPTIMAL)

    def test_matmul_with_function_right(self):
        np.random.seed(0)
        m, n, p = 5, 7, 11
        X = np.random.rand(m, n)
        Y = cp.Variable((n, p), bounds=[-2, 2], name='Y')
        Y.value = np.random.rand(n, p)
        obj = cp.sum(cp.matmul(X, cp.cos(Y)))
        problem = cp.Problem(cp.Minimize(obj))

        problem.solve(solver=cp.IPOPT, nlp=True, hessian_approximation='exact',
                    derivative_test='none', verbose=True)
        assert(problem.status == cp.OPTIMAL)

    def test_matmul_with_function_left(self):
        np.random.seed(0)
        m, n, p = 5, 7, 11
        X = cp.Variable((m, n), bounds=[-2, 2], name='X')
        Y = np.random.rand(n, p)
        X.value = np.random.rand(m, n)
        obj = cp.sum(cp.matmul(cp.cos(X), Y))
        problem = cp.Problem(cp.Minimize(obj))

        problem.solve(solver=cp.IPOPT, nlp=True, hessian_approximation='exact',
                    derivative_test='second-order', verbose=True)
        assert(problem.status == cp.OPTIMAL)

    def test_matmul_with_functions_both_sides(self):
        np.random.seed(0)
        m, n, p = 5, 7, 11
        X = cp.Variable((m, n), bounds=[-2, 2], name='X')
        Y = cp.Variable((n, p), bounds=[-2, 2], name='Y')
        X.value = np.random.rand(m, n)
        Y.value = np.random.rand(n, p)
        obj = cp.sum(cp.matmul(cp.cos(X), cp.sin(Y)))
        problem = cp.Problem(cp.Minimize(obj))

        problem.solve(solver=cp.IPOPT, nlp=True, hessian_approximation='exact',
                    derivative_test='second-order', verbose=True)
        assert(problem.status == cp.OPTIMAL)