import numpy as np
import pytest

import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestBroadcast():

    def test_scalar_to_matrix(self):
        np.random.seed(0)
        x = cp.Variable(name='x')
        A = np.random.randn(200, 6)
        obj = cp.sum(cp.square(x - A))
        problem = cp.Problem(cp.Minimize(obj))

        problem.solve(solver=cp.IPOPT, nlp=True, hessian_approximation='exact',
                    derivative_test='none', verbose=True)
        assert(problem.status == cp.OPTIMAL)
        assert(np.allclose(x.value, np.mean(A)))

    def test_row_broadcast(self):
        np.random.seed(0)
        x = cp.Variable(6, name='x')
        A = np.random.randn(5, 6)
        obj = cp.sum(cp.square(x - A))
        problem = cp.Problem(cp.Minimize(obj))

        problem.solve(solver=cp.IPOPT, nlp=True, hessian_approximation='exact',
                    derivative_test='none', verbose=True)
        assert(problem.status == cp.OPTIMAL)
        assert(np.allclose(x.value, np.mean(A, axis=0)))
                        
    def test_column_broadcast(self):
        np.random.seed(0)
        x = cp.Variable((5, 1), name='x')
        A = np.random.randn(5, 6)
        obj = cp.sum(cp.square(x - A))
        problem = cp.Problem(cp.Minimize(obj))

        problem.solve(solver=cp.IPOPT, nlp=True, hessian_approximation='exact',
                    derivative_test='none', verbose=True)
        assert(problem.status == cp.OPTIMAL)
        assert(np.allclose(x.value.flatten(), np.mean(A, axis=1)))
    
