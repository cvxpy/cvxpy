import numpy as np
import numpy.linalg as LA
import pytest

import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestAbs():

    def test_lasso_square_small(self):
        np.random.seed(0)
        m, n = 10, 10
        factors = np.linspace(0.1, 1, 20)

        for factor in factors:
            b = np.random.randn(m)
            A = np.random.randn(m, n)
            lmbda_max = 2 * LA.norm(A.T @ b, np.inf)
            lmbda = factor * lmbda_max

            x = cp.Variable((n, ), name='x')
            obj = cp.sum(cp.square((A @ x - b))) + lmbda * cp.sum(cp.abs(x))
            problem = cp.Problem(cp.Minimize(obj))
            problem.solve(solver=cp.CLARABEL)
            obj_star_dcp = obj.value

            x = cp.Variable((n, ), name='x')
            obj = cp.sum(cp.square((A @ x - b))) + lmbda * cp.sum(cp.abs(x))
            problem = cp.Problem(cp.Minimize(obj))
            problem.solve(solver=cp.IPOPT, nlp=True, hessian_approximation='exact',
                            derivative_test='none', verbose=False)
            obj_star_nlp = obj.value
            assert(np.abs(obj_star_nlp - obj_star_dcp) / obj_star_nlp <= 1e-4)

    @pytest.mark.skip(reason="Fails sometimes, needs investigation.")
    def test_lasso_square(self):
        np.random.seed(0)
        m, n = 50, 50
        factors = np.linspace(0.1, 1, 20)

        for factor in factors:
            b = np.random.randn(m)
            A = np.random.randn(m, n)
            lmbda_max = 2 * LA.norm(A.T @ b, np.inf)
            lmbda = factor * lmbda_max

            x = cp.Variable((n, ), name='x')
            obj = cp.sum(cp.square((A @ x - b))) + lmbda * cp.sum(cp.abs(x))
            problem = cp.Problem(cp.Minimize(obj))
            problem.solve(solver=cp.CLARABEL)
            obj_star_dcp = obj.value

            x = cp.Variable((n, ), name='x')
            obj = cp.sum(cp.square((A @ x - b))) + lmbda * cp.sum(cp.abs(x))
            problem = cp.Problem(cp.Minimize(obj))
            problem.solve(solver=cp.IPOPT, nlp=True, hessian_approximation='exact',
                            derivative_test='none', verbose=False)
            obj_star_nlp = obj.value
            assert(np.abs(obj_star_nlp - obj_star_dcp) / obj_star_nlp <= 1e-4)

    def test_lasso_underdetermined(self):
        np.random.seed(0)
        m, n = 100, 200
        factors = np.linspace(0.1, 1, 20)

        for factor in factors:
            b = np.random.randn(m)
            A = np.random.randn(m, n)
            lmbda_max = 2 * LA.norm(A.T @ b, np.inf)
            lmbda = factor * lmbda_max

            x = cp.Variable((n, ), name='x')
            obj = cp.sum(cp.square((A @ x - b))) + lmbda * cp.sum(cp.abs(x))
            problem = cp.Problem(cp.Minimize(obj))
            problem.solve(solver=cp.CLARABEL)
            obj_star_dcp = obj.value

            x = cp.Variable((n, ), name='x')
            obj = cp.sum(cp.square((A @ x - b))) + lmbda * cp.sum(cp.abs(x))
            problem = cp.Problem(cp.Minimize(obj))
            problem.solve(solver=cp.IPOPT, nlp=True, hessian_approximation='exact',
                            derivative_test='none', verbose=False)
            obj_star_nlp = obj.value
            assert(np.abs(obj_star_nlp - obj_star_dcp) / obj_star_nlp <= 1e-4)


    def test_lasso_overdetermined(self):
        np.random.seed(0)
        m, n = 200, 100
        factors = np.linspace(0.1, 1, 20)

        for factor in factors:
            b = np.random.randn(m)
            A = np.random.randn(m, n)
            lmbda_max = 2 * LA.norm(A.T @ b, np.inf)
            lmbda = factor * lmbda_max

            x = cp.Variable((n, ), name='x')
            obj = cp.sum(cp.square((A @ x - b))) + lmbda * cp.sum(cp.abs(x))
            problem = cp.Problem(cp.Minimize(obj))
            problem.solve(solver=cp.CLARABEL)
            obj_star_dcp = obj.value

            x = cp.Variable((n, ), name='x')
            obj = cp.sum(cp.square((A @ x - b))) + lmbda * cp.sum(cp.abs(x))
            problem = cp.Problem(cp.Minimize(obj))
            problem.solve(solver=cp.IPOPT, nlp=True, hessian_approximation='exact',
                            derivative_test='none', verbose=False)
            obj_star_nlp = obj.value
            assert(np.abs(obj_star_nlp - obj_star_dcp) / obj_star_nlp <= 1e-4)
