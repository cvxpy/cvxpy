import numpy as np
import numpy.linalg as LA
import cvxpy as cp


class TestAbs():

    def test_lasso_underdetermined(self):
        np.random.seed(0)
        m, n = 100, 200
        b = np.random.randn(m)
        A = np.random.randn(m, n)
        lmbda_max = 2 * LA.norm(A.T @ b, np.inf)

        for lmbda in np.linspace(lmbda_max, lmbda_max, 20):
            x = cp.Variable((n, ), name='x')
            obj = cp.sum(cp.square((A @ x - b))) + lmbda * cp.sum(cp.abs(x))
            problem = cp.Problem(cp.Minimize(obj))
            problem.solve(solver=cp.CLARABEL, verbose=False)
            obj_star_dcp = obj.value

            x = cp.Variable((n, ), name='x')
            x.value = LA.lstsq(A, b, rcond=None)[0]
            obj = cp.sum(cp.square((A @ x - b))) + lmbda * cp.sum(cp.abs(x))
            problem = cp.Problem(cp.Minimize(obj))
            problem.solve(solver=cp.IPOPT, nlp=True)
            obj_star_nlp = obj.value
            assert(np.abs(obj_star_nlp - obj_star_dcp) / obj_star_nlp <= 1e-4)


    def test_lasso_overdetermined(self):
        np.random.seed(0)
        m, n = 200, 100
        b = np.random.randn(m)
        A = np.random.randn(m, n)
        lmbda_max = 2 * LA.norm(A.T @ b, np.inf)

        for lmbda in np.linspace(lmbda_max, lmbda_max, 20):
            x = cp.Variable((n, ), name='x')
            obj = cp.sum(cp.square((A @ x - b))) + lmbda * cp.sum(cp.abs(x))
            problem = cp.Problem(cp.Minimize(obj))
            problem.solve(solver=cp.CLARABEL)
            obj_star_dcp = obj.value

            x = cp.Variable((n, ), name='x')
            x.value = LA.lstsq(A, b, rcond=None)[0]
            obj = cp.sum(cp.square((A @ x - b))) + lmbda * cp.sum(cp.abs(x))
            problem = cp.Problem(cp.Minimize(obj))
            problem.solve(solver=cp.IPOPT, nlp=True)
            obj_star_nlp = obj.value
            assert(np.abs(obj_star_nlp - obj_star_dcp) / obj_star_nlp <= 1e-4)
