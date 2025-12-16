import numpy as np
import pytest

import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS

np.random.seed(0)

@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestSharpeRatio():

    def test_formulation_one(self):
        n = 100
        Sigma = np.random.rand(n, n)
        Sigma = Sigma @ Sigma.T  
        mu = np.random.rand(n, )

        x = cp.Variable((n, ), nonneg=True)

        # This type of initialization makes ipopt muich more robust.
        # With no initialization it sometimes fails. Perhaps this is 
        # because we initialize in a very infeasible point?
        x.value = np.ones(n) / n

        obj = cp.square(mu @ x) / cp.quad_form(x, Sigma)
        constraints = [cp.sum(x) == 1]
        problem = cp.Problem(cp.Maximize(obj), constraints)
        problem.solve(solver=cp.IPOPT, nlp=True, verbose=True, hessian_approximation='exact')
        x_noncvx = x.value

        # global solution computed using convex optimization
        obj_convex = cp.quad_form(x, Sigma)
        constraints_convex = [mu @ x == 1]
        problem_convex = cp.Problem(cp.Minimize(obj_convex), constraints_convex)
        problem_convex.solve(solver=cp.CLARABEL, verbose=True)
        x_cvx = x.value / np.sum(x.value)

        sharpe_ratio1 = mu @ x_noncvx / np.sqrt(x_noncvx @ Sigma @ x_noncvx)
        sharpe_ratio2 = mu @ x_cvx / np.sqrt(x_cvx @ Sigma @ x_cvx)
        assert(np.abs(sharpe_ratio1 - sharpe_ratio2) < 1e-6)
