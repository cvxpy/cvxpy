import numpy as np
import pytest

import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestNonsmoothNontrivial():

    #  convex optimization, huber regression
    def test_huber(self):
        np.random.seed(1)
        n = 100
        SAMPLES = int(1.5 * n)
        beta_true = 5 * np.random.normal(size=(n, 1))
        X = np.random.randn(n, SAMPLES)
        Y = np.zeros((SAMPLES, 1))
        v = np.random.normal(size=(SAMPLES, 1)) 
        TESTS = 5
        p_vals = np.linspace(0, 0.15, num=TESTS)
        for idx, p in enumerate(p_vals):
            # generate the sign changes.
            factor = 2 * np.random.binomial(1, 1 - p, size=(SAMPLES, 1)) - 1
            Y = factor * X.T.dot(beta_true) + v 

            # form problem
            beta = cp.Variable((n, 1))
            cost = cp.sum(cp.huber(X.T @ beta - Y, 1))
            prob = cp.Problem(cp.Minimize(cost))    

            # solve using NLP solver
            prob.solve(nlp=True, solver=cp.IPOPT, verbose=False)
            nlp_value = prob.value  

            # solve using conic solver
            prob.solve()
            conic_value = prob.value    
            assert(np.abs(nlp_value - conic_value) <= 1e-4)

    # convex optimization, sum largest
    def test_sum_largest(self):
        x = cp.Variable(5)
        w = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        # form problem
        k = 2
        cost = cp.sum_largest(cp.multiply(x, w), k)
        prob = cp.Problem(cp.Minimize(cost), [cp.sum(x) == 1, x >= 0])

        # solve using NLP solver
        prob.solve(nlp=True, solver=cp.IPOPT)
        nlp_value = prob.value

        # solve using conic solver
        prob.solve()
        conic_value = prob.value

        assert(np.abs(nlp_value - conic_value) <= 1e-4)

    # convex optimization, sum smallest
    def test_sum_smallest(self):
        x = cp.Variable(5)
        w = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        # form problem
        k = 2
        cost = cp.sum_smallest(cp.multiply(x, w), k)
        prob = cp.Problem(cp.Maximize(cost), [cp.sum(x) == 1, x >= 0])

        # solve using NLP solver
        prob.solve(nlp=True, solver=cp.IPOPT)
        nlp_value = prob.value

        # solve using conic solver
        prob.solve()
        conic_value = prob.value

        assert(np.abs(nlp_value - conic_value) <= 1e-4)

