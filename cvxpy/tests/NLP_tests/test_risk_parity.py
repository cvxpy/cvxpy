import numpy as np
import numpy.linalg as LA
import pytest

import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS

np.random.seed(0)

@pytest.fixture
def Sigma():
    return 1e-5 * np.array([
        [41.16, 22.03, 18.64, -4.74,  6.27, 10.1 , 14.52,  3.18],
        [22.03, 58.57, 32.92, -5.04,  4.02,  3.7 , 26.76,  2.17],
        [18.64, 32.92, 81.02,  0.53,  6.05,  2.02, 25.52,  1.56],
        [-4.74, -5.04,  0.53, 20.6 ,  2.52,  0.57,  0.2 ,  3.6 ],
        [6.27,  4.02,  6.05,  2.52, 10.13,  2.59,  4.32,  3.13],
        [10.1 ,  3.7 ,  2.02,  0.57,  2.59, 22.89,  3.97,  3.26],
        [14.52, 26.76, 25.52,  0.2 ,  4.32,  3.97, 29.91,  3.25],
        [3.18,  2.17,  1.56,  3.6 ,  3.13,  3.26,  3.25, 13.63]
    ])

@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestRiskParity:
    def test_vanilla_risk_parity_formulation_one(self, Sigma):
        n = 8
        risk_target = (1 / n) * np.ones(n)

        w = cp.Variable((n,), nonneg=True, name='w')
        t = cp.Variable((n,), name='t')
        constraints = [cp.sum(w) == 1, t == Sigma @ w]

        term1 = cp.sum(cp.multiply(cp.square(w), cp.square(t))) / cp.quad_form(w, Sigma)
        term2 = (LA.norm(risk_target) ** 2) * cp.quad_form(w, Sigma)
        term3 = - 2 * cp.sum(cp.multiply(risk_target, cp.multiply(w, t)))
        obj = cp.Minimize(term1 + term2 + term3)
        problem = cp.Problem(obj, constraints)
        problem.solve(solver=cp.IPOPT, nlp=True, verbose=True, derivative_test='none')

        risk_contributions = w.value * (Sigma @ w.value)
        risk_contributions /= np.sum(risk_contributions)
        assert np.linalg.norm(risk_contributions - risk_target) < 1e-5
    
    # we do not expand the objective, and use square roots
    def test_vanilla_risk_parity_formulation_two(self):
        pass 

    # we expand the objective manually to get rid of the square root
    def test_group_risk_parity_formulation_one(self, Sigma):
        n = 8
        b = np.array([0.4, 0.6])

        w = cp.Variable((n, ), nonneg=True, name='w')
        t = cp.Variable((n, ), name='t')
        constraints = [cp.sum(w) == 1, t == Sigma @ w]
        w.value = np.ones(n) / n
        groups = [[0, 1, 5], [3, 4, 2, 6, 7]]

        term1 = 0
        term2 = 0
        term3 = 0

        for k, g in enumerate(groups):
            term1 += cp.square(cp.sum(cp.multiply(w[g], t[g]))) /  cp.quad_form(w, Sigma)
            term2 += (LA.norm(b[k]) ** 2) * cp.quad_form(w, Sigma)
            term3 += - 2 * b[k] * cp.sum(cp.multiply(w[g], t[g]))

        obj = cp.Minimize(term1 + term2 + term3)
        problem = cp.Problem(obj, constraints)
        problem.solve(solver=cp.IPOPT, nlp=True, verbose=True, derivative_test='none')
        risk_contributions = w.value * (Sigma @ w.value)
        risk_contributions /= np.sum(risk_contributions)
        risk_contributions = np.array([np.sum(risk_contributions[g]) for g in groups])
        assert np.linalg.norm(risk_contributions - b) < 1e-5

    # other formulation
    def test_group_risk_parity_formulation_two(self, Sigma):
        n = 8
        b = np.array([0.4, 0.6])
        
        w = cp.Variable((n, ), nonneg=True, name='w')
        t = cp.Variable((n, ), name='t')
        constraints = [cp.sum(w) == 1, t == Sigma @ w]
        w.value = np.ones(n) / n
        groups = [[0, 1, 5], [3, 4, 2, 6, 7]]

        obj = 0
        for k, g in enumerate(groups):
            obj += cp.square(cp.sum(cp.multiply(w[g], t[g])) / cp.quad_form(w, Sigma) - b[k])

        problem = cp.Problem(cp.Minimize(obj), constraints)
        problem.solve(solver=cp.IPOPT, nlp=True, verbose=True, derivative_test='none')

        risk_contributions = w.value * (Sigma @ w.value)
        risk_contributions /= np.sum(risk_contributions)
        risk_contributions = np.array([np.sum(risk_contributions[g]) for g in groups])
        assert np.linalg.norm(risk_contributions - b) < 1e-5