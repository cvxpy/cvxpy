import pytest

import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestHyperbolic():

    def test_sinh(self):
        n = 10
        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.sum(cp.sinh(cp.logistic(x * 2)))),
                           [x >= 0.1, cp.sum(x) == 10])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL

        
    def test_tanh(self):
        n = 10
        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.sum(cp.tanh(cp.logistic(x * 2)))),
                           [x >= 0.1, cp.sum(x) == 10])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL

    def test_asinh(self):
        n = 10
        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.sum(cp.asinh(cp.logistic(x * 3)))),
                           [x >= 0.1, cp.sum(x) == 10])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL

    def test_atanh(self):
        n = 10
        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.sum(cp.atanh(cp.logistic(x * 0.1)))),
                           [x >= 0.1, cp.sum(x) == 10])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL