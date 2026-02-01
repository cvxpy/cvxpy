import numpy as np
import pytest

import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
from cvxpy.reductions.solvers.nlp_solvers.nlp_solver import DerivativeChecker


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestLogSumExp():

    def test_one(self):
        x = cp.Variable(3, name='x')
        obj = cp.Minimize(cp.log_sum_exp(x))
        constraints = [x >= 1]
        prob = cp.Problem(obj, constraints)
        prob.solve(nlp=True, verbose=True, derivative_test='none')
        expected = np.log(3 * np.exp(1))
        assert np.isclose(obj.value, expected)

        checker = DerivativeChecker(prob)
        checker.run_and_assert()
       
    def test_two(self):
        m = 50
        n = 10
        A = np.random.randn(m, n)
        x = cp.Variable(n)
        obj = cp.Minimize(cp.log_sum_exp(A @ x))
        constraints = [x >= 0, cp.sum(x) == 1]
        prob = cp.Problem(obj, constraints)
        prob.solve(nlp=True, verbose=True, derivative_test='second-order')
        DNLP_opt_val = obj.value
        prob.solve(solver=cp.CLARABEL, verbose=True)
        DCP_opt_val = obj.value
        assert np.isclose(DNLP_opt_val, DCP_opt_val)

        x.value = x.value + 0.1  # perturb to avoid boundary issues
        checker = DerivativeChecker(prob)
        checker.run_and_assert()
    
    @pytest.mark.parametrize(
    "m, n",
    [(50, 25), (300, 100)]
    )
    def test_three(self, m, n):
        A = np.random.randn(m, n)
        x = cp.Variable(n)
        y = cp.Variable(n)
        obj = cp.Minimize(cp.log_sum_exp(cp.square(A @ x)))
        constraints = [x >= 0, x + y == 1, y >= 0]
        prob = cp.Problem(obj, constraints)
        prob.solve(nlp=True, verbose=True, derivative_test='none')
        DNLP_opt_val = obj.value
        prob.solve(solver=cp.CLARABEL, verbose=True)
        DCP_opt_val = obj.value
        assert np.isclose(DNLP_opt_val, DCP_opt_val)

        checker = DerivativeChecker(prob)
        checker.run_and_assert()

