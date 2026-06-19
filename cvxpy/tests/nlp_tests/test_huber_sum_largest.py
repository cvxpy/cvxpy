"""
Copyright, the CVXPY authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import pytest

import cvxpy as cp
from cvxpy.atoms.sum_largest import sum_largest
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.eliminate_pwl.canonicalizers.sum_largest_canon import sum_largest_canon
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
from cvxpy.tests.nlp_tests.derivative_checker import DerivativeChecker


class TestSumLargestCanonWarmStart:
    """Regression test for sum_largest_canon DNLP warm-start initialization."""

    def test_warmstart_feasibility(self):
        """Middle elements must satisfy x_i <= t_i + q after warm-start.

        Before the fix, q was set to max(k smallest) and t was only set for
        top-k elements, so elements between q and top-k violated x <= t + q.
        """
        x = Variable(6)
        # k=2, top-2 are {10, 9}. Middle elements 6, 8 triggered the bug.
        x.value = np.array([1.0, 10.0, 6.0, 8.0, 3.0, 9.0])
        expr = sum_largest(x, 2)

        obj, constraints = sum_largest_canon(expr, [x])
        # obj = sum(t) + k*q
        t = obj.args[0].args[0]
        q = obj.args[1].args[1]

        assert q.value == pytest.approx(8.0)  # max of non-top-k, i.e. x_{[k+1]}
        expected_t = np.zeros(6)
        expected_t[1] = 10.0 - 8.0  # top-k element
        expected_t[5] = 9.0 - 8.0   # top-k element
        np.testing.assert_allclose(t.value, expected_t)
        assert np.all(x.value <= t.value + q.value + 1e-10)
        assert np.all(t.value >= -1e-10)


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

            checker = DerivativeChecker(prob)
            checker.run_and_assert()

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

        checker = DerivativeChecker(prob)
        checker.run_and_assert()

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

        checker = DerivativeChecker(prob)
        checker.run_and_assert()

