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

import pytest

import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
from cvxpy.tests.nlp_tests.derivative_checker import DerivativeChecker


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestHyperbolic():

    def test_sinh(self):
        n = 10
        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.sum(cp.sinh(cp.logistic(x * 2)))),
                           [x >= 0.1, cp.sum(x) == 10])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL

        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_tanh(self):
        n = 10
        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.sum(cp.tanh(cp.logistic(x * 2)))),
                           [x >= 0.1, cp.sum(x) == 10])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL

        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_asinh(self):
        n = 10
        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.sum(cp.asinh(cp.logistic(x * 3)))),
                           [x >= 0.1, cp.sum(x) == 10])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL

        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_atanh(self):
        n = 10
        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.sum(cp.atanh(cp.logistic(x * 0.1)))),
                           [x >= 0.1, cp.sum(x) == 10])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL

        checker = DerivativeChecker(prob)
        checker.run_and_assert()
