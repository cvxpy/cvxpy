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
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
from cvxpy.tests.nlp_tests.derivative_checker import DerivativeChecker


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestKron:

    def test_left_kron(self):
        """cp.kron(A, f(X)) with a constant left operand containing zeros."""
        np.random.seed(0)
        A = np.array([[2.0, 0.0, -1.0],
                      [0.0, 3.0, 0.0]])
        X = cp.Variable((2, 2), bounds=[-1, 1], name='X')
        X.value = np.random.rand(2, 2)
        obj = cp.sum(cp.kron(A, cp.nlp.sin(X)))
        problem = cp.Problem(cp.Minimize(obj))

        problem.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        assert(problem.status == cp.OPTIMAL)

        checker = DerivativeChecker(problem)
        checker.run_and_assert()

    def test_right_kron(self):
        """cp.kron(f(X), B) with a constant right operand containing zeros."""
        np.random.seed(0)
        B = np.array([[1.0, 0.0],
                      [0.0, -2.0],
                      [0.5, 0.0]])
        X = cp.Variable((2, 3), bounds=[-1, 1], name='X')
        X.value = np.random.rand(2, 3)
        obj = cp.sum(cp.kron(cp.nlp.sin(X), B))
        problem = cp.Problem(cp.Minimize(obj))

        problem.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        assert(problem.status == cp.OPTIMAL)

        checker = DerivativeChecker(problem)
        checker.run_and_assert()
