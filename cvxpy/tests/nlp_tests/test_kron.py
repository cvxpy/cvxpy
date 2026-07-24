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

    def test_left_kron_parameter(self):
        """cp.kron(A, f(X)) with a parametric left operand.

        Solve with hardcoded A1, A2, then with a Parameter and mutate
        A.value; the solutions must match. A2 is nonzero exactly where A1
        is zero, so a converter that pruned blocks from the initial
        parameter value would drop every entry of the second solve.
        """
        np.random.seed(0)
        A1 = np.array([[2.0, 0.0, -1.0],
                       [0.0, 3.0, 0.0]])
        A2 = np.array([[0.0, 1.5, 0.0],
                       [2.5, 0.0, -2.0]])
        X0 = np.random.rand(2, 2)

        # Solve with hardcoded values.
        X = cp.Variable((2, 2), bounds=[-1, 1], name='X')
        prob1 = cp.Problem(cp.Minimize(cp.sum(cp.kron(A1, cp.nlp.sin(X)))))
        prob2 = cp.Problem(cp.Minimize(cp.sum(cp.kron(A2, cp.nlp.sin(X)))))
        X.value = X0
        prob1.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        assert prob1.status == cp.OPTIMAL
        hardcoded_sol1 = X.value
        X.value = X0
        prob2.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        assert prob2.status == cp.OPTIMAL
        hardcoded_sol2 = X.value

        # Solve with a parameter, then update its value and re-solve.
        A = cp.Parameter((2, 3), value=A1)
        prob = cp.Problem(cp.Minimize(cp.sum(cp.kron(A, cp.nlp.sin(X)))))
        X.value = X0
        prob.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        assert prob.status == cp.OPTIMAL
        checker = DerivativeChecker(prob)
        checker.run_and_assert()
        param_sol1 = X.value

        A.value = A2
        X.value = X0
        prob.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        assert prob.status == cp.OPTIMAL
        checker = DerivativeChecker(prob)
        checker.run_and_assert()
        param_sol2 = X.value

        assert np.linalg.norm(param_sol1 - hardcoded_sol1) == 0.0
        assert np.linalg.norm(param_sol2 - hardcoded_sol2) == 0.0

    def test_right_kron_parameter(self):
        """cp.kron(f(X), B) with a parametric right operand.

        Same structure as test_left_kron_parameter: B2's zero pattern is
        the complement of B1's, exercising the all-blocks-active path for
        parametric operands.
        """
        np.random.seed(0)
        B1 = np.array([[1.0, 0.0],
                       [0.0, -2.0],
                       [0.5, 0.0]])
        B2 = np.array([[0.0, 2.0],
                       [-1.5, 0.0],
                       [0.0, 1.0]])
        X0 = np.random.rand(2, 3)

        # Solve with hardcoded values.
        X = cp.Variable((2, 3), bounds=[-1, 1], name='X')
        prob1 = cp.Problem(cp.Minimize(cp.sum(cp.kron(cp.nlp.sin(X), B1))))
        prob2 = cp.Problem(cp.Minimize(cp.sum(cp.kron(cp.nlp.sin(X), B2))))
        X.value = X0
        prob1.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        assert prob1.status == cp.OPTIMAL
        hardcoded_sol1 = X.value
        X.value = X0
        prob2.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        assert prob2.status == cp.OPTIMAL
        hardcoded_sol2 = X.value

        # Solve with a parameter, then update its value and re-solve.
        B = cp.Parameter((3, 2), value=B1)
        prob = cp.Problem(cp.Minimize(cp.sum(cp.kron(cp.nlp.sin(X), B))))
        X.value = X0
        prob.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        assert prob.status == cp.OPTIMAL
        checker = DerivativeChecker(prob)
        checker.run_and_assert()
        param_sol1 = X.value

        B.value = B2
        X.value = X0
        prob.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        assert prob.status == cp.OPTIMAL
        checker = DerivativeChecker(prob)
        checker.run_and_assert()
        param_sol2 = X.value

        assert np.linalg.norm(param_sol1 - hardcoded_sol1) == 0.0
        assert np.linalg.norm(param_sol2 - hardcoded_sol2) == 0.0


class _FakeKron:
    """Stand-in for a kron expression: convert_kron only reads ``args``."""

    def __init__(self, a, b):
        self.args = [a, b]


class TestKronConverterGuards:
    """Converter-level validation; no solver required."""

    def _convert(self, a, b):
        registry = pytest.importorskip(
            "cvxpy.reductions.solvers.nlp_solvers.diff_engine.registry"
        )
        return registry.convert_kron(_FakeKron(a, b), [None, None])

    def test_rejects_two_variable_operands(self):
        with pytest.raises(ValueError, match="variable-free operand"):
            self._convert(cp.Variable((2, 2)), cp.Variable((2, 2)))
