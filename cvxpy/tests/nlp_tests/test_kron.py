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
from scipy import sparse

import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
from cvxpy.tests.nlp_tests.derivative_checker import DerivativeChecker

# Constant operands with zeros, so the converter prunes inactive blocks. The
# tests weight the kron output entrywise before summing: a bare sum would be
# blind to the block layout, since sum(kron(A, S)) == sum(A) * sum(S) whatever
# the layout.
A = np.array([[2.0, 0.0, -1.0],
              [0.0, 3.0, 0.0]])
B = np.array([[1.0, 0.0],
              [0.0, -2.0],
              [0.5, 0.0]])


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestKron():

    def test_left_kron(self):
        """cp.kron(A, f(X)) with a constant left operand containing zeros."""
        np.random.seed(0)
        X = cp.Variable((2, 2), bounds=[-1, 1], name='X')
        X.value = np.random.rand(2, 2)
        W = np.random.rand(4, 6)
        problem = cp.Problem(cp.Minimize(cp.sum(cp.multiply(W, cp.kron(A, cp.nlp.sin(X))))))

        problem.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        assert problem.status == cp.OPTIMAL
        assert problem.value == pytest.approx(np.sum(W * np.kron(A, np.sin(X.value))))

        DerivativeChecker(problem).run_and_assert()

    def test_right_kron(self):
        """cp.kron(f(X), B) with a constant right operand containing zeros."""
        np.random.seed(0)
        X = cp.Variable((2, 3), bounds=[-1, 1], name='X')
        X.value = np.random.rand(2, 3)
        W = np.random.rand(6, 6)
        problem = cp.Problem(cp.Minimize(cp.sum(cp.multiply(W, cp.kron(cp.nlp.sin(X), B)))))

        problem.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        assert problem.status == cp.OPTIMAL
        assert problem.value == pytest.approx(np.sum(W * np.kron(np.sin(X.value), B)))

        DerivativeChecker(problem).run_and_assert()

    def test_sparse_constant_operand(self):
        """A sparse constant operand, with an explicitly stored zero."""
        np.random.seed(0)
        A_sparse = sparse.csr_array(A)
        A_sparse.data[0] = 0.0  # stored but zero: not an active block
        X = cp.Variable((2, 2), bounds=[-1, 1], name='X')
        X.value = np.random.rand(2, 2)
        W = np.random.rand(4, 6)
        problem = cp.Problem(cp.Minimize(cp.sum(cp.multiply(W, cp.kron(A_sparse, cp.nlp.sin(X))))))

        problem.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        assert problem.status == cp.OPTIMAL
        expected = np.sum(W * np.kron(A_sparse.toarray(), np.sin(X.value)))
        assert problem.value == pytest.approx(expected)

        DerivativeChecker(problem).run_and_assert()

    @pytest.mark.parametrize('const_is_left', [True, False])
    def test_kron_parameter(self, const_is_left):
        """A parametric operand must give the same answer as a hardcoded one.

        C2 is nonzero exactly where C1 is zero, so a converter that pruned
        blocks from the initial parameter value would drop every entry of the
        second solve.
        """
        np.random.seed(0)
        if const_is_left:
            C1, C2 = A, np.array([[0.0, 1.5, 0.0],
                                  [2.5, 0.0, -2.0]])
            X = cp.Variable((2, 2), bounds=[-1, 1], name='X')
        else:
            C1, C2 = B, np.array([[0.0, 2.0],
                                  [-1.5, 0.0],
                                  [0.0, 1.0]])
            X = cp.Variable((2, 3), bounds=[-1, 1], name='X')
        X0 = np.random.rand(*X.shape)

        def objective(C):
            f = cp.nlp.sin(X)
            return cp.Minimize(cp.sum(cp.kron(C, f) if const_is_left else cp.kron(f, C)))

        # Solve with hardcoded values.
        hardcoded = []
        for C in (C1, C2):
            X.value = X0
            problem = cp.Problem(objective(C))
            problem.solve(solver=cp.IPOPT, nlp=True, verbose=False)
            assert problem.status == cp.OPTIMAL
            hardcoded.append(X.value)

        # Solve with a parameter, then update its value and re-solve.
        C = cp.Parameter(C1.shape)
        problem = cp.Problem(objective(C))
        for value, expected in zip((C1, C2), hardcoded):
            C.value = value
            X.value = X0
            problem.solve(solver=cp.IPOPT, nlp=True, verbose=False)
            assert problem.status == cp.OPTIMAL
            DerivativeChecker(problem).run_and_assert()
            np.testing.assert_allclose(X.value, expected, atol=1e-6)
