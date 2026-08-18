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


def _duplicate_gather_problem():
    """gh-3442 shape: multiply of gathers whose index arrays contain
    DUPLICATES, so each gather Jacobian has more nonzeros than its source
    variable. Corrupted the heap in derivative-structure initialization."""
    nb = 4
    rows = np.array([0, 0, 1, 2, 2, 3])
    cols = np.array([1, 2, 2, 0, 3, 1])
    theta = cp.Variable(nb, name='theta', bounds=[-0.5, 0.5])
    v = cp.Variable(nb, name='v', bounds=[0.9, 1.1])
    x = cp.Variable(rows.size, name='x')
    C = cp.nlp.cos(theta[rows] - theta[cols])
    vv = cp.multiply(v[rows], v[cols])
    prob = cp.Problem(cp.Minimize(cp.sum(x)), [x == cp.multiply(vv, C)])
    return prob, theta, v, x, rows, cols


class TestDuplicateIndexGathers:
    # Regression tests for gh-3442 (solver-free: the crash was in the diff
    # engine's jacobian/hessian structure init, before any solver ran).
    # The fix (gather-Jacobian allocation for duplicated indices,
    # SparseDiffEngine#105) ships in sparsediffpy 0.6.0, the pinned floor.
    def test_multiply_duplicate_index_gathers_derivatives(self):
        np.random.seed(0)
        prob, theta, v, x, rows, _ = _duplicate_gather_problem()
        theta.value = np.random.uniform(-0.4, 0.4, theta.size)
        v.value = np.random.uniform(0.95, 1.05, v.size)
        x.value = np.zeros(rows.size)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestAffineDiffEngine:
    # Stress tests for affine vector atoms in the diff engine.
    def test_row_broadcast(self):
        # x is 1 x n, Y is m x n
        np.random.seed(0)
        m, n = 3, 4
        x = cp.Variable((1, n), bounds=[-2, 2])
        Y = cp.Variable((m, n), bounds=[-1, 1])
        obj = cp.Minimize(cp.sum(x + Y))
        prob = cp.Problem(obj)
        x.value = np.random.rand(1, n)
        Y.value = np.random.rand(m, n)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()
        prob.solve(solver=cp.IPOPT, nlp=True)
        # Solution: x = -2, Y = -1
        assert prob.status == cp.OPTIMAL
        assert np.allclose(x.value, -2, atol=1e-4)
        assert np.allclose(Y.value, -1, atol=1e-4)

    def test_col_broadcast(self):
        # x is m x 1, Y is m x n
        np.random.seed(0)
        m, n = 3, 4
        x = cp.Variable((m, 1), bounds=[-2, 2])
        Y = cp.Variable((m, n), bounds=[-1, 1])
        obj = cp.Minimize(cp.sum(x + Y))
        prob = cp.Problem(obj)
        x.value = np.random.rand(m, 1)
        Y.value = np.random.rand(m, n)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()
        prob.solve(solver=cp.IPOPT, nlp=True)
        # Solution: x = -2, Y = -1
        assert prob.status == cp.OPTIMAL
        assert np.allclose(x.value, -2, atol=1e-4)
        assert np.allclose(Y.value, -1, atol=1e-4)

    def test_index_stress(self):
        np.random.seed(0)
        m, n = 3, 4
        X = cp.Variable((m, n), bounds=[-2, 2])
        expr = (cp.sum(X[0, :]) + cp.sum(X[0, :]) +
                cp.sum(X[1, :]) + cp.sum(X[:, 2]) + X[0, 1] + X[2, 2])
        obj = cp.Minimize(expr)
        prob = cp.Problem(obj)
        X.value = np.random.rand(m, n)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()
        prob.solve(solver=cp.IPOPT, nlp=True)
        # Solution: all X at lower bound
        assert prob.status == cp.OPTIMAL
        assert np.allclose(prob.value, -34.0)

    def test_duplicate_indices(self):
        np.random.seed(0)
        m, n = 3, 3
        X = cp.Variable((m, n), bounds=[-2, 2])
        # Use duplicate indices: X[[0,0],[1,1]] = [X[0,1], X[0,1]]
        expr = cp.sum(X[[0, 0], [1, 1]]) - 2 * X[0, 1] + cp.sum(X)
        obj = cp.Minimize(expr)
        prob = cp.Problem(obj)
        X.value = np.random.rand(m, n)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()
        prob.solve(solver=cp.IPOPT, nlp=True)
        assert prob.status == cp.OPTIMAL
        assert np.allclose(X.value, -2, atol=1e-4)

    def test_multiply_duplicate_index_gathers_solve(self):
        # gh-3442: the array-indexed formulation must solve and agree with
        # the scalar-indexed loop formulation (the issue's workaround).
        np.random.seed(0)
        prob, theta, v, x, rows, cols = _duplicate_gather_problem()
        theta.value = np.random.uniform(-0.4, 0.4, theta.size)
        v.value = np.random.uniform(0.95, 1.05, v.size)
        x.value = np.zeros(rows.size)
        prob.solve(solver=cp.IPOPT, nlp=True)
        assert prob.status == cp.OPTIMAL
        gathered_value = prob.value

        x2 = cp.Variable(rows.size)
        cons = [x2[k] == v[i] * v[j] * cp.nlp.cos(theta[i] - theta[j])
                for k, (i, j) in enumerate(zip(rows, cols))]
        prob2 = cp.Problem(cp.Minimize(cp.sum(x2)), cons)
        prob2.solve(solver=cp.IPOPT, nlp=True)
        assert prob2.status == cp.OPTIMAL
        assert np.isclose(gathered_value, prob2.value, atol=1e-5)

    def test_promote_row(self):
        # Promote scalar to row vector
        np.random.seed(0)
        n = 4
        x = cp.Variable(bounds=[-3, 3])
        Y = cp.Variable((1, n), bounds=[-2, 2])
        obj = cp.Minimize(cp.sum(x + Y))
        prob = cp.Problem(obj)
        x.value = 2.0
        Y.value = np.random.rand(1, n)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()
        prob.solve(solver=cp.IPOPT, nlp=True)
        # Solution: x = -3, Y = -2
        assert prob.status == cp.OPTIMAL
        assert np.allclose(x.value, -3, atol=1e-4)
        assert np.allclose(Y.value, -2, atol=1e-4)

    def test_promote_col(self):
        # Promote scalar to column vector
        np.random.seed(0)
        m = 4
        x = cp.Variable(bounds=[-3, 3])
        Y = cp.Variable((m, 1), bounds=[-2, 2])
        obj = cp.Minimize(cp.sum(x + Y))
        prob = cp.Problem(obj)
        x.value = 2.0
        Y.value = np.random.rand(m, 1)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()
        prob.solve(solver=cp.IPOPT, nlp=True)
        # Solution: x = -3, Y = -2
        assert prob.status == cp.OPTIMAL
        assert np.allclose(x.value, -3, atol=1e-4)
        assert np.allclose(Y.value, -2, atol=1e-4)

    def test_promote_add(self):
        # Scalar x, matrix Y, with bounds set via the bounds attribute
        np.random.seed(0)
        x = cp.Variable(bounds=[-1, 1])
        Y = cp.Variable((2, 2), bounds=[0, 2])
        obj = cp.Minimize(cp.sum(x + Y))
        prob = cp.Problem(obj)
        x.value = 0.0
        Y.value = np.random.rand(2, 2)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()
        prob.solve(solver=cp.IPOPT, nlp=True)
        # Solution: x = -1, Y = 0
        assert prob.status == cp.OPTIMAL
        assert np.allclose(x.value, -1, atol=1e-4)
        assert np.allclose(Y.value, 0, atol=1e-4)

    def test_reshape(self):
        x = cp.Variable(8, bounds=[-5, 5])
        A = np.random.rand(4, 2)
        obj = cp.Minimize(cp.sum_squares(cp.reshape(x, (4, 2), order='F') - A))
        prob = cp.Problem(obj)
        x.value = np.linspace(-2, 2, 8)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()
        prob.solve(solver=cp.IPOPT, nlp=True)
        assert prob.status == cp.OPTIMAL
        assert np.allclose(x.value, A.flatten(order='F'), atol=1e-4)

    def test_broadcast(self):
        np.random.seed(0)
        x = cp.Variable(8, bounds=[-5, 5])
        A = np.random.rand(8, 1)
        obj = cp.Minimize(cp.sum_squares(x - A))
        prob = cp.Problem(obj)
        x.value = np.linspace(-2, 2, 8)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()
        prob.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        assert prob.status == cp.OPTIMAL
        assert np.allclose(x.value, np.mean(A), atol=1e-4)

    def test_hstack(self):
        np.random.seed(0)
        m = 5
        n = 3
        x = cp.Variable((n, 1), bounds=[-3, 3])
        y = cp.Variable((n, 1), bounds=[-2, 2])
        A1 = np.random.rand(m, n)
        A2 = np.random.rand(m, n)
        b1 = np.random.rand(m, 1)
        b2 = np.random.rand(m, 1)
        obj = cp.Minimize(cp.sum_squares(cp.hstack([A1 @ x + A2 @ y - b1,
                                                    A1 @ y + A2 @ x - b2,
                                                    A2 @ x - A1 @ y])))

        prob = cp.Problem(obj)

        # check derivatives
        x.value = np.random.rand(n, 1)
        y.value = np.random.rand(n, 1)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

        # solve as an NLP
        prob.solve(solver=cp.IPOPT, nlp=True)
        nlp_sol_x = x.value
        nlp_sol_y = y.value
        nlp_sol_val = prob.value

        # solve as DCP
        prob.solve(solver=cp.CLARABEL)
        dcp_sol_x = x.value
        dcp_sol_y = y.value
        dcp_sol_val = prob.value

        assert np.allclose(nlp_sol_x, dcp_sol_x, atol=1e-4)
        assert np.allclose(nlp_sol_y, dcp_sol_y, atol=1e-4)
        assert np.allclose(nlp_sol_val, dcp_sol_val, atol=1e-4)

    def test_hstack_matrices(self):
        np.random.seed(0)
        m = 5
        n = 3
        X = cp.Variable((n, m), bounds=[-3, 3])
        Y = cp.Variable((n, m), bounds=[-2, 2])
        A1 = np.random.rand(m, n)
        A2 = np.random.rand(m, n)
        b1 = np.random.rand(m, m)
        b2 = np.random.rand(m, m)
        obj = cp.Minimize(cp.sum_squares(cp.hstack([A1 @ X + A2 @ Y - b1,
                                                    A1 @ Y + A2 @ X - b2,
                                                    A2 @ X - A1 @ Y])))

        prob = cp.Problem(obj)

        # check derivatives
        X.value = np.random.rand(n, m)
        Y.value = np.random.rand(n, m)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

        # solve as an NLP
        prob.solve(solver=cp.IPOPT, nlp=True)
        nlp_sol_x = X.value
        nlp_sol_y = Y.value
        nlp_sol_val = prob.value

        # solve as DCP
        prob.solve(solver=cp.CLARABEL)
        dcp_sol_x = X.value
        dcp_sol_y = Y.value
        dcp_sol_val = prob.value

        assert np.allclose(nlp_sol_x, dcp_sol_x, atol=1e-4)
        assert np.allclose(nlp_sol_y, dcp_sol_y, atol=1e-4)
        assert np.allclose(nlp_sol_val, dcp_sol_val, atol=1e-4)
