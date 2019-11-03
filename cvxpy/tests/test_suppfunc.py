"""
Copyright 2019, the cvxpy developers.

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

import cvxpy as cvx
import numpy as np
from cvxpy.tests.base_test import BaseTest
from cvxpy.error import SolverError


class TestSupportFunctions(BaseTest):
    """
    Test the implementation of support function atoms.

    Relevant source code includes:
        cvxpy.atoms.suppfunc
        cvxpy.transforms.suppfunc
        cvxpy.reductions.dcp2cone.atom_canonicalizers.suppfunc_canon
    """

    def test_Rn(self):
        np.random.seed(0)
        n = 5
        x = cvx.Variable(shape=(n,))
        sigma = cvx.suppfunc(x, [])
        a = np.random.randn(n,)
        y = cvx.Variable(shape=(n,))
        cons = [sigma(y - a) <= 0]  # "<= num" for any num >= 0 is valid.
        objective = cvx.Minimize(a @ y)
        prob = cvx.Problem(objective, cons)
        prob.solve(solver='ECOS')
        actual = prob.value
        expected = np.dot(a, a)
        assert abs(actual - expected) <= 1e-6
        actual = y.value
        expected = a
        assert np.linalg.norm(actual - expected, ord=2) <= 1e-6
        viol = cons[0].violation()
        assert viol <= 1e-8

    def test_vector1norm(self):
        n = 3
        np.random.seed(1)
        a = np.random.randn(n,)
        x = cvx.Variable(shape=(n,))
        sigma = cvx.suppfunc(x, [cvx.norm(x - a, 1) <= 1])
        y = np.random.randn(n,)
        y_var = cvx.Variable(shape=(n,))
        prob = cvx.Problem(cvx.Minimize(sigma(y_var)), [y == y_var])
        prob.solve(solver='ECOS')
        actual = prob.value
        expected = a @ y + np.linalg.norm(y, ord=np.inf)
        assert abs(actual - expected) <= 1e-6
        assert abs(prob.objective.expr.value - prob.value) <= 1e-6

    def test_vector2norm(self):
        n = 3
        np.random.seed(1)
        a = np.random.randn(n,)
        x = cvx.Variable(shape=(n,))
        sigma = cvx.suppfunc(x, [cvx.norm(x - a, 2) <= 1])
        y = np.random.randn(n,)
        y_var = cvx.Variable(shape=(n,))
        prob = cvx.Problem(cvx.Minimize(sigma(y_var)), [y == y_var])
        prob.solve(solver='ECOS')
        actual = prob.value
        expected = a @ y + np.linalg.norm(y, ord=2)
        assert abs(actual - expected) <= 1e-6
        assert abs(prob.objective.expr.value - prob.value) <= 1e-6

    def test_rectangular_variable(self):
        np.random.seed(2)
        rows, cols = 4, 2
        a = np.random.randn(rows, cols)
        x = cvx.Variable(shape=(rows, cols))
        sigma = cvx.suppfunc(x, [x[:, 0] == 0])
        y = cvx.Variable(shape=(rows, cols))
        cons = [sigma(y - a) <= 0]
        objective = cvx.Minimize(cvx.sum_squares(y.flatten()))
        prob = cvx.Problem(objective, cons)
        prob.solve(solver='ECOS')
        expect = np.hstack([np.zeros(shape=(rows, 1)), a[:, [1]]])
        actual = y.value
        assert np.linalg.norm(actual - expect, ord=2) <= 1e-6
        viol = cons[0].violation()
        assert viol <= 1e-6

    def test_psd_dualcone(self):
        np.random.seed(5)
        n = 3
        X = cvx.Variable(shape=(n, n))
        sigma = cvx.suppfunc(X, [X >> 0])
        A = np.random.randn(n, n)
        Y = cvx.Variable(shape=(n, n))
        objective = cvx.Minimize(cvx.norm(A.ravel(order='F') + Y.flatten()))
        cons = [sigma(Y) <= 0]  # Y is negative definite.
        prob = cvx.Problem(objective, cons)
        prob.solve(solver='SCS', eps=1e-8)
        viol = cons[0].violation()
        assert viol <= 1e-6
        eigs = np.linalg.eigh(Y.value)[0]
        assert np.max(eigs) <= 1e-6

    def test_largest_singvalue(self):
        np.random.seed(3)
        rows, cols = 3, 4
        A = np.random.randn(rows, cols)
        A_sv = np.linalg.svd(A, compute_uv=False)
        X = cvx.Variable(shape=(rows, cols))
        sigma = cvx.suppfunc(X, [cvx.sigma_max(X) <= 1])
        Y = cvx.Variable(shape=(rows, cols))
        cons = [Y == A]
        prob = cvx.Problem(cvx.Minimize(sigma(Y)), cons)
        prob.solve(solver='SCS', eps=1e-8)
        actual = prob.value
        expect = np.sum(A_sv)
        assert abs(actual - expect) <= 1e-6

    def test_expcone_1(self):
        x = cvx.Variable(shape=(1,))
        tempcons = [cvx.exp(x[0]) <= np.exp(1), cvx.exp(-x[0]) <= np.exp(1)]
        sigma = cvx.suppfunc(x, tempcons)
        y = cvx.Variable(shape=(1,))
        obj_expr = y[0]
        cons = [sigma(y) <= 1]
        # ^ That just means -1 <= y[0] <= 1
        prob = cvx.Problem(cvx.Minimize(obj_expr), cons)
        prob.solve(solver='ECOS')
        viol = cons[0].violation()
        assert viol <= 1e-6
        assert abs(y.value - (-1)) <= 1e-6

    def test_expcone_2(self):
        x = cvx.Variable(shape=(3,))
        tempcons = [cvx.sum(x) <= 1.0, cvx.sum(x) >= 0.1, x >= 0.01,
                    cvx.kl_div(x[1], x[0]) + x[1] - x[0] + x[2] <= 0]
        sigma = cvx.suppfunc(x, tempcons)
        y = cvx.Variable(shape=(3,))
        a = np.array([-3, -2, -1])  # this is negative of objective in mosek_conif.py example
        expr = -sigma(y)
        objective = cvx.Maximize(expr)
        cons = [y == a]
        prob = cvx.Problem(objective, cons)
        prob.solve(solver='ECOS')
        # Check for expected objective value
        epi_actual = prob.value
        direct_actual = expr.value
        expect = 0.235348211
        assert abs(epi_actual - expect) <= 1e-6
        assert abs(direct_actual - expect) <= 1e-6

    def test_basic_lmi(self):
        np.random.seed(4)
        n = 3
        A = np.random.randn(n, n)
        A = A.T @ A
        X = cvx.Variable(shape=(n, n))  # will fail if you try PSD=True, or symmetric=Trues
        sigma = cvx.suppfunc(X, [0 << X, cvx.lambda_max(X) <= 1])
        Y = cvx.Variable(shape=(n, n))
        cons = [Y == A]
        expr = sigma(Y)
        prob = cvx.Problem(cvx.Minimize(expr), cons)  # opt value of support func would be at X=I.
        prob.solve(solver='SCS', eps=1e-8)
        actual1 = prob.value  # computed with epigraph
        actual2 = expr.value  # computed by evaluating support function, as a maximization problem.
        assert abs(actual1 - actual2) <= 1e-6
        expect = np.trace(A)
        assert abs(actual1 - expect) <= 1e-4

    def test_invalid_solver(self):
        n = 3
        x = cvx.Variable(shape=(n,))
        sigma = cvx.suppfunc(x, [cvx.norm(x - np.random.randn(n,), 2) <= 1])
        y_var = cvx.Variable(shape=(n,))
        prob = cvx.Problem(cvx.Minimize(sigma(y_var)), [np.random.randn(n,) == y_var])
        try:
            prob.solve(solver='OSQP')
            assert False
        except SolverError as e:
            assert 'could not be reduced to a QP' in e.args[0]
        pass

    def test_invalid_variable(self):
        x = cvx.Variable(shape=(2, 2), symmetric=True)
        try:
            cvx.suppfunc(x, [])  # dead-store
            assert False
        except ValueError as e:
            assert 'attributes' in e.args[0]
        pass

    def test_invalid_constraint(self):
        x = cvx.Variable(shape=(3,))
        a = cvx.Parameter(shape=(3,))
        cons = [a @ x == 1]
        try:
            cvx.suppfunc(x, cons)  # dead-store
            assert False
        except ValueError as e:
            assert 'Parameter' in e.args[0]
        pass
