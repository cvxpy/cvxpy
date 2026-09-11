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

import numpy as np
import pytest

import cvxpy as cp
from cvxpy.error import SolverError
from cvxpy.tests.base_test import BaseTest


class TestSupportFunctions(BaseTest):
    """
    Test the implementation of support function atoms.

    Relevant source code includes:
        cvxpy.atoms.suppfunc
        cvxpy.transforms.suppfunc
        cvxpy.reductions.dcp2cone.canonicalizers.suppfunc_canon
    """

    def test_Rn(self) -> None:
        np.random.seed(0)
        n = 5
        x = cp.Variable(shape=(n,))
        sigma = cp.suppfunc(x, [])
        a = np.random.randn(n,)
        y = cp.Variable(shape=(n,))
        cons = [sigma(y - a) <= 0]  # "<= num" for any num >= 0 is valid.
        objective = cp.Minimize(a @ y)
        prob = cp.Problem(objective, cons)
        prob.solve(solver='CLARABEL')
        actual = prob.value
        expected = np.dot(a, a)
        self.assertLessEqual(abs(actual - expected), 1e-6)
        actual = y.value
        expected = a
        self.assertLessEqual(np.linalg.norm(actual - expected, ord=2), 1e-6)
        viol = cons[0].violation()
        self.assertLessEqual(viol, 1e-8)

    def test_vector1norm(self) -> None:
        n = 3
        np.random.seed(1)
        a = np.random.randn(n,)
        x = cp.Variable(shape=(n,))
        sigma = cp.suppfunc(x, [cp.norm(x - a, 1) <= 1])
        y = np.random.randn(n,)
        y_var = cp.Variable(shape=(n,))
        prob = cp.Problem(cp.Minimize(sigma(y_var)), [y == y_var])
        prob.solve(solver='CLARABEL')
        actual = prob.value
        expected = a @ y + np.linalg.norm(y, ord=np.inf)
        self.assertLessEqual(abs(actual - expected), 1e-5)
        self.assertLessEqual(abs(prob.objective.expr.value - prob.value), 1e-5)

    def test_vector2norm(self) -> None:
        n = 3
        np.random.seed(1)
        a = np.random.randn(n,)
        x = cp.Variable(shape=(n,))
        sigma = cp.suppfunc(x, [cp.norm(x - a, 2) <= 1])
        y = np.random.randn(n,)
        y_var = cp.Variable(shape=(n,))
        prob = cp.Problem(cp.Minimize(sigma(y_var)), [y == y_var])
        prob.solve(solver='CLARABEL')
        actual = prob.value
        expected = a @ y + np.linalg.norm(y, ord=2)
        self.assertLessEqual(abs(actual - expected), 1e-6)
        self.assertLessEqual(abs(prob.objective.expr.value - prob.value), 1e-6)

    def test_rectangular_variable(self) -> None:
        np.random.seed(2)
        rows, cols = 4, 2
        a = np.random.randn(rows, cols)
        x = cp.Variable(shape=(rows, cols))
        sigma = cp.suppfunc(x, [x[:, 0] == 0])
        y = cp.Variable(shape=(rows, cols))
        cons = [sigma(y - a) <= 0]
        objective = cp.Minimize(cp.sum_squares(y.flatten(order='F')))
        prob = cp.Problem(objective, cons)
        prob.solve(solver='CLARABEL')
        expect = np.hstack([np.zeros(shape=(rows, 1)), a[:, [1]]])
        actual = y.value
        self.assertLessEqual(np.linalg.norm(actual - expect, ord=2), 1e-6)
        viol = cons[0].violation()
        self.assertLessEqual(viol, 1e-6)

    def test_psd_dualcone(self) -> None:
        np.random.seed(5)
        n = 3
        X = cp.Variable(shape=(n, n))
        sigma = cp.suppfunc(X, [X >> 0])
        A = np.random.randn(n, n)
        Y = cp.Variable(shape=(n, n))
        objective = cp.Minimize(cp.norm(A.ravel(order='F') + Y.flatten(order='F')))
        cons = [sigma(Y) <= 0]  # Y is negative definite.
        prob = cp.Problem(objective, cons)
        prob.solve(solver='SCS', eps=1e-8)
        viol = cons[0].violation()
        self.assertLessEqual(viol, 1e-6)
        eigs = np.linalg.eigh(Y.value)[0]
        self.assertLessEqual(np.max(eigs), 1e-6)

    def test_largest_singvalue(self) -> None:
        np.random.seed(3)
        rows, cols = 3, 4
        A = np.random.randn(rows, cols)
        A_sv = np.linalg.svd(A, compute_uv=False)
        X = cp.Variable(shape=(rows, cols))
        sigma = cp.suppfunc(X, [cp.sigma_max(X) <= 1])
        Y = cp.Variable(shape=(rows, cols))
        cons = [Y == A]
        prob = cp.Problem(cp.Minimize(sigma(Y)), cons)
        prob.solve(solver='SCS', eps=1e-8)
        actual = prob.value
        expect = np.sum(A_sv)
        self.assertLessEqual(abs(actual - expect), 1e-6)

    def test_expcone_1(self) -> None:
        x = cp.Variable(shape=(1,))
        tempcons = [cp.exp(x[0]) <= np.exp(1), cp.exp(-x[0]) <= np.exp(1)]
        sigma = cp.suppfunc(x, tempcons)
        y = cp.Variable(shape=(1,))
        obj_expr = y[0]
        cons = [sigma(y) <= 1]
        # ^ That just means -1 <= y[0] <= 1
        prob = cp.Problem(cp.Minimize(obj_expr), cons)
        prob.solve(solver='CLARABEL')
        viol = cons[0].violation()
        self.assertLessEqual(viol, 1e-6)
        self.assertLessEqual(abs(y.value - (-1)), 1e-6)

    def test_expcone_2(self) -> None:
        x = cp.Variable(shape=(3,))
        tempcons = [cp.sum(x) <= 1.0, cp.sum(x) >= 0.1, x >= 0.01,
                    cp.kl_div(x[1], x[0]) + x[1] - x[0] + x[2] <= 0]
        sigma = cp.suppfunc(x, tempcons)
        y = cp.Variable(shape=(3,))
        a = np.array([-3, -2, -1])  # this is negative of objective in mosek_conif.py example
        expr = -sigma(y)
        objective = cp.Maximize(expr)
        cons = [y == a]
        prob = cp.Problem(objective, cons)
        prob.solve(solver='CLARABEL')
        # Check for expected objective value
        epi_actual = prob.value
        direct_actual = expr.value
        expect = 0.235348211
        self.assertLessEqual(abs(epi_actual - expect), 1e-6)
        self.assertLessEqual(abs(direct_actual - expect), 1e-6)

    def test_basic_lmi(self) -> None:
        np.random.seed(4)
        n = 3
        A = np.random.randn(n, n)
        A = A.T @ A
        X = cp.Variable(shape=(n, n))  # will fail if you try PSD=True, or symmetric=Trues
        sigma = cp.suppfunc(X, [0 << X, cp.lambda_max(X) <= 1])
        Y = cp.Variable(shape=(n, n))
        cons = [Y == A]
        expr = sigma(Y)
        prob = cp.Problem(cp.Minimize(expr), cons)  # opt value of support func would be at X=I.
        prob.solve(solver='SCS', eps=1e-8)
        actual1 = prob.value  # computed with epigraph
        actual2 = expr.value  # computed by evaluating support function, as a maximization problem.
        self.assertLessEqual(abs(actual1 - actual2), 1e-6)
        expect = np.trace(A)
        self.assertLessEqual(abs(actual1 - expect), 1e-4)

    def test_invalid_solver(self) -> None:
        n = 3
        x = cp.Variable(shape=(n,))
        sigma = cp.suppfunc(x, [cp.norm(x - np.random.randn(n,), 2) <= 1])
        y_var = cp.Variable(shape=(n,))
        prob = cp.Problem(cp.Minimize(sigma(y_var)), [np.random.randn(n,) == y_var])
        with self.assertRaises(SolverError):
            prob.solve(solver='OSQP')

    def test_invalid_variable(self) -> None:
        x = cp.Variable(shape=(2, 2), symmetric=True)
        with self.assertRaises(ValueError):
            cp.suppfunc(x, [])

    def test_invalid_constraint(self) -> None:
        x = cp.Variable(shape=(3,))
        a = cp.Parameter(shape=(3,))
        cons = [a @ x == 1]
        with self.assertRaises(ValueError):
            cp.suppfunc(x, cons)

    def test_support_function_atom_metadata_and_validation(self) -> None:
        x = cp.Variable(2)
        sigma = cp.suppfunc(x, [cp.norm(x, 2) <= 1])
        y = cp.Variable(2)
        atom = sigma(y)

        self.assertEqual(atom.variables(), [y])
        self.assertEqual(atom.parameters(), [])
        self.assertEqual(atom.constants(), [])
        self.assertEqual(atom.shape_from_args(), tuple())
        self.assertEqual(atom.sign_from_args(), (False, False))
        self.assertFalse(atom.is_nonneg())
        self.assertFalse(atom.is_nonpos())
        self.assertFalse(atom.is_imag())
        self.assertFalse(atom.is_complex())
        self.assertTrue(atom.is_atom_convex())
        self.assertFalse(atom.is_atom_concave())
        self.assertFalse(atom.is_atom_log_log_convex())
        self.assertFalse(atom.is_atom_log_log_concave())
        self.assertTrue(atom.is_atom_quasiconvex())
        self.assertFalse(atom.is_atom_quasiconcave())
        self.assertFalse(atom.is_incr(0))
        self.assertFalse(atom.is_decr(0))
        self.assertTrue(atom.is_convex())
        self.assertFalse(atom.is_concave())
        self.assertTrue(atom.is_quasiconvex())
        self.assertFalse(atom.is_quasiconcave())

        with pytest.raises(ValueError, match="cannot be complex"):
            sigma(cp.Variable(2, complex=True))
        with pytest.raises(ValueError, match="must be affine"):
            sigma(cp.square(y))
        with pytest.raises(NotImplementedError):
            atom < 1
        with pytest.raises(NotImplementedError):
            atom > 1

    def test_support_function_value_and_gradient_paths(self) -> None:
        x = cp.Variable(1)
        sigma = cp.suppfunc(x, [x <= 1, x >= -1])
        y = cp.Variable(1)
        atom = sigma(y)

        y.value = np.array([2.0])
        self.assertAlmostEqual(atom.value, 2.0, places=4)
        grad = atom._grad([np.array([-2.0])])[0]
        np.testing.assert_allclose(y.value, [2.0])
        np.testing.assert_allclose(grad.toarray(), [[-1.0]], atol=1e-4)

        grad = atom._grad([np.array([2.0])])[0]
        np.testing.assert_allclose(y.value, [2.0])
        np.testing.assert_allclose(grad.toarray(), [[1.0]], atol=1e-4)
