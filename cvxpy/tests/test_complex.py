"""
Copyright 2013 Steven Diamond

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
from cvxpy.expressions.variable import Variable
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.constants import Parameter
from cvxpy import Problem, Minimize
from cvxpy.tests.base_test import BaseTest
import numpy as np
import scipy.sparse as sp
import sys
PY35 = sys.version_info >= (3, 5)


class TestComplex(BaseTest):
    """ Unit tests for the expression/expression module. """

    def test_variable(self):
        """Test the Variable class.
        """
        x = Variable(2, complex=False)
        y = Variable(2, complex=True)
        z = Variable(2, imag=True)

        assert not x.is_complex()
        assert not x.is_imag()
        assert y.is_complex()
        assert not y.is_imag()
        assert z.is_complex()
        assert z.is_imag()

        with self.assertRaises(Exception) as cm:
            x.value = np.array([1j, 0.])
        self.assertEqual(str(cm.exception), "Variable value must be real.")

        y.value = np.array([1., 0.])
        y.value = np.array([1j, 0.])

        with self.assertRaises(Exception) as cm:
            z.value = np.array([1., 0.])
        self.assertEqual(str(cm.exception), "Variable value must be imaginary.")

    def test_parameter(self):
        """Test the parameter class.
        """
        x = Parameter(2, complex=False)
        y = Parameter(2, complex=True)
        z = Parameter(2, imag=True)

        assert not x.is_complex()
        assert not x.is_imag()
        assert y.is_complex()
        assert not y.is_imag()
        assert z.is_complex()
        assert z.is_imag()

        with self.assertRaises(Exception) as cm:
            x.value = np.array([1j, 0.])
        self.assertEqual(str(cm.exception), "Parameter value must be real.")

        y.value = np.array([1., 0.])
        y.value = np.array([1j, 0.])

        with self.assertRaises(Exception) as cm:
            z.value = np.array([1., 0.])
        self.assertEqual(str(cm.exception), "Parameter value must be imaginary.")

    def test_constant(self):
        """Test the parameter class.
        """
        x = Constant(2)
        y = Constant(2j+1)
        z = Constant(2j)

        assert not x.is_complex()
        assert not x.is_imag()
        assert y.is_complex()
        assert not y.is_imag()
        assert z.is_complex()
        assert z.is_imag()

    def test_objective(self):
        """Test objectives.
        """
        x = Variable(complex=True)
        with self.assertRaises(Exception) as cm:
            Minimize(x)
        self.assertEqual(str(cm.exception), "The 'minimize' objective must be real valued.")

        with self.assertRaises(Exception) as cm:
            cvx.Maximize(x)
        self.assertEqual(str(cm.exception), "The 'maximize' objective must be real valued.")

    def test_arithmetic(self):
        """Test basic arithmetic expressions.
        """
        x = Variable(complex=True)
        y = Variable(imag=True)
        z = Variable()

        expr = x + z
        assert expr.is_complex()
        assert not expr.is_imag()

        expr = y + z
        assert expr.is_complex()
        assert not expr.is_imag()

        expr = y*z
        assert expr.is_complex()
        assert expr.is_imag()

        expr = y*y
        assert not expr.is_complex()
        assert not expr.is_imag()

        expr = y/2
        assert expr.is_complex()
        assert expr.is_imag()

        expr = y/1j
        assert not expr.is_complex()
        assert not expr.is_imag()

        A = np.ones((2, 2))
        expr = A*y*A
        assert expr.is_complex()
        assert expr.is_imag()

    def test_real(self):
        """Test real.
        """
        A = np.ones((2, 2))
        expr = Constant(A) + 1j*Constant(A)
        expr = cvx.real(expr)
        assert expr.is_real()
        assert not expr.is_complex()
        assert not expr.is_imag()
        self.assertItemsAlmostEqual(expr.value, A)

        x = Variable(complex=True)
        expr = cvx.imag(x) + cvx.real(x)
        assert expr.is_real()

    def test_imag(self):
        """Test imag.
        """
        A = np.ones((2, 2))
        expr = Constant(A) + 2j*Constant(A)
        expr = cvx.imag(expr)
        assert expr.is_real()
        assert not expr.is_complex()
        assert not expr.is_imag()
        self.assertItemsAlmostEqual(expr.value, 2*A)

    def test_conj(self):
        """Test imag.
        """
        A = np.ones((2, 2))
        expr = Constant(A) + 1j*Constant(A)
        expr = cvx.conj(expr)
        assert not expr.is_real()
        assert expr.is_complex()
        assert not expr.is_imag()
        self.assertItemsAlmostEqual(expr.value, A - 1j*A)

    def test_affine_atoms_canon(self):
        """Test canonicalization for affine atoms.
        """
        # Scalar.
        x = Variable()
        expr = cvx.imag(x + 1j*x)
        prob = Problem(Minimize(expr), [x >= 0])
        result = prob.solve()
        self.assertAlmostEqual(result, 0)
        self.assertAlmostEqual(x.value, 0)

        x = Variable(imag=True)
        expr = 1j*x
        prob = Problem(Minimize(expr), [cvx.imag(x) <= 1])
        result = prob.solve()
        self.assertAlmostEqual(result, -1)
        self.assertAlmostEqual(x.value, 1j)

        x = Variable(2)
        expr = x/1j
        prob = Problem(Minimize(expr[0]*1j + expr[1]*1j), [cvx.real(x + 1j) >= 1])
        result = prob.solve()
        self.assertAlmostEqual(result, -np.inf)
        prob = Problem(Minimize(expr[0]*1j + expr[1]*1j), [cvx.real(x + 1j) <= 1])
        result = prob.solve()
        self.assertAlmostEqual(result, -2)
        self.assertItemsAlmostEqual(x.value, [1, 1])
        prob = Problem(Minimize(expr[0]*1j + expr[1]*1j), [cvx.real(x + 1j) >= 1, cvx.conj(x) <= 0])
        result = prob.solve()
        self.assertAlmostEqual(result, np.inf)

        x = Variable((2, 2))
        y = Variable((3, 2), complex=True)
        expr = cvx.vstack([x, y])
        prob = Problem(Minimize(cvx.sum(cvx.imag(cvx.conj(expr)))),
                       [x == 0, cvx.real(y) == 0, cvx.imag(y) <= 1])
        result = prob.solve()
        self.assertAlmostEqual(result, -6)
        self.assertItemsAlmostEqual(y.value, 1j*np.ones((3, 2)))
        self.assertItemsAlmostEqual(x.value, np.zeros((2, 2)))

        x = Variable((2, 2))
        y = Variable((3, 2), complex=True)
        expr = cvx.vstack([x, y])
        prob = Problem(Minimize(cvx.sum(cvx.imag(expr.H))),
                       [x == 0, cvx.real(y) == 0, cvx.imag(y) <= 1])
        result = prob.solve()
        self.assertAlmostEqual(result, -6)
        self.assertItemsAlmostEqual(y.value, 1j*np.ones((3, 2)))
        self.assertItemsAlmostEqual(x.value, np.zeros((2, 2)))

    def test_params(self):
        """Test with parameters.
        """
        p = cvx.Parameter(imag=True, value=1j)
        x = Variable(2, complex=True)
        prob = Problem(cvx.Maximize(cvx.sum(cvx.imag(x) + cvx.real(x))), [cvx.abs(p*x) <= 2])
        result = prob.solve()
        self.assertAlmostEqual(result, 4*np.sqrt(2))
        val = np.ones(2)*np.sqrt(2)
        self.assertItemsAlmostEqual(x.value, val + 1j*val)

    def test_abs(self):
        """Test with absolute value.
        """
        x = Variable(2, complex=True)
        prob = Problem(cvx.Maximize(cvx.sum(cvx.imag(x) + cvx.real(x))), [cvx.abs(x) <= 2])
        result = prob.solve()
        self.assertAlmostEqual(result, 4*np.sqrt(2))
        val = np.ones(2)*np.sqrt(2)
        self.assertItemsAlmostEqual(x.value, val + 1j*val)

    def test_pnorm(self):
        """Test complex with pnorm.
        """
        x = Variable((1, 2), complex=True)
        prob = Problem(cvx.Maximize(cvx.sum(cvx.imag(x) + cvx.real(x))), [cvx.norm1(x) <= 2])
        result = prob.solve()
        self.assertAlmostEqual(result, 2*np.sqrt(2))
        val = np.ones(2)*np.sqrt(2)/2
        # self.assertItemsAlmostEqual(x.value, val + 1j*val)

        x = Variable((2, 2), complex=True)
        prob = Problem(cvx.Maximize(cvx.sum(cvx.imag(x) + cvx.real(x))),
                       [cvx.pnorm(x, p=2) <= np.sqrt(8)])
        result = prob.solve()
        self.assertAlmostEqual(result, 8)
        val = np.ones((2, 2))
        self.assertItemsAlmostEqual(x.value, val + 1j*val)

    def test_matrix_norms(self):
        """Test matrix norms.
        """
        P = np.arange(8) - 2j*np.arange(8)
        P = np.reshape(P, (2, 4))
        sigma_max = np.linalg.norm(P, 2)
        X = Variable((2, 4), complex=True)
        prob = Problem(Minimize(cvx.norm(X, 2)), [X == P])
        result = prob.solve()
        self.assertAlmostEqual(result, sigma_max, places=1)

        norm_nuc = np.linalg.norm(P, 'nuc')
        X = Variable((2, 4), complex=True)
        prob = Problem(Minimize(cvx.norm(X, 'nuc')), [X == P])
        result = prob.solve(solver=cvx.SCS, eps=1e-4)
        self.assertAlmostEqual(result, norm_nuc, places=1)

    def test_log_det(self):
        """Test log det.
        """
        P = np.arange(9) - 2j*np.arange(9)
        P = np.reshape(P, (3, 3))
        P = np.conj(P.T).dot(P)/100 + np.eye(3)*.1
        value = cvx.log_det(P).value
        X = Variable((3, 3), complex=True)
        prob = Problem(cvx.Maximize(cvx.log_det(X)), [X == P])
        result = prob.solve(solver=cvx.SCS, eps=1e-6)
        self.assertAlmostEqual(result, value, places=2)

    def test_eigval_atoms(self):
        """Test eigenvalue atoms.
        """
        P = np.arange(9) - 2j*np.arange(9)
        P = np.reshape(P, (3, 3))
        P1 = np.conj(P.T).dot(P)/10 + np.eye(3)*.1
        P2 = np.array([[10, 1j, 0], [-1j, 10, 0], [0, 0, 1]])
        for P in [P1, P2]:
            value = cvx.lambda_max(P).value
            X = Variable(P.shape, complex=True)
            prob = Problem(cvx.Minimize(cvx.lambda_max(X)), [X == P])
            result = prob.solve(solver=cvx.SCS, eps=1e-5)
            self.assertAlmostEqual(result, value, places=2)

            eigs = np.linalg.eigvals(P).real
            value = cvx.sum_largest(eigs, 2).value
            X = Variable(P.shape, complex=True)
            prob = Problem(cvx.Minimize(cvx.lambda_sum_largest(X, 2)), [X == P])
            result = prob.solve(solver=cvx.SCS, eps=1e-8, verbose=True)
            self.assertAlmostEqual(result, value, places=3)
            self.assertItemsAlmostEqual(X.value, P, places=3)

            value = cvx.sum_smallest(eigs, 2).value
            X = Variable(P.shape, complex=True)
            prob = Problem(cvx.Maximize(cvx.lambda_sum_smallest(X, 2)), [X == P])
            result = prob.solve(solver=cvx.SCS, eps=1e-6)
            self.assertAlmostEqual(result, value, places=3)

    def test_quad_form(self):
        """Test quad_form atom.
        """
        # Create a random positive definite Hermitian matrix for all tests.
        np.random.seed(42)
        P = np.random.randn(3, 3) - 1j*np.random.randn(3, 3)
        P = np.conj(P.T).dot(P)

        # Solve a problem with real variable
        b = np.arange(3)
        x = Variable(3, complex=False)
        value = cvx.quad_form(b, P).value
        prob = Problem(cvx.Minimize(cvx.quad_form(x, P)), [x == b])
        result = prob.solve()
        self.assertAlmostEqual(result, value)

        # Solve a problem with complex variable
        b = np.arange(3) + 3j*(np.arange(3) + 10)
        x = Variable(3, complex=True)
        value = cvx.quad_form(b, P).value
        prob = Problem(cvx.Minimize(cvx.quad_form(x, P)), [x == b])
        result = prob.solve()
        normalization = max(abs(result), abs(value))
        self.assertAlmostEqual(result / normalization, value / normalization, places=5)

        # Solve a problem with an imaginary variable
        b = 3j*(np.arange(3) + 10)
        x = Variable(3, imag=True)
        value = cvx.quad_form(b, P).value
        expr = cvx.quad_form(x, P)
        prob = Problem(cvx.Minimize(expr), [x == b])
        result = prob.solve()
        normalization = max(abs(result), abs(value))
        self.assertAlmostEqual(result / normalization, value / normalization)

    def test_matrix_frac(self):
        """Test matrix_frac atom.
        """
        P = np.array([[10, 1j], [-1j, 10]])
        Y = Variable((2, 2), complex=True)
        b = np.arange(2)
        x = Variable(2, complex=False)
        value = cvx.matrix_frac(b, P).value
        expr = cvx.matrix_frac(x, Y)
        prob = Problem(cvx.Minimize(expr), [x == b, Y == P])
        result = prob.solve(solver=cvx.SCS, eps=1e-6, max_iters=7500, verbose=True)
        self.assertAlmostEqual(result, value, places=3)

        b = (np.arange(2) + 3j*(np.arange(2) + 10))
        x = Variable(2, complex=True)
        value = cvx.matrix_frac(b, P).value
        expr = cvx.matrix_frac(x, Y)
        prob = Problem(cvx.Minimize(expr), [x == b, Y == P])
        result = prob.solve(solver=cvx.SCS, eps=1e-6)
        self.assertAlmostEqual(result, value, places=3)

        b = (np.arange(2) + 10)/10j
        x = Variable(2, imag=True)
        value = cvx.matrix_frac(b, P).value
        expr = cvx.matrix_frac(x, Y)
        prob = Problem(cvx.Minimize(expr), [x == b, Y == P])
        result = prob.solve(solver=cvx.SCS, eps=1e-5, max_iters=7500)
        self.assertAlmostEqual(result, value, places=3)

    def test_quad_over_lin(self):
        """Test quad_over_lin atom.
        """
        P = np.array([[10, 1j], [-1j, 10]])
        X = Variable((2, 2), complex=True)
        b = 1
        y = Variable(complex=False)

        value = cvx.quad_over_lin(P, b).value
        expr = cvx.quad_over_lin(X, y)
        prob = Problem(cvx.Minimize(expr), [X == P, y == b])
        result = prob.solve(solver=cvx.SCS, eps=1e-6, max_iters=7500, verbose=True)
        self.assertAlmostEqual(result, value, places=3)

        expr = cvx.quad_over_lin(X - P, y)
        prob = Problem(cvx.Minimize(expr), [y == b])
        result = prob.solve(solver=cvx.SCS, eps=1e-6, max_iters=7500, verbose=True)
        self.assertAlmostEqual(result, 0, places=3)
        self.assertItemsAlmostEqual(X.value, P, places=3)

    def test_hermitian(self):
        """Test Hermitian variables.
        """
        X = Variable((2, 2), hermitian=True)
        prob = Problem(cvx.Minimize(cvx.imag(X[1, 0])),
                       [X[0, 0] == 2, X[1, 1] == 3, X[0, 1] == 1+1j])
        prob.solve()
        self.assertItemsAlmostEqual(X.value, [2, 1-1j, 1+1j, 3])

    def test_psd(self):
        """Test Hermitian variables.
        """
        X = Variable((2, 2), hermitian=True)
        prob = Problem(cvx.Minimize(cvx.imag(X[1, 0])),
                       [X >> 0, X[0, 0] == -1])
        prob.solve()
        assert prob.status is cvx.INFEASIBLE

    def test_promote(self):
        """Test promotion of complex variables.
        """
        v = Variable(complex=True)
        obj = cvx.Maximize(cvx.real(cvx.sum(v * np.ones((2, 2)))))
        con = [cvx.norm(v) <= 1]
        prob = cvx.Problem(obj, con)
        result = prob.solve()
        self.assertAlmostEqual(result, 4.0)

    def test_sparse(self):
        """Test problem with complex sparse matrix.
        """
        # define sparse matrix [[0, 1j],[-1j,0]]
        row = np.array([0, 1])
        col = np.array([1, 0])
        data = np.array([1j, -1j])
        A = sp.csr_matrix((data, (row, col)), shape=(2, 2))

        # Feasibility with sparse matrix
        rho = cvx.Variable((2, 2), complex=True)
        Id = np.identity(2)
        obj = cvx.Maximize(0)
        cons = [A*rho == Id]
        prob = cvx.Problem(obj, cons)
        prob.solve()
        rho_sparse = rho.value
        # infeasible here, which is wrong!

        # Feasibility with numpy array: just replace A with A.toarray()
        rho = cvx.Variable((2, 2), complex=True)
        Id = np.identity(2)
        obj = cvx.Maximize(0)
        cons = [A.toarray()*rho == Id]
        prob = cvx.Problem(obj, cons)
        prob.solve()
        self.assertItemsAlmostEqual(rho.value, rho_sparse)

    def test_special_idx(self):
        """Test with special index.
        """
        c = [0, 1]
        n = len(c)
        # Create optimization variables.
        f = cvx.Variable((n, n), hermitian=True)
        # Create constraints.
        constraints = [f >> 0]
        for k in range(1, n):
            indices = [(i * n) + i - (n - k) for i in range(n - k, n)]
            constraints += [cvx.sum(cvx.vec(f)[indices]) == c[n - k]]
        # Form objective.
        obj = cvx.Maximize(c[0] - cvx.real(cvx.trace(f)))
        # Form and solve problem.
        prob = cvx.Problem(obj, constraints)
        prob.solve()

    def test_validation(self):
        """Test that complex arguments are rejected.
        """
        x = Variable(complex=True)
        with self.assertRaises(Exception) as cm:
            (x >= 0)
        self.assertEqual(str(cm.exception), "Inequality constraints cannot be complex.")

        with self.assertRaises(Exception) as cm:
            cvx.quad_over_lin(x, x)
        self.assertEqual(str(cm.exception),
                         "The second argument to quad_over_lin cannot be complex.")

        with self.assertRaises(Exception) as cm:
            cvx.sum_largest(x, 2)
        self.assertEqual(str(cm.exception), "Arguments to sum_largest cannot be complex.")

        x = Variable(2, complex=True)
        for atom in [cvx.geo_mean, cvx.log_sum_exp, cvx.max,
                     cvx.entr, cvx.exp, cvx.huber,
                     cvx.log, cvx.log1p, cvx.logistic]:
            name = atom.__name__
            with self.assertRaises(Exception) as cm:
                print(name)
                atom(x)
            self.assertEqual(str(cm.exception), "Arguments to %s cannot be complex." % name)

        x = Variable(2, complex=True)
        for atom in [cvx.maximum, cvx.kl_div]:
            name = atom.__name__
            with self.assertRaises(Exception) as cm:
                print(name)
                atom(x, x)
            self.assertEqual(str(cm.exception), "Arguments to %s cannot be complex." % name)

        x = Variable(2, complex=True)
        for atom in [cvx.inv_pos, cvx.sqrt, lambda x: cvx.power(x, .2)]:
            with self.assertRaises(Exception) as cm:
                atom(x)
            self.assertEqual(str(cm.exception), "Arguments to power cannot be complex.")

        x = Variable(2, complex=True)
        for atom in [cvx.harmonic_mean, lambda x: cvx.pnorm(x, .2)]:
            with self.assertRaises(Exception) as cm:
                atom(x)
            self.assertEqual(str(cm.exception), "pnorm(x, p) cannot have x complex for p < 1.")
