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

import numpy as np
import scipy.sparse as sp

import cvxpy as cp
from cvxpy import Minimize, Problem
from cvxpy.expressions.constants import Constant, Parameter
from cvxpy.expressions.variable import Variable
from cvxpy.tests.base_test import BaseTest


class TestComplex(BaseTest):
    """ Unit tests for the expression/expression module. """

    def test_variable(self) -> None:
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

    def test_parameter(self) -> None:
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

    def test_constant(self) -> None:
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

    def test_objective(self) -> None:
        """Test objectives.
        """
        x = Variable(complex=True)
        with self.assertRaises(Exception) as cm:
            Minimize(x)
        self.assertEqual(str(cm.exception), "The 'minimize' objective must be real valued.")

        with self.assertRaises(Exception) as cm:
            cp.Maximize(x)
        self.assertEqual(str(cm.exception), "The 'maximize' objective must be real valued.")

    def test_arithmetic(self) -> None:
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
        expr = (A*A)*y
        assert expr.is_complex()
        assert expr.is_imag()

    def test_real(self) -> None:
        """Test real.
        """
        A = np.ones((2, 2))
        expr = Constant(A) + 1j*Constant(A)
        expr = cp.real(expr)
        assert expr.is_real()
        assert not expr.is_complex()
        assert not expr.is_imag()
        self.assertItemsAlmostEqual(expr.value, A)

        x = Variable(complex=True)
        expr = cp.imag(x) + cp.real(x)
        assert expr.is_real()

    def test_imag(self) -> None:
        """Test imag.
        """
        A = np.ones((2, 2))
        expr = Constant(A) + 2j*Constant(A)
        expr = cp.imag(expr)
        assert expr.is_real()
        assert not expr.is_complex()
        assert not expr.is_imag()
        self.assertItemsAlmostEqual(expr.value, 2*A)

    def test_conj(self) -> None:
        """Test imag.
        """
        A = np.ones((2, 2))
        expr = Constant(A) + 1j*Constant(A)
        expr = cp.conj(expr)
        assert not expr.is_real()
        assert expr.is_complex()
        assert not expr.is_imag()
        self.assertItemsAlmostEqual(expr.value, A - 1j*A)

    def test_affine_atoms_canon(self) -> None:
        """Test canonicalization for affine atoms.
        """
        # Scalar.
        x = Variable()
        expr = cp.imag(x + 1j*x)
        prob = Problem(Minimize(expr), [x >= 0])
        result = prob.solve(solver="SCS", eps=1e-6)
        self.assertAlmostEqual(result, 0)
        self.assertAlmostEqual(x.value, 0)

        x = Variable(imag=True)
        expr = 1j*x
        prob = Problem(Minimize(expr), [cp.imag(x) <= 1])
        result = prob.solve(solver="SCS", eps=1e-6)
        self.assertAlmostEqual(result, -1)
        self.assertAlmostEqual(x.value, 1j)

        x = Variable(2)
        expr = x*1j
        prob = Problem(Minimize(expr[0]*1j + expr[1]*1j), [cp.real(x + 1j) >= 1])
        result = prob.solve(solver="SCS", eps=1e-6)
        self.assertAlmostEqual(result, -np.inf)
        prob = Problem(Minimize(expr[0]*1j + expr[1]*1j), [cp.real(x + 1j) <= 1])
        result = prob.solve(solver="SCS", eps=1e-6)
        self.assertAlmostEqual(result, -2)
        self.assertItemsAlmostEqual(x.value, [1, 1])
        prob = Problem(Minimize(expr[0]*1j + expr[1]*1j), [cp.real(x + 1j) >= 1, cp.conj(x) <= 0])
        result = prob.solve(solver="SCS", eps=1e-6)
        self.assertAlmostEqual(result, np.inf)

        x = Variable((2, 2))
        y = Variable((3, 2), complex=True)
        expr = cp.vstack([x, y])
        prob = Problem(Minimize(cp.sum(cp.imag(cp.conj(expr)))),
                       [x == 0, cp.real(y) == 0, cp.imag(y) <= 1])
        result = prob.solve(solver="SCS", eps=1e-6)
        self.assertAlmostEqual(result, -6)
        self.assertItemsAlmostEqual(y.value, 1j*np.ones((3, 2)))
        self.assertItemsAlmostEqual(x.value, np.zeros((2, 2)))

        x = Variable((2, 2))
        y = Variable((3, 2), complex=True)
        expr = cp.vstack([x, y])
        prob = Problem(Minimize(cp.sum(cp.imag(expr.H))),
                       [x == 0, cp.real(y) == 0, cp.imag(y) <= 1])
        result = prob.solve(solver="SCS", eps=1e-6)
        self.assertAlmostEqual(result, -6)
        self.assertItemsAlmostEqual(y.value, 1j*np.ones((3, 2)))
        self.assertItemsAlmostEqual(x.value, np.zeros((2, 2)))

    def test_params(self) -> None:
        """Test with parameters.
        """
        p = cp.Parameter(imag=True, value=1j)
        x = Variable(2, complex=True)
        prob = Problem(cp.Maximize(cp.sum(cp.imag(x) + cp.real(x))), [cp.abs(p*x) <= 2])
        result = prob.solve(solver="SCS", eps=1e-6)
        self.assertAlmostEqual(result, 4*np.sqrt(2))
        val = np.ones(2)*np.sqrt(2)
        self.assertItemsAlmostEqual(x.value, val + 1j*val)

    def test_missing_imag(self) -> None:
        """Test problems where imaginary is missing.
        """
        Z = Variable((2, 2), hermitian=True)
        constraints = [cp.trace(cp.real(Z)) == 1]
        obj = cp.Minimize(0)
        prob = cp.Problem(obj, constraints)
        prob.solve(solver="SCS")

        Z = Variable((2, 2), imag=True)
        obj = cp.Minimize(cp.trace(cp.real(Z)))
        prob = cp.Problem(obj, constraints)
        result = prob.solve(solver="SCS")
        self.assertAlmostEqual(result, 0)

    def test_abs(self) -> None:
        """Test with absolute value.
        """
        x = Variable(2, complex=True)
        prob = Problem(cp.Maximize(cp.sum(cp.imag(x) + cp.real(x))), [cp.abs(x) <= 2])
        result = prob.solve(solver="SCS", eps=1e-6)
        self.assertAlmostEqual(result, 4*np.sqrt(2))
        val = np.ones(2)*np.sqrt(2)
        self.assertItemsAlmostEqual(x.value, val + 1j*val)

    def test_soc(self) -> None:
        """Test with SOC.
        """
        x = Variable(2, complex=True)
        t = Variable()
        prob = Problem(cp.Minimize(t), [cp.SOC(t, x), x == 2j])
        result = prob.solve(solver="SCS", eps=1e-6)
        self.assertAlmostEqual(result, 2*np.sqrt(2))
        self.assertItemsAlmostEqual(x.value, [2j, 2j])

    def test_pnorm(self) -> None:
        """Test complex with pnorm.
        """
        x = Variable((1, 2), complex=True)
        prob = Problem(cp.Maximize(cp.sum(cp.imag(x) + cp.real(x))), [cp.norm1(x) <= 2])
        result = prob.solve(solver="ECOS")
        self.assertAlmostEqual(result, 2*np.sqrt(2))
        val = np.ones(2)*np.sqrt(2)/2
        # self.assertItemsAlmostEqual(x.value, val + 1j*val)

        x = Variable((2, 2), complex=True)
        prob = Problem(cp.Maximize(cp.sum(cp.imag(x) + cp.real(x))),
                       [cp.pnorm(x, p=2) <= np.sqrt(8)])
        result = prob.solve(solver="ECOS")
        self.assertAlmostEqual(result, 8)
        val = np.ones((2, 2))
        self.assertItemsAlmostEqual(x.value, val + 1j*val)

    def test_matrix_norms(self) -> None:
        """Test matrix norms.
        """
        P = np.arange(8) - 2j*np.arange(8)
        P = np.reshape(P, (2, 4))
        sigma_max = np.linalg.norm(P, 2)
        X = Variable((2, 4), complex=True)
        prob = Problem(Minimize(cp.norm(X, 2)), [X == P])
        result = prob.solve(solver="SCS")
        self.assertAlmostEqual(result, sigma_max, places=1)

        norm_nuc = np.linalg.norm(P, 'nuc')
        X = Variable((2, 4), complex=True)
        prob = Problem(Minimize(cp.norm(X, 'nuc')), [X == P])
        result = prob.solve(solver=cp.SCS, eps=1e-4)
        self.assertAlmostEqual(result, norm_nuc, places=1)

    def test_log_det(self) -> None:
        """Test log det.
        """
        P = np.arange(9) - 2j*np.arange(9)
        P = np.reshape(P, (3, 3))
        P = np.conj(P.T).dot(P)/100 + np.eye(3)*.1
        value = cp.log_det(P).value
        X = Variable((3, 3), complex=True)
        prob = Problem(cp.Maximize(cp.log_det(X)), [X == P])
        result = prob.solve(solver=cp.SCS, eps=1e-6)
        self.assertAlmostEqual(result, value, places=2)

    def test_eigval_atoms(self) -> None:
        """Test eigenvalue atoms.
        """
        P = np.arange(9) - 2j*np.arange(9)
        P = np.reshape(P, (3, 3))
        P1 = np.conj(P.T).dot(P)/10 + np.eye(3)*.1
        P2 = np.array([[10, 1j, 0], [-1j, 10, 0], [0, 0, 1]])
        for P in [P1, P2]:
            value = cp.lambda_max(P).value
            X = Variable(P.shape, complex=True)
            prob = Problem(cp.Minimize(cp.lambda_max(X)), [X == P])
            result = prob.solve(solver=cp.SCS, eps=1e-6)
            self.assertAlmostEqual(result, value, places=2)

            eigs = np.linalg.eigvals(P).real
            value = cp.sum_largest(eigs, 2).value
            X = Variable(P.shape, complex=True)
            prob = Problem(cp.Minimize(cp.lambda_sum_largest(X, 2)), [X == P])
            result = prob.solve(solver=cp.SCS, eps=1e-8)
            self.assertAlmostEqual(result, value, places=3)
            self.assertItemsAlmostEqual(X.value, P, places=3)

            value = cp.sum_smallest(eigs, 2).value
            X = Variable(P.shape, complex=True)
            prob = Problem(cp.Maximize(cp.lambda_sum_smallest(X, 2)), [X == P])
            result = prob.solve(solver=cp.SCS, eps=1e-6)
            self.assertAlmostEqual(result, value, places=3)

    def test_quad_form(self) -> None:
        """Test quad_form atom.
        """
        # Create a random positive definite Hermitian matrix for all tests.
        np.random.seed(42)
        P = np.random.randn(3, 3) - 1j*np.random.randn(3, 3)
        P = np.conj(P.T).dot(P)

        # Solve a problem with real variable
        b = np.arange(3)
        x = Variable(3, complex=False)
        value = cp.quad_form(b, P).value
        prob = Problem(cp.Minimize(cp.quad_form(x, P)), [x == b])
        result = prob.solve(solver="ECOS")
        self.assertAlmostEqual(result, value)

        # Solve a problem with complex variable
        b = np.arange(3) + 3j*(np.arange(3) + 10)
        x = Variable(3, complex=True)
        value = cp.quad_form(b, P).value
        prob = Problem(cp.Minimize(cp.quad_form(x, P)), [x == b])
        result = prob.solve(solver="ECOS")
        normalization = max(abs(result), abs(value))
        self.assertAlmostEqual(result / normalization, value / normalization, places=5)

        # Solve a problem with an imaginary variable
        b = 3j*(np.arange(3) + 10)
        x = Variable(3, imag=True)
        value = cp.quad_form(b, P).value
        expr = cp.quad_form(x, P)
        prob = Problem(cp.Minimize(expr), [x == b])
        result = prob.solve(solver="ECOS")
        normalization = max(abs(result), abs(value))
        self.assertAlmostEqual(result / normalization, value / normalization)

    def test_matrix_frac(self) -> None:
        """Test matrix_frac atom.
        """
        P = np.array([[10, 1j], [-1j, 10]])
        Y = Variable((2, 2), complex=True)
        b = np.arange(2)
        x = Variable(2, complex=False)
        value = cp.matrix_frac(b, P).value
        expr = cp.matrix_frac(x, Y)
        prob = Problem(cp.Minimize(expr), [x == b, Y == P])
        result = prob.solve(solver=cp.SCS, eps=1e-6, max_iters=7500, verbose=True)
        self.assertAlmostEqual(result, value, places=3)

        b = (np.arange(2) + 3j*(np.arange(2) + 10))
        x = Variable(2, complex=True)
        value = cp.matrix_frac(b, P).value
        expr = cp.matrix_frac(x, Y)
        prob = Problem(cp.Minimize(expr), [x == b, Y == P])
        result = prob.solve(solver=cp.SCS, eps=1e-6)
        self.assertAlmostEqual(result, value, places=3)

        b = (np.arange(2) + 10)/10j
        x = Variable(2, imag=True)
        value = cp.matrix_frac(b, P).value
        expr = cp.matrix_frac(x, Y)
        prob = Problem(cp.Minimize(expr), [x == b, Y == P])
        result = prob.solve(solver=cp.SCS, eps=1e-5, max_iters=7500)
        self.assertAlmostEqual(result, value, places=3)

    def test_quad_over_lin(self) -> None:
        """Test quad_over_lin atom.
        """
        P = np.array([[10, 1j], [-1j, 10]])
        X = Variable((2, 2), complex=True)
        b = 1
        y = Variable(complex=False)

        value = cp.quad_over_lin(P, b).value
        expr = cp.quad_over_lin(X, y)
        prob = Problem(cp.Minimize(expr), [X == P, y == b])
        result = prob.solve(solver=cp.SCS, eps=1e-6, max_iters=7500, verbose=True)
        self.assertAlmostEqual(result, value, places=3)

        expr = cp.quad_over_lin(X - P, y)
        prob = Problem(cp.Minimize(expr), [y == b])
        result = prob.solve(solver=cp.SCS, eps=1e-6, max_iters=7500, verbose=True)
        self.assertAlmostEqual(result, 0, places=3)
        self.assertItemsAlmostEqual(X.value, P, places=3)

    def test_hermitian(self) -> None:
        """Test Hermitian variables.
        """
        X = Variable((2, 2), hermitian=True)
        prob = Problem(cp.Minimize(cp.imag(X[1, 0])),
                       [X[0, 0] == 2, X[1, 1] == 3, X[0, 1] == 1+1j])
        prob.solve(solver="SCS")
        self.assertItemsAlmostEqual(X.value, [2, 1-1j, 1+1j, 3])

    def test_psd(self) -> None:
        """Test Hermitian variables.
        """
        X = Variable((2, 2), hermitian=True)
        prob = Problem(cp.Minimize(cp.imag(X[1, 0])),
                       [X >> 0, X[0, 0] == -1])
        prob.solve(solver="SCS")
        assert prob.status is cp.INFEASIBLE

    def test_promote(self) -> None:
        """Test promotion of complex variables.
        """
        v = Variable(complex=True)
        obj = cp.Maximize(cp.real(cp.sum(v * np.ones((2, 2)))))
        con = [cp.norm(v) <= 1]
        prob = cp.Problem(obj, con)
        result = prob.solve(solver="ECOS")
        self.assertAlmostEqual(result, 4.0)

    def test_sparse(self) -> None:
        """Test problem with complex sparse matrix.
        """
        # define sparse matrix [[0, 1j],[-1j,0]]
        row = np.array([0, 1])
        col = np.array([1, 0])
        data = np.array([1j, -1j])
        A = sp.csr_matrix((data, (row, col)), shape=(2, 2))

        # Feasibility with sparse matrix
        rho = cp.Variable((2, 2), complex=True)
        Id = np.identity(2)
        obj = cp.Maximize(0)
        cons = [A @ rho == Id]
        prob = cp.Problem(obj, cons)
        prob.solve(solver="SCS")
        rho_sparse = rho.value
        # infeasible here, which is wrong!

        # Feasibility with numpy array: just replace A with A.toarray()
        rho = cp.Variable((2, 2), complex=True)
        Id = np.identity(2)
        obj = cp.Maximize(0)
        cons = [A.toarray() @ rho == Id]
        prob = cp.Problem(obj, cons)
        prob.solve(solver="SCS")
        self.assertItemsAlmostEqual(rho.value, rho_sparse)

    def test_special_idx(self) -> None:
        """Test with special index.
        """
        c = [0, 1]
        n = len(c)
        # Create optimization variables.
        f = cp.Variable((n, n), hermitian=True)
        # Create constraints.
        constraints = [f >> 0]
        for k in range(1, n):
            indices = [(i * n) + i - (n - k) for i in range(n - k, n)]
            constraints += [cp.sum(cp.vec(f)[indices]) == c[n - k]]
        # Form objective.
        obj = cp.Maximize(c[0] - cp.real(cp.trace(f)))
        # Form and solve problem.
        prob = cp.Problem(obj, constraints)
        prob.solve(solver="SCS")

    def test_validation(self) -> None:
        """Test that complex arguments are rejected.
        """
        x = Variable(complex=True)
        with self.assertRaises(Exception) as cm:
            (x >= 0)
        self.assertEqual(str(cm.exception), "Inequality constraints cannot be complex.")

        with self.assertRaises(Exception) as cm:
            cp.quad_over_lin(x, x)
        self.assertEqual(str(cm.exception),
                         "The second argument to quad_over_lin cannot be complex.")

        with self.assertRaises(Exception) as cm:
            cp.sum_largest(x, 2)
        self.assertEqual(str(cm.exception), "Arguments to sum_largest cannot be complex.")

        x = Variable(2, complex=True)
        for atom in [cp.geo_mean, cp.log_sum_exp, cp.max,
                     cp.entr, cp.exp, cp.huber,
                     cp.log, cp.log1p, cp.logistic]:
            name = atom.__name__
            with self.assertRaises(Exception) as cm:
                print(name)
                atom(x)
            self.assertEqual(str(cm.exception), "Arguments to %s cannot be complex." % name)

        x = Variable(2, complex=True)
        for atom in [cp.maximum, cp.kl_div]:
            name = atom.__name__
            with self.assertRaises(Exception) as cm:
                print(name)
                atom(x, x)
            self.assertEqual(str(cm.exception), "Arguments to %s cannot be complex." % name)

        x = Variable(2, complex=True)
        for atom in [cp.inv_pos, cp.sqrt, lambda x: cp.power(x, .2)]:
            with self.assertRaises(Exception) as cm:
                atom(x)
            self.assertEqual(str(cm.exception), "Arguments to power cannot be complex.")

        x = Variable(2, complex=True)
        for atom in [cp.harmonic_mean, lambda x: cp.pnorm(x, .2)]:
            with self.assertRaises(Exception) as cm:
                atom(x)
            self.assertEqual(str(cm.exception), "pnorm(x, p) cannot have x complex for p < 1.")

    def test_diag(self) -> None:
        """Test diag of mat, and of vector.
        """
        X = cp.Variable((2, 2), complex=True)
        obj = cp.Maximize(cp.trace(cp.real(X)))
        cons = [cp.diag(X) == 1]
        prob = cp.Problem(obj, cons)
        result = prob.solve(solver="SCS")
        self.assertAlmostEqual(result, 2)

        x = cp.Variable(2, complex=True)
        X = cp.diag(x)
        obj = cp.Maximize(cp.trace(cp.real(X)))
        cons = [cp.diag(X) == 1]
        prob = cp.Problem(obj, cons)
        result = prob.solve(solver="SCS")
        self.assertAlmostEqual(result, 2)

    def test_complex_qp(self) -> None:
        """Test a QP with a complex variable.
        """
        A0 = np.array([0+1j, 2-1j])
        A1 = np.array([[2, -1+1j], [4-3j, -3+2j]])
        Z = cp.Variable(complex=True)
        X = cp.Variable(2)
        B = np.array([2+1j, -2j])

        objective = cp.Minimize(cp.sum_squares(A0*Z + A1@X - B))
        prob = cp.Problem(objective)
        prob.solve(solver="SCS")
        self.assertEqual(prob.status, cp.OPTIMAL)

        constraints = [X >= 0]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver="SCS")
        self.assertEqual(prob.status, cp.OPTIMAL)
        assert constraints[0].dual_value is not None

    def test_quad_psd(self) -> None:
        """Test PSD checking from #1491.
        """
        x = cp.Variable(2, complex=True)
        P1 = np.eye(2)
        P2 = np.array([[1+0j, 0+0j],
                       [0-0j, 1+0j]])
        print("P1 is real:", cp.quad_form(x, P1).curvature)
        print("P2 is complex:", cp.quad_form(x, P2).curvature)
        assert cp.quad_form(x, P2).is_dcp()

    def test_bool(self) -> None:
        # The purpose of this test is to make sure
        # that we don't try to recover dual variables
        # unless they're actually present.
        #
        # Added as part of fixing GitHub issue 1133.
        bool_var = cp.Variable(boolean=True)
        complex_var = cp.Variable(complex=True)

        constraints = [
            cp.real(complex_var) <= bool_var,
        ]

        obj = cp.Maximize(cp.real(complex_var))
        prob = cp.Problem(obj, constraints)
        prob.solve(solver='ECOS_BB')
        self.assertAlmostEqual(prob.value, 1, places=4)

    def test_partial_trace(self) -> None:
        """
        Test a problem with partial_trace.
        rho_ABC = rho_A \\otimes rho_B \\otimes rho_C.
        Here \\otimes signifies Kronecker product.
        Each rho_i is normalized, i.e. Tr(rho_i) = 1.
        """
        # Set random state.
        np.random.seed(1)

        # Generate test case.
        rho_A = np.random.random((4, 4)) + 1j*np.random.random((4, 4))
        rho_A /= np.trace(rho_A)
        rho_B = np.random.random((3, 3)) + 1j*np.random.random((3, 3))
        rho_B /= np.trace(rho_B)
        rho_C = np.random.random((2, 2)) + 1j*np.random.random((2, 2))
        rho_C /= np.trace(rho_C)
        rho_AB = np.kron(rho_A, rho_B)
        rho_AC = np.kron(rho_A, rho_C)

        # Construct a cvxpy Variable with value equal to rho_A \otimes rho_B \otimes rho_C.
        rho_ABC_val = np.kron(rho_AB, rho_C)
        rho_ABC = cp.Variable(shape=rho_ABC_val.shape, complex=True)
        cons = [
            rho_ABC_val == rho_ABC,
            rho_AB == cp.partial_trace(rho_ABC, [4, 3, 2], axis=2),
            rho_AC == cp.partial_trace(rho_ABC, [4, 3, 2], axis=1),
        ]
        prob = cp.Problem(cp.Minimize(0), cons)
        prob.solve()

        print(rho_ABC_val)
        assert np.allclose(rho_ABC.value, rho_ABC_val)

    def test_partial_transpose(self) -> None:
        """
        Test a problem with partial_transpose.
        rho_ABC = rho_A \\otimes rho_B \\otimes rho_C.
        Here \\otimes signifies Kronecker product.
        Each rho_i is normalized, i.e. Tr(rho_i) = 1.
        """
        # Set random state.
        np.random.seed(1)

        # Generate three test cases
        rho_A = np.random.random((8, 8)) + 1j*np.random.random((8, 8))
        rho_A /= np.trace(rho_A)
        rho_B = np.random.random((6, 6)) + 1j*np.random.random((6, 6))
        rho_B /= np.trace(rho_B)
        rho_C = np.random.random((4, 4)) + 1j*np.random.random((4, 4))
        rho_C /= np.trace(rho_C)

        rho_TC = np.kron(np.kron(rho_A, rho_B), rho_C.T)
        rho_TB = np.kron(np.kron(rho_A, rho_B.T), rho_C)

        # Construct a cvxpy Variable with value equal to rho_A \otimes rho_B \otimes rho_C.
        rho_ABC_val = np.kron(np.kron(rho_A, rho_B), rho_C)
        rho_ABC = cp.Variable(shape=rho_ABC_val.shape, complex=True)
        cons = [
            rho_ABC_val == rho_ABC,
            rho_TC == cp.partial_transpose(rho_ABC, [8, 6, 4], axis=2),
            rho_TB == cp.partial_transpose(rho_ABC, [8, 6, 4], axis=1),
        ]
        prob = cp.Problem(cp.Minimize(0), cons)
        prob.solve()

        print(rho_ABC_val)
        assert np.allclose(rho_ABC.value, rho_ABC_val)
