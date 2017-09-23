"""
Copyright 2013 Steven Diamond

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""

import cvxpy as cvx
from cvxpy.expressions.variable import Variable
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.constants import Parameter
from cvxpy import Problem, Minimize
from cvxpy.tests.base_test import BaseTest
import numpy as np
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
        self.assertItemsAlmostEqual(x.value, val + 1j*val)

        x = Variable((2, 2), complex=True)
        prob = Problem(cvx.Maximize(cvx.sum(cvx.imag(x) + cvx.real(x))),
                       [cvx.norm2(x) <= np.sqrt(8)])
        result = prob.solve()
        self.assertAlmostEqual(result, 8)
        val = np.ones((2, 2))
        self.assertItemsAlmostEqual(x.value, val + 1j*val)
