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


class TestExpressions(BaseTest):
    """ Unit tests for the expression/expression module. """

    def setUp(self):
        self.a = Variable(name='a')

        self.x = Variable(2, name='x')
        self.y = Variable(3, name='y')
        self.z = Variable(2, name='z')

        self.A = Variable((2,2), name='A')
        self.B = Variable((2,2), name='B')
        self.C = Variable((3,2), name='C')

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
