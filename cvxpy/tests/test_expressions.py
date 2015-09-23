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

from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.expressions.expression import *
from cvxpy.expressions.variables import Variable, Semidef
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.constants import Parameter
from cvxpy import Problem, Minimize
import cvxpy.utilities as u
import cvxpy.interface.matrix_utilities as intf
import cvxpy.settings as s
from collections import deque
import unittest
from cvxpy.tests.base_test import BaseTest
from cvxopt import matrix
import numpy as np
import warnings

class TestExpressions(BaseTest):
    """ Unit tests for the expression/expression module. """
    def setUp(self):
        self.a = Variable(name='a')

        self.x = Variable(2, name='x')
        self.y = Variable(3, name='y')
        self.z = Variable(2, name='z')

        self.A = Variable(2,2,name='A')
        self.B = Variable(2,2,name='B')
        self.C = Variable(3,2,name='C')
        self.intf = intf.DEFAULT_INTF

    # Test the Variable class.
    def test_variable(self):
        x = Variable(2)
        y = Variable(2)
        assert y.name() != x.name()

        x = Variable(2, name='x')
        y = Variable()
        self.assertEqual(x.name(), 'x')
        self.assertEqual(x.size, (2,1))
        self.assertEqual(y.size, (1,1))
        self.assertEqual(x.curvature, u.Curvature.AFFINE_KEY)
        self.assertEqual(x.canonical_form[0].size, (2,1))
        self.assertEqual(x.canonical_form[1], [])

        self.assertEquals(repr(self.x), "Variable(2, 1)")
        self.assertEquals(repr(self.A), "Variable(2, 2)")

        # # Scalar variable
        # coeff = self.a.coefficients()
        # self.assertEqual(coeff[self.a.id], [1])

        # # Vector variable.
        # coeffs = x.coefficients()
        # self.assertItemsEqual(coeffs.keys(), [x.id])
        # vec = coeffs[x.id][0]
        # self.assertEqual(vec.shape, (2,2))
        # self.assertEqual(vec[0,0], 1)

        # # Matrix variable.
        # coeffs = self.A.coefficients()
        # self.assertItemsEqual(coeffs.keys(), [self.A.id])
        # self.assertEqual(len(coeffs[self.A.id]), 2)
        # mat = coeffs[self.A.id][1]
        # self.assertEqual(mat.shape, (2,4))
        # self.assertEqual(mat[0,2], 1)

    def test_assign_var_value(self):
        """Test assigning a value to a variable.
        """
        # Scalar variable.
        a = Variable()
        a.value = 1
        self.assertEqual(a.value, 1)
        with self.assertRaises(Exception) as cm:
            a.value = [2, 1]
        self.assertEqual(str(cm.exception), "Invalid dimensions (2, 1) for Variable value.")

        # Test assigning None.
        a.value = 1
        a.value = None
        assert a.value is None

        # Vector variable.
        x = Variable(2)
        x.value = [2, 1]
        self.assertItemsAlmostEqual(x.value, [2, 1])
        # Matrix variable.
        A = Variable(3, 2)
        A.value = np.ones((3, 2))
        self.assertItemsAlmostEqual(A.value, np.ones((3, 2)))

    # Test tranposing variables.
    def test_transpose_variable(self):
        var = self.a.T
        self.assertEquals(var.name(), "a")
        self.assertEquals(var.size, (1,1))

        self.a.save_value(2)
        self.assertEquals(var.value, 2)

        var = self.x.T
        self.assertEquals(var.name(), "x.T")
        self.assertEquals(var.size, (1,2))

        self.x.save_value( matrix([1,2]) )
        self.assertEquals(var.value[0,0], 1)
        self.assertEquals(var.value[0,1], 2)

        var = self.C.T
        self.assertEquals(var.name(), "C.T")
        self.assertEquals(var.size, (2,3))

        # coeffs = var.canonical_form[0].coefficients()
        # mat = coeffs.values()[0][0]
        # self.assertEqual(mat.size, (2,6))
        # self.assertEqual(mat[1,3], 1)

        index = var[1,0]
        self.assertEquals(index.name(), "C.T[1, 0]")
        self.assertEquals(index.size, (1,1))

        var = self.x.T.T
        self.assertEquals(var.name(), "x.T.T")
        self.assertEquals(var.size, (2,1))

    # Test the Constant class.
    def test_constants(self):
        c = Constant(2)
        self.assertEqual(c.name(), str(2))

        c = Constant(2)
        self.assertEqual(c.value, 2)
        self.assertEqual(c.size, (1,1))
        self.assertEqual(c.curvature, u.Curvature.CONSTANT_KEY)
        self.assertEqual(c.sign, u.Sign.POSITIVE_KEY)
        self.assertEqual(Constant(-2).sign, u.Sign.NEGATIVE_KEY)
        self.assertEqual(Constant(0).sign, u.Sign.ZERO_KEY)
        self.assertEqual(c.canonical_form[0].size, (1,1))
        self.assertEqual(c.canonical_form[1], [])

        # coeffs = c.coefficients()
        # self.assertEqual(coeffs.keys(), [s.CONSTANT])
        # self.assertEqual(coeffs[s.CONSTANT], [2])

        # Test the sign.
        c = Constant([[2], [2]])
        self.assertEqual(c.size, (1, 2))
        self.assertEqual(c.sign, u.Sign.POSITIVE_KEY)
        self.assertEqual((-c).sign, u.Sign.NEGATIVE_KEY)
        self.assertEqual((0*c).sign, u.Sign.ZERO_KEY)
        c = Constant([[2], [-2]])
        self.assertEqual(c.sign, u.Sign.UNKNOWN_KEY)

        # Test sign of a complex expression.
        c = Constant([1, 2])
        A = Constant([[1,1],[1,1]])
        exp = c.T*A*c
        self.assertEqual(exp.sign, u.Sign.POSITIVE_KEY)
        self.assertEqual((c.T*c).sign, u.Sign.POSITIVE_KEY)
        exp = c.T.T
        self.assertEqual(exp.sign, u.Sign.POSITIVE_KEY)
        exp = c.T*self.A
        self.assertEqual(exp.sign, u.Sign.UNKNOWN_KEY)

        # Test repr.
        self.assertEqual(repr(c), "Constant(CONSTANT, POSITIVE, (2, 1))")

    def test_1D_array(self):
        """Test NumPy 1D arrays as constants.
        """
        c = np.array([1,2])
        p = Parameter(2)
        p.value = [1,1]
        self.assertEquals((c*p).value, 3)
        self.assertEqual((c*self.x).size, (1,1))

    # Test the Parameter class.
    def test_parameters(self):
        p = Parameter(name='p')
        self.assertEqual(p.name(), "p")
        self.assertEqual(p.size, (1,1))

        p = Parameter(4, 3, sign="positive")
        with self.assertRaises(Exception) as cm:
            p.value = 1
        self.assertEqual(str(cm.exception), "Invalid dimensions (1, 1) for Parameter value.")

        val = -np.ones((4,3))
        val[0,0] = 2

        p = Parameter(4, 3, sign="positive")
        with self.assertRaises(Exception) as cm:
            p.value = val
        self.assertEqual(str(cm.exception), "Invalid sign for Parameter value.")

        p = Parameter(4, 3, sign="negative")
        with self.assertRaises(Exception) as cm:
            p.value = val
        self.assertEqual(str(cm.exception), "Invalid sign for Parameter value.")

        # No error for unknown sign.
        p = Parameter(4, 3)
        p.value = val

        # Initialize a parameter with a value.
        p = Parameter(value=10)
        self.assertEqual(p.value, 10)

        # Test assigning None.
        p.value = 10
        p.value = None
        assert p.value is None

        with self.assertRaises(Exception) as cm:
            p = Parameter(2, 1, sign="negative", value=[2,1])
        self.assertEqual(str(cm.exception), "Invalid sign for Parameter value.")

        with self.assertRaises(Exception) as cm:
            p = Parameter(4, 3, sign="positive", value=[1,2])
        self.assertEqual(str(cm.exception), "Invalid dimensions (2, 1) for Parameter value.")

        # Test repr.
        p = Parameter(4, 3, sign="negative")
        self.assertEqual(repr(p), 'Parameter(4, 3, sign="NEGATIVE")')

    # Test the AddExpresion class.
    def test_add_expression(self):
        # Vectors
        c = Constant([2,2])
        exp = self.x + c
        self.assertEqual(exp.curvature, u.Curvature.AFFINE_KEY)
        self.assertEqual(exp.sign, u.Sign.UNKNOWN_KEY)
        self.assertEqual(exp.canonical_form[0].size, (2,1))
        self.assertEqual(exp.canonical_form[1], [])
        # self.assertEqual(exp.name(), self.x.name() + " + " + c.name())
        self.assertEqual(exp.size, (2,1))

        z = Variable(2, name='z')
        exp = exp + z + self.x

        with self.assertRaises(Exception) as cm:
            (self.x + self.y)
        self.assertEqual(str(cm.exception), "Incompatible dimensions (2, 1) (3, 1)")

        # Matrices
        exp = self.A + self.B
        self.assertEqual(exp.curvature, u.Curvature.AFFINE_KEY)
        self.assertEqual(exp.size, (2,2))

        with self.assertRaises(Exception) as cm:
            (self.A + self.C)
        self.assertEqual(str(cm.exception), "Incompatible dimensions (2, 2) (3, 2)")

        with self.assertRaises(Exception) as cm:
            AddExpression([self.A, self.C])
        self.assertEqual(str(cm.exception), "Incompatible dimensions (2, 2) (3, 2)")

        # Test that sum is flattened.
        exp = self.x + c + self.x
        self.assertEqual(len(exp.args), 3)

        # Test repr.
        self.assertEqual(repr(exp), "Expression(AFFINE, UNKNOWN, (2, 1))")

    # Test the SubExpresion class.
    def test_sub_expression(self):
        # Vectors
        c = Constant([2,2])
        exp = self.x - c
        self.assertEqual(exp.curvature, u.Curvature.AFFINE_KEY)
        self.assertEqual(exp.sign, u.Sign.UNKNOWN_KEY)
        self.assertEqual(exp.canonical_form[0].size, (2,1))
        self.assertEqual(exp.canonical_form[1], [])
        # self.assertEqual(exp.name(), self.x.name() + " - " + Constant([2,2]).name())
        self.assertEqual(exp.size, (2,1))

        z = Variable(2, name='z')
        exp = exp - z - self.x

        with self.assertRaises(Exception) as cm:
            (self.x - self.y)
        self.assertEqual(str(cm.exception), "Incompatible dimensions (2, 1) (3, 1)")

        # Matrices
        exp = self.A - self.B
        self.assertEqual(exp.curvature, u.Curvature.AFFINE_KEY)
        self.assertEqual(exp.size, (2,2))

        with self.assertRaises(Exception) as cm:
            (self.A - self.C)
        self.assertEqual(str(cm.exception), "Incompatible dimensions (2, 2) (3, 2)")

        # Test repr.
        self.assertEqual(repr(self.x - c), "Expression(AFFINE, UNKNOWN, (2, 1))")

    # Test the MulExpresion class.
    def test_mul_expression(self):
        # Vectors
        c = Constant([[2],[2]])
        exp = c*self.x
        self.assertEqual(exp.curvature, u.Curvature.AFFINE_KEY)
        self.assertEqual((c[0]*self.x).sign, u.Sign.UNKNOWN_KEY)
        self.assertEqual(exp.canonical_form[0].size, (1,1))
        self.assertEqual(exp.canonical_form[1], [])
        # self.assertEqual(exp.name(), c.name() + " * " + self.x.name())
        self.assertEqual(exp.size, (1,1))

        with self.assertRaises(Exception) as cm:
            ([2,2,3]*self.x)
        self.assertEqual(str(cm.exception), "Incompatible dimensions (3, 1) (2, 1)")

        # Matrices
        with self.assertRaises(Exception) as cm:
            Constant([[2, 1],[2, 2]]) * self.C
        self.assertEqual(str(cm.exception), "Incompatible dimensions (2, 2) (3, 2)")

        with self.assertRaises(Exception) as cm:
            (self.A * self.B)
        self.assertEqual(str(cm.exception), "Cannot multiply two non-constants.")

        # Constant expressions
        T = Constant([[1,2,3],[3,5,5]])
        exp = (T + T) * self.B
        self.assertEqual(exp.curvature, u.Curvature.AFFINE_KEY)
        self.assertEqual(exp.size, (3,2))

        # Expression that would break sign multiplication without promotion.
        c = Constant([[2], [2], [-2]])
        exp = [[1], [2]] + c*self.C
        self.assertEqual(exp.sign, u.Sign.UNKNOWN_KEY)

        # Scalar constants on the right should be moved left.
        expr = self.C*2
        self.assertEqual(expr.args[0].value, 2)

        # Scalar variables on the left should be moved right.
        expr = self.a*[2,1]
        self.assertItemsAlmostEqual(expr.args[0].value, [2,1])

    # Test the DivExpresion class.
    def test_div_expression(self):
        # Vectors
        exp = self.x/2
        self.assertEqual(exp.curvature, u.Curvature.AFFINE_KEY)
        self.assertEqual(exp.sign, u.Sign.UNKNOWN_KEY)
        self.assertEqual(exp.canonical_form[0].size, (2,1))
        self.assertEqual(exp.canonical_form[1], [])
        # self.assertEqual(exp.name(), c.name() + " * " + self.x.name())
        self.assertEqual(exp.size, (2,1))

        with self.assertRaises(Exception) as cm:
            (self.x/[2,2,3])
        print(cm.exception)
        self.assertEqual(str(cm.exception), "Can only divide by a scalar constant.")

        # Constant expressions.
        c = Constant(2)
        exp = c/(3 - 5)
        self.assertEqual(exp.curvature, u.Curvature.CONSTANT_KEY)
        self.assertEqual(exp.size, (1,1))
        self.assertEqual(exp.sign, u.Sign.NEGATIVE_KEY)

        # Parameters.
        p = Parameter(sign="positive")
        exp = 2/p
        p.value = 2
        self.assertEquals(exp.value, 1)

        rho = Parameter(sign="positive")
        rho.value = 1

        self.assertEquals(rho.sign, u.Sign.POSITIVE_KEY)
        self.assertEquals(Constant(2).sign, u.Sign.POSITIVE_KEY)
        self.assertEquals((Constant(2)/Constant(2)).sign, u.Sign.POSITIVE_KEY)
        self.assertEquals((Constant(2)*rho).sign, u.Sign.POSITIVE_KEY)
        self.assertEquals((rho/2).sign, u.Sign.POSITIVE_KEY)

    # Test the NegExpression class.
    def test_neg_expression(self):
        # Vectors
        exp = -self.x
        self.assertEqual(exp.curvature, u.Curvature.AFFINE_KEY)
        assert exp.is_affine()
        self.assertEqual(exp.sign, u.Sign.UNKNOWN_KEY)
        assert not exp.is_positive()
        self.assertEqual(exp.canonical_form[0].size, (2,1))
        self.assertEqual(exp.canonical_form[1], [])
        # self.assertEqual(exp.name(), "-%s" % self.x.name())
        self.assertEqual(exp.size, self.x.size)

        # Matrices
        exp = -self.C
        self.assertEqual(exp.curvature, u.Curvature.AFFINE_KEY)
        self.assertEqual(exp.size, (3,2))

    # Test promotion of scalar constants.
    def test_scalar_const_promotion(self):
        # Vectors
        exp = self.x + 2
        self.assertEqual(exp.curvature, u.Curvature.AFFINE_KEY)
        assert exp.is_affine()
        self.assertEqual(exp.sign, u.Sign.UNKNOWN_KEY)
        assert not exp.is_negative()
        self.assertEqual(exp.canonical_form[0].size, (2,1))
        self.assertEqual(exp.canonical_form[1], [])
        # self.assertEqual(exp.name(), self.x.name() + " + " + Constant(2).name())
        self.assertEqual(exp.size, (2,1))

        self.assertEqual((4 - self.x).size, (2,1))
        self.assertEqual((4 * self.x).size, (2,1))
        self.assertEqual((4 <= self.x).size, (2,1))
        self.assertEqual((4 == self.x).size, (2,1))
        self.assertEqual((self.x >= 4).size, (2,1))

        # Matrices
        exp = (self.A + 2) + 4
        self.assertEqual(exp.curvature, u.Curvature.AFFINE_KEY)
        self.assertEqual((3 * self.A).size, (2,2))

        self.assertEqual(exp.size, (2,2))

    # Test indexing expression.
    def test_index_expression(self):
        # Tuple of integers as key.
        exp = self.x[1,0]
        # self.assertEqual(exp.name(), "x[1,0]")
        self.assertEqual(exp.curvature, u.Curvature.AFFINE_KEY)
        assert exp.is_affine()
        self.assertEquals(exp.size, (1,1))
        # coeff = exp.canonical_form[0].coefficients()[self.x][0]
        # self.assertEqual(coeff[0,1], 1)
        self.assertEqual(exp.value, None)

        exp = self.x[1,0].T
        # self.assertEqual(exp.name(), "x[1,0]")
        self.assertEqual(exp.curvature, u.Curvature.AFFINE_KEY)
        self.assertEquals(exp.size, (1,1))

        with self.assertRaises(Exception) as cm:
            (self.x[2,0])
        self.assertEqual(str(cm.exception), "Index/slice out of bounds.")

        # Slicing
        exp = self.C[0:2,1]
        # self.assertEquals(exp.name(), "C[0:2,1]")
        self.assertEquals(exp.size, (2,1))
        exp = self.C[0:,0:2]
        # self.assertEquals(exp.name(), "C[0:,0:2]")
        self.assertEquals(exp.size, (3,2))
        exp = self.C[0::2,0::2]
        # self.assertEquals(exp.name(), "C[0::2,0::2]")
        self.assertEquals(exp.size, (2,1))
        exp = self.C[:3,:1:2]
        # self.assertEquals(exp.name(), "C[0:3,0]")
        self.assertEquals(exp.size, (3,1))
        exp = self.C[0:,0]
        # self.assertEquals(exp.name(), "C[0:,0]")
        self.assertEquals(exp.size, (3,1))

        c = Constant([[1,-2],[0,4]])
        exp = c[1, 1]
        self.assertEqual(exp.curvature, u.Curvature.CONSTANT_KEY)
        self.assertEqual(exp.sign, u.Sign.UNKNOWN_KEY)
        self.assertEqual(c[0,1].sign, u.Sign.UNKNOWN_KEY)
        self.assertEqual(c[1,0].sign, u.Sign.UNKNOWN_KEY)
        self.assertEquals(exp.size, (1,1))
        self.assertEqual(exp.value, 4)

        c = Constant([[1,-2,3],[0,4,5],[7,8,9]])
        exp = c[0:3,0:4:2]
        self.assertEqual(exp.curvature, u.Curvature.CONSTANT_KEY)
        assert exp.is_constant()
        self.assertEquals(exp.size, (3,2))
        self.assertEqual(exp[0,1].value, 7)

        # Slice of transpose
        exp = self.C.T[0:2,1]
        self.assertEquals(exp.size, (2,1))

        # Arithmetic expression indexing
        exp = (self.x + self.z)[1,0]
        # self.assertEqual(exp.name(), "x[1,0] + z[1,0]")
        self.assertEqual(exp.curvature, u.Curvature.AFFINE_KEY)
        self.assertEqual(exp.sign, u.Sign.UNKNOWN_KEY)
        self.assertEquals(exp.size, (1,1))

        exp = (self.x + self.a)[1,0]
        # self.assertEqual(exp.name(), "x[1,0] + a")
        self.assertEqual(exp.curvature, u.Curvature.AFFINE_KEY)
        self.assertEquals(exp.size, (1,1))

        exp = (self.x - self.z)[1,0]
        # self.assertEqual(exp.name(), "x[1,0] - z[1,0]")
        self.assertEqual(exp.curvature, u.Curvature.AFFINE_KEY)
        self.assertEquals(exp.size, (1,1))

        exp = (self.x - self.a)[1,0]
        # self.assertEqual(exp.name(), "x[1,0] - a")
        self.assertEqual(exp.curvature, u.Curvature.AFFINE_KEY)
        self.assertEquals(exp.size, (1,1))

        exp = (-self.x)[1,0]
        # self.assertEqual(exp.name(), "-x[1,0]")
        self.assertEqual(exp.curvature, u.Curvature.AFFINE_KEY)
        self.assertEquals(exp.size, (1,1))

        c = Constant([[1,2],[3,4]])
        exp = (c*self.x)[1,0]
        # self.assertEqual(exp.name(), "[[2], [4]] * x[0:,0]")
        self.assertEqual(exp.curvature, u.Curvature.AFFINE_KEY)
        self.assertEquals(exp.size, (1,1))

        c = Constant([[1,2],[3,4]])
        exp = (c*self.a)[1,0]
        # self.assertEqual(exp.name(), "2 * a")
        self.assertEqual(exp.curvature, u.Curvature.AFFINE_KEY)
        self.assertEquals(exp.size, (1,1))

    def test_neg_indices(self):
        """Test negative indices.
        """
        c = Constant([[1,2],[3,4]])
        exp = c[-1, -1]
        self.assertEquals(exp.value, 4)
        self.assertEquals(exp.size, (1, 1))
        self.assertEquals(exp.curvature, u.Curvature.CONSTANT_KEY)

        c = Constant([1,2,3,4])
        exp = c[1:-1]
        self.assertItemsAlmostEqual(exp.value, [2, 3])
        self.assertEquals(exp.size, (2, 1))
        self.assertEquals(exp.curvature, u.Curvature.CONSTANT_KEY)

        c = Constant([1,2,3,4])
        exp = c[::-1]
        self.assertItemsAlmostEqual(exp.value, [4, 3, 2, 1])
        self.assertEquals(exp.size, (4, 1))
        self.assertEquals(exp.curvature, u.Curvature.CONSTANT_KEY)

        x = Variable(4)
        Problem(Minimize(0), [x[::-1] == c]).solve()
        self.assertItemsAlmostEqual(x.value, [4, 3, 2, 1])
        self.assertEquals(x[::-1].size, (4, 1))

        x = Variable(2)
        self.assertEquals(x[::-1].size, (2, 1))

        x = Variable(100, name="x")
        self.assertEquals("x[:-1, 0]", str(x[:-1]))

    def test_logical_indices(self):
        """Test indexing with logical arrays.
        """
        pass

    def test_powers(self):
        exp = self.x**2
        self.assertEqual(exp.curvature, u.Curvature.CONVEX_KEY)
        exp = self.x**0.5
        self.assertEqual(exp.curvature, u.Curvature.CONCAVE_KEY)
        exp = self.x**-1
        self.assertEqual(exp.curvature, u.Curvature.CONVEX_KEY)

    def test_sum(self):
        """Test built-in sum. Not good usage.
        """
        self.a.value = 1
        expr = sum(self.a)
        self.assertEquals(expr.value, 1)

        self.x.value = [1, 2]
        expr = sum(self.x)
        self.assertEquals(expr.value, 3)

    def test_var_copy(self):
        """Test the copy function for variable types.
        """
        x = Variable(3, 4, name="x")
        y = x.copy()
        self.assertEquals(y.size, (3, 4))
        self.assertEquals(y.name(), "x")

        x = Semidef(5, name="x")
        y = x.copy()
        self.assertEquals(y.size, (5, 5))

    def test_param_copy(self):
        """Test the copy function for Parameters.
        """
        x = Parameter(3, 4, name="x", sign="positive")
        y = x.copy()
        self.assertEquals(y.size, (3, 4))
        self.assertEquals(y.name(), "x")
        self.assertEquals(y.sign, "POSITIVE")

    def test_constant_copy(self):
        """Test the copy function for Constants.
        """
        x = Constant(2)
        y = x.copy()
        self.assertEquals(y.size, (1, 1))
        self.assertEquals(y.value, 2)
