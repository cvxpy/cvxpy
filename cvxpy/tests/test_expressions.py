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

from cvxpy.expressions.expression import *
from cvxpy.expressions.variables import Variable
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.constants import Parameter
import cvxpy.utilities as u
import cvxpy.interface.matrix_utilities as intf
import cvxpy.settings as s
from collections import deque
import unittest
from cvxopt import matrix

class TestExpressions(unittest.TestCase):
    """ Unit tests for the expression/expression module. """
    def setUp(self):
        self.a = Variable(name='a')

        self.x = Variable(2, name='x')
        self.y = Variable(3, name='y')
        self.z = Variable(2, name='z')

        self.A = Variable(2,2,name='A')
        self.B = Variable(2,2,name='B')
        self.C = Variable(3,2,name='C')
        self.intf = intf.DEFAULT_INTERFACE

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
        self.assertEqual(x.curvature, u.Curvature.AFFINE)
        self.assertEqual(x.canonicalize()[0].size, (2,1))
        self.assertEqual(x.canonicalize()[1], [])

        # Scalar variable
        coeff = self.a.coefficients(self.intf)
        self.assertEqual(coeff[self.a], 1)

        # Vector variable.
        coeffs = x.coefficients(self.intf)
        self.assertItemsEqual(coeffs.keys(), [x])
        vec = coeffs[x]
        self.assertEqual(vec.size, (2,2))
        self.assertEqual(list(vec), [1,0,0,1])

        # Matrix variable.
        coeffs = self.A.coefficients(self.intf)
        self.assertItemsEqual(coeffs.keys(), [self.A])
        mat = coeffs[self.A]
        self.assertEqual(mat.size, (2,2))
        self.assertEqual(list(mat), [1,0,0,1])

    # Test the TransposeVariable class.
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

        coeffs = var.coefficients(self.intf)
        self.assertItemsEqual(coeffs.keys(), [var])
        mat = coeffs[var]
        self.assertEqual(mat.size, (2,2))
        self.assertEqual(list(mat), [1,0,0,1])

        index = var[1,0]
        self.assertEquals(index.name(), "C[0,1]")
        self.assertEquals(index.size, (1,1))

        var = self.x.T.T
        self.assertEquals(var.name(), "x")
        self.assertEquals(var.size, (2,1))

    # Test the Constant class.
    def test_constants(self):
        c = Constant(2)
        self.assertEqual(c.name(), str(2))

        c = Constant(2, name="c")
        self.assertEqual(c.name(), "c")
        self.assertEqual(c.value, 2)
        self.assertEqual(c.size, (1,1))
        self.assertEqual(c.curvature, u.Curvature.CONSTANT)
        self.assertEqual(c.sign, u.Sign.POSITIVE)
        self.assertEqual(Constant(-2).sign, u.Sign.NEGATIVE)
        self.assertEqual(Constant(0).sign, u.Sign.ZERO)
        self.assertEqual(c.canonicalize()[0].size, (1,1))
        self.assertEqual(c.canonicalize()[1], [])
        
        coeffs = c.coefficients(self.intf)
        self.assertEqual(coeffs.keys(), [Constant])
        self.assertEqual(coeffs[Constant], 2)

        # Test the sign.
        c = Constant([[2],[2]])
        self.assertEqual(c.size, (1,2))
        self.assertEqual(c.sign.neg_mat.value.shape, (1,2))

        # Test sign of a complex expression.
        c = Constant([1, 2])
        A = Constant([[1,1],[1,1]])
        exp = c.T*A*c
        self.assertEqual(exp.sign, u.Sign.POSITIVE)
        self.assertEqual((c.T*c).sign, u.Sign.POSITIVE)
        exp = c.T.T
        self.assertEqual(exp.sign.pos_mat.value.ndim, 2)
        exp = c.T*self.A
        self.assertEqual(exp.sign.pos_mat.value.ndim, 2)

    # Test the Parameter class.
    def test_parameters(self):
        p = Parameter(name='p')
        self.assertEqual(p.name(), "p")
        self.assertEqual(p.size, (1,1))

    # Test the AddExpresion class.
    def test_add_expression(self):
        # Vectors
        c = Constant([2,2])
        exp = self.x + c
        self.assertEqual(exp.curvature, u.Curvature.AFFINE)
        self.assertEqual(exp.sign, u.Sign.UNKNOWN)
        self.assertEqual(exp.canonicalize()[0].size, (2,1))
        self.assertEqual(exp.canonicalize()[1], [])
        self.assertEqual(exp.name(), self.x.name() + " + " + c.name())
        self.assertEqual(exp.size, (2,1))

        z = Variable(2, name='z')
        exp = exp + z + self.x

        with self.assertRaises(Exception) as cm:
            (self.x + self.y)
        self.assertEqual(str(cm.exception), "Incompatible dimensions.")

        # Matrices
        exp = self.A + self.B
        self.assertEqual(exp.curvature, u.Curvature.AFFINE)
        self.assertEqual(exp.size, (2,2))

        with self.assertRaises(Exception) as cm:
            (self.A + self.C)
        self.assertEqual(str(cm.exception), "Incompatible dimensions.")


    # Test the SubExpresion class.
    def test_sub_expression(self):
        # Vectors
        c = Constant([2,2])
        exp = self.x - c
        self.assertEqual(exp.curvature, u.Curvature.AFFINE)
        self.assertEqual(exp.sign, u.Sign.UNKNOWN)
        self.assertEqual(exp.canonicalize()[0].size, (2,1))
        self.assertEqual(exp.canonicalize()[1], [])
        self.assertEqual(exp.name(), self.x.name() + " - " + Constant([2,2]).name())
        self.assertEqual(exp.size, (2,1))

        z = Variable(2, name='z')
        exp = exp - z - self.x

        with self.assertRaises(Exception) as cm:
            (self.x - self.y)
        self.assertEqual(str(cm.exception), "Incompatible dimensions.")

        # Matrices
        exp = self.A - self.B
        self.assertEqual(exp.curvature, u.Curvature.AFFINE)
        self.assertEqual(exp.size, (2,2))

        with self.assertRaises(Exception) as cm:
            (self.A - self.C)
        self.assertEqual(str(cm.exception), "Incompatible dimensions.")

    # Test the MulExpresion class.
    def test_mul_expression(self):
        # Vectors
        c = Constant([[2],[2]])
        exp = c*self.x
        self.assertEqual(exp.curvature, u.Curvature.AFFINE)
        self.assertEqual((c[0]*self.x).sign, u.Sign.UNKNOWN)
        self.assertEqual(exp.canonicalize()[0].size, (1,1))
        self.assertEqual(exp.canonicalize()[1], [])
        self.assertEqual(exp.name(), c.name() + " * " + self.x.name())
        self.assertEqual(exp.size, (1,1))

        with self.assertRaises(Exception) as cm:
            ([2,2,3]*self.x)
        const_name = Constant([2,2,3]).name()
        self.assertEqual(str(cm.exception), 
            "Incompatible dimensions.")

        # Matrices
        with self.assertRaises(Exception) as cm:
            Constant([[2, 1],[2, 2]]) * self.C
        self.assertEqual(str(cm.exception), "Incompatible dimensions.")

        with self.assertRaises(Exception) as cm:
            (self.A * self.B)
        self.assertEqual(str(cm.exception), "Cannot multiply two non-constants.")

        # Constant expressions
        T = Constant([[1,2,3],[3,5,5]])
        exp = (T + T) * self.B
        self.assertEqual(exp.curvature, u.Curvature.AFFINE)
        self.assertEqual(exp.size, (3,2))

        # Expression that would break sign multiplication without promotion.
        c = Constant([[2],[2],[-2]])
        exp = [[1],[2]] + c*self.C
        self.assertEqual(exp.sign.pos_mat.value.shape, (1,2))

    # Test the NegExpression class.
    def test_neg_expression(self):
        # Vectors
        exp = -self.x
        self.assertEqual(exp.curvature, u.Curvature.AFFINE)
        self.assertEqual(exp.sign, u.Sign.UNKNOWN)
        self.assertEqual(exp.canonicalize()[0].size, (2,1))
        self.assertEqual(exp.canonicalize()[1], [])
        self.assertEqual(exp.name(), "-%s" % self.x.name())
        self.assertEqual(exp.size, self.x.size)

        # Matrices
        exp = -self.C
        self.assertEqual(exp.curvature, u.Curvature.AFFINE)
        self.assertEqual(exp.size, (3,2))

    # Test promotion of scalar constants.
    def test_scalar_const_promotion(self):
        # Vectors
        exp = self.x + 2
        self.assertEqual(exp.curvature, u.Curvature.AFFINE)
        self.assertEqual(exp.sign, u.Sign.UNKNOWN)
        self.assertEqual(exp.canonicalize()[0].size, (2,1))
        self.assertEqual(exp.canonicalize()[1], [])
        self.assertEqual(exp.name(), self.x.name() + " + " + Constant(2).name())
        self.assertEqual(exp.size, (2,1))

        self.assertEqual((4 - self.x).size, (2,1))
        self.assertEqual((4 * self.x).size, (2,1))
        self.assertEqual((4 <= self.x).size, (2,1))
        self.assertEqual((4 == self.x).size, (2,1))
        self.assertEqual((self.x >= 4).size, (2,1))

        # Matrices
        exp = (self.A + 2) + 4
        self.assertEqual(exp.curvature, u.Curvature.AFFINE)
        self.assertEqual((3 * self.A).size, (2,2))

        self.assertEqual(exp.size, (2,2))

    # Test indexing expression.
    def test_index_expression(self):
        # Tuple of integers as key.
        exp = self.x[1,0]
        self.assertEqual(exp.name(), "x[1,0]")
        self.assertEqual(exp.curvature, u.Curvature.AFFINE)
        self.assertEquals(exp.size, (1,1))
        coeff = exp.coefficients(self.intf)
        self.assertEqual(coeff[exp], 1)
        self.assertEqual(exp.value, None)

        with self.assertRaises(Exception) as cm:
            (self.x[2,0])
        self.assertEqual(str(cm.exception), "Invalid indices 2,0 for 'x'.")

        # Slicing
        exp = self.C[0:2,1]
        self.assertEquals(exp.name(), "C[0:2,1]")
        self.assertEquals(exp.size, (2,1))
        exp = self.C[0:,0:2]
        self.assertEquals(exp.name(), "C[0:,0:2]")
        self.assertEquals(exp.size, (3,2))
        exp = self.C[0::2,0::2]
        self.assertEquals(exp.name(), "C[0::2,0::2]")
        self.assertEquals(exp.size, (2,1))
        exp = self.C[:3,:1:2]
        self.assertEquals(exp.name(), "C[0:3,0]")
        self.assertEquals(exp.size, (3,1))
        exp = self.C[0:,0]
        self.assertEquals(exp.name(), "C[0:,0]")
        self.assertEquals(exp.size, (3,1))

        c = Constant([[1,-2],[0,4]])
        exp = c[1,1]
        print exp
        self.assertEqual(exp.curvature, u.Curvature.CONSTANT)
        self.assertEqual(exp.sign, u.Sign.POSITIVE)
        self.assertEqual(c[0,1].sign, u.Sign.ZERO)
        self.assertEqual(c[1,0].sign, u.Sign.NEGATIVE)
        self.assertEquals(exp.size, (1,1))
        self.assertEqual(exp.value, 4)

        c = Constant([[1,-2,3],[0,4,5],[7,8,9]])
        exp = c[0:3,0:4:2]
        self.assertEqual(exp.curvature, u.Curvature.CONSTANT)
        self.assertEquals(exp.size, (3,2))
        self.assertEqual(exp[0,1].value, 7)

        # Arithmetic expression indexing
        exp = (self.x + self.z)[1,0]
        self.assertEqual(exp.name(), "x[1,0] + z[1,0]")
        self.assertEqual(exp.curvature, u.Curvature.AFFINE)
        self.assertEqual(exp.sign, u.Sign.UNKNOWN)
        self.assertEquals(exp.size, (1,1))

        exp = (self.x + self.a)[1,0]
        self.assertEqual(exp.name(), "x[1,0] + a")
        self.assertEqual(exp.curvature, u.Curvature.AFFINE)
        self.assertEquals(exp.size, (1,1))

        exp = (self.x - self.z)[1,0]
        self.assertEqual(exp.name(), "x[1,0] - z[1,0]")
        self.assertEqual(exp.curvature, u.Curvature.AFFINE)
        self.assertEquals(exp.size, (1,1))

        exp = (self.x - self.a)[1,0]
        self.assertEqual(exp.name(), "x[1,0] - a")
        self.assertEqual(exp.curvature, u.Curvature.AFFINE)
        self.assertEquals(exp.size, (1,1))

        exp = (-self.x)[1,0]
        self.assertEqual(exp.name(), "-x[1,0]")
        self.assertEqual(exp.curvature, u.Curvature.AFFINE)
        self.assertEquals(exp.size, (1,1))

        c = Constant([[1,2],[3,4]])
        exp = (c*self.x)[1,0]
        self.assertEqual(exp.name(), "[[2], [4]] * x[0:,0]")
        self.assertEqual(exp.curvature, u.Curvature.AFFINE)
        self.assertEquals(exp.size, (1,1))

        c = Constant([[1,2],[3,4]])
        exp = (c*self.a)[1,0]
        self.assertEqual(exp.name(), "2 * a")
        self.assertEqual(exp.curvature, u.Curvature.AFFINE)
        self.assertEquals(exp.size, (1,1))