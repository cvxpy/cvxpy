from cvxpy.expressions.expression import *
from cvxpy.expressions.variable import Variable
from cvxpy.expressions.constant import Constant
from cvxpy.expressions.parameter import Parameter
from cvxpy.expressions.curvature import Curvature
import cvxpy.interface.matrix_utilities as intf
import cvxpy.settings as s
from collections import deque
import unittest

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
        self.assertEqual(x.curvature, Curvature.AFFINE)
        self.assertEqual(x.canonicalize()[0].size, (2,1))
        self.assertEqual(x.canonicalize()[1], [])
        self.assertEqual(x.as_term(), (x,deque([x])))

        # Vector variable.
        coeffs = x.coefficients(self.intf)
        self.assertItemsEqual(coeffs.keys(), [x[0,0].id, x[1,0].id])
        vec = coeffs[x[1,0].id]
        self.assertEqual(vec.size, (2,1))
        self.assertEqual(vec[0,0], 0)
        self.assertEqual(vec[1,0], 1)

        # Matrix variable.
        coeffs = self.A.coefficients(self.intf)
        self.assertItemsEqual(coeffs.keys(), [self.A[0,0].id, self.A[1,0].id,
                                              self.A[0,1].id, self.A[1,1].id])
        mat = coeffs[self.A[1,0].id]
        self.assertEqual(mat.size, (2,2))
        self.assertEqual(mat[0,0], 0)
        self.assertEqual(mat[1,0], 1)

    # Test the Constant class.
    def test_constants(self):
        c = Constant(2)
        self.assertEqual(c.name(), str(2))

        c = Constant(2, name="c")
        self.assertEqual(c.name(), "c")
        self.assertEqual(c.value, 2)
        self.assertEqual(c.size, (1,1))
        self.assertEqual(c.curvature, Curvature.CONSTANT)
        self.assertEqual(c.canonicalize()[0].size, (1,1))
        self.assertEqual(c.canonicalize()[1], [])
        self.assertEqual(c.as_term(), (c,deque([c])))

        coeffs = c.coefficients(self.intf)
        self.assertEqual(coeffs.keys(), [s.CONSTANT])
        self.assertEqual(coeffs[s.CONSTANT], 2)

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
        self.assertEqual(exp.curvature, Curvature.AFFINE)
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
        self.assertEqual(exp.curvature, Curvature.AFFINE)
        self.assertEqual(exp.size, (2,2))

        with self.assertRaises(Exception) as cm:
            (self.A + self.C)
        self.assertEqual(str(cm.exception), "Incompatible dimensions.")


    # Test the SubExpresion class.
    def test_sub_expression(self):
        # Vectors
        c = Constant([2,2])
        exp = self.x - c
        self.assertEqual(exp.curvature, Curvature.AFFINE)
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
        self.assertEqual(exp.curvature, Curvature.AFFINE)
        self.assertEqual(exp.size, (2,2))

        with self.assertRaises(Exception) as cm:
            (self.A - self.C)
        self.assertEqual(str(cm.exception), "Incompatible dimensions.")

    # Test the MulExpresion class.
    def test_mul_expression(self):
        # Vectors
        c = Constant([[2],[2]])
        exp = c*self.x
        self.assertEqual(exp.curvature, Curvature.AFFINE)
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
        self.assertEqual(str(cm.exception), "Cannot multiply on the left by a non-constant.")

        # Constant expressions
        T = Constant([[1,2,3],[3,5,5]])
        exp = (T + T) * self.B
        self.assertEqual(exp.curvature, Curvature.AFFINE)
        self.assertEqual(exp.size, (3,2))

    # Test the NegExpression class.
    def test_neg_expression(self):
        # Vectors
        exp = -self.x
        self.assertEqual(exp.curvature, Curvature.AFFINE)
        self.assertEqual(exp.canonicalize()[0].size, (2,1))
        self.assertEqual(exp.canonicalize()[1], [])
        self.assertEqual(exp.name(), "-%s" % self.x.name())
        self.assertEqual(exp.size, self.x.size)

        # Matrices
        exp = -self.C
        self.assertEqual(exp.curvature, Curvature.AFFINE)
        self.assertEqual(exp.size, (3,2))

    # Test promotion of scalar constants.
    def test_scalar_const_promotion(self):
        # Vectors
        exp = self.x + 2
        self.assertEqual(exp.curvature, Curvature.AFFINE)
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
        self.assertEqual(exp.curvature, Curvature.AFFINE)
        self.assertEqual((3 * self.A).size, (2,2))

        self.assertEqual(exp.size, (2,2))

    # Test indexing expression.
    def test_index_expression(self):
        # Tuple of integers as key.
        exp = self.x[1,0]
        self.assertEqual(exp.name(), "x[1,0]")
        self.assertEqual(exp.curvature, Curvature.AFFINE)
        self.assertEquals(exp.size, (1,1))

        with self.assertRaises(Exception) as cm:
            (self.x[2,0])
        self.assertEqual(str(cm.exception), "Invalid indices 2,0 for 'x'.")

        c = Constant([[1,2],[3,4]])
        exp = c[1,1]
        self.assertEqual(exp.curvature, Curvature.CONSTANT)
        self.assertEquals(exp.size, (1,1))
        self.assertEqual(exp.value, 4)

        # Arithmetic expression indexing
        exp = (self.x + self.z)[1,0]
        self.assertEqual(exp.name(), "x[1,0] + z[1,0]")
        self.assertEqual(exp.curvature, Curvature.AFFINE)
        self.assertEquals(exp.size, (1,1))

        exp = (self.x + self.a)[1,0]
        self.assertEqual(exp.name(), "x[1,0] + a")
        self.assertEqual(exp.curvature, Curvature.AFFINE)
        self.assertEquals(exp.size, (1,1))

        exp = (self.x - self.z)[1,0]
        self.assertEqual(exp.name(), "x[1,0] - z[1,0]")
        self.assertEqual(exp.curvature, Curvature.AFFINE)
        self.assertEquals(exp.size, (1,1))

        exp = (self.x - self.a)[1,0]
        self.assertEqual(exp.name(), "x[1,0] - a")
        self.assertEqual(exp.curvature, Curvature.AFFINE)
        self.assertEquals(exp.size, (1,1))

        exp = (-self.x)[1,0]
        self.assertEqual(exp.name(), "-x[1,0]")
        self.assertEqual(exp.curvature, Curvature.AFFINE)
        self.assertEquals(exp.size, (1,1))

        c = Constant([[1,2],[3,4]])
        exp = (c*self.x)[1,0]
        self.assertEqual(exp.name(), "0 + 2 * x[0,0] + 4 * x[1,0]")
        self.assertEqual(exp.curvature, Curvature.AFFINE)
        self.assertEquals(exp.size, (1,1))

        c = Constant([[1,2],[3,4]])
        exp = (c*self.a)[1,0]
        self.assertEqual(exp.name(), "2 * a")
        self.assertEqual(exp.curvature, Curvature.AFFINE)
        self.assertEquals(exp.size, (1,1))