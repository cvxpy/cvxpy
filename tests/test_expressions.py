from cvxpy.expressions.expression import *
from cvxpy.expressions.variable import Variable
from cvxpy.expressions.containers import Variables
from cvxpy.expressions.curvature import Curvature
import cvxpy.interface.matrix_utilities as intf
import unittest

class TestExpressions(unittest.TestCase):
    """ Unit tests for the expression/expression module. """
    def setUp(self):
        self.x = Variable(2, name='x')
        self.y = Variable(3, name='y')

        self.A = Variable(2,2,name='A')
        self.B = Variable(2,2,name='B')
        self.C = Variable(3,2,name='C')
        self.intf = intf.DEFAULT_INTERFACE()

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
        self.assertEqual(x.canonicalize(), (x, []))

        self.assertEqual(x.variables()[x.id], x)
        identity = x.coefficients(self.intf)[x.id]
        self.assertEqual(identity.size, (2,2))
        self.assertEqual(identity[0,0], 1)
        self.assertEqual(identity[0,1], 0)
        self.assertEqual(identity[1,0], 0)
        self.assertEqual(identity[1,1], 1)

    # Test the Variables class.
    def test_variables(self):
        v = Variables(['y',3],'x','z',['A',3,4])
        self.assertEqual(v.y.name(), 'y')
        self.assertEqual(v.y.size, (3,1))
        self.assertEqual(v.x.name(), 'x')
        self.assertEqual(v.x.size, (1,1))
        self.assertEqual(v.z.name(), 'z')
        self.assertEqual(v.z.size, (1,1))
        self.assertEqual(v.A.name(), 'A')
        self.assertEqual(v.A.size, (3,4))

    # Test the Constant class.
    def test_constants(self):
        c = Constant(2)
        self.assertEqual(c.name(), str(2))

        c = Constant(2, name="c")
        self.assertEqual(c.name(), "c")
        self.assertEqual(c.value, 2)
        self.assertEqual(c.size, (1,1))
        self.assertEqual(c.variables(), {})
        self.assertEqual(c.curvature, Curvature.CONSTANT)
        self.assertEqual(c.canonicalize(), (c, []))

    # Test the AddExpresion class.
    def test_add_expression(self):
        # Vectors
        exp = self.x + [2,2]
        self.assertEqual(exp.curvature, Curvature.AFFINE)
        self.assertEqual(exp.canonicalize(), (exp, []))
        self.assertEqual(exp.name(), self.x.name() + " + " + Constant([2,2]).name())
        self.assertEqual(exp.size, (2,1))

        z = Variable(2, name='z')
        exp = exp + z + self.x
        self.assertItemsEqual(exp.variables().keys(), [self.x.id, z.id])
        self.assertEqual(exp.coefficients(self.intf)[self.x.id][0,0], 2)

        with self.assertRaises(Exception) as cm:
            (self.x + self.y).size
        self.assertEqual(str(cm.exception), "'x + y' has incompatible dimensions.")

        # Matrices
        exp = self.A + self.B
        self.assertEqual(exp.curvature, Curvature.AFFINE)
        self.assertEqual(exp.size, (2,2))

        with self.assertRaises(Exception) as cm:
            (self.A + self.C).size
        self.assertEqual(str(cm.exception), "'A + C' has incompatible dimensions.")


    # Test the SubExpresion class.
    def test_sub_expression(self):
        # Vectors
        exp = self.x - [2,2]
        self.assertEqual(exp.curvature, Curvature.AFFINE)
        self.assertEqual(exp.canonicalize(), (exp, []))
        self.assertEqual(exp.name(), self.x.name() + " - " + Constant([2,2]).name())
        self.assertEqual(exp.size, (2,1))

        z = Variable(2, name='z')
        exp = exp - z - self.x
        self.assertItemsEqual(exp.variables().keys(), [self.x.id, z.id])
        self.assertEqual(exp.coefficients(self.intf)[self.x.id][0,0], 0)

        with self.assertRaises(Exception) as cm:
            (self.x - self.y).size
        self.assertEqual(str(cm.exception), "'x - y' has incompatible dimensions.")

        # Matrices
        exp = self.A - self.B
        self.assertEqual(exp.curvature, Curvature.AFFINE)
        self.assertEqual(exp.size, (2,2))

        with self.assertRaises(Exception) as cm:
            (self.A - self.C).size
        self.assertEqual(str(cm.exception), "'A - C' has incompatible dimensions.")

    # Test the MulExpresion class.
    def test_mul_expression(self):
        # Vectors
        c = [[2],[2]]
        exp = c*self.x
        self.assertEqual(exp.curvature, Curvature.AFFINE)
        self.assertEqual(exp.canonicalize(), (exp, []))
        self.assertEqual(exp.name(), Constant(c).name() + " * " + self.x.name())
        self.assertEqual(exp.size, (1,1))

        new_exp = 2*(exp + 1)
        self.assertEqual(new_exp.variables(), exp.variables())
        self.assertEqual(new_exp.coefficients(self.intf)[self.x.id][0,0], 4)
        self.assertEqual(Expression.constant(new_exp.coefficients(self.intf)), 2)

        with self.assertRaises(Exception) as cm:
            ([2,2,3]*self.x).size
        const_name = Constant([2,2,3]).name()
        self.assertEqual(str(cm.exception), 
            "'%s * x' has incompatible dimensions." % const_name)

        # Matrices
        exp = self.C * self.B
        self.assertEqual(exp.curvature, Curvature.UNKNOWN)
        self.assertEqual(exp.size, (3,2))

        with self.assertRaises(Exception) as cm:
            (self.A * self.C).size
        self.assertEqual(str(cm.exception), "'A * C' has incompatible dimensions.")

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
        self.assertEqual(exp.canonicalize(), (exp, []))
        self.assertEqual(exp.name(), "-%s" % self.x.name())
        self.assertEqual(exp.size, self.x.size)
        exp = self.x+self.y
        self.assertEquals((-exp).variables(), exp.variables())

        # Matrices
        exp = -self.C
        self.assertEqual(exp.curvature, Curvature.AFFINE)
        self.assertEqual(exp.size, (3,2))

    # Test promotion of scalar constants.
    def test_scalar_const_promotion(self):
        # Vectors
        exp = self.x + 2
        self.assertEqual(exp.curvature, Curvature.AFFINE)
        self.assertEqual(exp.canonicalize(), (exp, []))
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

    # # Test indexing expression.
    # def test_index_expression(self):
    #     # Tuple of integers as key.
    #     exp = self.x[1,0]
    #     self.assertEqual(exp.name(), "x[1,0]")
    #     self.assertEqual(exp.curvature, Curvature.AFFINE)
    #     self.assertEquals(exp.size, (1,1))

    #     with self.assertRaises(Exception) as cm:
    #         (self.x[2,0]).canonicalize()
    #     self.assertEqual(str(cm.exception), "Invalid indices 2,0 for 'x'.")
