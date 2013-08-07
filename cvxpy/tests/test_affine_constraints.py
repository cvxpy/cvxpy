from cvxpy.expressions.variable import Variable
from cvxpy.expressions.constant import Constant
from cvxpy.expressions.parameter import Parameter
from cvxpy.expressions.shape import Shape
from cvxpy.expressions.affine import AffineObjective
from cvxpy.constraints.affine import AffEqConstraint, AffLeqConstraint
import cvxpy.interface.matrix_utilities as intf
import cvxpy.settings as s
from collections import deque
import unittest

class TestAffineConstraints(unittest.TestCase):
    """ Unit tests for the constraints.affine module. """
    def setUp(self):
        self.x = Variable(2, name='x')
        self.y = Variable(2, name='y')

        self.A = Constant([[1, 2], [1, 2]])

        self.xAff = AffineObjective([(self.x, deque([self.x]))], Shape(2,1))
        self.yAff = AffineObjective([(self.y, deque([self.y]))], Shape(2,1))
        self.constAff = AffineObjective([(self.A, deque([self.A]))], Shape(2,2))
        self.intf = intf.DEFAULT_INTERFACE

    # Test AffEqConstraint.
    def test_eq_constraint(self):
        constr = AffEqConstraint(self.xAff, self.yAff)
        self.assertItemsEqual(constr.variables().keys(), [self.x.id, self.y.id])

        coeffs = constr.coefficients(self.intf)
        exp = self.xAff - self.yAff
        expCoeffs = exp.coefficients(self.intf)
        self.assertItemsEqual(coeffs.keys(), expCoeffs.keys())
        self.assertEqual(list(coeffs[self.y[1,0].id]), 
                         list(expCoeffs[self.y[1,0].id]))

        with self.assertRaises(Exception) as cm:
            AffEqConstraint(self.xAff, self.constAff)
        self.assertEqual(str(cm.exception), "Incompatible dimensions.")

    # Test AffLeqConstraint.
    def test_eq_constraint(self):
        constr = AffLeqConstraint(self.xAff, self.yAff)
        self.assertItemsEqual(constr.variables().keys(), [self.x.id, self.y.id])

        coeffs = constr.coefficients(self.intf)
        exp = self.xAff - self.yAff
        expCoeffs = exp.coefficients(self.intf)
        self.assertItemsEqual(coeffs.keys(), expCoeffs.keys())
        self.assertEqual(list(coeffs[self.y[1,0].id]), 
                         list(expCoeffs[self.y[1,0].id]))

        with self.assertRaises(Exception) as cm:
            AffLeqConstraint(self.xAff, self.constAff)
        self.assertEqual(str(cm.exception), "Incompatible dimensions.")