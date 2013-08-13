from cvxpy.expressions.variable import Variable
from cvxpy.expressions.constant import Constant
from cvxpy.expressions.parameter import Parameter
from cvxpy.expressions.vstack import AffVstack
import cvxpy.utilities as u
import cvxpy.interface.matrix_utilities as intf
import cvxpy.settings as s
from collections import deque
import unittest

class TestAffVstack(unittest.TestCase):
    """ Unit tests for the expressions.affine module. """
    def setUp(self):
        self.x = Variable(2, name='x')
        self.y = Variable(2, name='y')

        self.A = Variable(3,2, name='A')
        self.B = Variable(5,2, name='B')

        self.C = Constant([[1, 2], [1, 2]])
        self.intf = intf.DEFAULT_INTERFACE

    # Test the variables method.
    def test_variables(self):
        exp = AffVstack(self.x, self.y, self.x+self.y)
        self.assertItemsEqual(exp.variables(), [self.x, self.y, self.x, self.y])
        exp = AffVstack(self.A, self.B, self.C)
        self.assertItemsEqual(exp.variables(), [self.A, self.B])

    # Test coefficients method.
    def test_coefficients(self):
        exp = AffVstack(self.x)
        coeffs = exp.coefficients(self.intf)
        self.assertEqual(coeffs.keys(), self.x.coefficients(self.intf).keys())

        exp = AffVstack(self.x, self.y)
        coeffs = exp.coefficients(self.intf)
        self.assertItemsEqual(coeffs.keys(), 
            self.x.coefficients(self.intf).keys() + \
            self.y.coefficients(self.intf).keys())
        for k,v in coeffs.items():
            self.assertEqual(intf.size(v), (4,1))

        exp = AffVstack(self.A, self.B, self.C)
        coeffs = exp.coefficients(self.intf)
        for k,v in coeffs.items():
            self.assertEqual(intf.size(v), (10,2))