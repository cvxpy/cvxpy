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

from cvxpy.expressions.variables import Variable
from cvxpy.expressions.constant import Constant
from cvxpy.expressions.parameter import Parameter
from cvxpy.expressions.affine import AffObjective
import cvxpy.utilities as u
import cvxpy.interface.matrix_utilities as intf
import cvxpy.settings as s
from collections import deque
import unittest

class TestAffObjective(unittest.TestCase):
    """ Unit tests for the expressions.affine module. """
    def setUp(self):
        self.x = Variable(2, name='x')
        self.y = Variable(2, name='y')

        self.A = Constant([[1, 2], [1, 2]])

        self.xAff = AffObjective([self.x], [deque([self.x])], u.Shape(2,1))
        self.yAff = AffObjective([self.y], [deque([self.y])], u.Shape(2,1))
        self.constAff = AffObjective([self.A], [deque([self.A])], u.Shape(2,2))
        self.intf = intf.DEFAULT_INTERFACE

    # Test adding AffObjectives.
    def test_add(self):
        add = self.xAff + self.yAff
        self.assertItemsEqual(add._terms, [deque([self.x]), 
                                           deque([self.y])])
        self.assertItemsEqual(add.variables(), [self.x, self.y])

        add = self.xAff + self.xAff
        self.assertItemsEqual(add._terms, [deque([self.x]), 
                                           deque([self.x])])
        self.assertItemsEqual(add.variables(), [self.x, self.x])

        with self.assertRaises(Exception) as cm:
            self.xAff + self.constAff
        self.assertEqual(str(cm.exception), "Incompatible dimensions.")

    # Test multiplying AffObjectives.
    def test_mul(self):
        mul = self.constAff * self.xAff
        self.assertItemsEqual(mul._terms, [deque([self.x, self.A])])
        self.assertItemsEqual(mul.variables(), [self.x])

        mul = self.constAff * (self.yAff + self.xAff)
        self.assertItemsEqual(mul._terms, [deque([self.x, self.A]),
                                           deque([self.y, self.A])])
        self.assertItemsEqual(mul.variables(), [self.x, self.y])

        with self.assertRaises(Exception) as cm:
            self.xAff * self.yAff
        self.assertEqual(str(cm.exception), "Incompatible dimensions.")

    # Test negating AffObjectives.
    def test_neg(self):
        neg = -self.xAff
        self.assertEqual(neg._terms[0][1].name(), "-1")
        self.assertItemsEqual(neg.variables(), [self.x])

    # Test subtracting AffObjectives.
    def test_sub(self):
        sub = self.xAff - self.yAff
        self.assertItemsEqual(sub._terms, (self.xAff + -self.yAff)._terms)
        self.assertItemsEqual(sub.variables(), [self.x, self.y])

        with self.assertRaises(Exception) as cm:
            self.xAff - self.constAff
        self.assertEqual(str(cm.exception), "Incompatible dimensions.")

    # Test coefficients method.
    def test_coefficients(self):
        # Leaf
        coeffs = self.xAff.coefficients(self.intf)
        self.assertEqual(coeffs.keys(), self.x.coefficients(self.intf).keys())

        # Sum with different keys
        exp = self.xAff + self.yAff
        coeffs = exp.coefficients(self.intf)
        keys = self.x.coefficients(self.intf).keys() + \
               self.y.coefficients(self.intf).keys()
        self.assertItemsEqual(coeffs.keys(), keys)

        # Sum with same keys
        exp = self.xAff + self.xAff
        coeffs = exp.coefficients(self.intf)
        xCoeffs = self.x.coefficients(self.intf)
        self.assertItemsEqual(coeffs.keys(), xCoeffs)
        self.assertEqual(list(coeffs[self.x]), [2,0,0,2])

        # Product
        exp = self.constAff * self.yAff
        coeffs = exp.coefficients(self.intf)
        yCoeffs = self.y.coefficients(self.intf)
        self.assertItemsEqual(coeffs.keys(), yCoeffs)
        self.assertEqual(list(coeffs[self.y]), [1,2,1,2])

        # Distributed product
        exp = self.constAff * (self.xAff + self.yAff)
        coeffs = exp.coefficients(self.intf)
        xCoeffs = self.x.coefficients(self.intf)
        yCoeffs = self.y.coefficients(self.intf)
        self.assertItemsEqual(coeffs.keys(), xCoeffs.keys() + yCoeffs.keys())
        self.assertEqual(list(coeffs[self.x]), [1,2,1,2])        