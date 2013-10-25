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
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.constants import Parameter
from cvxpy.expressions.affine import AffExpression
from cvxpy.constraints.affine import AffEqConstraint, AffLeqConstraint
import cvxpy.utilities as u
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

        self.xAff = AffExpression([self.x], [deque([self.x])], u.Shape(2,1))
        self.yAff = AffExpression([self.y], [deque([self.y])], u.Shape(2,1))
        self.constAff = AffExpression([self.A], [deque([self.A])], u.Shape(2,2))
        self.intf = intf.DEFAULT_INTERFACE

    # Test AffEqConstraint.
    def test_eq_constraint(self):
        constr = AffEqConstraint(self.xAff, self.yAff)
        self.assertItemsEqual(constr.variables(), [self.x, self.y])

        coeffs = constr.coefficients(self.intf)
        exp = self.xAff - self.yAff
        expCoeffs = exp.coefficients(self.intf)
        self.assertItemsEqual(coeffs.keys(), expCoeffs.keys())
        self.assertEqual(list(coeffs[self.y]), 
                         list(expCoeffs[self.y]))

        with self.assertRaises(Exception) as cm:
            AffEqConstraint(self.xAff, self.constAff)
        self.assertEqual(str(cm.exception), "Incompatible dimensions.")

    # Test AffLeqConstraint.
    def test_leq_constraint(self):
        constr = AffLeqConstraint(self.xAff, self.yAff)
        self.assertItemsEqual(constr.variables(), [self.x, self.y])

        coeffs = constr.coefficients(self.intf)
        exp = self.xAff - self.yAff
        expCoeffs = exp.coefficients(self.intf)
        self.assertItemsEqual(coeffs.keys(), expCoeffs.keys())
        self.assertEqual(list(coeffs[self.y]), 
                         list(expCoeffs[self.y]))

        with self.assertRaises(Exception) as cm:
            AffLeqConstraint(self.xAff, self.constAff)
        self.assertEqual(str(cm.exception), "Incompatible dimensions.")