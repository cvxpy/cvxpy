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
            self.assertEqual(intf.size(v), (4,2))

        exp = AffVstack(self.A, self.B, self.C)
        coeffs = exp.coefficients(self.intf)
        v = coeffs[self.A]
        self.assertEqual(intf.size(v), (10,3))