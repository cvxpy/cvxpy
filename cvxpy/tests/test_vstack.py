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
from cvxpy.atoms.affine.vstack import vstack
import cvxpy.utilities as u
import cvxpy.interface as intf
import cvxpy.settings as s
import unittest

class TestVstack(unittest.TestCase):
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
        exp,constr = vstack(self.x, self.y, self.x+self.y).canonical_form
        self.assertEquals(constr, [])
        self.assertItemsEqual(exp.variables(), [self.x, self.y])
        exp = vstack(self.A, self.B, self.C).canonical_form[0]
        self.assertItemsEqual(exp.variables(), [self.A, self.B])

    # Test coefficients method.
    def test_coefficients(self):
        exp = vstack(self.x).canonical_form[0]
        coeffs = exp.coefficients()
        self.assertEqual(coeffs.keys(), self.x.coefficients().keys())

        exp = vstack(self.x, self.y).canonical_form[0]
        coeffs = exp.coefficients()
        self.assertItemsEqual(coeffs.keys(),
            self.x.coefficients().keys() + \
            self.y.coefficients().keys())
        for k,blocks in coeffs.items():
            self.assertEqual(len(blocks), 1)
            for block in blocks:
                self.assertEqual(intf.size(block), (4,2))

        exp = vstack(self.A, self.B, self.C).canonical_form[0]
        coeffs = exp.coefficients()
        blocks = coeffs[self.A.id]
        self.assertEqual(len(blocks), 2)
        for block in blocks:
            self.assertEqual(intf.size(block), (10,6))
