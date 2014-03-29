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
import cvxpy.utilities as u
import cvxpy.utilities.coefficient_utils as cu
import cvxpy.utilities.key_utils as ku
import cvxpy.interface as intf
import cvxpy.settings as s
import unittest

class test_coefficients(unittest.TestCase):
    """ Unit tests for the expressions.affine module. """
    def setUp(self):
        self.a = Variable()

        self.x = Variable(2, name='x')
        self.y = Variable(2, name='y')

        self.A = Variable(2,2)

        self.c = Constant(3)
        self.C = Constant([[1, 2], [1, 2]])

    def test_leaf_coeffs(self):
        """Test the coefficients for Variables and Constants.
        """
        # Scalars
        coeffs = self.a.coefficients()
        self.assertItemsEqual(coeffs.keys(), [self.a.id])
        blocks = coeffs[self.a.id]
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0], 1)

        # Vectors
        coeffs = self.x.coefficients()
        self.assertItemsEqual(coeffs.keys(), [self.x.id])
        blocks = coeffs[self.x.id]
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].shape, (2, 2))

        # Matrices
        coeffs = self.A.coefficients()
        self.assertItemsEqual(coeffs.keys(), [self.A.id])
        blocks = coeffs[self.A.id]
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0].shape, (2,4))

        # Constants
        coeffs = self.c.coefficients()
        self.assertItemsEqual(coeffs.keys(), [s.CONSTANT])
        blocks = coeffs[s.CONSTANT]
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0], 3)

        coeffs = self.C.coefficients()
        self.assertItemsEqual(coeffs.keys(), [s.CONSTANT])
        blocks = coeffs[s.CONSTANT]
        self.assertEqual(len(blocks), 2)
        print blocks[0]
        self.assertEqual(blocks[0].shape, (2,1))
        self.assertEqual(blocks[0][0,0], 1)

    def test_add(self):
        """Test adding coefficients.
        """
        coeffs = cu.add(self.x.coefficients(), self.y.coefficients())
        self.assertItemsEqual(coeffs.keys(), [self.x.id, self.y.id])
        blocks = coeffs[self.x.id]
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].shape, (2,2))

        coeffs = cu.add(coeffs, coeffs)
        self.assertItemsEqual(coeffs.keys(), [self.x.id, self.y.id])
        blocks = coeffs[self.x.id]
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].shape, (2,2))
        self.assertEqual(blocks[0][0,0], 2)

        coeffs = cu.add(coeffs, self.C.coefficients())
        self.assertItemsEqual(coeffs.keys(), [self.x.id, self.y.id, s.CONSTANT])
        blocks = coeffs[s.CONSTANT]
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0].shape, (2,1))
        self.assertEqual(blocks[0][0,0], 1)

    def test_neg(self):
        """Test negating coefficients.
        """
        coeffs = cu.neg(self.a.coefficients())
        self.assertItemsEqual(coeffs.keys(), [self.a.id])
        blocks = coeffs[self.a.id]
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0], -1)

        coeffs = cu.neg(self.A.coefficients())
        self.assertItemsEqual(coeffs.keys(), [self.A.id])
        blocks = coeffs[self.A.id]
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0].shape, (2,4))
        self.assertEqual(blocks[0][0,0], -1)

    def test_sub(self):
        """Test subtracting coefficients.
        """
        coeffs = cu.sub(self.x.coefficients(), self.y.coefficients())
        self.assertItemsEqual(coeffs.keys(), [self.x.id, self.y.id])
        blocks = coeffs[self.y.id]
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].shape, (2,2))
        self.assertEqual(blocks[0][0,0], -1)

        coeffs = cu.sub(coeffs, self.x.coefficients())
        self.assertItemsEqual(coeffs.keys(), [self.x.id, self.y.id])
        blocks = coeffs[self.x.id]
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].shape, (2,2))
        self.assertEqual(blocks[0][0,0], 0)

    def test_mul(self):
        """Test multiplying coefficients.
        """
        coeffs = cu.add(self.x.coefficients(), self.y.coefficients())
        coeffs = cu.mul(self.C.coefficients(), coeffs)
        self.assertItemsEqual(coeffs.keys(), [self.x.id, self.y.id])
        blocks = coeffs[self.y.id]
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].shape, (2,2))
        self.assertEqual(blocks[0][1,0], 2)

        # Scalar by Matrix multiplication.
        coeffs = cu.add(self.x.coefficients(), self.y.coefficients())
        coeffs = cu.mul(self.c.coefficients(), coeffs)
        self.assertItemsEqual(coeffs.keys(), [self.x.id, self.y.id])
        blocks = coeffs[self.y.id]
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].shape, (2,2))
        self.assertEqual(blocks[0][0,0], 3)

        # Matrix by Scalar multiplication.
        coeffs = self.a.coefficients()
        coeffs = cu.mul(self.C.coefficients(), coeffs)
        self.assertItemsEqual(coeffs.keys(), [self.a.id])
        blocks = coeffs[self.a.id]
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0].shape, (2,1))
        self.assertEqual(blocks[0][1,0], 2)
        self.assertEqual(blocks[1].shape, (2,1))
        self.assertEqual(blocks[1][1,0], 2)

    def test_index(self):
        """Test indexing/slicing into coefficients.
        """
        # Index.
        sum_coeffs = cu.add(self.x.coefficients(), self.y.coefficients())
        key = ku.validate_key((1, 0), self.x.shape)
        coeffs = cu.index(sum_coeffs, key)
        self.assertItemsEqual(coeffs.keys(), [self.x.id, self.y.id])
        blocks = coeffs[self.y.id]
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].shape, (1,2))
        self.assertEqual(blocks[0][0,0], 0)

        # Slice.
        sum_coeffs = cu.add(self.A.coefficients(), self.C.coefficients())
        key = ku.validate_key((slice(None, None, None), 1), self.A.shape)
        coeffs = cu.index(sum_coeffs, key)
        self.assertItemsEqual(coeffs.keys(), [self.A.id, s.CONSTANT])
        # Variable.
        blocks = coeffs[self.A.id]
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].shape, (2,4))
        # Constant.
        blocks = coeffs[s.CONSTANT]
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].shape, (2,1))
