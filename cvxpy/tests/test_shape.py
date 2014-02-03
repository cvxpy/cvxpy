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
from cvxpy.utilities import Shape
import unittest

class TestShape(unittest.TestCase):
    """ Unit tests for the expressions/shape module. """
    def setUp(self):
        pass

    # Test the size method.
    def test_size(self):
        self.assertEqual(Shape(1,3).size, (1,3))
        self.assertEqual(Shape(2,1).size, (2,1))

    # Test adding two shapes.
    def test_add(self):
        self.assertEqual((Shape(3,4) + Shape(3,4)).size, (3,4))

        with self.assertRaises(Exception) as cm:
            (Shape(1,3) + Shape(4,3))
        self.assertEqual(str(cm.exception), "Incompatible dimensions (1, 3) (4, 3)")

        # Promotion
        self.assertEqual((Shape(3,4) + Shape(1,1)).size, (3,4))
        self.assertEqual((Shape(1,1) + Shape(3,4)).size, (3,4))

    # Test multiplying two shapes.
    def test_mul(self):
        self.assertEqual((Shape(5,9) * Shape(9,2)).size, (5,2))

        with self.assertRaises(Exception) as cm:
            (Shape(5,3) * Shape(9,2))
        self.assertEqual(str(cm.exception), "Incompatible dimensions (5, 3) (9, 2)")

        # Promotion
        self.assertEqual((Shape(3,4) * Shape(1,1)).size, (3,4))
        self.assertEqual((Shape(1,1) * Shape(3,4)).size, (3,4))