"""
Copyright 2017 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from cvxpy.expressions.variables import Variable
from cvxpy.utilities import shape
import unittest


class TestShape(unittest.TestCase):
    """ Unit tests for the expressions/shape module. """

    def setUp(self):
        pass

    # Test adding two shapes.
    def test_add(self):
        self.assertEqual(shape.sum_shapes([(3, 4), (3, 4)]), (3, 4))

        with self.assertRaises(Exception) as cm:
            shape.sum_shapes([(1, 3), (4, 3)])
        self.assertEqual(str(cm.exception), "Incompatible dimensions (1, 3) (4, 3)")

        # Promotion
        self.assertEqual(shape.sum_shapes([(3, 4), (1, 1)]), (3, 4))
        self.assertEqual(shape.sum_shapes([(1, 1), (3, 4)]), (3, 4))

    # Test multiplying two shapes.
    def test_mul(self):
        self.assertEqual(shape.mul_shapes((5, 9), (9, 2)), (5, 2))

        with self.assertRaises(Exception) as cm:
            shape.mul_shapes((5, 3), (9, 2))
        self.assertEqual(str(cm.exception), "Incompatible dimensions (5, 3) (9, 2)")

        # Promotion
        self.assertEqual(shape.mul_shapes((3, 4), (1, 1)), (3, 4))
        self.assertEqual(shape.mul_shapes((1, 1), (3, 4)), (3, 4))
