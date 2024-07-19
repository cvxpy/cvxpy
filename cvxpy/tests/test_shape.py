"""
Copyright 2013 Steven Diamond

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

import numpy as np
import pytest
from hypothesis import given
from hypothesis.extra.numpy import mutually_broadcastable_shapes

from cvxpy.atoms.affine.reshape import reshape
from cvxpy.expressions.variable import Variable
from cvxpy.utilities import shape


class TestShape():
    """ Unit tests for the expressions/shape module. """

    def setUp(self) -> None:
        pass

    # Test adding two shapes.
    def test_add_matching(self) -> None:
        """Test addition of matching shapes.
        """
        self.assertEqual(shape.sum_shapes([(3, 4), (3, 4)]), (3, 4))
        self.assertEqual(shape.sum_shapes([(3, 4)] * 5), (3, 4))

    def test_add_broadcasting(self) -> None:
        """Test broadcasting of shapes during addition.
        """
        # Broadcasting with scalars is permitted.
        self.assertEqual(shape.sum_shapes([(3, 4), (1, 1)]), (3, 4))
        self.assertEqual(shape.sum_shapes([(1, 1), (3, 4)]), (3, 4))

        self.assertEqual(shape.sum_shapes([(1,), (3, 4)]), (3, 4))
        self.assertEqual(shape.sum_shapes([(3, 4), (1,)]), (3, 4))

        self.assertEqual(shape.sum_shapes([tuple(), (3, 4)]), (3, 4))
        self.assertEqual(shape.sum_shapes([(3, 4), tuple()]), (3, 4))

        self.assertEqual(shape.sum_shapes([(1, 1), (4,)]), (1, 4))
        self.assertEqual(shape.sum_shapes([(4,), (1, 1)]), (1, 4))

        # All other types of broadcasting is not permitted.
        with self.assertRaises(ValueError):
            shape.sum_shapes([(4, 1), (4,)])
        with self.assertRaises(ValueError):
            shape.sum_shapes([(4,), (4, 1)])

        with self.assertRaises(ValueError):
            shape.sum_shapes([(4, 2), (2,)])
        with self.assertRaises(ValueError):
            shape.sum_shapes([(2,), (4, 2)])

        with self.assertRaises(ValueError):
            shape.sum_shapes([(4, 2), (4, 1)])
        with self.assertRaises(ValueError):
            shape.sum_shapes([(4, 1), (4, 2)])

    def test_add_incompatible(self) -> None:
        """
        Test addition of incompatible shapes raises a ValueError.
        """
        with pytest.raises(ValueError):
            shape.sum_shapes([(4, 2), (4,)])

    def test_mul_scalars(self) -> None:
        """
        Test multiplication by scalars raises a ValueError.
        """
        with pytest.raises(ValueError):
            shape.mul_shapes(tuple(), (5, 9))
        with pytest.raises(ValueError):
            shape.mul_shapes((5, 9), tuple())
        with pytest.raises(ValueError):
            shape.mul_shapes(tuple(), tuple())

    def test_reshape_with_lists(self) -> None:
        n = 2
        a = Variable([n, n])
        b = Variable(n**2)
        c = reshape(b, [n, n])
        assert (a + c).shape == (n, n)
        d = reshape(b, (n, n))
        assert (a + d).shape == (n, n)
