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

import cvxpy as cp
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.expressions.variable import Variable
from cvxpy.utilities import shape


class TestShape():
    
    @given(s=mutually_broadcastable_shapes(num_shapes=7))
    def test_add_broadcasting(self, s) -> None:
        assert shape.sum_shapes(s.input_shapes) == s.result_shape

    @given(s=mutually_broadcastable_shapes(signature=np.matmul.signature))
    def test_mul_broadcasting(self, s) -> None:
        x, y = s.input_shapes
        assert shape.mul_shapes(x, y) == s.result_shape

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
        c = reshape(b, [n, n], order='F')
        assert (a + c).shape == (n, n)
        d = reshape(b, (n, n), order='F')
        assert (a + d).shape == (n, n)

    def test_negative_axis(self) -> None:
        a = cp.Parameter(shape=(7, 3, 9, 5))
        b = cp.sum(a, axis=-1)
        assert b.shape == (7, 3, 9)
        b = cp.sum(a, axis=-2)
        assert b.shape == (7, 3, 5)
        b = cp.sum(a, axis=-3)
        assert b.shape == (7, 9, 5)

        b = cp.sum(a, axis=-1, keepdims=True)
        assert b.shape == (7, 3, 9, 1)
        b = cp.sum(a, axis=-2, keepdims=True)
        assert b.shape == (7, 3, 1, 5)
        b = cp.sum(a, axis=-3, keepdims=True)
        assert b.shape == (7, 1, 9, 5)
        with pytest.raises(ValueError):
            cp.sum(a, axis=-5)
        with pytest.raises(ValueError):
            cp.sum(a, axis=-10)
        with pytest.raises(ValueError):
            cp.sum(a, axis=-5, keepdims=True)
        with pytest.raises(ValueError):
            cp.sum(a, axis=-10, keepdims=True)

        b = cp.sum(a, axis=(-2, -1, 0), keepdims=True)
        assert b.shape == (1, 3, 1, 1)
        b = cp.sum(a, axis=(-2, -1, 0))
        assert b.shape == (3,)
        with pytest.raises(ValueError):
            cp.sum(a, axis=(-5, 3))
