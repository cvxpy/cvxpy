"""
Copyright, the CVXPY authors

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

import cvxpy as cp
from cvxpy.expressions.expression import (
    __BINARY_EXPRESSION_UFUNCS__,
    __NUMPY_UFUNC_ERROR__,
)
from cvxpy.tests.base_test import BaseTest


class TestNumpy(BaseTest):
    """ Unit tests for using NumPy ufuncs on CVXPY objects should cause errors. """

    def setUp(self) -> None:

        self.x = cp.Variable((2, 2), name='x')
        self.x.value = [[10., 11], [12, 13]]
        self.y = cp.Variable(1, name='y')

    def test_np_ufunc_errors(self) -> None:
        with pytest.raises(RuntimeError, match=__NUMPY_UFUNC_ERROR__):
            np.sqrt(self.x)

        with pytest.raises(RuntimeError, match=__NUMPY_UFUNC_ERROR__):
            np.log(self.x)

    def test_some_np_ufunc_works(self) -> None:
        a = np.array([[1., 3.], [3., 1.]])
        b = np.int64(1)

        for ufunc in __BINARY_EXPRESSION_UFUNCS__:
            if ufunc is np.multiply:
                continue  # We don't want to deal with the multiply warnings
            if ufunc is np.power:
                continue  # We don't implement __rpow__ yet.
            with pytest.raises(RuntimeError, match=__NUMPY_UFUNC_ERROR__):
                ufunc(self.x, a)
            with pytest.raises(RuntimeError, match=__NUMPY_UFUNC_ERROR__):
                ufunc(a, self.x, out=a)

            if ufunc is np.left_shift or \
                    ufunc is np.right_shift or \
                    ufunc is np.equal or \
                    ufunc is np.less_equal or \
                    ufunc is np.greater_equal or \
                    ufunc is np.less or \
                    ufunc is np.greater:
                continue
            self.assertItemsAlmostEqual(ufunc(a, self.x).value, ufunc(a, self.x.value))

        for ufunc in __BINARY_EXPRESSION_UFUNCS__:
            if ufunc is np.matmul:
                continue  # matmul doesn't play nice with scalars
            if ufunc is np.power:
                continue  # We don't implement __rpow__ yet.

            with pytest.raises(RuntimeError, match=__NUMPY_UFUNC_ERROR__):
                ufunc(self.x, b)

            with pytest.raises(RuntimeError, match=__NUMPY_UFUNC_ERROR__):
                ufunc(b, self.x, out=b)

            if ufunc is np.left_shift or \
                    ufunc is np.right_shift or \
                    ufunc is np.equal or \
                    ufunc is np.less_equal or \
                    ufunc is np.greater_equal or \
                    ufunc is np.less or \
                    ufunc is np.greater:
                continue

            self.assertItemsAlmostEqual(ufunc(b, self.x).value, ufunc(b, self.x.value))

    def test_working_numpy_functions(self) -> None:
        hstack = np.hstack([self.x])
        self.assertEqual(hstack.shape, (1,))
        self.assertEqual(hstack.dtype, object)
        vstack = np.vstack([self.x])
        self.assertEqual(vstack.shape, (1, 1))
        self.assertEqual(vstack.dtype, object)

    def test_broken_numpy_functions(self) -> None:
        with pytest.raises(RuntimeError, match=__NUMPY_UFUNC_ERROR__):
            np.linalg.norm(self.x)
