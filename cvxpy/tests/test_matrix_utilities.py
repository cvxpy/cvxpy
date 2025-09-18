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

import unittest

import numpy as np

from cvxpy.interface import matrix_utilities
from cvxpy.interface.numpy_interface.ndarray_interface import COMPLEX_TYPES


class TestMatrixUtilities(unittest.TestCase):
    def test_convert(self):
        underlying_values = [0, 1]

        # List of values
        values = underlying_values
        with self.subTest(array=values):
            actual = matrix_utilities.convert(values)
            expected = np.array(values, dtype=float)
            np.testing.assert_array_equal(actual, expected, strict=True)

        # List of list
        values = [underlying_values]
        with self.subTest(array=values):
            actual = matrix_utilities.convert(values)
            expected = np.array(values, dtype=float).T
            np.testing.assert_array_equal(actual, expected, strict=True)

        for dtype in [int, float, bool] + COMPLEX_TYPES:
            expected_dtype = dtype if dtype in COMPLEX_TYPES else float

            # 1D arrays
            values = np.array(underlying_values, dtype=dtype)
            with self.subTest(array=values, dtype=dtype):
                actual = matrix_utilities.convert(values)
                expected = values.astype(expected_dtype)
                np.testing.assert_array_equal(actual, expected, strict=True)

            # 2D arrays
            for values in (
                np.array([underlying_values], dtype=dtype),
                np.matrix(underlying_values, dtype=dtype),
            ):
                with self.subTest(array=values, dtype=dtype):
                    actual = matrix_utilities.convert(values)
                    expected = np.array([underlying_values], dtype=expected_dtype)
                    np.testing.assert_array_equal(actual, expected, strict=True)
