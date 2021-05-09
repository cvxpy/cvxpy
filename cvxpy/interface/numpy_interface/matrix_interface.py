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


from cvxpy.interface.numpy_interface.ndarray_interface import NDArrayInterface
import scipy.sparse as sp
import numpy as np


class MatrixInterface(NDArrayInterface):
    """
    An interface to convert constant values to the numpy matrix class.
    """
    TARGET_MATRIX = np.matrix

    @NDArrayInterface.scalar_const
    def const_to_matrix(self, value, convert_scalars: bool = False):
        """Convert an arbitrary value into a matrix of type self.target_matrix.

        Args:
            value: The constant to be converted.
            convert_scalars: Should scalars be converted?

        Returns:
            A matrix of type self.target_matrix or a scalar.
        """
        # Lists and 1D arrays become column vectors.
        if isinstance(value, list) or \
                isinstance(value, np.ndarray) and value.ndim == 1:
            value = np.asmatrix(value, dtype='float64').T
        # First convert sparse to dense.
        elif sp.issparse(value):
            value = value.todense()
        return np.asmatrix(value, dtype='float64')

    # Return an identity matrix.
    def identity(self, size):
        return np.asmatrix(np.eye(size))

    # A matrix with all entries equal to the given scalar value.
    def scalar_matrix(self, value, shape):
        mat = np.zeros(shape, dtype='float64') + value
        return np.asmatrix(mat)

    def reshape(self, matrix, size):
        return np.reshape(matrix, size, order='F')
