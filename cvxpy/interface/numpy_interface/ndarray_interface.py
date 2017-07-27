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


from .. import base_matrix_interface as base
import numpy
import scipy.sparse


class NDArrayInterface(base.BaseMatrixInterface):
    """
    An interface to convert constant values to the numpy ndarray class.
    """
    TARGET_MATRIX = numpy.ndarray

    def const_to_matrix(self, value, convert_scalars=False):
        """Convert an arbitrary value into a matrix of type self.target_matrix.

        Args:
            value: The constant to be converted.
            convert_scalars: Should scalars be converted?

        Returns:
            A matrix of type self.target_matrix or a scalar.
        """
        if isinstance(value, list):
            value = numpy.atleast_2d(value)
            value = value.T
        elif scipy.sparse.issparse(value):
            value = value.A
        elif isinstance(value, numpy.matrix):
            value = value.A
        return numpy.atleast_2d(value)

    # Return an identity matrix.
    def identity(self, size):
        return numpy.eye(size)

    # Return the dimensions of the matrix.
    def size(self, matrix):
        # Scalars.
        if len(matrix.shape) == 0:
            return (1, 1)
        # 1D arrays are treated as column vectors.
        elif len(matrix.shape) == 1:
            return (int(matrix.size), 1)
        # 2D arrays.
        else:
            rows = int(matrix.shape[0])
            cols = int(matrix.shape[1])
            return (rows, cols)

    # Get the value of the passed matrix, interpreted as a scalar.
    def scalar_value(self, matrix):
        return numpy.asscalar(matrix)

    # A matrix with all entries equal to the given scalar value.
    def scalar_matrix(self, value, rows, cols):
        return numpy.zeros((rows, cols), dtype='float64') + value

    def reshape(self, matrix, size):
        return numpy.reshape(matrix, size, order='F')
