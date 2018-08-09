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
        if scipy.sparse.issparse(value):
            result = value.A
        elif isinstance(value, numpy.matrix):
            result = value.A
        elif isinstance(value, list):
            result = numpy.asarray(value).T
        else:
            result = numpy.asarray(value)
        if result.dtype in [numpy.complex, numpy.float64]:
            return result
        else:
            return result.astype(numpy.float64)

    # Return an identity matrix.
    def identity(self, size):
        return numpy.eye(size)

    # Return the dimensions of the matrix.
    def shape(self, matrix):
        return tuple(int(d) for d in matrix.shape)

    def size(self, matrix):
        """Returns the number of elements in the matrix.
        """
        return numpy.prod(self.shape(matrix))

    # Get the value of the passed matrix, interpreted as a scalar.
    def scalar_value(self, matrix):
        return numpy.asscalar(matrix)

    # A matrix with all entries equal to the given scalar value.
    def scalar_matrix(self, value, shape):
        return numpy.zeros(shape, dtype='float64') + value

    def reshape(self, matrix, size):
        return numpy.reshape(matrix, size, order='F')
