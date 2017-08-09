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
            return value.A.astype(numpy.float64)
        elif isinstance(value, numpy.matrix):
            return value.A.astype(numpy.float64)
        elif isinstance(value, list):
            return numpy.asarray(value, dtype=numpy.float64).T
        else:
            return numpy.asarray(value, dtype=numpy.float64)

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
