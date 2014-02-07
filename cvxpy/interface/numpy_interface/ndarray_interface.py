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
import cvxopt
import numbers

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
        # Convert cvxopt sparse to dense.
        if isinstance(value, cvxopt.spmatrix):
            value = cvxopt.matrix(value)
        mat = numpy.array(value, dtype='float64')
        if isinstance(value, list):
            mat = numpy.atleast_2d(mat)
            return mat.T
        return numpy.atleast_2d(mat)

    # Return an identity matrix.
    def identity(self, size):
        return numpy.eye(size)

    # Return the dimensions of the matrix.
    def size(self, matrix):
        # Slicing drops the second dimension.
        if len(matrix.shape) == 1:
            dim = matrix.shape[0]
            return (dim, matrix.size/dim)
        else:
            return matrix.shape

    # Get the value of the passed matrix, interpreted as a scalar.
    def scalar_value(self, matrix):
        return numpy.asscalar(matrix)

    # A matrix with all entries equal to the given scalar value.
    def scalar_matrix(self, value, rows, cols):
        return numpy.zeros((rows,cols), dtype='float64') + value

    def reshape(self, matrix, size):
        return numpy.reshape(matrix, size, order='F')
