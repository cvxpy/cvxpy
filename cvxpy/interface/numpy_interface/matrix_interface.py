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

from ndarray_interface import NDArrayInterface
import cvxopt
import numpy

class MatrixInterface(NDArrayInterface):
    """
    An interface to convert constant values to the numpy matrix class.
    """
    TARGET_MATRIX = numpy.matrix
    # Convert an arbitrary value into a matrix of type self.target_matrix.
    @NDArrayInterface.scalar_const
    def const_to_matrix(self, value):
        if isinstance(value, list):
            mat = numpy.asmatrix(value, dtype='float64')
            return mat.T
        return numpy.asmatrix(value, dtype='float64')

    # Return an identity matrix.
    def identity(self, size):
        return numpy.asmatrix(numpy.eye(size))

    # A matrix with all entries equal to the given scalar value.
    def scalar_matrix(self, value, rows, cols):
        mat = numpy.zeros((rows,cols), dtype='float64') + value
        return numpy.asmatrix(mat)

    def reshape(self, matrix, size):
        return numpy.reshape(matrix, size, order='F')