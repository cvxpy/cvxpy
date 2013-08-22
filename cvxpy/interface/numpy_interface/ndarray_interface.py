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

import cvxpy.interface.base_matrix_interface as base
import cvxopt
import numpy
import numbers

class DenseMatrixInterface(base.BaseMatrixInterface):
    """ 
    An interface to convert constant values to the numpy ndarray class. 
    """
    TARGET_MATRIX = cvxopt.matrix
    # Convert an arbitrary value into a matrix of type self.target_matrix.
    def const_to_matrix(self, value):
        if isinstance(value, numbers.Number):
            return value
        return numpy.array(value, dtype='float64')

    # Return an identity matrix.
    def identity(self, size):
        return numpy.eye(size)

    # A matrix with all entries equal to the given scalar value.
    def scalar_matrix(self, value, rows, cols):
        return numpy.zeros((rows,cols), dtype='float64') + value

    def reshape(self, matrix, size):
        arr = numpy.array(list(matrix), dtype='float64')
        return numpy.reshape(arr, size)