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

from .. import base_matrix_interface
import cvxopt
import numbers

class DenseMatrixInterface(base_matrix_interface.BaseMatrixInterface):
    """ 
    An interface to convert constant values to the cvxopt dense matrix class. 
    """
    TARGET_MATRIX = cvxopt.matrix
    # Convert an arbitrary value into a matrix of type self.target_matrix.
    def const_to_matrix(self, value):
        if isinstance(value, numbers.Number):
            return value
        return cvxopt.matrix(value, tc='d')

    # Return an identity matrix.
    def identity(self, size):
        matrix = self.zeros(size, size)
        for i in range(size):
            matrix[i,i] = 1
        return matrix

    # A matrix with all entries equal to the given scalar value.
    def scalar_matrix(self, value, rows, cols):
        return cvxopt.matrix(value, (rows,cols), tc='d')

    def reshape(self, matrix, size):
        return cvxopt.matrix(list(matrix), size, tc='d')