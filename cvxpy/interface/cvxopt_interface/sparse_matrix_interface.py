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

from cvxpy.interface.cvxopt_interface.dense_matrix_interface import DenseMatrixInterface
import scipy.sparse as sp
import cvxopt
import numpy
import numbers

class SparseMatrixInterface(DenseMatrixInterface):
    """
    An interface to convert constant values to the cvxopt sparse matrix class.
    """
    TARGET_MATRIX = cvxopt.spmatrix

    @DenseMatrixInterface.scalar_const
    def const_to_matrix(self, value, convert_scalars=False):
        """Convert an arbitrary value into a matrix of type self.target_matrix.

        Args:
            value: The constant to be converted.
            convert_scalars: Should scalars be converted?

        Returns:
            A matrix of type self.target_matrix or a scalar.
        """
        if isinstance(value, (numpy.ndarray, numbers.Number)):
            return cvxopt.sparse(cvxopt.matrix(value), tc='d')
        # Convert scipy sparse matrices to coo form first.
        if sp.issparse(value):
            value = value.tocoo()
            V = value.data
            I = value.row
            J = value.col
            return cvxopt.spmatrix(V, I, J, value.shape, tc='d')
        return cvxopt.sparse(value, tc='d')

    # Return an identity matrix.
    def identity(self, size):
        return cvxopt.spmatrix(1, range(size), range(size))

    # A matrix with all entries equal to the given scalar value.
    def scalar_matrix(self, value, rows, cols):
        if isinstance(rows, numbers.Number):
            rows = int(rows)
        if isinstance(cols, numbers.Number):
            cols = int(cols)

        if value == 0:
            return cvxopt.spmatrix(0, [], [], size=(rows,cols))
        else:
            dense = cvxopt.matrix(value, (rows,cols), tc='d')
            return cvxopt.sparse(dense)

    def reshape(self, matrix, size):
        old_size = matrix.size
        new_mat = self.zeros(*size)
        for v,i,j in zip(matrix.V, matrix.I, matrix.J):
            pos = i + old_size[0]*j
            new_row = pos % size[0]
            new_col = pos // size[0]
            new_mat[new_row, new_col] = v
        return new_mat
