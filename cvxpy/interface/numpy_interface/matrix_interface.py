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


from cvxpy.interface.numpy_interface.ndarray_interface import NDArrayInterface
import scipy.sparse as sp
import numpy as np
import cvxopt

class MatrixInterface(NDArrayInterface):
    """
    An interface to convert constant values to the numpy matrix class.
    """
    TARGET_MATRIX = np.matrix

    @NDArrayInterface.scalar_const
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
        # Lists and 1D arrays become column vectors.
        elif isinstance(value, list) or \
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
    def scalar_matrix(self, value, rows, cols):
        mat = np.zeros((rows, cols), dtype='float64') + value
        return np.asmatrix(mat)

    def reshape(self, matrix, size):
        return np.reshape(matrix, size, order='F')

