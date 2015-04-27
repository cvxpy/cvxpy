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
import numbers
import cvxopt


class SparseMatrixInterface(NDArrayInterface):
    """
    An interface to convert constant values to the scipy sparse CSC class.
    """
    TARGET_MATRIX = sp.csc_matrix

    @NDArrayInterface.scalar_const
    def const_to_matrix(self, value, convert_scalars=False):
        """Convert an arbitrary value into a matrix of type self.target_matrix.

        Args:
            value: The constant to be converted.
            convert_scalars: Should scalars be converted?

        Returns:
            A matrix of type self.target_matrix or a scalar.
        """
        # Convert cvxopt sparse to coo matrix.
        if isinstance(value, cvxopt.spmatrix):
            Vp, Vi, Vx = value.CCS
            Vp, Vi = (np.fromiter(iter(x),
                                  dtype=np.int32,
                                  count=len(x))
                      for x in (Vp, Vi))
            Vx = np.fromiter(iter(Vx), dtype=np.double)
            m, n = value.size
            return sp.csc_matrix((Vx, Vi, Vp), shape=(m, n))
        if isinstance(value, list):
            return sp.csc_matrix(value, dtype=np.double).T
        return sp.csc_matrix(value, dtype=np.double)

    def identity(self, size):
        """Return an identity matrix.
        """
        return sp.eye(size, size, format="csc")

    def size(self, matrix):
        """Return the dimensions of the matrix.
        """
        return matrix.shape

    def scalar_value(self, matrix):
        """Get the value of the passed matrix, interpreted as a scalar.
        """
        return matrix[0, 0]

    def zeros(self, rows, cols):
        """Return a matrix with all 0's.
        """
        return sp.csc_matrix((rows, cols), dtype='float64')

    def reshape(self, matrix, size):
        """Change the shape of the matrix.
        """
        matrix = matrix.todense()
        matrix = super(SparseMatrixInterface, self).reshape(matrix, size)
        return self.const_to_matrix(matrix, convert_scalars=True)

    def block_add(self, matrix, block, vert_offset, horiz_offset, rows, cols,
                  vert_step=1, horiz_step=1):
        """Add the block to a slice of the matrix.

        Args:
            matrix: The matrix the block will be added to.
            block: The matrix/scalar to be added.
            vert_offset: The starting row for the matrix slice.
            horiz_offset: The starting column for the matrix slice.
            rows: The height of the block.
            cols: The width of the block.
            vert_step: The row step size for the matrix slice.
            horiz_step: The column step size for the matrix slice.
        """
        block = self._format_block(matrix, block, rows, cols)
        slice_ = [slice(vert_offset, rows+vert_offset, vert_step),
                  slice(horiz_offset, horiz_offset+cols, horiz_step)]
        # Convert to lil before changing sparsity structure.
        matrix[slice_[0], slice_[1]] = matrix[slice_[0], slice_[1]] + block
