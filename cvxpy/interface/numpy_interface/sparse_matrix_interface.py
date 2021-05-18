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

from cvxpy.interface.numpy_interface.ndarray_interface import NDArrayInterface
import scipy.sparse as sp
import numpy as np


class SparseMatrixInterface(NDArrayInterface):
    """
    An interface to convert constant values to the scipy sparse CSC class.
    """
    TARGET_MATRIX = sp.csc_matrix

    @NDArrayInterface.scalar_const
    def const_to_matrix(self, value, convert_scalars: bool = False):
        """Convert an arbitrary value into a matrix of type self.target_matrix.

        Args:
            value: The constant to be converted.
            convert_scalars: Should scalars be converted?

        Returns:
            A matrix of type self.target_matrix or a scalar.
        """
        # Convert cvxopt sparse to coo matrix.
        if isinstance(value, list):
            return sp.csc_matrix(value, dtype=np.double).T
        if value.dtype in [np.double, np.complex]:
            dtype = value.dtype
        else:
            # Cast bool, int, etc to double
            dtype = np.double
        return sp.csc_matrix(value, dtype=dtype)

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
                  vert_step: int = 1, horiz_step: int = 1) -> None:
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
