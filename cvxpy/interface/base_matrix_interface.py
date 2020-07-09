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

import cvxpy.interface.matrix_utilities
import abc
import numpy as np


class BaseMatrixInterface(object):
    """
    An interface between constants' internal values
    and the target matrix used internally.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def const_to_matrix(self, value, convert_scalars=False):
        """Convert an arbitrary value into a matrix of type self.target_matrix.

        Args:
            value: The constant to be converted.
            convert_scalars: Should scalars be converted?

        Returns:
            A matrix of type self.target_matrix or a scalar.
        """
        return NotImplemented

    # Adds a case for scalars to const_to_matrix methods.
    @staticmethod
    def scalar_const(converter):
        def new_converter(self, value, convert_scalars=False):
            if not convert_scalars and cvxpy.interface.matrix_utilities.is_scalar(value):
                return cvxpy.interface.matrix_utilities.scalar_value(value)
            else:
                return converter(self, value)
        return new_converter

    # Return an identity matrix.
    @abc.abstractmethod
    def identity(self, size):
        return NotImplemented

    # Return the number of elements of the matrix.
    def size(self, matrix):
        return np.prod(self.shape(matrix), dtype=int)

    # Return the dimensions of the matrix.
    @abc.abstractmethod
    def shape(self, matrix):
        return NotImplemented

    # Get the matrix interpreted as a scalar.
    @abc.abstractmethod
    def scalar_value(self, matrix):
        return NotImplemented

    # Return a matrix with all 0's.
    def zeros(self, shape):
        return self.scalar_matrix(0, shape)

    # Return a matrix with all 1's.
    def ones(self, shape):
        return self.scalar_matrix(1, shape)

    # A matrix with all entries equal to the given scalar value.
    @abc.abstractmethod
    def scalar_matrix(self, value, shape):
        return NotImplemented

    # Return the value at the given index in the matrix.
    def index(self, matrix, key):
        value = matrix[key]
        # Reduce to a scalar if possible.
        if cvxpy.interface.matrix_utilities.shape(value) == (1, 1):
            return cvxpy.interface.matrix_utilities.scalar_value(value)
        else:
            return value

    # Coerce the matrix into the given shape.
    @abc.abstractmethod
    def reshape(self, matrix, shape):
        return NotImplemented

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
        matrix[vert_offset:(rows+vert_offset):vert_step,
               horiz_offset:(horiz_offset+cols):horiz_step] += block

    def _format_block(self, matrix, block, rows, cols):
        """Formats the block for block_add.

        Args:
            matrix: The matrix the block will be added to.
            block: The matrix/scalar to be added.
            rows: The height of the block.
            cols: The width of the block.
        """
        # If the block is a scalar, promote it.
        if cvxpy.interface.matrix_utilities.is_scalar(block):
            block = self.scalar_matrix(
                cvxpy.interface.matrix_utilities.scalar_value(block), rows, cols)
        # If the block is a vector coerced into a matrix, promote it.
        elif cvxpy.interface.matrix_utilities.is_vector(block) and cols > 1:
            block = self.reshape(block, (rows, cols))
        # If the block is a matrix coerced into a vector, vectorize it.
        elif not cvxpy.interface.matrix_utilities.is_vector(block) and cols == 1:
            block = self.reshape(block, (rows, cols))
        # Ensure the block is the same type as the matrix.
        elif type(block) != type(matrix):
            block = self.const_to_matrix(block)
        return block
