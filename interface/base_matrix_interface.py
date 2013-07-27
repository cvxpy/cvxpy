import matrix_utilities as intf
import abc

class BaseMatrixInterface(object):
    """
    An interface between constants' internal values
    and the target matrix used internally.
    """
    __metaclass__ = abc.ABCMeta
    # Convert an arbitrary value into a matrix of type self.target_matrix.
    @abc.abstractmethod
    def const_to_matrix(self, value):
        return NotImplemented

    # Return an identity matrix.
    @abc.abstractmethod
    def identity(self, size):
        return NotImplemented

    # Return a matrix with all 0's.
    def zeros(self, rows, cols):
        return self.scalar_matrix(0, rows, cols)

    # Return a matrix with all 1's.
    def ones(self, rows, cols):
        return self.scalar_matrix(1, rows, cols)

    # A matrix with all entries equal to the given scalar value.
    @abc.abstractmethod
    def scalar_matrix(self, value, rows, cols):
        return NotImplemented

    # A matrix with all entries equal to the given scalar value.
    @abc.abstractmethod
    def list_to_matrix(self, values, size):
        return NotImplemented

    # Copy the block into the matrix at the given offset.
    def block_copy(self, matrix, block, vert_offset, horiz_offset, rows, cols):
        # If the block is a scalar, promote it.
        if intf.is_scalar(block):
            block = self.scalar_matrix(intf.scalar_value(block), rows, cols)
        # If the block is a vector coerced into a matrix, promote it.
        elif intf.is_vector(block) and cols > 1:
            block = self.list_to_matrix(list(block), (rows, cols))
        # If the block is a matrix coerced into a vector, vectorize it.
        elif not intf.is_vector(block) and cols == 1:
            block = self.list_to_matrix(list(block), (rows*cols, 1))
        matrix[vert_offset:(rows+vert_offset), horiz_offset:(horiz_offset+cols)] = block