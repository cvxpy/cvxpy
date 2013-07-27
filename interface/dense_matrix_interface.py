import base_matrix_interface
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

    def list_to_matrix(self, values, size):
        return cvxopt.matrix(values, size, tc='d')