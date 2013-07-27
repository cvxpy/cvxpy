import base_matrix_interface
import cvxopt
import numbers
import numpy

class SparseMatrixInterface(base_matrix_interface.BaseMatrixInterface):
    """ 
    An interface to convert constant values to the cvxopt sparse matrix class. 
    """
    TARGET_MATRIX = cvxopt.spmatrix
    # Convert an arbitrary value into a matrix of type self.target_matrix.
    def const_to_matrix(self, value):
        if isinstance(value, numbers.Number):
            return value
        if isinstance(value, numpy.ndarray):
            return cvxopt.sparse(cvxopt.matrix(value), tc='d')
        return cvxopt.sparse(value, tc='d')

    # Return an identity matrix.
    def identity(self, size):
        return cvxopt.spmatrix(1, range(size), range(size))

    # A matrix with all entries equal to the given scalar value.
    def scalar_matrix(self, value, rows, cols):
        return cvxopt.sparse( cvxopt.matrix(value, (rows,cols)), tc='d' )

    def list_to_matrix(self, values, size):
        return cvxopt.sparse( cvxopt.matrix(values, size), tc='d' )