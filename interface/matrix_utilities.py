import dense_matrix_interface as cvxopt_dense
import sparse_matrix_interface as cvxopt_sparse
import cvxopt
import numbers
import numpy

DENSE_TARGET = cvxopt.matrix
SPARSE_TARGET = cvxopt.spmatrix
DEFAULT_INTERFACE = cvxopt_dense.DenseMatrixInterface

# Returns an interface between constants' internal values
# and the target matrix used internally.
def get_matrix_interface(target_matrix):
    if target_matrix is cvxopt.matrix:
        return cvxopt_dense.DenseMatrixInterface()
    else:
        return cvxopt_sparse.SparseMatrixInterface()

# Get the dimensions of the constant.
def size(constant):
    if isinstance(constant, numbers.Number):
        return (1,1)
    elif isinstance(constant, list):
        if len(constant) == 0:
            return (0,0)
        elif isinstance(constant[0], numbers.Number): # Vector
            return (len(constant),1)
        else: # Matrix
            return (len(constant[0]),len(constant))
    elif isinstance(constant, cvxopt.matrix) or \
         isinstance(constant, cvxopt.spmatrix):
        return constant.size
    elif isinstance(constant, numpy.ndarray):
        return constant.shape

# Is the constant a vector?
def is_vector(constant):
    return size(constant)[1] == 1

# Is the constant a scalar?
def is_scalar(constant):
    return size(constant) == (1,1)

# Get the value of the passed constant, interpreted as a scalar.
def scalar_value(constant):
    assert is_scalar(constant)
    if isinstance(constant, numbers.Number):
        return constant
    elif isinstance(constant, list):
        return constant[0]
    elif isinstance(constant, cvxopt.matrix) or \
         isinstance(constant, cvxopt.spmatrix) or \
         isinstance(constant, numpy.ndarray) or \
         isinstance(constant, numpy.matrix):
        return constant[0,0]

# Return a dense matrix with all 0's.
# Needed for constant vectors in conelp.
def dense_zeros(rows, cols):
    return cvxopt_dense.DenseMatrixInterface().zeros(rows, cols)