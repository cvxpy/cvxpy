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

import dense_matrix_interface as cvxopt_dense
import sparse_matrix_interface as cvxopt_sparse
import numpy_interface as np_intf
import cvxpy.utilities as u
import cvxopt
import scipy
import numbers
import numpy

DENSE_TARGET = cvxopt.matrix
SPARSE_TARGET = cvxopt.spmatrix
CVXOPT_DENSE_INTERFACE = cvxopt_dense.DenseMatrixInterface()
NDARRAY_INTERFACE = np_intf.DenseMatrixInterface()
DEFAULT_INTERFACE = cvxopt_dense.DenseMatrixInterface()

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
    elif isinstance(constant, (cvxopt.matrix, cvxopt.spmatrix)):
        return constant.size
    elif isinstance(constant, (numpy.ndarray, numpy.matrix)):
        # Slicing drops the second dimension.
        if len(constant.shape) == 1:
            dim = constant.shape[0]
            return (dim,constant.size/dim)
        else:
            return constant.shape
    else:
        raise Exception("%s is not a valid type for a Constant value." % type(constant))

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
    elif isinstance(constant, (cvxopt.matrix, cvxopt.spmatrix)):
        return constant[0,0]
    elif isinstance(constant, (numpy.ndarray, numpy.matrix)):
        return constant[0]

# Return a matrix of signs based on the constant's values.
# TODO sparse matrices.
def sign(constant):
    if isinstance(constant, numbers.Number):
        return u.Sign(constant < 0, constant > 0)
    elif isinstance(cvxopt.sparse, scipy.sparse):
        return NotImplemented
    else:
        cvxopt_mat = CVXOPT_DENSE_INTERFACE.const_to_matrix(constant)
        mat = numpy.array(cvxopt_mat)
        return u.Sign(u.BoolMat(mat < 0), u.BoolMat(mat > 0))
    
# Get the value at the given index.
def index(constant, key):
    if isinstance(constant, list):
        if is_vector(constant):
            return constant[key[0]]
        else:
            return constant[key[1]][key[0]]
    elif isinstance(constant, (cvxopt.matrix, cvxopt.spmatrix, 
                               numpy.ndarray, numpy.matrix)):
        return constant[key] 

# Return a dense matrix with all 0's.
# Needed for constant vectors in conelp.
def dense_zeros(rows, cols):
    return cvxopt_dense.DenseMatrixInterface().zeros(rows, cols)