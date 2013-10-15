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

import cvxopt_interface as co_intf
import numpy_interface as np_intf
import numpy_wrapper
import cvxpy.utilities as u
import cvxopt
import scipy.sparse as sp
import numbers
import numpy as np

# A mapping of class to interface.
INTERFACES = {cvxopt.matrix: co_intf.DenseMatrixInterface(),
              cvxopt.spmatrix: co_intf.SparseMatrixInterface(),
              np.ndarray: np_intf.NDArrayInterface(),
              numpy_wrapper.ndarray: np_intf.NDArrayInterface(),
              np.matrix: np_intf.MatrixInterface(),
              numpy_wrapper.matrix: np_intf.MatrixInterface(),
}
# Default dense and sparse matrix interfaces.
DEFAULT_INTERFACE = INTERFACES[cvxopt.matrix]
DEFAULT_SPARSE_INTERFACE = INTERFACES[cvxopt.spmatrix]

# Returns the interface for interacting with the target matrix class.
def get_matrix_interface(target_class):
    return INTERFACES[target_class]

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
    elif constant.__class__ in INTERFACES:
        return INTERFACES[constant.__class__].size(constant)
    else:
        raise Exception("%s is not a valid type for a Constant value." % type(constant))

# Is the constant a column vector?
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
    elif constant.__class__ in INTERFACES:
        return INTERFACES[constant.__class__].scalar_value(constant)
    else:
        raise Exception("%s is not a valid type for a Constant value." % type(constant))

# Return a matrix of signs based on the constant's values.
# TODO scipy sparse matrices.
def sign(constant):
    if isinstance(constant, numbers.Number):
        return u.Sign(constant < 0, constant > 0)
    elif isinstance(constant, cvxopt.spmatrix):
        # Convert to COO matrix.
        V = np.array(list(constant.V))
        I = list(constant.I)
        J = list(constant.J)
        # Check if entries > 0 for pos_mat, < 0 for neg_mat.
        neg_mat = sp.coo_matrix((V < 0,(I,J)), shape=constant.size, dtype='bool')
        pos_mat = sp.coo_matrix((V > 0,(I,J)), shape=constant.size, dtype='bool')
        return u.Sign(u.SparseBoolMat(neg_mat), u.SparseBoolMat(pos_mat))
    else:
        mat = INTERFACES[np.ndarray].const_to_matrix(constant)
        return u.Sign(u.BoolMat(mat < 0), u.BoolMat(mat > 0))
    
# Get the value at the given index.
def index(constant, key):
    if isinstance(constant, list):
        if is_vector(constant):
            return constant[key[0]]
        else:
            selection = [l[key[0]] for l in constant[key[1]]]
            for i in range(2): # Unpack column vectors and scalars.
                if len(selection) == 1:
                    selection = selection[0]
            return selection
    elif constant.__class__ in INTERFACES:
        return INTERFACES[constant.__class__].index(constant, key)

# Get the tranpose of the given constant.
def transpose(constant):
    const_size = size(constant)
    if const_size == (1,1):
        return constant
    elif isinstance(constant, list):
        # Column vector
        if const_size[1] == 1:
            return [[elem] for elem in constant]
        # Row vector
        elif const_size[0] == 1:
            return [elem[0] for elem in constant]
        else:
            # http://stackoverflow.com/questions/6473679/python-list-of-lists-transpose-without-zipm-thing
            return map(list, zip(*constant))
    elif constant.__class__ in INTERFACES:
        return INTERFACES[constant.__class__].transpose(constant)