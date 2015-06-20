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

from cvxpy.interface import cvxopt_interface as co_intf
from cvxpy.interface import numpy_interface as np_intf
import scipy.sparse as sp
import numbers
import numpy as np
from cvxpy.utilities.sign import Sign
import cvxopt

# A mapping of class to interface.
INTERFACES = {cvxopt.matrix: co_intf.DenseMatrixInterface(),
              cvxopt.spmatrix: co_intf.SparseMatrixInterface(),
              np.ndarray: np_intf.NDArrayInterface(),
              np.matrix: np_intf.MatrixInterface(),
              sp.csc_matrix: np_intf.SparseMatrixInterface(),
}
# Default Numpy interface.
DEFAULT_NP_INTF = INTERFACES[np.ndarray]
# Default dense and sparse matrix interfaces.
DEFAULT_INTF = INTERFACES[np.matrix]
DEFAULT_SPARSE_INTF = INTERFACES[sp.csc_matrix]
# CVXOPT interfaces.
CVXOPT_DENSE_INTF = INTERFACES[cvxopt.matrix]
CVXOPT_SPARSE_INTF = INTERFACES[cvxopt.spmatrix]

# Returns the interface for interacting with the target matrix class.
def get_matrix_interface(target_class):
    return INTERFACES[target_class]

def is_sparse(constant):
    """Is the constant a sparse matrix?
    """
    return sp.issparse(constant) or isinstance(constant, cvxopt.spmatrix)

# Get the dimensions of the constant.
def size(constant):
    if isinstance(constant, numbers.Number) or np.isscalar(constant):
        return (1, 1)
    elif isinstance(constant, list):
        if len(constant) == 0:
            return (0,0)
        elif isinstance(constant[0], numbers.Number): # Vector
            return (len(constant),1)
        else: # Matrix
            return (len(constant[0]), len(constant))
    elif constant.__class__ in INTERFACES:
        return INTERFACES[constant.__class__].size(constant)
    # Direct all sparse matrices to CSC interface.
    elif is_sparse(constant):
        return INTERFACES[sp.csc_matrix].size(constant)
    else:
        raise TypeError("%s is not a valid type for a Constant value." % type(constant))

# Is the constant a column vector?
def is_vector(constant):
    return size(constant)[1] == 1

# Is the constant a scalar?
def is_scalar(constant):
    return size(constant) == (1, 1)

def from_2D_to_1D(constant):
    """Convert 2D Numpy matrices or arrays to 1D.
    """
    if isinstance(constant, np.ndarray):
        return np.asarray(constant)[:, 0]
    else:
        return constant

def from_1D_to_2D(constant):
    """Convert 1D Numpy arrays to matrices.
    """
    if isinstance(constant, np.ndarray) and constant.ndim == 1:
        return np.mat(constant).T
    else:
        return constant

# Get the value of the passed constant, interpreted as a scalar.
def scalar_value(constant):
    assert is_scalar(constant)
    if isinstance(constant, numbers.Number) or np.isscalar(constant):
        return constant
    elif isinstance(constant, list):
        return constant[0]
    elif constant.__class__ in INTERFACES:
        return INTERFACES[constant.__class__].scalar_value(constant)
    # Direct all sparse matrices to CSC interface.
    elif is_sparse(constant):
        return INTERFACES[sp.csc_matrix].scalar_value(constant.tocsc())
    else:
        raise TypeError("%s is not a valid type for a Constant value." % type(constant))

# Return the collective sign of the matrix entries.
def sign(constant):
    if isinstance(constant, numbers.Number):
        return Sign.val_to_sign(constant)
    elif isinstance(constant, cvxopt.spmatrix):
        max_val = max(constant.V)
        min_val = min(constant.V)
    elif sp.issparse(constant):
        max_val = constant.max()
        min_val = constant.min()
    else: # Convert to Numpy array.
        mat = INTERFACES[np.ndarray].const_to_matrix(constant)
        max_val = mat.max()
        min_val = mat.min()
    max_sign = Sign.val_to_sign(max_val)
    min_sign = Sign.val_to_sign(min_val)
    return max_sign + min_sign

# Get the value at the given index.
def index(constant, key):
    if is_scalar(constant):
        return constant
    elif constant.__class__ in INTERFACES:
        return INTERFACES[constant.__class__].index(constant, key)
    # Use CSC interface for all sparse matrices.
    elif is_sparse(constant):
        interface = INTERFACES[sp.csc_matrix]
        constant = interface.const_to_matrix(constant)
        return interface.index(constant, key)
