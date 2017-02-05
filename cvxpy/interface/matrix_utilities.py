"""
Copyright 2017 Steven Diamond

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

from cvxpy.interface import numpy_interface as np_intf
import scipy.sparse as sp
import numbers
import numpy as np

# A mapping of class to interface.
INTERFACES = {np.ndarray: np_intf.NDArrayInterface(),
              np.matrix: np_intf.MatrixInterface(),
              sp.csc_matrix: np_intf.SparseMatrixInterface(),
              }
# Default Numpy interface.
DEFAULT_NP_INTF = INTERFACES[np.ndarray]
# Default dense and sparse matrix interfaces.
DEFAULT_INTF = INTERFACES[np.matrix]
DEFAULT_SPARSE_INTF = INTERFACES[sp.csc_matrix]


# Returns the interface for interacting with the target matrix class.
def get_matrix_interface(target_class):
    return INTERFACES[target_class]


def get_cvxopt_dense_intf():
    """Dynamic import of CVXOPT dense interface.
    """
    import cvxpy.interface.cvxopt_interface.valuerix_interface as dmi
    return dmi.DenseMatrixInterface()


def get_cvxopt_sparse_intf():
    """Dynamic import of CVXOPT sparse interface.
    """
    import cvxpy.interface.cvxopt_interface.sparse_matrix_interface as smi
    return smi.SparseMatrixInterface()

# Tools for handling CVXOPT matrices.


def sparse2cvxopt(value):
    """Converts a SciPy sparse matrix to a CVXOPT sparse matrix.

    Parameters
    ----------
    sparse_mat : SciPy sparse matrix
        The matrix to convert.

    Returns
    -------
    CVXOPT spmatrix
        The converted matrix.
    """
    import cvxopt
    if isinstance(value, (np.ndarray, np.matrix)):
        return cvxopt.sparse(cvxopt.matrix(value.astype('float64')), tc='d')
    # Convert scipy sparse matrices to coo form first.
    elif sp.issparse(value):
        value = value.tocoo()
        return cvxopt.spmatrix(value.data.tolist(), value.row.tolist(),
                               value.col.tolist(), size=value.shape, tc='d')


def dense2cvxopt(value):
    """Converts a NumPy matrix to a CVXOPT matrix.

    Parameters
    ----------
    value : NumPy matrix/ndarray
        The matrix to convert.

    Returns
    -------
    CVXOPT matrix
        The converted matrix.
    """
    import cvxopt
    return cvxopt.matrix(value, tc='d')


def cvxopt2dense(value):
    """Converts a CVXOPT matrix to a NumPy ndarray.

    Parameters
    ----------
    value : CVXOPT matrix
        The matrix to convert.

    Returns
    -------
    NumPy ndarray
        The converted matrix.
    """
    return np.array(value)


def is_sparse(constant):
    """Is the constant a sparse matrix?
    """
    return sp.issparse(constant)

# Get the dimensions of the constant.


def size(constant):
    if isinstance(constant, numbers.Number) or np.isscalar(constant):
        return (1, 1)
    elif isinstance(constant, list):
        if len(constant) == 0:
            return (0, 0)
        elif isinstance(constant[0], numbers.Number):  # Vector
            return (len(constant), 1)
        else:  # Matrix
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


def sign(constant, tol=1e-5):
    """Return (is positive, is negative).

    Parameters
    ----------
    constant : numeric type
        The numeric value to evaluate the sign of.
    tol : float, optional
        The largest (smallest) value considered positive (negative).

    Returns
    -------
    tuple
        The sign of the constant.
    """
    if isinstance(constant, numbers.Number):
        max_val = constant
        min_val = constant
    elif sp.issparse(constant):
        max_val = constant.max()
        min_val = constant.min()
    else:  # Convert to Numpy array.
        mat = INTERFACES[np.ndarray].const_to_matrix(constant)
        max_val = mat.max()
        min_val = mat.min()
    return (min_val >= -tol, max_val <= tol)

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
