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
DEFAULT_INTF = INTERFACES[np.ndarray]
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


def shape(constant):
    if isinstance(constant, numbers.Number) or np.isscalar(constant):
        return tuple()
    elif isinstance(constant, list):
        if len(constant) == 0:
            return (0,)
        elif isinstance(constant[0], numbers.Number):  # Vector
            return (len(constant),)
        else:  # Matrix
            return (len(constant[0]), len(constant))
    elif constant.__class__ in INTERFACES:
        return INTERFACES[constant.__class__].shape(constant)
    # Direct all sparse matrices to CSC interface.
    elif is_sparse(constant):
        return INTERFACES[sp.csc_matrix].shape(constant)
    else:
        raise TypeError("%s is not a valid type for a Constant value." % type(constant))

# Is the constant a column vector?


def is_vector(constant):
    return shape(constant)[1] == 1

# Is the constant a scalar?


def is_scalar(constant):
    return shape(constant) == (1, 1)


def from_2D_to_1D(constant):
    """Convert 2D Numpy matrices or arrays to 1D.
    """
    if isinstance(constant, np.ndarray) and constant.ndim == 2:
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


def convert(constant, sparse=False, convert_scalars=False):
    """Convert to appropriate type.
    """
    if isinstance(constant, (list, np.matrix)):
        return DEFAULT_INTF.const_to_matrix(constant,
                                            convert_scalars=convert_scalars)
    elif sparse:
        return DEFAULT_SPARSE_INTF.const_to_matrix(constant,
                                                   convert_scalars=convert_scalars)
    else:
        return constant

# Get the value of the passed constant, interpreted as a scalar.


def scalar_value(constant):
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
    """Return (is positive, is negative).

    Parameters
    ----------
    constant : numeric type
        The numeric value to evaluate the sign of.

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
    return (min_val >= 0, max_val <= 0)


def is_complex(constant, tol=1e-5):
    """Return (is real, is imaginary).

    Parameters
    ----------
    constant : numeric type
        The numeric value to evaluate the sign of.
    tol : float, optional
        The largest magnitude considered nonzero.

    Returns
    -------
    tuple
        The sign of the constant.
    """
    complex_type = np.iscomplexobj(constant)
    if not complex_type:
        return True, False
    if isinstance(constant, numbers.Number):
        real_max = np.abs(np.real(constant))
        imag_max = np.abs(np.imag(constant))
    elif sp.issparse(constant):
        real_max = np.abs(constant.real).max()
        imag_max = np.abs(constant.imag).max()
    else:  # Convert to Numpy array.
        constant = INTERFACES[np.ndarray].const_to_matrix(constant)
        real_max = np.abs(constant.real).max()
        imag_max = np.abs(constant.imag).max()
    return (real_max >= tol, imag_max >= tol)

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


def is_hermitian(constant):
    """Check if a matrix is Hermitian and/or symmetric.
    """
    complex_type = np.iscomplexobj(constant)
    if complex_type:
        # TODO catch complex symmetric but not Hermitian?
        is_symm = False
        if sp.issparse(constant):
            is_herm = is_sparse_symmetric(constant, complex=True)
        else:
            is_herm = np.allclose(constant, np.conj(constant.T))
    else:
        if sp.issparse(constant):
            is_symm = is_sparse_symmetric(constant, complex=False)
        else:
            is_symm = np.allclose(constant, constant.T)
        is_herm = is_symm
    return is_symm, is_herm


def is_sparse_symmetric(m, complex=False):
    """Check if a sparse matrix is symmetric

    Parameters
    ----------
    m : array or sparse matrix
        A square matrix.

    Returns
    -------
    check : bool
        The check result.

    """
    # https://mail.scipy.org/pipermail/scipy-dev/2014-October/020101.html
    if m.shape[0] != m.shape[1]:
        raise ValueError('m must be a square matrix')

    if not isinstance(m, sp.coo_matrix):
        m = sp.coo_matrix(m)

    r, c, v = m.row, m.col, m.data
    tril_no_diag = r > c
    triu_no_diag = c > r

    if triu_no_diag.sum() != tril_no_diag.sum():
        return False

    rl = r[tril_no_diag]
    cl = c[tril_no_diag]
    vl = v[tril_no_diag]
    ru = r[triu_no_diag]
    cu = c[triu_no_diag]
    vu = v[triu_no_diag]

    sortl = np.lexsort((cl, rl))
    sortu = np.lexsort((ru, cu))
    vl = vl[sortl]
    vu = vu[sortu]

    if complex:
        check = np.allclose(vl, np.conj(vu))
    else:
        check = np.allclose(vl, vu)

    return check
