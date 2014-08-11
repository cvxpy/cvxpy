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

import cvxpy.lin_ops.lin_op as lo
import cvxpy.interface as intf
import numpy as np
import scipy.sparse as sp
import scipy.linalg as sp_la

# Utility functions for converting LinOps into matrices.

def flatten(matrix):
    """Converts the matrix into a column vector.

    Parameters
    ----------
    matrix :
        The matrix to flatten.
    """
    np_mat = intf.DEFAULT_INTERFACE
    matrix = np_mat.const_to_matrix(matrix, convert_scalars=True)
    size = intf.size(matrix)
    return np_mat.reshape(matrix, (size[0]*size[1], 1))

def get_coefficients(lin_op):
    """Converts a linear op into coefficients.

    Parameters
    ----------
    lin_op : LinOp
        The linear op to convert.

    Returns
    -------
    list
        A list of (id, coefficient) tuples.
    """
    # VARIABLE converts to a giant identity matrix.
    if lin_op.type is lo.VARIABLE:
        coeffs = var_coeffs(lin_op)
    # Constants convert directly to their value.
    elif lin_op.type in CONSTANT_TYPES:
        mat = const_mat(lin_op)
        coeffs = [(lo.CONSTANT_ID, flatten(mat))]
    # Sum concatenates coefficients.
    elif lin_op.type is lo.SUM:
        coeffs = sum_coeffs(lin_op)
    # All other types operate via a coefficient matrix.
    elif lin_op.type in TYPE_TO_FUNC:
        coeff_mat = TYPE_TO_FUNC[lin_op.type](lin_op)
        rh_coeffs = get_coefficients(lin_op.args[0])
        coeffs = mul_by_const(coeff_mat, rh_coeffs)
    else:
        raise Exception("Unknown linear operator.")
    return coeffs

def get_constant_coeff(lin_op):
    """Converts a linear op into coefficients and returns the constant term.

    Parameters
    ----------
    lin_op : LinOp
        The linear op to convert.

    Returns
    -------
    The constant coefficient or None if none present.
    """
    coeffs = get_coefficients(lin_op)
    for id_, coeff in coeffs:
        if id_ is lo.CONSTANT_ID:
            return coeff
    return None

def var_coeffs(lin_op):
    """Returns the coefficients for a VARIABLE.

    Parameters
    ----------
    lin_op : LinOp
        The variable linear op.

    Returns
    -------
    list
       A list of (id, size, coefficient) tuples.
    """
    id_ = lin_op.data
    coeff = sp.eye(lin_op.size[0]*lin_op.size[1]).tocsc()
    return [(id_, coeff)]

def const_mat(lin_op):
    """Returns the matrix for a constant type.

    Parameters
    ----------
    lin_op : LinOp
        The linear op.

    Returns
    -------
    A numerical constant.
    """
    if lin_op.type is lo.PARAM:
        coeff = lin_op.data.value
    elif lin_op.type in [lo.SCALAR_CONST, lo.DENSE_CONST, lo.SPARSE_CONST]:
        coeff = lin_op.data
    return coeff

def mul_by_const(constant, rh_coeffs):
    """Multiplies a constant by a list of coefficients.

    Parameters
    ----------
    constant : numeric type
        The constant to multiply by.
    rh_coeffs : list
        The coefficients of the right hand side.

    Returns
    -------
    list
        A list of (id, size, coefficient) tuples.
    """
    new_coeffs = []
    # Multiply all right-hand terms by the left-hand constant.
    for (id_, coeff) in rh_coeffs:
        new_coeffs.append((id_, constant*coeff))

    return new_coeffs

def sum_coeffs(lin_op):
    """Returns the coefficients for SUM linear op.

    Parameters
    ----------
    lin_op : LinOp
        The sum linear op.

    Returns
    -------
    list
       A list of (id, size, coefficient) tuples.
    """
    coeffs = []
    for arg in lin_op.args:
        coeffs += get_coefficients(arg)
    return coeffs

def sum_entries_mat(lin_op):
    """Returns the coefficient matrix for SUM_ENTRIES linear op.

    Parameters
    ----------
    lin_op : LinOp
        The sum entries linear op.

    Returns
    -------
    NumPy matrix
        The matrix representing the sum_entries operation.
    """
    rows, cols = lin_op.args[0].size
    coeff = np.ones((1, rows*cols))
    return np.matrix(coeff)

def trace_mat(lin_op):
    """Returns the coefficient matrix for TRACE linear op.

    Parameters
    ----------
    lin_op : LinOp
        The trace linear op.

    Returns
    -------
    NumPy matrix
        The matrix representing the trace operation.
    """
    rows, _ = lin_op.args[0].size
    mat = np.zeros((1, rows**2))
    for i in xrange(rows):
        mat[0, i*rows + i] = 1
    return np.matrix(mat)

def neg_mat(lin_op):
    """Returns the coefficient matrix for NEG linear op.

    Parameters
    ----------
    lin_op : LinOp
        The neg linear op.

    Returns
    -------
    SciPy CSC matrix
        The matrix representing the neg operation.
    """
    mat = -sp.eye(lin_op.size[0]*lin_op.size[1])
    return mat.tocsc()

def div_mat(lin_op):
    """Returns the coefficient matrix for DIV linear op.

    Assumes dividing by scalar constants.

    Parameters
    ----------
    lin_op : LinOp
        The div linear op.

    Returns
    -------
    SciPy CSC matrix
        The matrix representing the div operation.
    """
    divisor = const_mat(lin_op.data)
    mat = sp.eye(lin_op.size[0]*lin_op.size[1])/divisor
    return mat.tocsc()

def mul_elemwise_mat(lin_op):
    """Returns the coefficient matrix for MUL_ELEM linear op.

    Parameters
    ----------
    lin_op : LinOp
        The mul_elem linear op.

    Returns
    -------
    SciPy CSC matrix
        The matrix representing the mul_elemwise operation.
    """
    constant = const_mat(lin_op.data)
    # Convert the constant to a giant diagonal matrix.
    vectorized = intf.from_2D_to_1D(flatten(constant))
    return sp.diags(vectorized, 0).tocsc()

def promote_mat(lin_op):
    """Returns the coefficient matrix for PROMOTE linear op.

    Parameters
    ----------
    lin_op : LinOp
        The promote linear op.

    Returns
    -------
    NumPy matrix
        The matrix for scalar promotion.
    """
    num_entries = lin_op.size[0]*lin_op.size[1]
    coeff = np.ones((num_entries, 1))
    return np.matrix(coeff)

def mul_mat(lin_op):
    """Returns the coefficient matrix for MUL linear op.

    Parameters
    ----------
    lin_op : LinOp
        The mul linear op.

    Returns
    -------
    SciPy CSC matrix or scalar.
        The matrix for the multiplication on the left operator.
    """
    constant = const_mat(lin_op.data)
    # Scalars don't need to be replicated.
    if not intf.is_scalar(constant):
        constant = sp.block_diag(lin_op.size[1]*[constant]).tocsc()
    return constant

def index_mat(lin_op):
    """Returns the coefficient matrix for indexing.

    Parameters
    ----------
    lin_op : LinOp
        The index linear op.

    Returns
    -------
    SciPy CSC matrix
        The matrix for the index/slice operation.
    """
    key = lin_op.data
    rows, cols = lin_op.args[0].size
    row_selection = range(rows)[key[0]]
    col_selection = range(cols)[key[1]]
    # Construct a coo matrix.
    val_arr = []
    row_arr = []
    col_arr = []
    counter = 0
    for col in col_selection:
        for row in row_selection:
            val_arr.append(1.0)
            row_arr.append(counter)
            col_arr.append(col*rows + row)
            counter += 1
    block_rows = lin_op.size[0]*lin_op.size[1]
    block_cols = rows*cols
    return sp.coo_matrix((val_arr, (row_arr, col_arr)),
                         (block_rows, block_cols)).tocsc()

def transpose_mat(lin_op):
    """Returns the coefficient matrix for TRANSPOSE linear op.

    Parameters
    ----------
    lin_op : LinOp
        The transpose linear op.

    Returns
    -------
    SciPy CSC matrix
        The matrix for the transpose operation.
    """
    rows, cols = lin_op.size
    # Create a sparse matrix representing the transpose.
    val_arr = []
    row_arr = []
    col_arr = []
    for col in xrange(cols):
        for row in xrange(rows):
            # Index in transposed coeff.
            row_arr.append(col*rows + row)
            # Index in original coeff.
            col_arr.append(row*cols + col)
            val_arr.append(1.0)

    return sp.coo_matrix((val_arr, (row_arr, col_arr)),
                         (rows*cols, rows*cols)).tocsc()

def diag_vec_mat(lin_op):
    """Returns the coefficient matrix for DIAG_VEC linear op.

    Parameters
    ----------
    lin_op : LinOp
        The diag vec linear op.

    Returns
    -------
    SciPy CSC matrix
        The matrix representing placing a vector on a diagonal.
    """
    rows, _ = lin_op.size

    val_arr = []
    row_arr = []
    col_arr = []
    for i in xrange(rows):
        # Index in the diagonal matrix.
        row_arr.append(i*rows + i)
        # Index in the original vector.
        col_arr.append(i)
        val_arr.append(1.0)

    return sp.coo_matrix((val_arr, (row_arr, col_arr)),
                         (rows**2, rows)).tocsc()

def diag_mat_mat(lin_op):
    """Returns the coefficients matrix for DIAG_MAT linear op.

    Parameters
    ----------
    lin_op : LinOp
        The diag mat linear op.

    Returns
    -------
    SciPy CSC matrix
        The matrix to extract the diagonal from a matrix.
    """
    rows, _ = lin_op.size

    val_arr = []
    row_arr = []
    col_arr = []
    for i in xrange(rows):
        # Index in the original matrix.
        col_arr.append(i*rows + i)
        # Index in the extracted vector.
        row_arr.append(i)
        val_arr.append(1.0)

    return sp.coo_matrix((val_arr, (row_arr, col_arr)),
                         (rows, rows**2)).tocsc()

def conv_mat(lin_op):
    """Returns the coefficient matrix for CONV linear op.

    Parameters
    ----------
    lin_op : LinOp
        The conv linear op.

    Returns
    -------
    NumPy matrix
        The matrix representing the convolution operation.
    """
    constant = const_mat(lin_op.data)
    # Cast to 1D.
    constant = intf.from_2D_to_1D(constant)

    # Create a Toeplitz matrix with constant as columns.
    rows = lin_op.size[0]
    nonzeros = lin_op.data.size[0]
    toeplitz_col = np.zeros(rows)
    toeplitz_col[0:nonzeros] = constant

    cols = lin_op.args[0].size[0]
    toeplitz_row = np.zeros(cols)
    toeplitz_row[0] = constant[0]
    coeff = sp_la.toeplitz(toeplitz_col, toeplitz_row)

    return np.matrix(coeff)


# A list of all the linear operator types for constants.
CONSTANT_TYPES = [lo.PARAM, lo.SCALAR_CONST, lo.DENSE_CONST, lo.SPARSE_CONST]

# A map of LinOp type to the coefficient matrix function.
TYPE_TO_FUNC = {
    lo.PROMOTE: promote_mat,
    lo.NEG: neg_mat,
    lo.MUL: mul_mat,
    lo.MUL_ELEM: mul_elemwise_mat,
    lo.DIV: div_mat,
    lo.SUM_ENTRIES: sum_entries_mat,
    lo.TRACE: trace_mat,
    lo.INDEX: index_mat,
    lo.TRANSPOSE: transpose_mat,
    lo.RESHAPE: lambda x: 1,
    lo.DIAG_VEC: diag_vec_mat,
    lo.DIAG_MAT: diag_mat_mat,
    lo.CONV: conv_mat,
}
