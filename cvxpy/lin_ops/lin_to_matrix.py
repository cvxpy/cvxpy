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
        A list of (id, size, coefficient) tuples.
    """
    # VARIABLE converts to a giant identity matrix.
    if lin_op.type is lo.VARIABLE:
        coeffs = var_coeffs(lin_op)
    # Constants convert directly to their value.
    elif lin_op.type is lo.PARAM:
        coeffs = [(lo.CONSTANT_ID, lin_op.size, lin_op.data.value)]
    elif lin_op.type in [lo.SCALAR_CONST, lo.DENSE_CONST, lo.SPARSE_CONST]:
        coeffs = [(lo.CONSTANT_ID, lin_op.size, lin_op.data)]
    # For non-leaves, recurse on args.
    elif lin_op.type is lo.SUM:
        coeffs = sum_coeffs(lin_op)
    elif lin_op.type is lo.NEG:
        coeffs = neg_coeffs(lin_op)
    elif lin_op.type is lo.MUL:
        coeffs = mul_coeffs(lin_op)
    elif lin_op.type is lo.DIV:
        coeffs = div_coeffs(lin_op)
    elif lin_op.type is lo.SUM_ENTRIES:
        coeffs = sum_entries_coeffs(lin_op)
    elif lin_op.type is lo.INDEX:
        coeffs = index_coeffs(lin_op)
    elif lin_op.type is lo.TRANSPOSE:
        coeffs = transpose_coeffs(lin_op)
    elif lin_op.type is lo.CONV:
        coeffs = conv_coeffs(lin_op)
    else:
        raise Exception("Unknown linear operator.")
    return coeffs

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
    size = lin_op.size
    coeff = sp.eye(lin_op.size[0]*lin_op.size[1]).tocsc()
    return [(id_, size, coeff)]

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

def sum_entries_coeffs(lin_op):
    """Returns the coefficients for SUM_ENTRIES linear op.

    Parameters
    ----------
    lin_op : LinOp
        The sum entries linear op.

    Returns
    -------
    list
       A list of (id, size, coefficient) tuples.
    """
    coeffs = get_coefficients(lin_op.args[0])
    new_coeffs = []
    for id_, size, block in coeffs:
        # Sum all elements if constant.
        if id_ is lo.CONSTANT_ID:
            size = (1, 1)
            block = np.sum(block)
        # Sum columns if variable.
        else:
            block = block.sum(axis=0)
        new_coeffs.append((id_, size, block))
    return new_coeffs

def neg_coeffs(lin_op):
    """Returns the coefficients for NEG linear op.

    Parameters
    ----------
    lin_op : LinOp
        The neg linear op.

    Returns
    -------
    list
        A list of (id, size, coefficient) tuples.
    """
    coeffs = get_coefficients(lin_op.args[0])
    new_coeffs = []
    for id_, size, block in coeffs:
        new_coeffs.append((id_, size, -block))
    return new_coeffs

def merge_constants(coeffs):
    """Sums all the constant coefficients.

    Parameters
    ----------
    coeffs : list
        A list of (id, size, coefficient) tuples.

    Returns
    -------
    The constant term.
    """
    constant = None
    for id_, size, block in coeffs:
        # Sum constants.
        if id_ is lo.CONSTANT_ID:
            if constant is None:
                constant = block
            else:
                constant += block
    return constant

def div_coeffs(lin_op):
    """Returns the coefficients for DIV linea op.

    Assumes dividing by scalar constants.

    Parameters
    ----------
    lin_op : LinOp
        The div linear op.

    Returns
    -------
    list
        A list of (id, size, coefficient) tuples.
    """
    rh_coeffs = get_coefficients(lin_op.data)
    divisor = merge_constants(rh_coeffs)

    lh_coeffs = get_coefficients(lin_op.args[0])
    new_coeffs = []
    # Divide all right-hand constants by left-hand constant.
    for (id_, lh_size, coeff) in lh_coeffs:
        new_coeffs.append((id_, lh_size, coeff/divisor))
    return new_coeffs

def mul_coeffs(lin_op):
    """Returns the coefficients for MUL linear op.

    Parameters
    ----------
    lin_op : LinOp
        The mul linear op.

    Returns
    -------
    list
        A list of (id, size, coefficient) tuples.
    """
    lh_coeffs = get_coefficients(lin_op.data)
    constant = merge_constants(lh_coeffs)
    rh_coeffs = get_coefficients(lin_op.args[0])

    return mul_by_const(constant, rh_coeffs, lin_op.size)

def mul_by_const(constant, rh_coeffs, size):
    """Multiplies a constant by a list of coefficients.

    Parameters
    ----------
    constant : numeric type
        The constant to multiply by.
    rh_coeffs : list
        The coefficients of the right hand side.
    size : tuple
        (product rows, product columns)

    Returns
    -------
    list
        A list of (id, size, coefficient) tuples.
    """
    new_coeffs = []
    # Multiply all left-hand constants by right-hand terms.
    for (id_, rh_size, coeff) in rh_coeffs:
        rep_mat = sp.block_diag(size[1]*[constant]).tocsc()
        # For scalar right hand constants
        # or single column, just multiply.
        if intf.is_scalar(constant) or \
           id_ is lo.CONSTANT_ID or size[1] == 1:
            product = constant*coeff
        # For promoted variables with matrix coefficients,
        # flatten the matrix.
        elif size != (1, 1) and intf.is_scalar(coeff):
            flattened_const = flatten(constant)
            product = flattened_const*coeff
        # Otherwise replicate the matrix.
        else:
            product = rep_mat*coeff
        new_coeffs.append((id_, rh_size, product))
    rh_coeffs = new_coeffs

    return new_coeffs

def index_var(lin_op):
    """Returns the coefficients from indexing a raw variable.

    Parameters
    ----------
    lin_op : LinOp
        The index linear op.

    Returns
    -------
    list
        A list of (id, size, coefficient) tuples.
    """
    key = lin_op.data
    var_rows, var_cols = lin_op.args[0].size
    row_selection = range(var_rows)[key[0]]
    col_selection = range(var_cols)[key[1]]
    # Construct a coo matrix.
    val_arr = []
    row_arr = []
    col_arr = []
    counter = 0
    for col in col_selection:
        for row in row_selection:
            val_arr.append(1.0)
            row_arr.append(counter)
            col_arr.append(col*var_rows + row)
            counter += 1
    block_rows = lin_op.size[0]*lin_op.size[1]
    block_cols = var_rows*var_cols
    block = sp.coo_matrix((val_arr, (row_arr, col_arr)),
                          (block_rows, block_cols)).tocsc()
    return [(lin_op.args[0].data, lin_op.args[0].size, block)]

def index_coeffs(lin_op):
    """Returns the coefficients for INDEX linear op.

    Parameters
    ----------
    lin_op : LinOp
        The index linear op.

    Returns
    -------
    list
        A list of (id, size, coefficient) tuples.
    """
    # Special case if variable.
    if lin_op.args[0].type is lo.VARIABLE:
        return index_var(lin_op)
    key = lin_op.data
    coeffs = get_coefficients(lin_op.args[0])
    new_coeffs = []
    for id_, size, block in coeffs:
        # Index/slice constants normally.
        if id_ is lo.CONSTANT_ID:
            size = lin_op.size
            block = intf.index(block, key)
        # Split into column blocks, slice column blocks list,
        # then index each column block and merge.
        else:
            block = get_index_block(block, lin_op.args[0].size, key)
        new_coeffs.append((id_, size, block))

    return new_coeffs

def get_index_block(block, idx_size, key):
    """Transforms a coefficient into an indexed coefficient.

    Parameters
    ----------
    block : matrix
        The coefficient matrix.
    idx_size : tuple
        The dimensions of the indexed expression.
    key : tuple
        (row slice, column slice)

    Returns
    -------
    The indexed/sliced coefficient matrix.
    """
    rows, cols = idx_size
    # Number of rows in each column block.
    # and number of column blocks.
    col_selection = range(cols)[key[1]]
    # Split into column blocks.
    col_blocks = get_col_blocks(rows, block, col_selection)
    # Select rows from each remaining column block.
    row_key = (key[0], slice(None, None, None))
    # Short circuit for single column.
    if len(col_blocks) == 1:
        block = intf.index(col_blocks[0], row_key)
    else:
        indexed_blocks = []
        for col_block in col_blocks:
            idx_block = intf.index(col_block, row_key)
            # Convert to sparse CSC matrix.
            sp_intf = intf.DEFAULT_SPARSE_INTERFACE
            idx_block = sp_intf.const_to_matrix(idx_block)
            indexed_blocks.append(idx_block)
        block = sp.vstack(indexed_blocks)
    return block

def get_col_blocks(rows, coeff, col_selection):
    """Selects column blocks from a matrix.

    Parameters
    ----------
    rows : int
        The number of rows in the expression.
    coeff : NumPy matrix or SciPy sparse matrix
        The coefficient matrix to split.
    col_selection : list
        The indices of the columns to select.

    Returns
    -------
    list
        A list of column blocks from the coeff matrix.
    """
    col_blocks = []
    for col in col_selection:
        key = (slice(col*rows, (col+1)*rows, 1),
               slice(None, None, None))
        block = intf.index(coeff, key)
        col_blocks.append(block)
    return col_blocks

def transpose_coeffs(lin_op):
    """Returns the coefficients for TRANSPOSE linear op.

    Assumes lin_op's arg is a single variable.

    Parameters
    ----------
    lin_op : LinOp
        The transpose linear op.

    Returns
    -------
    list
        A list of (id, size, coefficient) tuples.
    """
    coeffs = get_coefficients(lin_op.args[0])
    assert len(coeffs) == 1
    id_, size, _ = coeffs[0]
    rows, cols = size
    # Create a sparse matrix representing the transpose.
    val_arr = []
    row_arr = []
    col_arr = []
    for row in xrange(rows):
        for col in xrange(cols):
            # Row in transpose coeff.
            row_arr.append(row*cols + col)
            # Row in original coeff.
            col_arr.append(col*rows + row)
            val_arr.append(1.0)

    new_size = (rows*cols, rows*cols)
    new_block = sp.coo_matrix((val_arr, (row_arr, col_arr)), new_size)
    return [(id_, size, new_block.tocsc())]

def conv_coeffs(lin_op):
    """Returns the coefficients for CONV linear op.

    Parameters
    ----------
    lin_op : LinOp
        The conv linear op.

    Returns
    -------
    list
        A list of (id, size, coefficient) tuples.
    """
    lh_coeffs = get_coefficients(lin_op.data)
    constant = merge_constants(lh_coeffs)
    # Cast to 1D.
    constant = intf.from_2D_to_1D(constant)
    rh_coeffs = get_coefficients(lin_op.args[0])

    # Create a Toeplitz matrix with constant as columns.
    rows = lin_op.size[0]
    nonzeros = lin_op.data.size[0]
    toeplitz_col = np.zeros(rows)
    toeplitz_col[0:nonzeros] = constant

    cols = lin_op.args[0].size[0]
    toeplitz_row = np.zeros(cols)
    toeplitz_row[0] = constant[0]
    coeff = sp_la.toeplitz(toeplitz_col, toeplitz_row)

    # Multiply the right hand terms by the toeplitz matrix.
    return mul_by_const(coeff, rh_coeffs, (rows, 1))
