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
        coeffs = [(lin_op.data,
                   lin_op.size,
                   sp.eye(lin_op.size[0]*lin_op.size[1]))]
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
    else:
        raise Exception("Unknown linear operator.")
    return coeffs

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
    rh_coeffs = get_coefficients(lin_op.args[1])
    # Sum all left hand coeffs before dividing.
    divisor = 0
    for (_, _, const) in rh_coeffs:
        divisor += const

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
    lh_coeffs = get_coefficients(lin_op.args[0])
    rh_coeffs = get_coefficients(lin_op.args[1])
    new_coeffs = []
    cols = lin_op.size[1]
    # Multiply all left-hand constants by right-hand terms.
    for (_, lh_size, constant) in lh_coeffs:
        for (id_, rh_size, coeff) in rh_coeffs:
            rep_mat = sp.block_diag(cols*[constant])
            # For scalar right hand, constants
            # or single column, just multiply.
            if intf.is_scalar(constant) or \
               id_ is lo.CONSTANT_ID or cols == 1:
                product = constant*coeff
            # For promoted variables, flatten the matrix.
            elif lh_size != (1, 1) and rh_size == (1, 1):
                flattened_const = flatten(constant)
                product = flattened_const*coeff
            # Otherwise replicate the matrix.
            else:
                product = rep_mat*coeff
            new_coeffs.append((id_, rh_size, product))
        rh_coeffs = new_coeffs

    return new_coeffs

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
            # Number of rows in each column block.
            # and number of column blocks.
            rows, cols = lin_op.args[0].size
            # Split into column blocks.
            col_blocks = split_into_col_blocks(rows, cols, block)
            # Select column blocks.
            col_blocks = col_blocks[key[1]]
            # Select rows from each remaining column block.
            indexed_blocks = []
            for col_block in col_blocks:
                idx_block = intf.index(col_block, (key[0],
                                                   slice(None, None, None)))
                # Convert to sparse CSC matrix.
                sp_intf = intf.DEFAULT_SPARSE_INTERFACE
                idx_block = sp_intf.const_to_matrix(idx_block)
                indexed_blocks.append(idx_block)
            block = sp.vstack(indexed_blocks)
        new_coeffs.append((id_, size, block))

    return new_coeffs

def split_into_col_blocks(rows, cols, coeff):
    """Splits a coefficient matrix into one block for each column.

    Parameters
    ----------
    rows : int
        The number of rows in the expression.
    cols : int
        The number of columns in the expression.
    coeff : NumPy matrix or SciPy sparse matrix
        The coefficient matrix to split.
    """
    col_blocks = []
    for col in range(cols):
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
    sp_intf = intf.DEFAULT_SPARSE_INTERFACE
    new_block = sp_intf.zeros(rows*cols, rows*cols).tolil()
    for row in xrange(rows):
        for col in xrange(cols):
            # Row in transpose coeff.
            t_row = row*cols + col
            # Row in original coeff.
            t_col = col*rows + row
            new_block[t_row, t_col] = 1.0

    return [(id_, size, new_block.tocsc())]
