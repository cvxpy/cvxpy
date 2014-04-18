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
import scipy.sparse as sp
import numpy as np

# Utility functions for converting LinOps into matrices.

CONSTANT_ID = "constant_id"

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
        return [(lin_op.data,
                 lin_op.size,
            sp.eye(lin_op.size[0]*lin_op.size[1]))]
    # Constants convert directly to their value.
    elif lin_op.type is lo.PARAM:
        return [(CONSTANT_ID, lin_op.size, lin_op.data.value)]
    elif lin_op.type in [lo.DENSE_MUL, lo.SPARSE_MUL]:
        return [(CONSTANT_ID, lin_op.size, lin_op.data)]

    # Otherwise, recurse on args.
    elif lin_op.type is lo.SUM:
        return sum_coeffs(lin_op)
    elif lin_op.type is lo.NEG:
        return neg_coeffs(lin_op)
    elif lin_op.type is lo.MUL:
        return mul_coeffs(lin_op)

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
    coeffs = get_coefficients(lin_op)
    new_coeffs = []
    for id_, size, block in coeffs:
        new_coeffs.append((id_, size, -block))
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
    # Multiply all left-hand constants by right-hand terms.
    for (_, size, constant) in lh_coeffs:
        for (id_, size, coeff) in rh_coeffs:
