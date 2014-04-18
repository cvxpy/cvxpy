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

def get_matrix(lin_op):
    """Converts a linear op into a matrix block.

    Parameters
    ----------
    lin_op : LinOp
        The linear op to convert.

    Returns
    -------
    A NumPy or SciPy matrix.
    """
    # EYE_MUL converts to an identity matrix.
    if lin_op.type is lo.EYE_MUL:
        block = sp.eye(lin_op.var_size[0]*lin_op.var_size[1])
    # Other MUL types convert to a block diagonal or
    # a vector if promoted.
    elif lin_op.type in [lo.DENSE_MUL, lo.SPARSE_MUL, lo.PARAM_MUL]:
        # Evaluate the parameter.
        if lin_op.type is lo.PARAM_MUL:
            value = lin_op.data.value
        else:
            value = lin_op.data
        # Flatten if promoted.
        if lin_op.var_size == (1, 1):
            block = flatten(value)
        # Make block diagonal if normal.
        else:
            block = sp.block_diag(lin_op.var_size[1]*[value])
    # Constants convert to a flattened vector.
    elif lin_op.type in [lo.SCALAR_CONST,
                         lo.DENSE_CONST,
                         lo.SPARSE_CONST,
                         lo.PARAM]:
        if lin_op.type is lo.PARAM:
            value = lin_op.data.value
        else:
            value = lin_op.data
        block = flatten(value)
    elif lin_op.type is lo.SUM_ENTRIES:
        block = np.ones((1, lin_op.var_size[0]*lin_op.var_size[1]))
    # TODO index, transpose

    return block*lin_op.scalar_coeff
