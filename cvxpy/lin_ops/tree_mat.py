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
import numpy as np

# Utility functions for treating an expression tree as a matrix
# and multiplying by it and it's transpose.

def mul(lin_op, val_dict):
    """Multiply the expression tree by a vector.

    Parameters
    ----------
    lin_op : LinOp
        The root of an expression tree.
    val_dict : dict
        A map of variable id to value.

    Returns
    -------
    NumPy matrix
        The result of the multiplication.
    """
    # Look up the value for a variable.
    if lin_op.type is lo.VARIABLE:
        if lin_op.data in val_dict:
            return val_dict[lin_op.data]
        # Defaults to zero if no value given.
        else:
            return np.mat(np.zeros(lin_op.size))
    else:
        eval_args = []
        for arg in lin_op.args:
            eval_args.append(mul(arg, val_dict))
        return op_mul(lin_op, eval_args)

def tmul(lin_op, value):
    """Multiply the transpose of the expression tree by a vector.

    Parameters
    ----------
    lin_op : LinOp
        The root of an expression tree.
    value : NumPy matrix
        The vector to multiply by.

    Returns
    -------
    dict
        A map of variable id to value.
    """
    # Store the value as the variable.
    if lin_op.type is lo.VARIABLE:
        return {lin_op.data: value}
    # Do nothing for constant leaves.
    elif lin_op.type in [lo.SCALAR_CONST,
                        lo.DENSE_CONST,
                        lo.SPARSE_CONST]:
        return {}
    else:
        result = op_tmul(lin_op, value)
        val_dict = {}
        for arg in lin_op.args:
            result_dict = tmul(arg, result)
            # Sum repeated entries.
            for id_, value in result_dict:
                if id_ in val_dict:
                    val_dict[id_] = val_dict[id_] + value
                else:
                    val_dict[id_] = value
        return val_dict

def op_mul(lin_op, args):
    """Applies the linear operator to the arguments.

    Parameters
    ----------
    lin_op : LinOp
        A linear operator.
    args : list
        The arguments to the operator.

    Returns
    -------
    NumPy matrix or SciPy sparse matrix.
        The result of applying the linear operator.
    """
    # Constants convert directly to their value.
    if lin_op.type in [lo.SCALAR_CONST, lo.DENSE_CONST, lo.SPARSE_CONST]:
        result = lin_op.data
    # For non-leaves, recurse on args.
    elif lin_op.type is lo.SUM:
        result = sum(args)
    elif lin_op.type is lo.NEG:
        result = -args[0]
    elif lin_op.type is lo.MUL:
        result = args[0]*args[1]
    elif lin_op.type is lo.DIV:
        result = args[0]/args[1]
    elif lin_op.type is lo.SUM_ENTRIES:
        result = np.sum(args[0])
    elif lin_op.type is lo.INDEX:
        row_slc, col_slc = lin_op.data
        result = args[0][row_slc, col_slc]
    elif lin_op.type is lo.TRANSPOSE:
        result = args[0].T
    else:
        raise Exception("Unknown linear operator.")
    return result

def op_tmul(lin_op, value):
    """Applies the transpose of the linear operator to the arguments.

    Parameters
    ----------
    lin_op : LinOp
        A linear operator.
    value : NumPy matrix
        A numeric value to apply the operator's transpose to.

    Returns
    -------
    NumPy matrix or SciPy sparse matrix.
        The result of applying the linear operator.
    """
    if lin_op.type is lo.SUM:
        result = value
    elif lin_op.type is lo.NEG:
        result = -value
    elif lin_op.type is lo.MUL:
        result = lin_op.args[0].T*value
    elif lin_op.type is lo.DIV:
        result = value/lin_op.args[1]
    elif lin_op.type is lo.SUM_ENTRIES:
        result = np.mat(np.ones(lin_op.args[0].size))*value
    elif lin_op.type is lo.INDEX:
        row_slc, col_slc = lin_op.data
        result = np.mat(np.zeros(lin_op.args[0].size))
        result[row_slc, col_slc] = value
    elif lin_op.type is lo.TRANSPOSE:
        result = value.T
    else:
        raise Exception("Unknown linear operator.")
    return result
