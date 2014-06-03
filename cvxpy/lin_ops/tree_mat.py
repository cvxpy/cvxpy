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

import cvxpy.interface as intf
import cvxpy.lin_ops.lin_op as lo
import copy
import numpy as np
from numpy.fft import fft, ifft

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
    # Return all zeros for NO_OP.
    elif lin_op.type is lo.NO_OP:
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
    # Do nothing for NO_OP.
    elif lin_op.type is lo.NO_OP:
        return {}
    else:
        result = op_tmul(lin_op, value)
        result_dicts = []
        for arg in lin_op.args:
            result_dicts.append(tmul(arg, result))
        # Sum repeated ids.
        return sum_dicts(result_dicts)

def sum_dicts(dicts):
    """Sums the dictionaries entrywise.

    Parameters
    ----------
    dicts : list
        A list of dictionaries with numeric entries.

    Returns
    -------
    dict
        A dict with the sum.
    """
    # Sum repeated entries.
    sum_dict = {}
    for val_dict in dicts:
        for id_, value in val_dict.items():
            if id_ in sum_dict:
                sum_dict[id_] = sum_dict[id_] + value
            else:
                sum_dict[id_] = value
    return sum_dict

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
    #print lin_op.type
    # Constants convert directly to their value.
    if lin_op.type in [lo.SCALAR_CONST, lo.DENSE_CONST, lo.SPARSE_CONST]:
        result = lin_op.data
    # No-op is not evaluated.
    elif lin_op.type is lo.NO_OP:
        return None
    # For non-leaves, recurse on args.
    elif lin_op.type is lo.SUM:
        result = sum(args)
    elif lin_op.type is lo.NEG:
        result = -args[0]
    elif lin_op.type is lo.MUL:
        coeff = mul(lin_op.data, {})
        result = coeff*args[0]
    elif lin_op.type is lo.DIV:
        divisor = mul(lin_op.data, {})
        result = args[0]/divisor
    elif lin_op.type is lo.SUM_ENTRIES:
        result = np.sum(args[0])
    elif lin_op.type is lo.INDEX:
        row_slc, col_slc = lin_op.data
        result = args[0][row_slc, col_slc]
    elif lin_op.type is lo.TRANSPOSE:
        result = args[0].T
    elif lin_op.type is lo.CONV:
        result = conv_mul(lin_op, args[0])
    elif lin_op.type is lo.PROMOTE:
        result = np.ones(lin_op.size)*args[0]
    elif lin_op.type is lo.DIAG_VEC:
        val = intf.from_2D_to_1D(args[0])
        result = np.diag(val)
    else:
        raise Exception("Unknown linear operator.")
    #print result
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
        coeff = mul(lin_op.data, {})
        # Scalar coefficient, no need to transpose.
        if np.isscalar(coeff):
            result = coeff*value
        else:
            result = coeff.T*value
    elif lin_op.type is lo.DIV:
        divisor = mul(lin_op.data, {})
        result = value/divisor
    elif lin_op.type is lo.SUM_ENTRIES:
        result = np.mat(np.ones(lin_op.args[0].size))*value
    elif lin_op.type is lo.INDEX:
        row_slc, col_slc = lin_op.data
        result = np.mat(np.zeros(lin_op.args[0].size))
        result[row_slc, col_slc] = value
    elif lin_op.type is lo.TRANSPOSE:
        result = value.T
    elif lin_op.type is lo.PROMOTE:
        result = np.ones(lin_op.size[0]).dot(value)
    elif lin_op.type is lo.DIAG_VEC:
        result = np.diag(value)
    elif lin_op.type is lo.CONV:
        return conv_mul(lin_op, value, True)
    else:
        raise Exception("Unknown linear operator.")
    return result

def conv_mul(lin_op, rh_val, transpose=False):
    """Multiply by a convolution operator.
    """
    # F^-1{F{left hand}*F(right hand)}
    length = lin_op.size[0]
    constant = mul(lin_op.data, {})
    # Convert to 2D
    constant, rh_val = map(intf.from_1D_to_2D, [constant, rh_val])
    lh_term = fft(constant, length, axis=0)
    rh_term = fft(rh_val, length, axis=0)
    # Transpose equivalent to taking conjugate
    # and keeping only first m terms.
    if transpose:
        lh_term = np.conjugate(lh_term)
    product = np.multiply(lh_term, rh_term)
    result = ifft(product, length, axis=0).real

    if transpose:
        rh_length = lin_op.args[0].size[0]
        return result[:rh_length]
    else:
        return result

def get_constant(lin_op):
    """Returns the constant term in the expression.

    Parameters
    ----------
    lin_op : LinOp
        The root linear operator.

    Returns
    -------
    NumPy NDArray
        The constant term as a flattened vector.
    """
    constant = mul(lin_op, {})
    const_size = constant.shape[0]*constant.shape[1]
    return np.reshape(constant, const_size, 'F')

def get_constr_constant(constraints):
    """Returns the constant term for the constraints matrix.

    Parameters
    ----------
    constraints : list
        The constraints that form the matrix.

    Returns
    -------
    NumPy NDArray
        The constant term as a flattened vector.
    """
    # TODO what if constraints is empty?
    constants = [get_constant(c.expr) for c in constraints]
    return np.hstack(constants)

def prune_constants(constraints):
    """Returns a new list of constraints with constant terms removed.

    Parameters
    ----------
    constraints : list
        The constraints that form the matrix.

    Returns
    -------
    list
        The pruned constraints.
    """
    pruned_constraints = []
    for constr in constraints:
        constr_type = type(constr)
        expr = copy.deepcopy(constr.expr)
        is_constant = prune_expr(expr)
        # Replace a constant root with a NO_OP.
        if is_constant:
            expr = lo.LinOp(lo.NO_OP, expr.size, [], None)
        pruned = constr_type(expr, constr.constr_id, constr.size)
        pruned_constraints.append(pruned)
    return pruned_constraints

def prune_expr(lin_op):
    """Prunes constant branches from the expression.

    Parameters
    ----------
    lin_op : LinOp
        The root linear operator.

    Returns
    -------
    bool
        Were all the expression's arguments pruned?
    """
    if lin_op.type is lo.VARIABLE:
        return False
    elif lin_op.type in [lo.SCALAR_CONST,
                         lo.DENSE_CONST,
                         lo.SPARSE_CONST,
                         lo.PARAM]:
        return True

    pruned_args = []
    is_constant = True
    for arg in lin_op.args:
        arg_constant = prune_expr(arg)
        if not arg_constant:
            is_constant = False
            pruned_args.append(arg)
    # Overwrite old args with only non-constant args.
    lin_op.args[:] = pruned_args[:]
    return is_constant
