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
from cvxpy.lin_ops.lin_constraints import LinEqConstr, LinLeqConstr

# Utility functions for dealing with LinOp.

class Counter(object):
    """A counter for ids.

    Attributes
    ----------
    count : int
        The current count.
    """
    def __init__(self):
        self.count = 0

ID_COUNTER = Counter()

def get_id():
    """Returns a new id and updates the id counter.

    Returns
    -------
    int
        A new id.
    """
    new_id = ID_COUNTER.count
    ID_COUNTER.count += 1
    return new_id

def create_var(size, var_id=None):
    """Creates a new internal variable.

    Parameters
    ----------
    size : tuple
        The (rows, cols) dimensions of the variable.
    var_id : int
        The id of the variable.

    Returns
    -------
    LinOP
        A LinOp representing the new variable.
    """
    if var_id is None:
        var_id = get_id()
    return lo.LinOp(lo.VARIABLE, size, [], var_id)

def create_param(value, size):
    """Wraps a parameter.

    Parameters
    ----------
    value : CVXPY Expression
        A function of parameters.
    size : tuple
        The (rows, cols) dimensions of the operator.

    Returns
    -------
    LinOP
        A LinOp wrapping the parameter.
    """
    return lo.LinOp(lo.PARAM, size, [], value)

def create_const(value, size, sparse=False):
    """Wraps a constant.

    Parameters
    ----------
    value : scalar, NumPy matrix, or SciPy sparse matrix.
        The numeric constant to wrap.
    size : tuple
        The (rows, cols) dimensions of the constant.
    sparse : bool
        Is the constant a SciPy sparse matrix?

    Returns
    -------
    LinOP
        A LinOp wrapping the constant.
    """
    # Check if scalar.
    if size == (1, 1):
        op_type = lo.SCALAR_CONST
    # Check if sparse.
    elif sparse:
        op_type = lo.SPARSE_CONST
    else:
        op_type = lo.DENSE_CONST
    return lo.LinOp(op_type, size, [], value)

def sum_expr(operators):
    """Add linear operators.

    Parameters
    ----------
    operators : list
        A list of linear operators.

    Returns
    -------
    LinOp
        A LinOp representing the sum of the operators.
    """
    return lo.LinOp(lo.SUM, operators[0].size, operators, None)

def neg_expr(operator):
    """Negate an operator.

    Parameters
    ----------
    expr : LinOp
        The operator to be negated.

    Returns
    -------
    LinOp
        The negated operator.
    """
    return lo.LinOp(lo.NEG, operator.size, [operator], None)

def sub_expr(lh_op, rh_op):
    """Difference of linear operators.

    Parameters
    ----------
    lh_op : LinOp
        The left-hand operator in the difference.
    rh_op : LinOp
        The right-hand operator in the difference.

    Returns
    -------
    LinOp
        A LinOp representing the difference of the operators.
    """
    return sum_expr([lh_op, neg_expr(rh_op)])

def mul_expr(lh_op, rh_op, size):
    """Multiply two linear operators, with the constant on the left.

    Parameters
    ----------
    lh_op : LinOp
        The left-hand operator in the product.
    rh_op : LinOp
        The right-hand operator in the product.
    size : tuple
        The size of the product.

    Returns
    -------
    LinOp
        A linear operator representing the product.
    """
    return lo.LinOp(lo.MUL, size, [rh_op], lh_op)

def rmul_expr(lh_op, rh_op, size):
    """Multiply two linear operators, with the constant on the right.

    Parameters
    ----------
    lh_op : LinOp
        The left-hand operator in the product.
    rh_op : LinOp
        The right-hand operator in the product.
    size : tuple
        The size of the product.

    Returns
    -------
    LinOp
        A linear operator representing the product.
    """
    return lo.LinOp(lo.RMUL, size, [lh_op], rh_op)

def mul_elemwise(lh_op, rh_op):
    """Multiply two linear operators elementwise.

    Parameters
    ----------
    lh_op : LinOp
        The left-hand operator in the product.
    rh_op : LinOp
        The right-hand operator in the product.

    Returns
    -------
    LinOp
        A linear operator representing the product.
    """
    return lo.LinOp(lo.MUL_ELEM, lh_op.size, [rh_op], lh_op)

def kron(lh_op, rh_op, size):
    """Kronecker product of two matrices.

    Parameters
    ----------
    lh_op : LinOp
        The left-hand operator in the product.
    rh_op : LinOp
        The right-hand operator in the product.

    Returns
    -------
    LinOp
        A linear operator representing the Kronecker product.
    """
    return lo.LinOp(lo.KRON, size, [rh_op], lh_op)

def div_expr(lh_op, rh_op):
    """Divide one linear operator by another.

    Assumes rh_op is a scalar constant.

    Parameters
    ----------
    lh_op : LinOp
        The left-hand operator in the quotient.
    rh_op : LinOp
        The right-hand operator in the quotient.
    size : tuple
        The size of the quotient.

    Returns
    -------
    LinOp
        A linear operator representing the quotient.
    """
    return lo.LinOp(lo.DIV, lh_op.size, [lh_op], rh_op)

def promote(operator, size):
    """Promotes a scalar operator to the given size.

    Parameters
    ----------
    operator : LinOp
        The operator to promote.
    size : tuple
        The dimensions to promote to.

    Returns
    -------
    LinOp
        A linear operator representing the promotion.
    """
    return lo.LinOp(lo.PROMOTE, size, [operator], None)

def sum_entries(operator):
    """Sum the entries of an operator.

    Parameters
    ----------
    expr : LinOp
        The operator to sum the entries of.

    Returns
    -------
    LinOp
        An operator representing the sum.
    """
    return lo.LinOp(lo.SUM_ENTRIES, (1, 1), [operator], None)

def trace(operator):
    """Sum the diagonal entries of an operator.

    Parameters
    ----------
    expr : LinOp
        The operator to sum the diagonal entries of.

    Returns
    -------
    LinOp
        An operator representing the sum of the diagonal entries.
    """
    return lo.LinOp(lo.TRACE, (1, 1), [operator], None)

def index(operator, size, keys):
    """Indexes/slices an operator.

    Parameters
    ----------
    operator : LinOp
        The expression to index.
    keys : tuple
        (row slice, column slice)
    size : tuple
        The size of the expression after indexing.

    Returns
    -------
    LinOp
        An operator representing the indexing.
    """
    return lo.LinOp(lo.INDEX, size, [operator], keys)

def conv(lh_op, rh_op, size):
    """1D discrete convolution of two vectors.

    Parameters
    ----------
    lh_op : LinOp
        The left-hand operator in the convolution.
    rh_op : LinOp
        The right-hand operator in the convolution.
    size : tuple
        The size of the convolution.

    Returns
    -------
    LinOp
        A linear operator representing the convolution.
    """
    return lo.LinOp(lo.CONV, size, [rh_op], lh_op)

def transpose(operator):
    """Transposes an operator.

    Parameters
    ----------
    operator : LinOp
        The operator to transpose.

    Returns
    -------
    LinOp
       A linear operator representing the transpose.
    """
    size = (operator.size[1], operator.size[0])
    return lo.LinOp(lo.TRANSPOSE, size, [operator], None)

def reshape(operator, size):
    """Reshapes an operator.

    Parameters
    ----------
    operator : LinOp
        The operator to reshape.
    size : tuple
        The (rows, cols) of the reshaped operator.

    Returns
    -------
    LinOp
       LinOp representing the reshaped expression.
    """
    return lo.LinOp(lo.RESHAPE, size, [operator], None)

def diag_vec(operator):
    """Converts a vector to a diagonal matrix.

    Parameters
    ----------
    operator : LinOp
        The operator to convert to a diagonal matrix.

    Returns
    -------
    LinOp
       LinOp representing the diagonal matrix.
    """
    size = (operator.size[0], operator.size[0])
    return lo.LinOp(lo.DIAG_VEC, size, [operator], None)

def diag_mat(operator):
    """Converts the diagonal of a matrix to a vector.

    Parameters
    ----------
    operator : LinOp
        The operator to convert to a vector.

    Returns
    -------
    LinOp
       LinOp representing the matrix diagonal.
    """
    size = (operator.size[0], 1)
    return lo.LinOp(lo.DIAG_MAT, size, [operator], None)

def upper_tri(operator):
    """Vectorized upper triangular portion of a square matrix.

    Parameters
    ----------
    operator : LinOp
        The matrix operator.

    Returns
    -------
    LinOp
       LinOp representing the vectorized upper triangle.
    """
    entries = operator.size[0]*operator.size[1]
    size = ((entries - operator.size[0])//2, 1)
    return lo.LinOp(lo.UPPER_TRI, size, [operator], None)

def hstack(operators, size):
    """Concatenates operators horizontally.

    Parameters
    ----------
    operator : list
        The operators to stack.
    size : tuple
        The (rows, cols) of the stacked operators.

    Returns
    -------
    LinOp
       LinOp representing the stacked expression.
    """
    return lo.LinOp(lo.HSTACK, size, operators, None)

def vstack(operators, size):
    """Concatenates operators vertically.

    Parameters
    ----------
    operator : list
        The operators to stack.
    size : tuple
        The (rows, cols) of the stacked operators.

    Returns
    -------
    LinOp
       LinOp representing the stacked expression.
    """
    return lo.LinOp(lo.VSTACK, size, operators, None)

def get_constr_expr(lh_op, rh_op):
    """Returns the operator in the constraint.
    """
    # rh_op defaults to 0.
    if rh_op is None:
        return lh_op
    else:
        return sum_expr([lh_op, neg_expr(rh_op)])

def create_eq(lh_op, rh_op=None, constr_id=None):
    """Creates an internal equality constraint.

    Parameters
    ----------
    lh_term : LinOp
        The left-hand operator in the equality constraint.
    rh_term : LinOp
        The right-hand operator in the equality constraint.
    constr_id : int
        The id of the CVXPY equality constraint creating the constraint.

    Returns
    -------
    LinEqConstr
    """
    if constr_id is None:
        constr_id = get_id()
    expr = get_constr_expr(lh_op, rh_op)
    return LinEqConstr(expr, constr_id, lh_op.size)

def create_leq(lh_op, rh_op=None, constr_id=None):
    """Creates an internal less than or equal constraint.

    Parameters
    ----------
    lh_term : LinOp
        The left-hand operator in the <= constraint.
    rh_term : LinOp
        The right-hand operator in the <= constraint.
    constr_id : int
        The id of the CVXPY equality constraint creating the constraint.

    Returns
    -------
    LinLeqConstr
    """
    if constr_id is None:
        constr_id = get_id()
    expr = get_constr_expr(lh_op, rh_op)
    return LinLeqConstr(expr, constr_id, lh_op.size)

def create_geq(lh_op, rh_op=None, constr_id=None):
    """Creates an internal greater than or equal constraint.

    Parameters
    ----------
    lh_term : LinOp
        The left-hand operator in the >= constraint.
    rh_term : LinOp
        The right-hand operator in the >= constraint.
    constr_id : int
        The id of the CVXPY equality constraint creating the constraint.

    Returns
    -------
    LinLeqConstr
    """
    if rh_op is not None:
        rh_op = neg_expr(rh_op)
    return create_leq(neg_expr(lh_op), rh_op, constr_id)

def get_expr_vars(operator):
    """Get a list of the variables in the operator and their sizes.

    Parameters
    ----------
    operator : LinOp
        The operator to extract the variables from.

    Returns
    -------
    list
        A list of (var id, var size) pairs.
    """
    if operator.type == lo.VARIABLE:
        return [(operator.data, operator.size)]
    else:
        vars_ = []
        for arg in operator.args:
            vars_ += get_expr_vars(arg)
        return vars_

def get_expr_params(operator):
    """Get a list of the parameters in the operator.

    Parameters
    ----------
    operator : LinOp
        The operator to extract the parameters from.

    Returns
    -------
    list
        A list of parameter objects.
    """
    if operator.type == lo.PARAM:
        return operator.data.parameters()
    else:
        params = []
        for arg in operator.args:
            params += get_expr_params(arg)
        # Some LinOps have a param as data.
        if isinstance(operator.data, lo.LinOp):
            params += get_expr_params(operator.data)
        return params

def copy_constr(constr, func):
    """Creates a copy of the constraint modified according to func.

    Parameters
    ----------
    constr : LinConstraint
        The constraint to modify.
    func : function
        Function to modify the constraint expression.

    Returns
    -------
    LinConstraint
        A copy of the constraint with the specified changes.
    """
    expr = func(constr.expr)
    return type(constr)(expr, constr.constr_id, constr.size)

def replace_new_vars(expr, id_to_new_var):
    """Replaces the given variables in the expression.

    Parameters
    ----------
    expr : LinOp
        The expression to replace variables in.
    id_to_new_var : dict
        A map of id to new variable.

    Returns
    -------
    LinOp
        An LinOp identical to expr, but with the given variables replaced.
    """
    if expr.type == lo.VARIABLE and expr.data in id_to_new_var:
        return id_to_new_var[expr.data]
    else:
        new_args = []
        for arg in expr.args:
            new_args.append(
                replace_new_vars(arg, id_to_new_var)
            )
        return lo.LinOp(expr.type, expr.size, new_args, expr.data)

def replace_params_with_consts(expr):
    """Replaces parameters with constant nodes.

    Parameters
    ----------
    expr : LinOp
        The expression to replace parameters in.

    Returns
    -------
    LinOp
        An LinOp identical to expr, but with the parameters replaced.
    """
    if expr.type == lo.PARAM:
        return create_const(expr.data.value, expr.size)
    else:
        new_args = []
        for arg in expr.args:
            new_args.append(replace_params_with_consts(arg))
        # Data could also be a parameter.
        if isinstance(expr.data, lo.LinOp) and expr.data.type == lo.PARAM:
            data_lin_op = expr.data
            data = create_const(data_lin_op.data.value, data_lin_op.size)
        else:
            data = expr.data
        return lo.LinOp(expr.type, expr.size, new_args, data)
