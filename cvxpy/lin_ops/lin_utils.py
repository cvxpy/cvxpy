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

THIS FILE IS DEPRECATED AND MAY BE REMOVED WITHOUT WARNING!
DO NOT CALL THESE FUNCTIONS IN YOUR CODE!
"""
import cvxpy.lin_ops.lin_op as lo
from cvxpy.lin_ops.lin_constraints import LinEqConstr, LinLeqConstr
import cvxpy.utilities as u
import numpy as np

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


def create_var(shape, var_id=None):
    """Creates a new internal variable.

    Parameters
    ----------
    shape : tuple
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
    return lo.LinOp(lo.VARIABLE, shape, [], var_id)


def create_param(value, shape):
    """Wraps a parameter.

    Parameters
    ----------
    value : CVXPY Expression
        A function of parameters.
    shape : tuple
        The (rows, cols) dimensions of the operator.

    Returns
    -------
    LinOP
        A LinOp wrapping the parameter.
    """
    return lo.LinOp(lo.PARAM, shape, [], value)


def create_const(value, shape, sparse=False):
    """Wraps a constant.

    Parameters
    ----------
    value : scalar, NumPy matrix, or SciPy sparse matrix.
        The numeric constant to wrap.
    shape : tuple
        The (rows, cols) dimensions of the constant.
    sparse : bool
        Is the constant a SciPy sparse matrix?

    Returns
    -------
    LinOP
        A LinOp wrapping the constant.
    """
    # Check if scalar.
    if shape == (1, 1):
        op_type = lo.SCALAR_CONST
        if not np.isscalar(value):
            value = value[0, 0]
    # Check if sparse.
    elif sparse:
        op_type = lo.SPARSE_CONST
    else:
        op_type = lo.DENSE_CONST
    return lo.LinOp(op_type, shape, [], value)


def is_scalar(operator):
    """Returns whether a LinOp is a scalar.

    Parameters
    ----------
    operator : LinOp
        The LinOp to test.

    Returns
    -------
        True if the LinOp is a scalar, False otherwise.
    """
    return len(operator.shape) == 0 or np.prod(operator.shape, dtype=int) == 1


def is_const(operator):
    """Returns whether a LinOp is constant.

    Parameters
    ----------
    operator : LinOp
        The LinOp to test.

    Returns
    -------
        True if the LinOp is a constant, False otherwise.
    """
    return operator.type in [lo.SCALAR_CONST, lo.SPARSE_CONST, lo.DENSE_CONST]


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
    return lo.LinOp(lo.SUM, operators[0].shape, operators, None)


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
    return lo.LinOp(lo.NEG, operator.shape, [operator], None)


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


def promote_lin_ops_for_mul(lh_op, rh_op):
    """Promote arguments for multiplication.

    Parameters
    ----------
    lh_op : LinOp
        The left-hand operator in the multiplication.
    rh_op : LinOp
        The right-hand operator in the multiplication.

    Returns
    -------
    LinOp
       Promoted left-hand operator.
    LinOp
       Promoted right-hand operator.
    tuple
       Shape of the product
    """
    lh_shape, rh_shape, shape = u.shape.mul_shapes_promote(
        lh_op.shape, rh_op.shape)
    lh_op = lo.LinOp(lh_op.type, lh_shape, lh_op.args,
                     lh_op.data)
    rh_op = lo.LinOp(rh_op.type, rh_shape, rh_op.args,
                     rh_op.data)
    return lh_op, rh_op, shape


def mul_expr(lh_op, rh_op, shape):
    """Multiply two linear operators, with the constant on the left.

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
    return lo.LinOp(lo.MUL, shape, [rh_op], lh_op)


def rmul_expr(lh_op, rh_op, shape):
    """Multiply two linear operators, with the constant on the right.

    Parameters
    ----------
    lh_op : LinOp
        The left-hand operator in the product.
    rh_op : LinOp
        The right-hand operator in the product.
    shape : tuple
        The shape of the product.

    Returns
    -------
    LinOp
        A linear operator representing the product.
    """
    return lo.LinOp(lo.RMUL, shape, [lh_op], rh_op)


def multiply(lh_op, rh_op):
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
    return lo.LinOp(lo.MUL_ELEM, lh_op.shape, [rh_op], lh_op)


def kron(lh_op, rh_op, shape):
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
    return lo.LinOp(lo.KRON, shape, [rh_op], lh_op)


def div_expr(lh_op, rh_op):
    """Divide one linear operator by another.

    Assumes rh_op is a scalar constant.

    Parameters
    ----------
    lh_op : LinOp
        The left-hand operator in the quotient.
    rh_op : LinOp
        The right-hand operator in the quotient.
    shape : tuple
        The shape of the quotient.

    Returns
    -------
    LinOp
        A linear operator representing the quotient.
    """
    return lo.LinOp(lo.DIV, lh_op.shape, [lh_op], rh_op)


def promote(operator, shape):
    """Promotes a scalar operator to the given shape.

    Parameters
    ----------
    operator : LinOp
        The operator to promote.
    shape : tuple
        The dimensions to promote to.

    Returns
    -------
    LinOp
        A linear operator representing the promotion.
    """
    return lo.LinOp(lo.PROMOTE, shape, [operator], None)


def sum_entries(operator, shape):
    """Sum the entries of an operator.

    Parameters
    ----------
    expr : LinOp
        The operator to sum the entries of.
    shape : tuple
        The shape of the sum.

    Returns
    -------
    LinOp
        An operator representing the sum.
    """
    return lo.LinOp(lo.SUM_ENTRIES, shape, [operator], None)


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


def index(operator, shape, keys):
    """Indexes/slices an operator.

    Parameters
    ----------
    operator : LinOp
        The expression to index.
    keys : tuple
        (row slice, column slice)
    shape : tuple
        The shape of the expression after indexing.

    Returns
    -------
    LinOp
        An operator representing the indexing.
    """
    return lo.LinOp(lo.INDEX, shape, [operator], keys)


def conv(lh_op, rh_op, shape):
    """1D discrete convolution of two vectors.

    Parameters
    ----------
    lh_op : LinOp
        The left-hand operator in the convolution.
    rh_op : LinOp
        The right-hand operator in the convolution.
    shape : tuple
        The shape of the convolution.

    Returns
    -------
    LinOp
        A linear operator representing the convolution.
    """
    return lo.LinOp(lo.CONV, shape, [rh_op], lh_op)


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
    if len(operator.shape) < 2:
        return operator
    elif len(operator.shape) > 2:
        return NotImplemented
    else:
        shape = (operator.shape[1], operator.shape[0])
        return lo.LinOp(lo.TRANSPOSE, shape, [operator], None)


def reshape(operator, shape):
    """Reshapes an operator.

    Parameters
    ----------
    operator : LinOp
        The operator to reshape.
    shape : tuple
        The (rows, cols) of the reshaped operator.

    Returns
    -------
    LinOp
       LinOp representing the reshaped expression.
    """
    return lo.LinOp(lo.RESHAPE, shape, [operator], None)


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
    shape = (operator.shape[0], operator.shape[0])
    return lo.LinOp(lo.DIAG_VEC, shape, [operator], None)


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
    shape = (operator.shape[0], 1)
    return lo.LinOp(lo.DIAG_MAT, shape, [operator], None)


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
    entries = operator.shape[0]*operator.shape[1]
    shape = ((entries - operator.shape[0])//2, 1)
    return lo.LinOp(lo.UPPER_TRI, shape, [operator], None)


def hstack(operators, shape):
    """Concatenates operators horizontally.

    Parameters
    ----------
    operator : list
        The operators to stack.
    shape : tuple
        The (rows, cols) of the stacked operators.

    Returns
    -------
    LinOp
       LinOp representing the stacked expression.
    """
    return lo.LinOp(lo.HSTACK, shape, operators, None)


def vstack(operators, shape):
    """Concatenates operators vertically.

    Parameters
    ----------
    operator : list
        The operators to stack.
    shape : tuple
        The (rows, cols) of the stacked operators.

    Returns
    -------
    LinOp
       LinOp representing the stacked expression.
    """
    return lo.LinOp(lo.VSTACK, shape, operators, None)


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
    return LinEqConstr(expr, constr_id, lh_op.shape)


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
    return LinLeqConstr(expr, constr_id, lh_op.shape)


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
    """Get a list of the variables in the operator and their shapes.

    Parameters
    ----------
    operator : LinOp
        The operator to extract the variables from.

    Returns
    -------
    list
        A list of (var id, var shape) pairs.
    """
    if operator.type == lo.VARIABLE:
        return [(operator.data, operator.shape)]
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
    return type(constr)(expr, constr.constr_id, constr.shape)


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
        return lo.LinOp(expr.type, expr.shape, new_args, expr.data)


def check_param_val(param):
    """Wrapper on accessing a parameter.

    Parameters
    ----------
    param : Parameter
        The parameter whose value is being accessed.

    Returns
    -------
    The numerical value of the parameter.

    Raises
    ------
    ValueError
        Raises error if parameter value is None.
    """
    val = param.value
    if val is None:
        raise ValueError("Problem has missing parameter value.")
    else:
        return val


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
        return create_const(check_param_val(expr.data), expr.shape)
    else:
        new_args = []
        for arg in expr.args:
            new_args.append(replace_params_with_consts(arg))
        # Data could also be a parameter.
        if isinstance(expr.data, lo.LinOp) and expr.data.type == lo.PARAM:
            data_lin_op = expr.data
            assert isinstance(data_lin_op.shape, tuple)
            val = check_param_val(data_lin_op.data)
            data = create_const(val, data_lin_op.shape)
        else:
            data = expr.data
        return lo.LinOp(expr.type, expr.shape, new_args, data)
