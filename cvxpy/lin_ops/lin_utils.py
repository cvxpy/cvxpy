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

import numpy as np

import cvxpy.lin_ops.lin_op as lo

# Utility functions for dealing with LinOp.


class Counter:
    """A counter for ids.

    Attributes
    ----------
    count : int
        The current count.
    """

    def __init__(self) -> None:
        self.count = 1


ID_COUNTER = Counter()


def get_id() -> int:
    """Returns a new id and updates the id counter.

    Returns
    -------
    int
        A new id.
    """
    new_id = ID_COUNTER.count
    ID_COUNTER.count += 1
    return new_id


def create_var(shape: tuple[int, ...], var_id=None):
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


def create_param(shape: tuple[int, ...], param_id=None):
    """Wraps a parameter.

    Parameters
    ----------
    shape : tuple
        The (rows, cols) dimensions of the operator.

    Returns
    -------
    LinOP
        A LinOp wrapping the parameter.
    """
    if param_id is None:
        param_id = get_id()
    return lo.LinOp(lo.PARAM, shape, [], param_id)


def create_const(value, shape: tuple[int, ...], sparse: bool = False):
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


def is_scalar(operator) -> bool:
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
    operator : LinOp
        The operator to be negated.

    Returns
    -------
    LinOp
        The negated operator.
    """
    return lo.LinOp(lo.NEG, operator.shape, [operator], None)


def mul_expr(lh_op, rh_op, shape: tuple[int, ...]):
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


def rmul_expr(lh_op, rh_op, shape: tuple[int, ...]):
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
    shape = max(lh_op.shape, rh_op.shape)
    return lo.LinOp(lo.MUL_ELEM, shape, [rh_op], lh_op)


def kron_r(lh_op, rh_op, shape: tuple[int, ...]):
    """Kronecker product of two matrices, where the right operand is a Variable

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
    return lo.LinOp(lo.KRON_R, shape, [rh_op], lh_op)


def kron_l(lh_op, rh_op, shape: tuple[int, ...]):
    """Kronecker product of two matrices, where the left operand is a Variable

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
    return lo.LinOp(lo.KRON_L, shape, [lh_op], rh_op)


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


def promote(operator, shape: tuple[int, ...]):
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


def broadcast_to(operators, shape: tuple[int, ...]):
    """Broadcasts operators to a common shape.
    """
    return lo.LinOp(lo.BROADCAST_TO, shape, operators, [])


def sum_entries(operator, shape: tuple[int, ...], axis=None, keepdims=None):
    """Sum the entries of an operator.

    Parameters
    ----------
    operator : LinOp
        The operator to sum the entries of.
    shape : tuple
        The shape of the sum.
    axis : int, optional
        The axis to sum along.
    keepdims : bool, optional
        Whether to keep the reduced dimensions.

    Returns
    -------
    LinOp
        An operator representing the sum.
    """
    return lo.LinOp(lo.SUM_ENTRIES, shape, [operator], data=[axis, keepdims])


def trace(operator, shape: tuple[int, ...] = ()):
    """Sum the diagonal entries of an operator.

    Parameters
    ----------
    operator : LinOp
        The operator to sum the diagonal entries of.
    shape : tuple
        The shape of the output (scalar for 2D, batch shape for ND).

    Returns
    -------
    LinOp
        An operator representing the sum of the diagonal entries.
    """
    return lo.LinOp(lo.TRACE, shape, [operator], None)


def index(operator, shape: tuple[int, ...], keys):
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


def conv(lh_op, rh_op, shape: tuple[int, ...]):
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


def transpose(operator, axes=None):
    """Transposes an operator.

    Parameters
    ----------
    operator : LinOp
        The operator to transpose.
    axes : None or tuple of ints
        If specified, it must be a tuple or list which contains a permutation
        of [0,1,..,N-1] where N is the number of dimensions of the operator.

    Returns
    -------
    LinOp
       A linear operator representing the transpose.
    """
    if len(operator.shape) < 2:
        return operator
    else:
        if axes is None:
            # Default reverse all axes
            shape = operator.shape[::-1]
        else:
            shape = tuple(operator.shape[i] for i in axes)
        return lo.LinOp(lo.TRANSPOSE, shape, [operator], data=[axes])


def reshape(operator, shape: tuple[int, ...]):
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


def diag_vec(operator, k: int = 0):
    """Converts a vector to a diagonal matrix.

    Parameters
    ----------
    operator : LinOp
        The operator to convert to a diagonal matrix.
    k : int
        The offset of the diagonal.

    Returns
    -------
    LinOp
       LinOp representing the diagonal matrix.
    """
    rows = operator.shape[0] + abs(k)
    shape = (rows, rows)
    return lo.LinOp(lo.DIAG_VEC, shape, [operator], k)


def diag_mat(operator, k: int = 0):
    """Converts the diagonal of a matrix to a vector.

    Parameters
    ----------
    operator : LinOp
        The operator to convert to a vector.
    k : int
        The offset of the diagonal.

    Returns
    -------
    LinOp
       LinOp representing the matrix diagonal.
    """
    shape = (operator.shape[0] - abs(k), 1)
    return lo.LinOp(lo.DIAG_MAT, shape, [operator], k)


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
    n = operator.shape[-1]
    batch_shape = operator.shape[:-2]
    tri = n * (n - 1) // 2
    shape = batch_shape + (tri, 1)
    return lo.LinOp(lo.UPPER_TRI, shape, [operator], None)


def hstack(operators, shape: tuple[int, ...]):
    """Concatenates operators horizontally.

    Parameters
    ----------
    operators : list
        The operators to stack.
    shape : tuple
        The (rows, cols) of the stacked operators.

    Returns
    -------
    LinOp
       LinOp representing the stacked expression.
    """
    return lo.LinOp(lo.HSTACK, shape, operators, None)


def vstack(operators, shape: tuple[int, ...]):
    """Concatenates operators vertically.

    Parameters
    ----------
    operators : list
        The operators to stack.
    shape : tuple
        The (rows, cols) of the stacked operators.

    Returns
    -------
    LinOp
       LinOp representing the stacked expression.
    """
    return lo.LinOp(lo.VSTACK, shape, operators, None)


def concatenate(operators, shape: tuple[int, ...], axis: int | None = 0):
    """Concatenate operators on axis.

    Parameters
    ----------
    operators : list
        The operators to concatenate.
    shape : tuple
        The (rows, cols) of the concatenated operators.
    axis : int, optional
        The axis along which the operators will be joined.

    Returns
    -------
    LinOp
       LinOp representing the stacked expression.
    """
    return lo.LinOp(lo.CONCATENATE, shape, operators, [axis])
