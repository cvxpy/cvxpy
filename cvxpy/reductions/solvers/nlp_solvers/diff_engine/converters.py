"""
Copyright 2025, the CVXPY developers

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Main entry point for converting CVXPY expressions to C diff engine expressions.
"""
from scipy import sparse
from sparsediffpy import _sparsediffengine as _diffengine

import cvxpy as cp
from cvxpy.reductions.solvers.nlp_solvers.diff_engine.helpers import (
    make_dense_left_matmul,
    make_dense_right_matmul,
    make_sparse_left_matmul,
    make_sparse_right_matmul,
    normalize_shape,
    to_dense_float,
)
from cvxpy.reductions.solvers.nlp_solvers.diff_engine.registry import ATOM_CONVERTERS


def convert_matmul(expr, children, var_dict, n_vars, param_dict):
    """Convert matrix multiplication A @ f(x), f(x) @ A, or X @ Y."""
    left_arg, right_arg = expr.args

    if left_arg.is_constant():
        A = left_arg.value
        if isinstance(left_arg, cp.Parameter):
            param_node = param_dict[left_arg.id]
        else:
            param_node = None
        if sparse.issparse(A):
            return make_sparse_left_matmul(param_node, children[1], A)
        return make_dense_left_matmul(param_node, children[1], A)

    elif right_arg.is_constant():
        A = right_arg.value
        if isinstance(right_arg, cp.Parameter):
            param_node = param_dict[right_arg.id]
        else:
            param_node = None
        if sparse.issparse(A):
            return make_sparse_right_matmul(param_node, children[0], A)
        return make_dense_right_matmul(param_node, children[0], A)

    else:
        return _diffengine.make_matmul(children[0], children[1])

# TODO we should support sparse elementwise multiply at some point.
def convert_multiply(expr, children, var_dict, n_vars, param_dict):
    """Convert elementwise multiplication."""
    left_arg, right_arg = expr.args

    # TODO: would be nice to catch promote here so we correctly create a
    # a scalar multiply. What is even the convention with promoting a parameter?
    # This is a very deep question.

    if left_arg.is_constant():
        if left_arg.size == 1:
            return _diffengine.make_param_scalar_mult(children[0], children[1])
        else:
            return _diffengine.make_param_vector_mult(children[0], children[1])
    elif right_arg.is_constant():
        if right_arg.size == 1:
            return _diffengine.make_param_scalar_mult(children[1], children[0])
        else:
            return _diffengine.make_param_vector_mult(children[1], children[0])
    else:
        return _diffengine.make_multiply(children[0], children[1])


def convert_expr(expr, var_dict, n_vars, param_dict=None):
    """Convert a CVXPY expression to a C diff engine expression.

    Args:
        expr: CVXPY expression tree node
        var_dict: {var_id: C variable capsule} mapping
        n_vars: total number of scalar variables
        param_dict: optional {param_id: C parameter capsule} mapping
    """
    # Base case: variable lookup
    if isinstance(expr, cp.Variable):
        return var_dict[expr.id]

    # Base case: parameter lookup
    if isinstance(expr, cp.Parameter):
        return param_dict[expr.id]

    # Base case: constant (in the diff engine, a constant is a parameter with ID -1)
    if isinstance(expr, cp.Constant):
        c = to_dense_float(expr.value)
        d1, d2 = normalize_shape(expr.shape)
        return _diffengine.make_parameter(d1, d2, -1, n_vars, c.flatten(order='F'))

    # Recursive case: atoms
    atom_name = type(expr).__name__
    children = [convert_expr(arg, var_dict, n_vars, param_dict) for arg in expr.args]

    # matmul and multiply need param_dict for parameter support
    # TODO: maybe multiply doesn't need parameter dict special case
    if atom_name == "MulExpression":
        C_expr = convert_matmul(expr, children, var_dict, n_vars, param_dict)
    elif atom_name == "multiply":
        C_expr = convert_multiply(expr, children, var_dict, n_vars, param_dict)
    elif atom_name in ATOM_CONVERTERS:
        C_expr = ATOM_CONVERTERS[atom_name](expr, children)
    else:
        raise NotImplementedError(f"Atom '{atom_name}' not supported")

    # check that python dimension is consistent with C dimension
    d1_C, d2_C = _diffengine.get_expr_dimensions(C_expr)
    d1_Python, d2_Python = normalize_shape(expr.shape)

    if d1_C != d1_Python or d2_C != d2_Python:
        raise ValueError(
            f"Dimension mismatch for atom '{atom_name}': "
            f"C dimensions ({d1_C}, {d2_C}) vs Python dimensions ({d1_Python}, {d2_Python})"
        )

    return C_expr
