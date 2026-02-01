
"""Converters from CVXPY expressions to C diff engine expressions.

This module provides the mapping between CVXPY atom types and their
corresponding C diff engine constructors.

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
"""

import numpy as np
from scipy import sparse

import cvxpy as cp
from cvxpy.reductions.inverse_data import InverseData

# Import the low-level C bindings
try:
    import _diffengine
except ImportError as e:
    raise ImportError(
        "NLP support requires diff-engine. Rebuild with: pip install -e ."
    ) from e


def _chain_add(children):
    """Chain multiple children with binary adds: a + b + c -> add(add(a, b), c)."""
    result = children[0]
    for child in children[1:]:
        result = _diffengine.make_add(result, child)
    return result


def _convert_matmul(expr, children):
    """Convert matrix multiplication A @ f(x), f(x) @ A, or X @ Y."""
    left_arg, right_arg = expr.args

    if left_arg.is_constant():
        A = left_arg.value
    
        if not isinstance(A, sparse.csr_matrix):
          A = sparse.csr_matrix(A)
          
        return _diffengine.make_left_matmul(
            children[1],  
            A.data.astype(np.float64),
            A.indices.astype(np.int32),
            A.indptr.astype(np.int32),
            A.shape[0],
            A.shape[1],
        )
    elif right_arg.is_constant():
        A = right_arg.value
       
        if not isinstance(A, sparse.csr_matrix):
            A = sparse.csr_matrix(A)

        return _diffengine.make_right_matmul(
            children[0],  
            A.data.astype(np.float64),
            A.indices.astype(np.int32),
            A.indptr.astype(np.int32),
            A.shape[0],
            A.shape[1],
        )
    else:
        return _diffengine.make_matmul(children[0], children[1])  

def _convert_hstack(expr, children):
    """Convert horizontal stack (hstack) of expressions."""
    return _diffengine.make_hstack(children)

def _convert_multiply(expr, children):
    """Convert multiplication based on argument types."""
    left_arg, right_arg = expr.args

    if left_arg.is_constant():
        a = left_arg.value
        # we only support dense constants for elementwise multiplication
        if sparse.issparse(a):
            a = a.todense()
        a = np.asarray(a, dtype=np.float64)

        # Scalar constant
        if a.size == 1:
            scalar = float(a.flat[0])
            if scalar == 1.0:
                return children[1]  
            else:
                return _diffengine.make_const_scalar_mult(children[1], scalar)

        # non-scalar constant
        return _diffengine.make_const_vector_mult(children[1], a.flatten(order='F'))

    elif right_arg.is_constant():
        a = right_arg.value
        # we only support dense constants for elementwise multiplication
        if sparse.issparse(a):
            a = a.todense()
        a = np.asarray(a, dtype=np.float64)

        # Scalar constant
        if a.size == 1:
            scalar = float(a.flat[0])
            if scalar == 1.0:
                return children[0]  
            else:
                return _diffengine.make_const_scalar_mult(children[0], scalar)

        # non-scalar constant
        return _diffengine.make_const_vector_mult(children[0], a.flatten(order='F'))

    # Neither is constant, use general multiply
    return _diffengine.make_multiply(children[0], children[1])


def _extract_flat_indices_from_index(expr):
    """Extract flattened indices from CVXPY index expression."""
    parent_shape = expr.args[0].shape
    indices_per_dim = [np.arange(s.start, s.stop, s.step) for s in expr.key]

    if len(indices_per_dim) == 1:
        return indices_per_dim[0].astype(np.int32)
    elif len(indices_per_dim) == 2:
        # Fortran order: idx = row + col * n_rows
        return (
            np.add.outer(indices_per_dim[0], indices_per_dim[1] * parent_shape[0])
            .flatten(order="F")
            .astype(np.int32)
        )
    else:
        raise NotImplementedError("index with >2 dimensions not supported")


def _extract_flat_indices_from_special_index(expr):
    """Extract flattened indices from CVXPY special_index expression."""
    return np.reshape(expr._select_mat, expr._select_mat.size, order="F").astype(np.int32)


def _convert_rel_entr(expr, children):
    """Convert rel_entr(x, y) = x * log(x/y) elementwise.
    
    Uses specialized functions based on argument shapes:
    - Both scalar or both same size: make_rel_entr (elementwise)
    - First arg vector, second scalar: make_rel_entr_vector_scalar
    - First arg scalar, second vector: make_rel_entr_scalar_vector
    """
    x_arg, y_arg = expr.args
    x_size = x_arg.size
    y_size = y_arg.size
    
    # Determine which variant to use based on sizes
    if x_size == y_size:
        return _diffengine.make_rel_entr(children[0], children[1])
    elif x_size > 1 and y_size == 1:
        return _diffengine.make_rel_entr_vector_scalar(children[0], children[1])
    elif x_size == 1 and y_size > 1:
        return _diffengine.make_rel_entr_scalar_vector(children[0], children[1])
    else:
        raise ValueError(
            f"rel_entr requires arguments to be either both scalars, both same size, "
            f"or one scalar and one vector. Got sizes: x={x_size}, y={y_size}"
        )


def _convert_quad_form(expr, children):
    """Convert quadratic form x.T @ P @ x."""

    P = expr.args[1]

    if not isinstance(P, cp.Constant):
        raise NotImplementedError("quad_form requires P to be a constant matrix")

    P = P.value

    if not isinstance(P, sparse.csr_matrix):
          P = sparse.csr_matrix(P)

    return _diffengine.make_quad_form(
        children[0],  
        P.data.astype(np.float64),
        P.indices.astype(np.int32),
        P.indptr.astype(np.int32),
        P.shape[0],
        P.shape[1],
    )


def _convert_reshape(expr, children):
    """Convert reshape - only Fortran order is supported.

    Note: Only order='F' (Fortran/column-major) is supported.
    """
    if expr.order != "F":
        raise NotImplementedError(
            f"reshape with order='{expr.order}' not supported. "
            "Only order='F' (Fortran) is currently supported."
        )

    x_shape = tuple(expr.shape)
    x_shape = (1,) * (2 - len(x_shape)) + x_shape
    d1, d2 = x_shape
    return _diffengine.make_reshape(children[0], d1, d2)

def _convert_broadcast(expr, children):
    d1, d2 = expr.broadcast_shape
    d1_C, d2_C = _diffengine.get_expr_dimensions(children[0])
    if d1_C == d1 and d2_C == d2:
        return children[0]

    return _diffengine.make_broadcast(children[0], d1, d2)

def _convert_sum(expr, children):
    axis = expr.axis
    if axis is None:
        axis = -1
    return _diffengine.make_sum(children[0], axis)

def _convert_promote(expr, children):
    x_shape = tuple(expr.shape)
    x_shape = (1,) * (2 - len(x_shape)) + x_shape
    d1, d2 = x_shape
    return _diffengine.make_promote(children[0], d1, d2)

def _convert_NegExpression(_expr, children):
    return _diffengine.make_neg(children[0])

def _convert_quad_over_lin(_expr, children):
    return _diffengine.make_quad_over_lin(children[0], children[1])

def _convert_index(expr, children):
    idxs = _extract_flat_indices_from_index(expr)
    x_shape = tuple(expr.shape)
    x_shape = (1,) * (2 - len(x_shape)) + x_shape
    d1, d2 = x_shape

    return _diffengine.make_index(children[0], d1, d2, idxs)

def _convert_special_index(expr, children):
    idxs = _extract_flat_indices_from_special_index(expr)
    x_shape = tuple(expr.shape)
    x_shape = (1,) * (2 - len(x_shape)) + x_shape
    d1, d2 = x_shape

    return _diffengine.make_index(children[0], d1, d2, idxs)

def _convert_prod(expr, children):
    axis = expr.axis
    if axis is None:
        return _diffengine.make_prod(children[0])
    elif axis == 0:
        return _diffengine.make_prod_axis_zero(children[0])
    elif axis == 1:
        return _diffengine.make_prod_axis_one(children[0])
    
def _convert_transpose(expr, children):
    # If the child is a vector (shape (n,) or (n,1) or (1,n)), use reshape to transpose
    child_shape = tuple(expr.args[0].shape)
    child_shape = (1,) * (2 - len(child_shape)) + child_shape
    
    if 1 in child_shape:
        return _diffengine.make_reshape(children[0], child_shape[1], child_shape[0])
    else:
        return _diffengine.make_transpose(children[0])
   
def _convert_trace(_expr, children):
    return _diffengine.make_trace(children[0])

def _convert_diag_vec(expr, children):
    # C implementation only supports k=0 (main diagonal)
    if expr.k != 0:
        raise NotImplementedError("diag_vec with k != 0 not supported in diff engine")
    return _diffengine.make_diag_vec(children[0])

# Mapping from CVXPY atom names to C diff engine functions
# Converters receive (expr, children) where expr is the CVXPY expression
ATOM_CONVERTERS = {
    # Elementwise unary
    "log": lambda _expr, children: _diffengine.make_log(children[0]),
    "exp": lambda _expr, children: _diffengine.make_exp(children[0]),
    # Affine unary
    "NegExpression": _convert_NegExpression,
    "Promote": _convert_promote,
    # N-ary (handles 2+ args)
    "AddExpression": lambda _expr, children: _chain_add(children),
    # Reductions
    "Sum": _convert_sum,
    # Bivariate
    "multiply": _convert_multiply,
    "QuadForm": _convert_quad_form,
    "quad_over_lin": _convert_quad_over_lin,
    "rel_entr": _convert_rel_entr,
    # Matrix multiplication
    "MulExpression": _convert_matmul,
    # Elementwise univariate with parameter
    "Power": lambda expr, children: _diffengine.make_power(children[0], float(expr.p.value)),
    "PowerApprox": lambda expr, children: _diffengine.make_power(children[0], float(expr.p.value)),
    # Trigonometric
    "sin": lambda _expr, children: _diffengine.make_sin(children[0]),
    "cos": lambda _expr, children: _diffengine.make_cos(children[0]),
    "tan": lambda _expr, children: _diffengine.make_tan(children[0]),
    # Hyperbolic
    "sinh": lambda _expr, children: _diffengine.make_sinh(children[0]),
    "tanh": lambda _expr, children: _diffengine.make_tanh(children[0]),
    "asinh": lambda _expr, children: _diffengine.make_asinh(children[0]),
    "atanh": lambda _expr, children: _diffengine.make_atanh(children[0]),
    # Other elementwise
    "entr": lambda _expr, children: _diffengine.make_entr(children[0]),
    "logistic": lambda _expr, children: _diffengine.make_logistic(children[0]),
    "xexp": lambda _expr, children: _diffengine.make_xexp(children[0]),
    # Indexing/slicing
    "index": _convert_index,
    "special_index": _convert_special_index,
    "reshape": _convert_reshape,
    "broadcast_to": _convert_broadcast,
    # Reductions returning scalar
    "Prod": _convert_prod,
    "transpose": _convert_transpose,
    # Horizontal stack
    "Hstack": _convert_hstack,
    "Trace": _convert_trace,
    # Diagonal
    "diag_vec": _convert_diag_vec,
}


def build_variable_dict(variables: list) -> tuple[dict, int]:
    """
    Build dictionary mapping CVXPY variable ids to C variables.

    Args:
        variables: list of CVXPY Variable objects

    Returns:
        var_dict: {var.id: c_variable} mapping
        n_vars: total number of scalar variables
    """
    id_map, _, n_vars, var_shapes = InverseData.get_var_offsets(variables)

    var_dict = {}
    for var in variables:
        offset, _ = id_map[var.id]
        shape = var_shapes[var.id]
        if len(shape) == 2:
            d1, d2 = shape[0], shape[1]
        elif len(shape) == 1:
            # NuMPy and CVXPY broadcasting rules treat a (n, ) vector as (1, n),
            # not as (n, 1)
            d1, d2 = 1, shape[0]
        else:  # scalar
            d1, d2 = 1, 1
        c_var = _diffengine.make_variable(d1, d2, offset, n_vars)
        var_dict[var.id] = c_var

    return var_dict, n_vars


def convert_expr(expr, var_dict: dict, n_vars: int):
    """Convert CVXPY expression using pre-built variable dictionary."""
    # Base case: variable lookup
    if isinstance(expr, cp.Variable):
        return var_dict[expr.id]

    # Base case: constant
    if isinstance(expr, cp.Constant):
        c = expr.value

        # we only support dense constants for now
        if sparse.issparse(c):
            c = c.todense()
        
        c = np.asarray(c, dtype=np.float64)
        x_shape = tuple(expr.shape)
        x_shape = (1,) * (2 - len(x_shape)) + x_shape
        d1, d2 = x_shape
        return _diffengine.make_constant(d1, d2, n_vars, c.flatten(order='F'))

    # Recursive case: atoms
    atom_name = type(expr).__name__


    if atom_name in ATOM_CONVERTERS:
        children = [convert_expr(arg, var_dict, n_vars) for arg in expr.args]
        C_expr = ATOM_CONVERTERS[atom_name](expr, children)

        # check that python dimension is consistent with C dimension
        d1_C, d2_C = _diffengine.get_expr_dimensions(C_expr)
        x_shape = tuple(expr.shape)
        x_shape = (1,) * (2 - len(x_shape)) + x_shape
        d1_Python, d2_Python = x_shape

        if d1_C != d1_Python or d2_C != d2_Python:
            raise ValueError(
                f"Dimension mismatch for atom '{atom_name}': "
                f"C dimensions ({d1_C}, {d2_C}) vs Python dimensions ({d1_Python}, {d2_Python})"
            )
            
        return C_expr
    
    raise NotImplementedError(f"Atom '{atom_name}' not supported")


def convert_expressions(problem: cp.Problem) -> tuple:
    """
    Convert CVXPY Problem to C expressions (low-level).

    Args:
        problem: CVXPY Problem object

    Returns:
        c_objective: C expression for objective
        c_constraints: list of C expressions for constraints
    """
    var_dict, n_vars = build_variable_dict(problem.variables())

    # Convert objective
    c_objective = convert_expr(problem.objective.expr, var_dict, n_vars)

    # Convert constraints (expression part only for now)
    c_constraints = []
    for constr in problem.constraints:
        c_expr = convert_expr(constr.expr, var_dict, n_vars)
        c_constraints.append(c_expr)

    return c_objective, c_constraints
