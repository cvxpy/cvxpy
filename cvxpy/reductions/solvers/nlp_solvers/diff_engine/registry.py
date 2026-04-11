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

Atom converter registry: maps CVXPY atom names to C diff engine constructors.

Each converter receives (expr, children) where expr is the CVXPY expression
and children are already-converted C nodes. matmul and multiply are handled
separately in converters.py (they need param_dict for parameter support).
"""
import numpy as np
from scipy import sparse
from sparsediffpy import _sparsediffengine as _diffengine

import cvxpy as cp
from cvxpy.reductions.solvers.nlp_solvers.diff_engine.helpers import (
    chain_add,
    normalize_shape,
)


def convert_hstack(expr, children):
    """Convert horizontal stack (hstack) of expressions."""
    return _diffengine.make_hstack(children)


def extract_flat_indices_from_index(expr):
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


def extract_flat_indices_from_special_index(expr):
    """Extract flattened indices from CVXPY special_index expression."""
    return np.reshape(expr._select_mat, expr._select_mat.size, order="F").astype(np.int32)


def convert_rel_entr(expr, children):
    """Convert rel_entr(x, y) = x * log(x/y).

    The C engine auto-dispatches based on argument dimensions
    (elementwise, vector-scalar, or scalar-vector).
    """
    x_size = expr.args[0].size
    y_size = expr.args[1].size
    if x_size > 1 and y_size > 1 and x_size != y_size:
        raise ValueError(
            f"rel_entr requires compatible argument sizes. "
            f"Got: x={x_size}, y={y_size}"
        )
    return _diffengine.make_rel_entr(children[0], children[1])


def convert_quad_form(expr, children):
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


def convert_reshape(expr, children):
    """Convert reshape - only Fortran order is supported.

    Note: Only order='F' (Fortran/column-major) is supported.
    """
    if expr.order != "F":
        raise NotImplementedError(
            f"reshape with order='{expr.order}' not supported. "
            "Only order='F' (Fortran) is currently supported."
        )

    d1, d2 = normalize_shape(expr.shape)
    return _diffengine.make_reshape(children[0], d1, d2)

def convert_broadcast(expr, children):
    d1, d2 = expr.broadcast_shape
    d1_C, d2_C = _diffengine.get_expr_dimensions(children[0])
    if d1_C == d1 and d2_C == d2:
        return children[0]

    return _diffengine.make_broadcast(children[0], d1, d2)

def convert_sum(expr, children):
    axis = expr.axis
    if axis is None:
        axis = -1
    return _diffengine.make_sum(children[0], axis)

def convert_promote(expr, children):
    d1, d2 = normalize_shape(expr.shape)
    return _diffengine.make_promote(children[0], d1, d2)

def convert_NegExpression(_expr, children):
    return _diffengine.make_neg(children[0])

def convert_quad_over_lin(_expr, children):
    return _diffengine.make_quad_over_lin(children[0], children[1])

def convert_index(expr, children):
    idxs = extract_flat_indices_from_index(expr)
    d1, d2 = normalize_shape(expr.shape)
    return _diffengine.make_index(children[0], d1, d2, idxs)

def convert_special_index(expr, children):
    idxs = extract_flat_indices_from_special_index(expr)
    d1, d2 = normalize_shape(expr.shape)
    return _diffengine.make_index(children[0], d1, d2, idxs)

def convert_prod(expr, children):
    axis = expr.axis
    if axis is None:
        return _diffengine.make_prod(children[0])
    elif axis == 0:
        return _diffengine.make_prod_axis_zero(children[0])
    elif axis == 1:
        return _diffengine.make_prod_axis_one(children[0])

def convert_transpose(expr, children):
    # If the child is a vector (shape (n,) or (n,1) or (1,n)), use reshape to transpose
    child_shape = normalize_shape(expr.args[0].shape)

    if 1 in child_shape:
        return _diffengine.make_reshape(children[0], child_shape[1], child_shape[0])
    else:
        return _diffengine.make_transpose(children[0])

def convert_trace(_expr, children):
    return _diffengine.make_trace(children[0])

def convert_diag_vec(expr, children):
    # C implementation only supports k=0 (main diagonal)
    if expr.k != 0:
        raise NotImplementedError("diag_vec with k != 0 not supported in diff engine")
    return _diffengine.make_diag_vec(children[0])


ATOM_CONVERTERS = {
    # Elementwise unary
    "log": lambda _expr, children: _diffengine.make_log(children[0]),
    "exp": lambda _expr, children: _diffengine.make_exp(children[0]),
    # Affine unary
    "NegExpression": convert_NegExpression,
    "Promote": convert_promote,
    # N-ary (handles 2+ args)
    "AddExpression": lambda _expr, children: chain_add(children),
    # Reductions
    "Sum": convert_sum,
    # Bivariate
    "QuadForm": convert_quad_form,
    "quad_over_lin": convert_quad_over_lin,
    "rel_entr": convert_rel_entr,
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
    "normcdf": lambda _expr, children: _diffengine.make_normal_cdf(children[0]),
    # Indexing/slicing
    "index": convert_index,
    "special_index": convert_special_index,
    "reshape": convert_reshape,
    "broadcast_to": convert_broadcast,
    # Reductions returning scalar
    "Prod": convert_prod,
    "transpose": convert_transpose,
    # Horizontal stack
    "Hstack": convert_hstack,
    "Trace": convert_trace,
    # Diagonal
    "diag_vec": convert_diag_vec,
}
