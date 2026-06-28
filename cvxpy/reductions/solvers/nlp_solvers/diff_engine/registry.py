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

import cvxpy.settings as s
from cvxpy.reductions.solvers.nlp_solvers.diff_engine.helpers import (
    chain_add,
    normalize_shape,
    to_dense_float,
)


def convert_hstack(expr, children):
    """Convert horizontal stack (hstack) of expressions."""
    return _diffengine.make_hstack(children)


def convert_vstack(expr, children):
    """Convert vertical stack (vstack) of expressions."""
    return _diffengine.make_vstack(children)


def convert_conv(expr, children):
    """Convert cp.conv / cp.convolve (full 1D convolution)."""
    return _diffengine.make_convolve(children[0], children[1])


def convert_kron(expr, children):
    """Convert cp.kron(A, B). One operand is variable-free (kron requires it);
    it is the "parameter" side that scales the variable-carrying operand. The
    native node re-evaluates that operand each solve, so a parametric A or B is
    supported (not just a constant). ``const_is_left`` tells the engine which
    operand is the parameter; (p, q) and (r, s) are A's and B's dims."""
    a, b = expr.args
    const_is_left = a.is_constant()
    param_node = children[0] if const_is_left else children[1]
    var_node = children[1] if const_is_left else children[0]
    p, q = a.shape
    r, s = b.shape
    return _diffengine.make_kron(param_node, var_node, int(const_is_left), p, q, r, s)


def convert_div(expr, children):
    """Convert x / d by multiplying x by the elementwise reciprocal of d.

    The divisor is variable-free (DivExpression requires a constant denominator),
    so it is the "parameter" side of the product -- exactly like the constant
    operand in convert_multiply. A constant divisor bakes ``1/d`` into a
    reciprocal parameter node (and rejects zero entries explicitly, matching
    coo_backend.div). A parametric divisor -- e.g. ``|c|^2`` produced by
    Complex2Real when dividing by a complex Parameter -- instead uses a
    ``make_power(d, -1)`` node that the diff engine re-evaluates from the current
    parameter value each solve.
    """
    divisor_expr = expr.args[1]
    if divisor_expr.parameters():
        recip_node = _diffengine.make_power(children[1], -1.0)
        size = divisor_expr.size
    else:
        divisor = to_dense_float(divisor_expr.value)
        if np.any(divisor == 0):
            raise ValueError("Division by zero encountered in divisor")
        recip = 1.0 / divisor
        d1, d2 = normalize_shape(recip.shape)
        recip_node = _diffengine.make_parameter(d1, d2, -1, 0, recip.flatten(order='F'))
        size = recip.size
    if size == 1:
        return _diffengine.make_param_scalar_mult(recip_node, children[0])
    return _diffengine.make_param_vector_mult(recip_node, children[0])


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
    """Convert the scalar quadratic form ``x.T @ P @ x``.

    ``children[0]`` is the converted ``x`` node and ``children[1]`` the converted ``P``.
    A constant ``P`` goes to the engine's sparse (CSR) or dense ``make_quad_form``
    binding; a dense-but-mostly-zero ``P`` is routed to the sparse binding (density
    check) to avoid a dense Hessian block. A parametric ``P`` (depends on parameters,
    not on ``x``) is fed to the dense path as the matrix-valued child ``children[1]``,
    which the engine re-evaluates each solve.
    """
    P = expr.args[1]
    n = expr.args[0].size

    if P.parameters():
        return _diffengine.make_quad_form(children[1], children[0], "dense", None, n)

    if not P.is_constant():
        raise NotImplementedError("quad_form requires P to be a constant matrix")

    P_val = P.value
    if P_val is None:
        raise NotImplementedError(
            "quad_form with a symbolic P (e.g. eye/parameter without a value) "
            "is not supported by the diff engine."
        )

    if not sparse.issparse(P_val):
        P_dense = np.asarray(P_val, dtype=np.float64)
        # A dense but mostly-zero P (e.g. a diagonal written as np.eye) would build a
        # dense Hessian block; route it to the sparse binding instead, mirroring the
        # matmul sparse-dispatch.
        density = np.count_nonzero(P_dense) / P_dense.size if P_dense.size else 1.0
        if density >= s.SPARSE_DENSITY_THRESHOLD:
            return _diffengine.make_quad_form(
                None, children[0], "dense", P_dense.flatten(order='F'), P_dense.shape[0]
            )
        P_val = sparse.csr_array(P_dense)

    P_csr = P_val.tocsr()
    return _diffengine.make_quad_form(
        None,
        children[0],
        "sparse",
        P_csr.data.astype(np.float64),
        P_csr.indices.astype(np.int32),
        P_csr.indptr.astype(np.int32),
        P_csr.shape[0],
        P_csr.shape[1],
    )


def convert_reshape(expr, children):
    """Convert reshape. C-order via transpose(F-reshape(transpose(x))).

    Identity: reshape(x, (m, n), C) == transpose(reshape(transpose(x), (n, m), F)).
    """
    d1, d2 = normalize_shape(expr.shape)
    if expr.order == "F":
        return _diffengine.make_reshape(children[0], d1, d2)
    transposed = _diffengine.make_transpose(children[0])
    reshaped = _diffengine.make_reshape(transposed, d2, d1)
    return _diffengine.make_transpose(reshaped)

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
    # If a user calls transpose on a (n, ) expression, CVXPY treats it as a
    # no-op and keeps the shape as (n, ). The diff engine represents all 1D
    # expressions as (1, n), so we need to check for this case and avoid
    # transposing if it's just a 1D expression.
    if len(expr.args[0].shape) <= 1:
        return children[0]

    # If the child is a vector (shape (n,1) or (1,n), use reshape to transpose
    child_shape = normalize_shape(expr.args[0].shape)
    if 1 in child_shape:
        return _diffengine.make_reshape(children[0], child_shape[1], child_shape[0])
    else:
        return _diffengine.make_transpose(children[0])

def convert_trace(_expr, children):
    return _diffengine.make_trace(children[0])

def convert_diag_vec(expr, children):
    # C implementation only supports k=0 (main diagonal)
    # TODO add support for diag vec with k
    if expr.k != 0:
        raise NotImplementedError("diag_vec with k != 0 not supported in diff engine")
    return _diffengine.make_diag_vec(children[0])


def convert_diag_mat(expr, children):
    """Convert diag_mat: extract diagonal from square matrix."""
    if expr.k != 0:
        raise NotImplementedError("diag_mat with k != 0 not supported in diff engine")
    node = _diffengine.make_diag_mat(children[0])
    # C produces (n, 1) but CVXPY shape is (n,) which normalizes to (1, n)
    # TODO add support for producing (1, n) directly in C and remove this reshape
    # TODO also raise error that the k should be zero, since that's the only supported case
    n = expr.args[0].shape[0]
    return _diffengine.make_reshape(node, 1, n)


def convert_upper_tri(_expr, children):
    """Convert upper_tri: extract strict upper triangular elements."""
    return _diffengine.make_upper_tri(children[0])


ATOM_CONVERTERS = {
    # Elementwise unary
    "log": lambda _expr, children: _diffengine.make_log(children[0]),
    "exp": lambda _expr, children: _diffengine.make_exp(children[0]),
    # Affine unary
    "NegExpression": convert_NegExpression,
    "Promote": convert_promote,
    # Pass-through wrappers (no-ops for real-valued expressions)
    "conj": lambda _expr, children: children[0],
    "nonneg_wrap": lambda _expr, children: children[0],
    "nonpos_wrap": lambda _expr, children: children[0],
    "psd_wrap": lambda _expr, children: children[0],
    "nsd_wrap": lambda _expr, children: children[0],
    "hermitian_wrap": lambda _expr, children: children[0],
    "skew_symmetric_wrap": lambda _expr, children: children[0],
    "symmetric_wrap": lambda _expr, children: children[0],
    # Division by constant
    "DivExpression": convert_div,
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
    # Horizontal/vertical stack
    "Hstack": convert_hstack,
    "Vstack": convert_vstack,
    # 1D full convolution
    "conv": convert_conv,
    "convolve": convert_conv,
    # Kronecker product
    "kron": convert_kron,
    "Trace": convert_trace,
    # Diagonal and triangular
    "diag_vec": convert_diag_vec,
    "diag_mat": convert_diag_mat,
    "upper_tri": convert_upper_tri,
}
