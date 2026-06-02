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
import numpy as np
from scipy import sparse
from sparsediffpy import _sparsediffengine as _diffengine

import cvxpy as cp
from cvxpy.atoms.affine.wraps import Wrap
from cvxpy.atoms.elementwise.power import Power
from cvxpy.atoms.quad_form import QuadForm
from cvxpy.atoms.quad_over_lin import quad_over_lin
from cvxpy.reductions.solvers.nlp_solvers.diff_engine.helpers import (
    make_dense_left_matmul,
    make_dense_right_matmul,
    make_sparse_left_matmul,
    make_sparse_right_matmul,
    normalize_shape,
    to_dense_float,
)
from cvxpy.reductions.solvers.nlp_solvers.diff_engine.registry import ATOM_CONVERTERS


def convert_matmul(expr, var_dict, n_vars, param_dict):
    """Convert matrix multiplication A @ f(x), f(x) @ A, or X @ Y.

    Follows numpy's matmul broadcasting rules for 1D operands.

    Operands are converted lazily: a *pure* constant matrix operand (constant
    with no parameters) is consumed directly from its `.value` (sparse-aware) and
    is never converted into a node. Converting it would densify a large sparse
    constant -- e.g. the symmetrization matrix M in a parametric P = reshape(M @
    theta) -- which dominates compile time. A parametric matrix operand is still
    converted (its node feeds the engine's parameter refresh).
    """
    left_arg, right_arg = expr.args

    if left_arg.is_constant():
        A = left_arg.value
        if A.ndim == 1:
            A = A.reshape(1, -1)
        child = convert_expr(right_arg, var_dict, n_vars, param_dict)
        param_node = (convert_expr(left_arg, var_dict, n_vars, param_dict)
                      if left_arg.parameters() else None)
        if sparse.issparse(A):
            return make_sparse_left_matmul(param_node, child, A)
        return make_dense_left_matmul(param_node, child, A)

    elif right_arg.is_constant():
        A = right_arg.value
        if A.ndim == 1:
            A = A.reshape(-1, 1)
        child = convert_expr(left_arg, var_dict, n_vars, param_dict)
        param_node = (convert_expr(right_arg, var_dict, n_vars, param_dict)
                      if right_arg.parameters() else None)
        if sparse.issparse(A):
            return make_sparse_right_matmul(param_node, child, A)
        return make_dense_right_matmul(param_node, child, A)

    else:
        left_node = convert_expr(left_arg, var_dict, n_vars, param_dict)
        right_node = convert_expr(right_arg, var_dict, n_vars, param_dict)
        # The diffengine doesn't natively support a 1D right operand in matmul,
        # so reshape (n,) -> (n, 1) here to match numpy's column-vector convention.
        if len(right_arg.shape) == 1:
            right_node = _diffengine.make_reshape(right_node, right_arg.shape[0], 1)
        return _diffengine.make_matmul(left_node, right_node)

# TODO we should support sparse elementwise multiply at some point.
def convert_multiply(expr, children, var_dict, n_vars, param_dict):
    """Convert elementwise multiplication."""
    left_arg, right_arg = expr.args

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


def _lower_symbolic_power(expr):
    """Rewrite a ``power``/``PowerApprox`` SymbolicQuadForm as the elementwise square.

    ``power(., 2)`` (incl. ``huber``'s internal square) is the elementwise
    *diagonal* quad ``x .* x`` -- a vector matching the original shape, which the
    native ``quad_form`` node (a *scalar* ``x' P x``) cannot represent. The scalar
    ``QuadForm``/``quad_over_lin``/``sum_squares`` cases instead use the native
    ``quad_form`` binding in ``convert_symbolic_quad_form``.

    We rebuild over ``expr.args[0]`` -- the quad form's first arg, which the
    canonicalizer guarantees is a *leaf* (the original variable, or an auxiliary
    ``t`` with an added affine ``t == affine_expr`` constraint). This matters: the
    diff engine's ``init_jacobian`` segfaults on a quad form over a *compound*
    argument, so we must not reuse ``original_expression`` (which holds the
    compound arg). The aux variable keeps the compound part in the affine
    constraint, where the engine handles it fine.
    """
    return cp.multiply(expr.args[0], expr.args[0])


def convert_symbolic_quad_form(expr, var_dict, n_vars, param_dict):
    """Convert a ``SymbolicQuadForm`` (Dcp2Cone quadratic-objective placeholder).

    The scalar ``x' P x`` cases -- ``QuadForm`` and ``quad_over_lin``/
    ``sum_squares`` -- are built with the native ``quad_form`` binding, choosing
    the path by ``P`` (``expr.args[1]``): a **sparse** node for a sparse constant
    ``P`` (e.g. the identity of ``sum_squares``), a **dense** (``permuted_dense``)
    node for a dense or *parametric* ``P``. Building the node directly (rather
    than a ``multiply``/``sum`` graph) lets the engine assemble the Hessian
    natively, handles a matrix-variable leaf (the binding flattens it via
    ``n = x.size``), and feeds a dense/parametric ``P`` straight into a dense
    Hessian instead of a sparse autodiff cross-term.

    ``power``/``PowerApprox`` is the elementwise diagonal square (vector-valued),
    which ``quad_form`` (scalar) cannot represent, so it keeps the cheap
    ``multiply`` lowering via ``_lower_symbolic_power``.
    """
    if expr.block_indices is not None:
        raise NotImplementedError(
            "SymbolicQuadForm with block_indices (axis-reduced quad form) is not "
            "supported by the diff engine."
        )

    orig = expr.original_expression
    if isinstance(orig, (QuadForm, quad_over_lin)):
        x = expr.args[0]
        P = expr.args[1]
        x_c = convert_expr(x, var_dict, n_vars, param_dict)
        n = x.size
        if P.parameters():
            # Parametric P canonicalizes to an affine expression of the
            # parameters (e.g. psd_wrap(reshape(M @ theta))). Since P never
            # depends on x, the Hessian is still 2P -- so we feed P as a
            # matrix-valued child expression and let the engine evaluate it each
            # solve, avoiding the dense autodiff cross-term. Peel value-identity
            # Wrap atoms (psd_wrap) the converter can't build directly.
            P_inner = P
            while isinstance(P_inner, Wrap):
                P_inner = P_inner.args[0]
            P_c = convert_expr(P_inner, var_dict, n_vars, param_dict)
            return _diffengine.make_quad_form(P_c, x_c, "dense", None, n)
        P_val = P.value
        if sparse.issparse(P_val):
            P_csr = P_val.tocsr()
            return _diffengine.make_quad_form(
                None, x_c, "sparse",
                P_csr.data.astype(np.float64),
                P_csr.indices.astype(np.int32),
                P_csr.indptr.astype(np.int32),
                P_csr.shape[0], P_csr.shape[1])
        P_dense = to_dense_float(P_val)
        return _diffengine.make_quad_form(
            None, x_c, "dense", P_dense.flatten(order='F'), n)

    if isinstance(orig, Power):  # PowerApprox subclasses Power; canon only p == 2
        # Elementwise diagonal square (vector-valued); quad_form (scalar) can't
        # represent it, so keep the cheap multiply lowering.
        return convert_expr(
            _lower_symbolic_power(expr), var_dict, n_vars, param_dict)

    raise NotImplementedError(
        f"SymbolicQuadForm over '{type(orig).__name__}' is not supported by the "
        "diff engine."
    )


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

    # SymbolicQuadForm is a placeholder Dcp2Cone(quad_obj=True) leaves in the
    # objective. Lower it to equivalent atoms before converting its args, since
    # its P arg (e.g. eye/y) may itself be unconvertible (parametric divisor).
    if atom_name == "SymbolicQuadForm":
        return convert_symbolic_quad_form(expr, var_dict, n_vars, param_dict)

    # matmul converts its operands lazily (see convert_matmul) so a pure-constant
    # matrix operand is never densified into a node; the others consume every
    # child, so convert eagerly for them.
    if atom_name == "MulExpression":
        C_expr = convert_matmul(expr, var_dict, n_vars, param_dict)
    elif atom_name == "multiply":
        children = [convert_expr(arg, var_dict, n_vars, param_dict)
                    for arg in expr.args]
        C_expr = convert_multiply(expr, children, var_dict, n_vars, param_dict)
    elif atom_name in ATOM_CONVERTERS:
        children = [convert_expr(arg, var_dict, n_vars, param_dict)
                    for arg in expr.args]
        C_expr = ATOM_CONVERTERS[atom_name](expr, children)
    else:
        raise NotImplementedError(f"Atom '{atom_name}' not supported")

    # check that python dimension is consistent with C dimension
    d1_C, d2_C = _diffengine.get_expr_dimensions(C_expr)
    d1_Python, d2_Python = normalize_shape(expr.shape)

    if d1_C != d1_Python or d2_C != d2_Python:
        # 1D shape (n,) normalizes to (1, n) but C may produce (n, 1); reshape.
        if len(expr.shape) <= 1 and d1_C * d2_C == d1_Python * d2_Python:
            C_expr = _diffengine.make_reshape(C_expr, d1_Python, d2_Python)
        else:
            raise ValueError(
                f"Dimension mismatch for atom '{atom_name}': "
                f"C dimensions ({d1_C}, {d2_C}) vs "
                f"Python dimensions ({d1_Python}, {d2_Python})"
            )

    return C_expr
