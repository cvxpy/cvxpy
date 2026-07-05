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
import operator
from functools import reduce

import numpy as np
from scipy import sparse
from sparsediffpy import _sparsediffengine as _diffengine

import cvxpy as cp
import cvxpy.settings as s
from cvxpy.atoms.affine.wraps import Wrap
from cvxpy.atoms.elementwise.power import Power
from cvxpy.atoms.quad_form import QuadForm
from cvxpy.atoms.quad_over_lin import quad_over_lin
from cvxpy.expressions.constants import Constant
from cvxpy.reductions.solvers.nlp_solvers.diff_engine.helpers import (
    make_dense_left_matmul,
    make_dense_right_matmul,
    make_sparse_left_matmul,
    make_sparse_right_matmul,
    normalize_shape,
    to_dense_float,
)
from cvxpy.reductions.solvers.nlp_solvers.diff_engine.registry import ATOM_CONVERTERS


def _is_plain_constant(expr):
    """Variable-free and parameter-free: a value that never changes between solves."""
    return expr.is_constant() and not expr.parameters()


def _is_vector(expr):
    """1-D, or 2-D with a singleton dimension."""
    return len(expr.shape) <= 1 or min(expr.shape) == 1


def _apply_constant_right(left, c):
    """Build an expression equivalent to ``left @ c`` for a plain-constant
    ``c``, pushing ``c`` toward the leaves so constants multiply constants.

    Each engine node's Jacobian has one row per node *output*, so a matmul
    chain must not drag wide intermediates when it ends in a constant:
    ``(A @ E @ B) @ x`` builds a rows x cols matrix Jacobian at every node
    (issue #2205: dense DFT sandwiches made one values fill take 33s), while
    pushed right-to-left every intermediate has only cols(c) columns.
    Structural recursion on ``left``:

      constant @ c   -> Constant(value)      the fold that shrinks the chain
      (A @ B) @ c    -> A @ (B @ c)          recursing into both factors
      (-E) @ c       -> -(E @ c)
      (E1 + E2) @ c  -> E1 @ c + E2 @ c      only for vector c: for a matrix
                                             tail this duplicates rather than
                                             shrinks the intermediates

    Only parameter-free constants fold, so parametric factors are never
    frozen. Shapes are preserved (matmul associativity), so the dimension
    check in convert_expr still applies.
    """
    if _is_plain_constant(left):
        return Constant(left.value @ c.value)
    name = type(left).__name__
    if name == "MulExpression":
        a, b = left.args
        if _is_plain_constant(b):
            return _apply_constant_right(a, Constant(b.value @ c.value))
        if _is_vector(c):
            return a @ _apply_constant_right(b, c)
        if _is_plain_constant(a):
            # (C1 @ E) @ C2 -> C1 @ (E @ C2): frees the tail to keep folding
            # when (E @ C2) is normalized on its own visit.
            return a @ (b @ c)
        return left @ c
    if name == "NegExpression":
        return -_apply_constant_right(left.args[0], c)
    if (name == "AddExpression" and _is_vector(c)
            and all(arg.shape == left.shape for arg in left.args)):
        return reduce(operator.add, [_apply_constant_right(arg, c) for arg in left.args])
    return left @ c


def _normalize_matmul(expr):
    """Reassociate ``expr`` when a matmul chain ends in a plain-constant
    factor, so the constant folds into adjacent constants and the
    intermediates shrink. General matrix-matrix chain reordering (and the
    row-vector-head mirror) is deliberately not attempted."""
    left, right = expr.args
    if _is_plain_constant(right) and not left.is_constant():
        return _apply_constant_right(left, right)
    return expr


def convert_matmul(expr, children, var_dict, n_vars, param_dict):
    """Convert matrix multiplication A @ f(x), f(x) @ A, or X @ Y.

    Follows numpy's matmul broadcasting rules for 1D operands.
    """
    left_arg, right_arg = expr.args

    if left_arg.is_constant():
        A = left_arg.value
        if A.ndim == 1:
            A = A.reshape(1, -1)
        param_node = children[0] if left_arg.parameters() else None
        if sparse.issparse(A):
            return make_sparse_left_matmul(param_node, children[1], A)
        # A constant dense matrix that is mostly zeros: route it to the sparse CSR binding
        # to avoid building a dense Jacobian/Hessian. Restricted to constants
        # (param_node is None) because sparsifying a parametric matrix would freeze its
        # sparsity pattern to the current value.
        density = np.count_nonzero(A) / A.size if A.size else 1.0
        if param_node is None and density < s.SPARSE_DENSITY_THRESHOLD:
            return make_sparse_left_matmul(None, children[1], sparse.csr_array(A))
        return make_dense_left_matmul(param_node, children[1], A)

    elif right_arg.is_constant():
        A = right_arg.value
        if A.ndim == 1:
            A = A.reshape(-1, 1)
        param_node = children[1] if right_arg.parameters() else None
        if sparse.issparse(A):
            return make_sparse_right_matmul(param_node, children[0], A)
        return make_dense_right_matmul(param_node, children[0], A)

    else:
        # The diffengine doesn't natively support a 1D right operand in matmul,
        # so reshape (n,) -> (n, 1) here to match numpy's column-vector convention.
        right_node = children[1]
        if len(right_arg.shape) == 1:
            right_node = _diffengine.make_reshape(right_node, right_arg.shape[0], 1)
        return _diffengine.make_matmul(children[0], right_node)

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
    """Lower a power/PowerApprox SymbolicQuadForm to the elementwise square x .* x.

    Rebuild over expr.args[0] (a leaf, per the canonicalizer): the engine's
    init_jacobian segfaults on a quad form over a compound argument.
    """
    return cp.multiply(expr.args[0], expr.args[0])


def convert_symbolic_quad_form(expr, var_dict, n_vars, param_dict):
    """Convert a SymbolicQuadForm (Dcp2Cone quadratic-objective placeholder).

    Scalar x'Px (QuadForm / quad_over_lin / sum_squares) uses the native quad_form
    binding -- the sparse path for a sparse constant P, the dense path for a dense or
    parametric P. power/PowerApprox is the elementwise square, lowered to multiply.
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
            # P is affine in the parameters and independent of x (Hessian still 2P):
            # feed it as a matrix-valued child evaluated each solve. Peel value-identity
            # Wrap atoms (e.g. psd_wrap) the converter can't build directly.
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

    # Reassociate matmul chains before converting children, and skip converting
    # a plain-constant matmul operand: convert_matmul consumes its scipy/numpy
    # value directly (sparse where possible), so the dense make_parameter node
    # the base case would build is pure waste (issue #2205 / quantum benchmark).
    skip_child = [False] * len(expr.args)
    if atom_name == "MulExpression":
        expr = _normalize_matmul(expr)
        if type(expr).__name__ != "MulExpression":
            return convert_expr(expr, var_dict, n_vars, param_dict)
        left_arg, right_arg = expr.args
        if _is_plain_constant(left_arg) and not right_arg.is_constant():
            skip_child[0] = True
        elif _is_plain_constant(right_arg) and not left_arg.is_constant():
            skip_child[1] = True

    children = [
        None if skip_child[i] else convert_expr(arg, var_dict, n_vars, param_dict)
        for i, arg in enumerate(expr.args)
    ]

    # matmul and multiply need param_dict for parameter support
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
