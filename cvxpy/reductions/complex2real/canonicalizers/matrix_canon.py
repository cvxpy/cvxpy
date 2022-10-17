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
"""

import numpy as np

from cvxpy.expressions.expression import Expression
from cvxpy.atoms import bmat, reshape, vstack, von_neumann_entr
from cvxpy.constraints.exponential import OpRelConeQuad
from cvxpy.atoms.affine.wraps import psd_wrap

# We expand the matrix A to B = [[Re(A), -Im(A)], [Im(A), Re(A)]]
# B has the same eigenvalues as A (if A is Hermitian).
# If x is an eigenvector of A, then [Re(x), Im(x)] and [Im(x), -Re(x)]
# are eigenvectors with same eigenvalue.
# Thus each eigenvalue is repeated twice.


def hermitian_dilation(expr, real_part, imag_part, force_lift=False):
    # All arguments are Expression or None.
    lift = force_lift or (imag_part is not None)
    if not lift:
        matrix = real_part
    else:
        if real_part is None:
            real_part = np.zeros(imag_part.shape)
        matrix = bmat([[real_part, -imag_part],
                       [imag_part, real_part]])
    return expr.copy([matrix]), lift


def hermitian_canon(expr, real_args, imag_args, real2imag):
    """Canonicalize functions that take a Hermitian matrix.
    """
    assert len(real_args) == 1 and len(imag_args) == 1
    expr_canon, _ = hermitian_dilation(expr, real_args[0], imag_args[0])
    return expr_canon, None


def norm_nuc_canon(expr, real_args, imag_args, real2imag):
    """Canonicalize nuclear norm with Hermitian matrix input.
    """
    # Divide by two because each eigenvalue is repeated twice.
    real, imag = hermitian_canon(expr, real_args, imag_args, real2imag)
    if imag_args[0] is not None:
        real /= 2
    return real, imag


def lambda_sum_largest_canon(expr, real_args, imag_args, real2imag):
    """Canonicalize sum of k largest eigenvalues with Hermitian matrix input.
    """
    # Divide by two because each eigenvalue is repeated twice.
    real, imag = hermitian_canon(expr, real_args, imag_args, real2imag)
    real.k *= 2
    if imag_args[0] is not None:
        real /= 2
    return real, imag


def von_neumann_entr_canon(expr: von_neumann_entr, real_args, imag_args, real2imag):
    """
    The von Neumann entropy of X is sum(entr(eigvals(X)).
    Each eigenvalue of X appears twice as an eigenvalue of the Hermitian dilation of X.
    """
    args = expr.args
    X_dilation, lift = hermitian_dilation(args[0], real_args[0], imag_args[0])
    canon_expr = expr.copy([X_dilation])
    if lift:
        # Note: we aren't dividing X_dilation by 2, we're dividing
        # von_neumann_entr(X_dilation) by 2.
        canon_expr /= 2
    return canon_expr, None


def quantum_rel_entr_canon(expr, real_args, imag_args, real2imag):
    """expr is a quantum_rel_entr Atom."""
    args = expr.args
    force = any(a is not None for a in imag_args)
    X_dilation = hermitian_dilation(args[0], real_args[0], imag_args[0], force)
    Y_dilation = hermitian_dilation(args[1], real_args[1], imag_args[1], force)
    canon_expr = expr.copy([X_dilation, Y_dilation])
    if force:
        # TODO: verify that this dividing by 2 is appropriate.
        #   Riley thinks it is, since quantum relative entropy is homogeneous.
        canon_expr /= 2
    return canon_expr, None


def op_rel_cone_canon(expr: OpRelConeQuad, real_args, imag_args, real2imag):
    """Transform Hermitian input for OpRelConeQuad into equivalent
    symmetric input for OpRelConeQuad
    """
    args = expr.args
    force = any(a is not None for a in imag_args)
    X_dilation = hermitian_dilation(args[0], real_args[0], imag_args[0], force)
    Y_dilation = hermitian_dilation(args[1], real_args[1], imag_args[1], force)
    Z_dilation = hermitian_dilation(args[2], real_args[2], imag_args[2], force)
    canon_expr = expr.copy([X_dilation, Y_dilation, Z_dilation])
    return canon_expr, None


def at_least_2D(expr):
    """Upcast 0D and 1D to 2D.
    """
    if expr.ndim < 2:
        return reshape(expr, (expr.size, 1))
    else:
        return expr


def quad_canon(expr, real_args, imag_args, real2imag):
    """Convert quad_form to real.
    """
    if imag_args[0] is None:
        vec = real_args[0]
        matrix = real_args[1]
    elif real_args[0] is None:
        vec = imag_args[0]
        matrix = real_args[1]
    else:
        vec = vstack([at_least_2D(real_args[0]),
                      at_least_2D(imag_args[0])])
        if real_args[1] is None:
            real_args[1] = np.zeros(imag_args[1].shape)
        elif imag_args[1] is None:
            imag_args[1] = np.zeros(real_args[1].shape)
        matrix = bmat([[real_args[1], -imag_args[1]],
                       [imag_args[1], real_args[1]]])
        matrix = psd_wrap(matrix)
    return expr.copy([vec, matrix]), None


def quad_over_lin_canon(expr, real_args, imag_args, real2imag):
    """Convert quad_over_lin to real.
    """
    if imag_args[0] is None:
        matrix = real_args[0]
    else:
        matrix = bmat([real_args[0], imag_args[0]])
    return expr.copy([matrix, real_args[1]]), None


def matrix_frac_canon(expr, real_args, imag_args, real2imag):
    """Convert matrix_frac to real.
    """
    if real_args[0] is None:
        real_args[0] = np.zeros(imag_args[0].shape)
    if imag_args[0] is None:
        imag_args[0] = np.zeros(real_args[0].shape)
    vec = vstack([at_least_2D(real_args[0]),
                  at_least_2D(imag_args[0])])
    if real_args[1] is None:
        real_args[1] = np.zeros(imag_args[1].shape)
    elif imag_args[1] is None:
        imag_args[1] = np.zeros(real_args[1].shape)
    matrix = bmat([[real_args[1], -imag_args[1]],
                   [imag_args[1], real_args[1]]])
    return expr.copy([vec, matrix]), None
