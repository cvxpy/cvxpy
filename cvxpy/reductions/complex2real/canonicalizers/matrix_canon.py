"""
Copyright 2013 Steven Diamond, 2022 the CVXPY Authors

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

from typing import List, Optional, Union

import numpy as np

from cvxpy.atoms import (
    bmat,
    lambda_sum_largest,
    normNuc,
    reshape,
    symmetric_wrap,
    von_neumann_entr,
    vstack,
)
from cvxpy.atoms.affine.wraps import psd_wrap
from cvxpy.constraints.exponential import OpRelEntrConeQuad
from cvxpy.expressions.constants.constant import Constant
from cvxpy.expressions.expression import Expression


def expand_complex(real_part: Optional[Expression],
                   imag_part: Optional[Expression]):
    """
    We expand the matrix A to B = [[Re(A), -Im(A)], [Im(A), Re(A)]].

    The resulting matrix has special structure if A is Hermitian.
    Specifically, if x is an eigenvector of A, then [Re(x), Im(x)]
    and [Im(x), -Re(x)] are eigenvectors of B with same eigenvalue.
    Therefore, the eigenvalues of B are the same as those of A,
    repeated twice.
    """
    if real_part is None:
        real_part = Constant(np.zeros(imag_part.shape))
    elif imag_part is None:
        # This is a strange code path to hit.
        imag_part = Constant(np.zeros(real_part.shape))
    matrix = bmat([[real_part, -imag_part],
                   [imag_part, real_part]])
    if real_part.is_symmetric() and imag_part.is_skew_symmetric():
        matrix = symmetric_wrap(matrix)
    return matrix


def expand_and_reapply(expr: Expression,
                       real_part: Optional[Expression],
                       imag_part: Optional[Expression]):
    if imag_part is None:
        # A weird code path to hit.
        matrix = real_part
    else:
        matrix = expand_complex(real_part, imag_part)
    return expr.copy([matrix])


def hermitian_canon(expr: Expression,
                    real_args: List[Union[Expression, None]],
                    imag_args: List[Union[Expression, None]], real2imag):
    """Canonicalize functions that take a Hermitian matrix.
    """
    assert len(real_args) == 1 and len(imag_args) == 1
    expr_canon = expand_and_reapply(expr, real_args[0], imag_args[0])
    return expr_canon, None


def trace_canon(expr: Expression,
                real_args: List[Union[Expression, None]],
                imag_args: List[Union[Expression, None]], real2imag):
    if real_args[0] is None:
        real_part = None
    else:
        real_part = expr.copy([real_args[0]])
    if (imag_args[0] is None) or expr.is_hermitian():
        imag_part = None
    else:
        imag_part = expr.copy([imag_args[0]])
    return real_part, imag_part


def norm_nuc_canon(expr: normNuc,
                   real_args: List[Union[Expression, None]],
                   imag_args: List[Union[Expression, None]], real2imag):
    """Canonicalize nuclear norm with Hermitian matrix input.
    """
    # Divide by two because each eigenvalue is repeated twice.
    real, imag = hermitian_canon(expr, real_args, imag_args, real2imag)
    if imag_args[0] is not None:
        real /= 2
    return real, imag


def lambda_sum_largest_canon(expr: lambda_sum_largest,
                             real_args: List[Union[Expression, None]],
                             imag_args: List[Union[Expression, None]], real2imag):
    """Canonicalize sum of k largest eigenvalues with Hermitian matrix input.
    """
    # Divide by two because each eigenvalue is repeated twice.
    real, imag = hermitian_canon(expr, real_args, imag_args, real2imag)
    real.k *= 2
    if imag_args[0] is not None:
        real /= 2
    return real, imag


def von_neumann_entr_canon(expr: von_neumann_entr,
                           real_args: List[Union[Expression, None]],
                           imag_args: List[Union[Expression, None]], real2imag):
    """
    The von Neumann entropy of X is sum(entr(eigvals(X)).
    Each eigenvalue of X appears twice as an eigenvalue of the Hermitian dilation of X.
    """
    canon_expr = expand_and_reapply(expr, real_args[0], imag_args[0])
    if imag_args[0] is not None:
        canon_expr /= 2
    return canon_expr, None


def op_rel_entr_cone_canon(expr: OpRelEntrConeQuad,
                           real_args: List[Union[Expression, None]],
                           imag_args: List[Union[Expression, None]], real2imag):
    """Transform Hermitian input for OpRelEntrConeQuad into equivalent
    symmetric input for OpRelEntrConeQuad
    """
    must_expand = any(a is not None for a in imag_args)
    if must_expand:
        X_dilation = expand_complex(real_args[0], imag_args[0])
        Y_dilation = expand_complex(real_args[1], imag_args[1])
        Z_dilation = expand_complex(real_args[2], imag_args[2])
        canon_expr = expr.copy([X_dilation, Y_dilation, Z_dilation])
    else:
        canon_expr = expr.copy(real_args)
    return [canon_expr], None


def at_least_2D(expr: Expression):
    """Upcast 0D and 1D to 2D.
    """
    if expr.ndim < 2:
        return reshape(expr, (expr.size, 1))
    else:
        return expr


def quad_canon(expr,
               real_args: List[Union[Expression, None]],
               imag_args: List[Union[Expression, None]], real2imag):
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


def quad_over_lin_canon(expr,
                        real_args: List[Union[Expression, None]],
                        imag_args: List[Union[Expression, None]], real2imag):
    """Convert quad_over_lin to real.
    """
    if imag_args[0] is None:
        matrix = real_args[0]
    else:
        matrix = bmat([real_args[0], imag_args[0]])
    return expr.copy([matrix, real_args[1]]), None


def matrix_frac_canon(expr,
                      real_args: List[Union[Expression, None]],
                      imag_args: List[Union[Expression, None]], real2imag):
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
