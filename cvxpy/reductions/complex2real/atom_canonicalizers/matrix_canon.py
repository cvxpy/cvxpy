"""
Copyright 2013 Steven Diamond

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""

from cvxpy.atoms import bmat, reshape, vstack
from cvxpy.expressions.constants import Constant
import numpy as np

# We expand the matrix A to B = [[Re(A), -Im(A)], [Im(A), Re(A)]]
# B has the same eigenvalues as A (if A is Hermitian).
# If x is an eigenvector of A, then [Re(x), Im(x)] and [Im(x), -Re(x)]
# are eigenvectors with same eigenvalue.
# Thus each eigenvalue is repeated twice.


def hermitian_canon(expr, real_args, imag_args, real2imag):
    """Canonicalize functions that take a Hermitian matrix.
    """
    if imag_args[0] is None:
        matrix = real_args[0]
    else:
        if real_args[0] is None:
            real_args[0] = np.zeros(imag_args[0].shape)
        matrix = bmat([[real_args[0], -imag_args[0]],
                       [imag_args[0], real_args[0]]])
    return expr.copy([matrix]), None


def norm_nuc_canon(expr, real_args, imag_args, real2imag):
    """Canonicalize nuclear norm with Hermitian matrix input.
    """
    # Divide by two because each eigenvalue is repeated twice.
    real, imag = hermitian_canon(expr, real_args, imag_args, real2imag)
    if imag_args[0] is not None:
        real /= 2
    return real, imag


def lambda_sum_largest_canon(expr, real_args, imag_args, real2imag):
    """Canonicalize nuclear norm with Hermitian matrix input.
    """
    # Divide by two because each eigenvalue is repeated twice.
    real, imag = hermitian_canon(expr, real_args, imag_args, real2imag)
    real.k *= 2
    if imag_args[0] is not None:
        real /= 2
    return real, imag


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
        matrix = Constant(matrix.value)
    return expr.copy([vec, matrix]), None


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
