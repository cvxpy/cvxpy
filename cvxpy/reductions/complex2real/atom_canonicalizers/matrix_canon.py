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

from cvxpy.atoms import bmat, vstack
import numpy as np


def hermitian_canon(expr, real_args, imag_args, real2imag):
    """Canonicalize functions that take a Hermitian matrix.
    """
    if imag_args[0] is None:
        matrix = real_args[0]
    else:
        if real_args[0] is None:
            real_args[0] = np.zeros(real_args[0].shape)
        matrix = bmat([[real_args[0], -imag_args[0]],
                       [imag_args[0], real_args[0]]])
    return expr.copy([matrix]), None


def quad_canon(expr, real_args, imag_args, real2imag):
    """Canonicalize atoms that take a vector and Hermitian matrix.
    """
    if imag_args[1] is None:
        vec = real_args[1]
        matrix = real_args[0]
    elif real_args[1] is None:
        vec = imag_args[1]
        matrix = real_args[0]
    else:
        vec = vstack([real_args[0], imag_args[0]])
        if real_args[0] is None:
            real_args[0] = np.zeros(real_args[0].shape)
        elif imag_args[0] is None:
            imag_args[0] = np.zeros(imag_args[0].shape)
        matrix = bmat([[real_args[0], -imag_args[0]],
                       [imag_args[0], real_args[0]]])
    return expr.copy([vec, matrix]), None
