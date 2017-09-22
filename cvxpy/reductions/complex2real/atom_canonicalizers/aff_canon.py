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

from cvxpy.expressions.constants import Constant
import numpy as np


def separable_canon(expr, real_args, imag_args):
    """Canonicalize linear functions that are seprable
       in real and imaginary parts.
    """
    if all([val is None for val in imag_args]):
        outputs = (expr.copy(real_args), None)
    elif all([val is None for val in real_args]):
        outputs = (None, expr.copy(imag_args))
    else:  # Mixed real_args and imaginaries.
        for idx, real_val in enumerate(real_args):
            if real_val is None:
                real_args[idx] = Constant(np.zeros(imag_args[idx].shape))
            elif imag_args[idx] is None:
                imag_args[idx] = Constant(np.zeros(real_args[idx].shape))
        outputs = (expr.copy(real_args), expr.copy(imag_args))
    return outputs


def real_canon(expr, real_args, imag_args):
    return real_args[0], None


def imag_canon(expr, real_args, imag_args):
    return imag_args[0], None


def conj_canon(expr, real_args, imag_args):
    return real_args[0], -imag_args[0]


def binary_canon(expr, real_args, imag_args):
    """Canonicalize functions like multiplication.
    """
    if all([val is None for val in imag_args]):
        outputs = (expr.copy(real_args), None)
    elif all([val is None for val in real_args]):
        outputs = (-expr.copy(imag_args), None)
    else:  # Mixed real_args and imaginaries.
        for idx, real_val in enumerate(real_args):
            if real_val is None:
                real_args[idx] = Constant(np.zeros(imag_args[idx].shape))
            elif imag_args[idx] is None:
                imag_args[idx] = Constant(np.zeros(real_args[idx].shape))
        real_part = expr.copy([real_args[0], real_args[1]])
        real_part -= expr.copy([imag_args[0], imag_args[1]])
        imag_part = expr.copy([real_args[0], imag_args[1]])
        imag_part += expr.copy([imag_args[0], real_args[1]])
        outputs = (real_part, imag_part)
    return outputs
