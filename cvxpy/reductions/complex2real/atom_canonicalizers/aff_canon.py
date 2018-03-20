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


def separable_canon(expr, real_args, imag_args, real2imag):
    """Canonicalize linear functions that are seprable
       in real and imaginary parts.
    """
    if all(val is None for val in imag_args):
        outputs = (expr.copy(real_args), None)
    elif all(val is None for val in real_args):
        outputs = (None, expr.copy(imag_args))
    else:  # Mixed real_args and imaginaries.
        for idx, real_val in enumerate(real_args):
            if real_val is None:
                real_args[idx] = Constant(np.zeros(imag_args[idx].shape))
            elif imag_args[idx] is None:
                imag_args[idx] = Constant(np.zeros(real_args[idx].shape))
        outputs = (expr.copy(real_args), expr.copy(imag_args))
    return outputs


def real_canon(expr, real_args, imag_args, real2imag):
    return real_args[0], None


def imag_canon(expr, real_args, imag_args, real2imag):
    return imag_args[0], None


def conj_canon(expr, real_args, imag_args, real2imag):
    if imag_args[0] is None:
        imag_arg = None
    else:
        imag_arg = -imag_args[0]
    return real_args[0], imag_arg


def join(expr, lh_arg, rh_arg):
    """Helper function to combine arguments.
    """
    if lh_arg is None or rh_arg is None:
        return None
    else:
        return expr.copy([lh_arg, rh_arg])


def add(lh_arg, rh_arg, neg=False):
    """Helper function to sum arguments.
       Negates rh_arg if neg is True.
    """
    if rh_arg is not None and neg:
        rh_arg = -rh_arg

    if lh_arg is None and rh_arg is None:
        return None
    elif lh_arg is None:
        return rh_arg
    elif rh_arg is None:
        return lh_arg
    else:
        return lh_arg + rh_arg


def binary_canon(expr, real_args, imag_args, real2imag):
    """Canonicalize functions like multiplication.
    """
    real_by_real = join(expr, real_args[0], real_args[1])
    imag_by_imag = join(expr, imag_args[0], imag_args[1])
    real_by_imag = join(expr, real_args[0], imag_args[1])
    imag_by_real = join(expr, imag_args[0], real_args[1])
    real_output = add(real_by_real, imag_by_imag, neg=True)
    imag_output = add(real_by_imag, imag_by_real, neg=False)
    return real_output, imag_output
