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

from cvxpy.atoms.affine.binary_operators import multiply
from cvxpy.atoms.affine.wraps import skew_symmetric_wrap, symmetric_wrap
from cvxpy.expressions.constants import Constant


def separable_canon(expr, real_args, imag_args, real2imag):
    """Canonicalize linear functions that are separable
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
    # If no real arguments, return zero.
    if real_args[0] is None:
        return 0*imag_args[0], None
    else:
        return real_args[0], None


def imag_canon(expr, real_args, imag_args, real2imag):
    # If no real arguments, return zero.
    if imag_args[0] is None:
        return 0*real_args[0], None
    else:
        return imag_args[0], None


def hermitian_wrap_canon(expr, real_args, imag_args, real2imag):
    if imag_args[0] is not None:
        imag_arg = skew_symmetric_wrap(imag_args[0])
    else:
        # This is a weird code path to hit.
        imag_arg = None
    real_arg = symmetric_wrap(real_args[0])
    return real_arg, imag_arg


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


def add(lh_arg, rh_arg, neg: bool = False):
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


def div_canon(expr, real_args, imag_args, real2imag):
    """Canonicalize division by a (possibly complex) constant or parameter.

    For a complex divisor c, z/c = z*conj(c)/|c|^2, i.e.
        Re(z/c) = (Re(z)*Re(c) + Im(z)*Im(c))/|c|^2,
        Im(z/c) = (Im(z)*Re(c) - Re(z)*Im(c))/|c|^2.
    The multiplication identity used by binary_canon is incorrect here.
    """
    re_c, im_c = real_args[1], imag_args[1]
    if im_c is None:
        # Real divisor: real and imaginary parts divide separately.
        return binary_canon(expr, real_args, imag_args, real2imag)
    if re_c is None:
        re_c = Constant(np.zeros(im_c.shape))
    abs_sq = multiply(re_c, re_c) + multiply(im_c, im_c)
    # Real and imaginary parts of 1/c = conj(c)/|c|^2.
    recip_real = re_c / abs_sq
    recip_imag = -im_c / abs_sq

    def mul(lh_arg, rh_arg):
        return None if lh_arg is None else multiply(lh_arg, rh_arg)

    real_output = add(mul(real_args[0], recip_real),
                      mul(imag_args[0], recip_imag), neg=True)
    imag_output = add(mul(real_args[0], recip_imag),
                      mul(imag_args[0], recip_real), neg=False)
    return real_output, imag_output
