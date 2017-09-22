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

import numpy as np


def separable_canon(expr, args):
    """Canonicalize linear functions that are seprable
       in real and imaginary parts.
    """
    reals = []
    imags = []
    for real, imag in args:
        reals.append(real)
        imags.append(imag)

    if all([val is None for val in imags]):
        outputs = (expr.copy(reals), None)
    elif all([val is None for val in reals]):
        outputs = (None, expr.copy(imags))
    else:  # Mixed reals and imaginaries.
        for idx, real_val in enumerate(reals):
            if real_val is None:
                reals[idx] = np.zeros(imags[idx].shape)
            elif imags[idx] is None:
                imags[idx] = np.zeros(reals[idx].shape)
        outputs = (expr.copy(reals), expr.copy(imags))
    return outputs


def real_canon(expr, args):
    return args[0][0], None


def imag_canon(expr, args):
    return None, args[0][1]


def conj_canon(expr, args):
    return args[0][0], -args[0][1]
