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
import cvxpy
from cvxpy.expressions.expression import Expression
from cvxpy.atoms.norm_nuc import normNuc
from cvxpy.atoms.sigma_max import sigma_max
from cvxpy.atoms.pnorm import pnorm
from cvxpy.atoms.norm1 import norm1
from cvxpy.atoms.norm_inf import norm_inf
from cvxpy.atoms.affine.vec import vec


def norm(x, p=2, axis=None):
    """Wrapper on the different norm atoms.

    Parameters
    ----------
    x : Expression or numeric constant
        The value to take the norm of.
    p : int or str, optional
        The type of norm.

    Returns
    -------
    Expression
        An Expression representing the norm.
    """
    x = Expression.cast_to_const(x)
    # matrix norms take precedence
    if axis is None and x.ndim == 2:
        if p == 1:  # matrix 1-norm
            return cvxpy.atoms.max(norm1(x, axis=0))
        elif p == 2:  # matrix 2-norm is largest singular value
            return sigma_max(x)
        elif p == 'nuc':  # the nuclear norm (sum of singular values)
            return normNuc(x)
        elif p == 'fro':  # Frobenius norm
            return pnorm(vec(x), 2)
        elif p in [np.inf, "inf", "Inf"]:  # the matrix infinity-norm
            return cvxpy.atoms.max(norm1(x, axis=1))
        else:
            raise RuntimeError('Unsupported matrix norm.')
    else:
        if p == 1 or x.is_scalar():
            return norm1(x, axis=axis)
        elif p in [np.inf, "inf", "Inf"]:
            return norm_inf(x, axis)
        else:
            return pnorm(x, p, axis)
