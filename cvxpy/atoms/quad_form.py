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

import cvxpy.interface as intf
from cvxpy.expressions.expression import Expression
from cvxpy.expressions.constants import Constant
from .norm import norm
from .elementwise.square import square
from scipy import linalg as LA
import numpy as np

class CvxPyDomainError(Exception):
    pass

def _decomp_quad(P, cond=None, rcond=None, lower=True, check_finite=True):
    """
    Compute a matrix decomposition.

    Compute sgn, scale, M such that P = sgn * scale * dot(M, M.T).
    The strategy of determination of eigenvalue negligibility follows
    the pinvh contributions from the scikit-learn project to scipy.

    Parameters
    ----------
    P : matrix or ndarray
        A real symmetric positive or negative (semi)definite input matrix
    cond, rcond : float, optional
        Cutoff for small eigenvalues.
        Singular values smaller than rcond * largest_eigenvalue
        are considered negligible.
        If None or -1, suitable machine precision is used (default).
    lower : bool, optional
        Whether the array data is taken from the lower or upper triangle of P.
        The default is to take it from the lower triangle.
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        The default is True; disabling may give a performance gain
        but may result in problems (crashes, non-termination) if the inputs
        contain infinities or NaNs.

    Returns
    -------
    sgn : -1 or 1
        1 if P is positive (semi)definite otherwise -1
    scale : float
        induced matrix 2-norm of P
    M : 2d ndarray
        A rectangular ndarray such that P = sgn * scale * dot(M, M.T)

    """
    w, V = LA.eigh(P, lower=lower, check_finite=check_finite)
    abs_w = np.absolute(w)
    sgn_w = np.sign(w)
    scale, sgn = max(zip(np.absolute(w), np.sign(w)))
    if rcond is not None:
        cond = rcond
    if cond in (None, -1):
        t = V.dtype.char.lower()
        factor = {'f': 1e3, 'd':1e6}
        cond = factor[t] * np.finfo(t).eps
    scaled_abs_w = abs_w / scale
    mask = scaled_abs_w > cond
    if np.any(w[mask] * sgn < 0):
        msg = 'P has both positive and negative eigenvalues.'
        raise CvxPyDomainError(msg)
    M = V[:, mask] * np.sqrt(scaled_abs_w[mask])
    return sgn, scale, M

def quad_form(x, P):
    """ Alias for :math:`x^T P x`.

    """
    x, P = map(Expression.cast_to_const, (x, P))
    # Check dimensions.
    n = P.size[0]
    if P.size[1] != n or x.size != (n,1):
        raise Exception("Invalid dimensions for arguments.")
    if x.is_constant():
        return x.T * P * x
    elif P.is_constant():
        np_intf = intf.get_matrix_interface(np.ndarray)
        P = np_intf.const_to_matrix(P.value)
        # P must be symmetric.
        if not np.allclose(P, P.T):
            msg = "P is not symmetric."
            raise CvxPyDomainError(msg)
        sgn, scale, M = _decomp_quad(P)
        return sgn * scale * square(norm(Constant(M.T) * x))
    else:
        raise Exception("At least one argument to quad_form must be constant.")
