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

from __future__ import division
import cvxpy.interface as intf
import warnings
from cvxpy.atoms.atom import Atom
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.expression import Expression
from .sum_squares import sum_squares
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
    scale : float
        induced matrix 2-norm of P
    M1, M2 : 2d ndarray
        A rectangular ndarray such that P = scale * (dot(M1, M1.T) - dot(M2, M2.T))

    """

    w, V = LA.eigh(P, lower=lower, check_finite=check_finite)

    if rcond is not None:
        cond = rcond
    if cond in (None, -1):
        t = V.dtype.char.lower()
        factor = {'f': 1e3, 'd': 1e6}
        cond = factor[t] * np.finfo(t).eps

    scale = max(np.absolute(w))
    w_scaled = w / scale
    maskp = w_scaled > cond
    maskn = w_scaled < -cond
    # TODO: allow indefinite quad_form
    if np.any(maskp) and np.any(maskn):
        warnings.warn("Forming a nonconvex expression quad_form(x, indefinite).")
    M1 = V[:, maskp] * np.sqrt(w_scaled[maskp])
    M2 = V[:, maskn] * np.sqrt(-w_scaled[maskn])
    return scale, M1, M2

def quad_form(x, P):
    """ Alias for :math:`x^T P x`.

    """
    x, P = map(Expression.cast_to_const, (x, P))
    # Check dimensions.
    n = P.shape[0]
    if P.shape[1] != n or x.shape != (n, 1):
        raise Exception("Invalid dimensions for arguments.")
    # P cannot be a parameter.
    if len(P.parameters()) > 0:
        raise Exception("P cannot be a parameter.")
    if x.is_constant():
        return x.T * P * x
    elif P.is_constant():
        P = intf.DEFAULT_NP_INTF.const_to_matrix(P.value)
        # Force symmetry
        P = (P + P.T) / 2.0
        scale, M1, M2 = _decomp_quad(P)
        ret = 0
        if M1.shape > 0:
            ret += scale * sum_squares(Constant(M1.T) * x)
        if M2.shape > 0:
            ret -= scale * sum_squares(Constant(M2.T) * x)
        return ret
    else:
        raise Exception("At least one argument to quad_form must be constant.")

class QuadForm(Atom):

    def __init__(self, x, P):
        self.x = x
        self.P = P
        self.P_eigvals = LA.eigvals(P) # cache eigenvalues

    @Atom.numpy_numeric
    def numeric(self, values):
        return np.dot(np.dot(values[0], values[1]), values[0])

    def validate_arguments(self):
        super(Quadratic, self).validate_arguments()
        if not P.is_constant():
            raise ValueError("P must be a constant matrix.")
        n = P.shape[0]
        if P.shape[1] != n or x.shape != (n, 1):
            raise ValueError("Invalid dimensions for arguments.")

    def sign_from_args(self):
        """Returns sign (is positive, is negative) of the expression.
        """
        return (self.is_atom_convex(), self.is_atom_concave())

    def is_atom_convex(self):
        """Is the atom convex?
        """
        return np.all(self.P_eigvals >= 0)

    def is_atom_concave(self):
        """Is the atom concave?
        """
        return np.all(self.P_eigvals <= 0)

    def is_incr(self, idx):
        """Is the composition non-decreasing in argument idx?
        """
        return self.is_pwl()

    def is_decr(self, idx):
        """Is the composition non-increasing in argument idx?
        """
        return self.is_pwl()

    def is_pwl(self):
        """Is the atom piecewise linear?
        """
        return np.count_nonzero(P) == 0

    def get_data(self):
        return [self.x, self.P]

    def name(self):
        return "%s(%s, %s)" % (self.__class__.__name__,
                               self.x,
                               self.P)
