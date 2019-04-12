"""
Copyright 2013 Steven Diamond, 2017 Robin Verschueren

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

from __future__ import division

import warnings

import numpy as np
from scipy import linalg as LA
from cvxpy.atoms.atom import Atom
from cvxpy.expressions.expression import Expression
from cvxpy.interface.matrix_utilities import is_sparse
import scipy.sparse as sp


class CvxPyDomainError(Exception):
    pass


class QuadForm(Atom):
    _allow_complex = True

    def __init__(self, x, P):
        super(QuadForm, self).__init__(x, P)

    def numeric(self, values):
        prod = values[1].dot(values[0])
        if self.args[0].is_complex():
            return np.dot(np.conj(values[0]).T, prod)
        else:
            return np.dot(np.transpose(values[0]), prod)

    def validate_arguments(self):
        super(QuadForm, self).validate_arguments()
        n = self.args[1].shape[0]
        if self.args[1].shape[1] != n or self.args[0].shape not in [(n, 1), (n,)]:
            raise ValueError("Invalid dimensions for arguments.")

    def sign_from_args(self):
        """Returns sign (is positive, is negative) of the expression.
        """
        return (self.is_atom_convex(), self.is_atom_concave())

    def is_atom_convex(self):
        """Is the atom convex?
        """
        return self.args[1].is_psd()

    def is_atom_concave(self):
        """Is the atom concave?
        """
        return self.args[1].is_nsd()

    def is_atom_log_log_convex(self):
        """Is the atom log-log convex?
        """
        return True

    def is_atom_log_log_concave(self):
        """Is the atom log-log concave?
        """
        return False

    def is_incr(self, idx):
        """Is the composition non-decreasing in argument idx?
        """
        return (self.args[0].is_nonneg() and self.args[1].is_nonneg()) or \
               (self.args[0].is_nonpos() and self.args[1].is_nonneg())

    def is_decr(self, idx):
        """Is the composition non-increasing in argument idx?
        """
        return (self.args[0].is_nonneg() and self.args[1].is_nonpos()) or \
               (self.args[0].is_nonpos() and self.args[1].is_nonpos())

    def is_quadratic(self):
        """Is the atom quadratic?
        """
        return True

    def is_pwl(self):
        """Is the atom piecewise linear?
        """
        return False

    def name(self):
        return "%s(%s, %s)" % (self.__class__.__name__,
                               self.args[0],
                               self.args[1])

    def _grad(self, values):
        x = np.array(values[0])
        P = np.array(values[1])
        D = 2 * np.dot(P, x.T)
        return [sp.csc_matrix(D.ravel(order='F')).T]

    def shape_from_args(self):
        return tuple() if self.args[0].ndim == 0 else (1, 1)


class SymbolicQuadForm(Atom):
    """
    Symbolic form of QuadForm when quadratic matrix is not known (yet).
    """
    def __init__(self, x, P, expr):
        self.original_expression = expr
        super(SymbolicQuadForm, self).__init__(x, P)
        self.P = self.args[1]

    def get_data(self):
        return [self.original_expression]

    def _grad(self, values):
        return NotImplemented

    def is_atom_concave(self):
        return self.original_expression.is_atom_concave()

    def is_atom_convex(self):
        return self.original_expression.is_atom_convex()

    def is_decr(self, idx):
        return self.original_expression.is_decr(idx)

    def is_incr(self, idx):
        return self.original_expression.is_incr(idx)

    def shape_from_args(self):
        return self.original_expression.shape_from_args()

    def sign_from_args(self):
        return self.original_expression.sign_from_args()

    def is_quadratic(self):
        return True


def decomp_quad(P, cond=None, rcond=None, lower=True, check_finite=True):
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
    if is_sparse(P):
        P = np.array(P.todense())  # make dense (needs to happen for eigh).
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
    if not P.ndim == 2 or P.shape[0] != P.shape[1] or max(x.shape, (1,))[0] != P.shape[0]:
        raise Exception("Invalid dimensions for arguments.")
    # P cannot be a parameter.
    if x.is_constant():
        return x.H * P * x
    elif P.is_constant():
        return QuadForm(x, P)
    else:
        raise Exception("At least one argument to quad_form must be constant.")
