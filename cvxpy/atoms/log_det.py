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

from cvxpy.atoms.atom import Atom
import numpy as np
from numpy import linalg as LA
import scipy.sparse as sp


class log_det(Atom):
    """:math:`\log\det A`

    """

    def __init__(self, A):
        super(log_det, self).__init__(A)

    def numeric(self, values):
        """Returns the logdet of PSD matrix A.

        For PSD matrix A, this is the sum of logs of eigenvalues of A
        and is equivalent to the nuclear norm of the matrix logarithm of A.
        """
        sign, logdet = LA.slogdet(values[0])
        if sign == 1:
            return logdet
        else:
            return -np.inf

    # Any argument shape is valid.
    def validate_arguments(self):
        n, m = self.args[0].shape
        if n != m:
            raise TypeError("The argument to log_det must be a square matrix.")

    def shape_from_args(self):
        """Returns the (row, col) shape of the expression.
        """
        return tuple()

    def sign_from_args(self):
        """Returns sign (is positive, is negative) of the expression.
        """
        return (True, False)

    def is_atom_convex(self):
        """Is the atom convex?
        """
        return False

    def is_atom_concave(self):
        """Is the atom concave?
        """
        return True

    def is_incr(self, idx):
        """Is the composition non-decreasing in argument idx?
        """
        return False

    def is_decr(self, idx):
        """Is the composition non-increasing in argument idx?
        """
        return False

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        X = np.matrix(values[0])
        eigen_val = LA.eigvals(X)
        if np.min(eigen_val) > 0:
            # Grad: X^{-1}.T
            D = np.linalg.inv(X).T
            return [sp.csc_matrix(D.A.ravel(order='F')).T]
        # Outside domain.
        else:
            return [None]

    def _domain(self):
        """Returns constraints describing the domain of the node.
        """
        return [self.args[0] >> 0]
