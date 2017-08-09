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
from numpy import linalg as LA
import numpy as np
import scipy.sparse as sp
from cvxpy.atoms.quad_form import QuadForm


def matrix_frac(X, P):
    if isinstance(P, np.ndarray):
        return QuadForm(X, LA.inv(P))
    else:
        return MatrixFrac(X, P)


class MatrixFrac(Atom):
    """ tr X.T*P^-1*X """

    def __init__(self, X, P):
        super(MatrixFrac, self).__init__(X, P)

    @Atom.numpy_numeric
    def numeric(self, values):
        """Returns tr X.T*P^-1*X.
        """
        # TODO raise error if not invertible?
        X = values[0]
        P = values[1]
        product = X.T.dot(LA.inv(P)).dot(X)
        return product.trace() if len(product.shape) == 2 else product

    def _domain(self):
        """Returns constraints describing the domain of the node.
        """
        return [self.args[1] >> 0]

    def _grad(self, values):
        """
        Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        X = np.array(values[0])
        print X
        if X.ndim == 1:
            X = X[:, None]
        P = np.array(values[1])
        try:
            P_inv = LA.inv(P)
        except LA.LinAlgError:
            return [None, None]
        # partial_X = (P^-1+P^-T)X
        # partial_P = - (P^-1 * X * X^T * P^-1)^T
        else:
            DX = np.dot(P_inv+np.transpose(P_inv), X)
            DX = DX.T.ravel(order='F')
            DX = sp.csc_matrix(DX).T

            DP = np.dot(P_inv, X)
            DP = np.dot(DP, X.T)
            DP = np.dot(DP, P_inv)
            DP = -DP.T
            DP = sp.csc_matrix(DP.T.ravel(order='F')).T
            return [DX, DP]

    def validate_arguments(self):
        """Checks that the dimensions of x and P match.
        """
        X = self.args[0]
        P = self.args[1]
        if P.ndim != 2 or P.shape[0] != P.shape[1]:
            raise ValueError(
                "The second argument to matrix_frac must be a square matrix."
            )
        elif X.shape[0] != P.shape[0]:
            raise ValueError(
                "The arguments to matrix_frac have incompatible dimensions."
            )

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
        return True

    def is_atom_concave(self):
        """Is the atom concave?
        """
        return False

    def is_incr(self, idx):
        """Is the composition non-decreasing in argument idx?
        """
        return False

    def is_decr(self, idx):
        """Is the composition non-increasing in argument idx?
        """
        return False

    def is_quadratic(self):
        """Quadratic if x is affine and P is constant.
        """
        return self.args[0].is_affine() and self.args[1].is_constant()

    def is_qpwa(self):
        """Quadratic of piecewise affine if x is PWL and P is constant.
        """
        return self.args[0].is_pwl() and self.args[1].is_constant()
