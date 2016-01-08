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

import cvxpy.utilities as u
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.atom import Atom
from cvxpy.atoms.affine.index import index
from cvxpy.atoms.affine.transpose import transpose
from cvxpy.constraints.semidefinite import SDP
import scipy.sparse as sp
from numpy import linalg as LA

class matrix_frac(Atom):
    """ tr X.T*P^-1*X """
    def __init__(self, X, P):
        super(matrix_frac, self).__init__(X, P)

    @Atom.numpy_numeric
    def numeric(self, values):
        """Returns tr X.T*P^-1*X.
        """
        # TODO raise error if not invertible?
        X = values[0]
        P = values[1]
        return (X.T.dot(LA.inv(P)).dot(X)).trace()

    def validate_arguments(self):
        """Checks that the dimensions of x and P match.
        """
        X = self.args[0]
        P = self.args[1]
        if P.size[0] != P.size[1]:
            raise ValueError(
                "The second argument to matrix_frac must be a square matrix."
            )
        elif X.size[0] != P.size[0]:
            raise ValueError(
                "The arguments to matrix_frac have incompatible dimensions."
            )

    def shape_from_args(self):
        """Resolves to a scalar.
        """
        return u.Shape(1, 1)

    def sign_from_args(self):
        """Always positive.
        """
        return u.Sign.POSITIVE

    def func_curvature(self):
        """Default curvature.
        """
        return u.Curvature.CONVEX

    def monotonicity(self):
        """Neither increasing nor decreasing.
        """
        return len(self.args)*[u.monotonicity.NONMONOTONIC]

    @staticmethod
    def graph_implementation(arg_objs, size, data=None):
        """Reduces the atom to an affine expression and list of constraints.

        Parameters
        ----------
        arg_objs : list
            LinExpr for each argument.
        size : tuple
            The size of the resulting expression.
        data :
            Additional data required by the atom.

        Returns
        -------
        tuple
            (LinOp for objective, list of constraints)
        """
        X = arg_objs[0] # n by m matrix.
        P = arg_objs[1] # n by n matrix.
        n, m = X.size
        # Create a matrix with Schur complement T - X.T*P^-1*X.
        M = lu.create_var((n + m, n + m))
        T = lu.create_var((m, m))
        constraints = []
        # Fix M using the fact that P must be affine by the DCP rules.
        # M[0:n, 0:n] == P.
        index.block_eq(M, P, constraints,
                       0, n, 0, n)
        # M[0:n, n:n+m] == X
        index.block_eq(M, X, constraints,
                       0, n, n, n+m)
        # M[n:n+m, n:n+m] == T
        index.block_eq(M, T, constraints,
                       n, n+m, n, n+m)
        # Add SDP constraint.
        return (lu.trace(T), constraints + [SDP(M)])
