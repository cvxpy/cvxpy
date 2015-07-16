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

class sigma_max(Atom):
    """ Maximum singular value. """
    def __init__(self, A):
        super(sigma_max, self).__init__(A)

    @Atom.numpy_numeric
    def numeric(self, values):
        """Returns the largest singular value of A.
        """
        return LA.norm(values[0], 2)

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
        return [u.monotonicity.NONMONOTONIC]

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
        A = arg_objs[0] # n by m matrix.
        n, m = A.size
        # Create a matrix with Schur complement I*t - (1/t)*A.T*A.
        X = lu.create_var((n+m, n+m))
        t = lu.create_var((1, 1))
        constraints = []
        # Fix X using the fact that A must be affine by the DCP rules.
        # X[0:n, 0:n] == I_n*t
        prom_t = lu.promote(t, (n, 1))
        index.block_eq(X, lu.diag_vec(prom_t), constraints,
                       0, n, 0, n)
        # X[0:n, n:n+m] == A
        index.block_eq(X, A, constraints,
                       0, n, n, n+m)
        # X[n:n+m, n:n+m] == I_m*t
        prom_t = lu.promote(t, (m, 1))
        # prom_t = lu.promote(lu.create_const(1, (1,1)), (m, 1))
        index.block_eq(X, lu.diag_vec(prom_t), constraints,
                       n, n+m, n, n+m)
        # Add SDP constraint.
        return (t, constraints + [SDP(X)])
