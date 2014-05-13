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
from cvxpy.constraints.semi_definite import SDP
import scipy.sparse as sp
from numpy import linalg as LA

class sigma_max(Atom):
    """ Maximum singular value. """
    def __init__(self, A):
        super(sigma_max, self).__init__(A)

    # Returns the largest singular value of A.
    @Atom.numpy_numeric
    def numeric(self, values):
        return LA.norm(values[0], 2)

    # Resolves to a scalar.
    def shape_from_args(self):
        return u.Shape(1,1)

    # Always unknown.
    def sign_from_args(self):
        return u.Sign.POSITIVE

    # Default curvature.
    def func_curvature(self):
        return u.Curvature.CONVEX

    def monotonicity(self):
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
        A = arg_objs[0] # m by n matrix.
        n, m = A.size
        # Create a matrix with Schur complement I*t - (1/t)*A.T*A.
        X = lu.create_var((n+m, n+m))
        t = lu.create_var((1, 1))
        I_n = lu.create_const(sp.eye(n), (n, n))
        I_m = lu.create_const(sp.eye(m), (m, m))
        # Expand A.T.
        obj, constraints = transpose.graph_implementation([A], (m, n))
        # Fix X using the fact that A must be affine by the DCP rules.
        # X[0:n, 0:n] == I_n*t
        index.block_eq(X, lu.mul_expr(I_n, t, (n, n)), constraints,
                       0, n, 0, n)
        # X[0:n, n:n+m] == A
        index.block_eq(X, A, constraints,
                       0, n, n, n+m)
        # X[n:n+m, 0:n] == obj
        index.block_eq(X, obj, constraints,
                       n, n+m, 0, n)
        # X[n:n+m, n:n+m] == I_m*t
        index.block_eq(X, lu.mul_expr(I_m, t, (m, m)), constraints,
                       n, n+m, n, n+m)
        # Add SDP constraint.
        return (t, constraints + [SDP(X)])
