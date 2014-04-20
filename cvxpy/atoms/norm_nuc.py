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
from numpy import linalg as LA

class normNuc(Atom):
    """ Sum of the singular values. """
    def __init__(self, A):
        super(normNuc, self).__init__(A)

    # Returns the nuclear norm (i.e. the sum of the singular values) of A.
    @Atom.numpy_numeric
    def numeric(self, values):
        U,s,V = LA.svd(values[0])
        return sum(s)

    # Resolves to a scalar.
    def shape_from_args(self):
        return u.Shape(1,1)

    # Always positive.
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
        # Create the equivalent problem:
        #   minimize (trace(U) + trace(V))/2
        #   subject to:
        #            [U A; A.T V] is positive semidefinite
        X = lu.create_var((n+m, n+m))
        # Expand A.T.
        obj, constraints = transpose.graph_implementation([A], (m, n))
        # Fix X using the fact that A must be affine by the DCP rules.
        # X[0:n,n:n+m] == A
        index.block_eq(X, A, constraints,
                       0, n, n, n+m)
        # X[n:n+m,0:n] == obj
        index.block_eq(X, obj, constraints,
                       n, n+m, 0, n)
        diag = [index.get_index(X, constraints, i, i) for i in range(n+m)]
        half = lu.create_const(0.5, (1, 1))
        trace = lu.mul_expr(half, lu.sum_expr(diag), (1, 1))
        # Add SDP constraint.
        return (trace, [SDP(X)] + constraints)
