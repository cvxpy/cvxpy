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

from atom import Atom
from elementwise.log import log
from .. import utilities as u
from .. import interface as intf
from ..expressions.variables import Variable
from ..constraints.semi_definite import SDP
import numpy as np
from numpy import linalg as LA

class log_det(Atom):
    """:math:`\log\det A`

    """
    def __init__(self, A):
        super(log_det, self).__init__(A)

    # Returns the nuclear norm (i.e. the sum of the singular values) of A.
    @Atom.numpy_numeric
    def numeric(self, values):
        return np.log(LA.det(values[0]))

    # Resolves to a scalar.
    def shape_from_args(self):
        return u.Shape(1,1)

    # Always positive.
    def sign_from_args(self):
        return u.Sign.UNKNOWN

    # Any argument size is valid.
    def validate_arguments(self):
        n, m = self.args[0].size
        if n != m:
            raise TypeError("The argument to log_det must be a square matrix." )

    # Default curvature.
    def func_curvature(self):
        return u.Curvature.CONCAVE

    def monotonicity(self):
        return [u.monotonicity.NONMONOTONIC]

    def graph_implementation(self, arg_objs):
        """Creates the equivalent problem::

           maximize    sum(log(D[i, i]))
           subject to: D diagonal
                       diag(D) = diag(Z)
                       Z is upper triangular.
                       [D Z; Z.T A] is positive semidefinite

        The problem computes the LDL factorization:

        .. math::

           A = (Z^TD^{-1})D(D^{-1}Z)

        This follows from the inequality:

        .. math::

           \det(A) >= \det(D) + \det([D, Z; Z^T, A])/\det(D)
                   >= \det(D)

        because (Z^TD^{-1})D(D^{-1}Z) is a feasible D, Z that achieves
        det(A) = det(D) and the objective maximizes det(D).
        """
        A = arg_objs[0] # n by n matrix.
        n, _ = A.size
        X = Variable(2*n, 2*n)
        Z = Variable(n, n)
        D = Variable(n, n)
        # Require that X is symmetric (which implies
        # A is symmetric).
        constraints = (X == X.T).canonical_form[1]
        # Require that X and A are PSD.
        constraints += [SDP(X), SDP(A)]
        # Fix Z as upper triangular, D as diagonal,
        # and diag(D) as diag(Z).
        for i in xrange(n):
            for j in xrange(n):
                if i == j:
                    constraints.append( D[i, j] == Z[i, j] )
                if i != j:
                    constraints.append( D[i, j] == 0 )
                if i > j:
                    constraints.append( Z[i, j] == 0 )
        # Fix X using the fact that A must be affine by the DCP rules.
        constraints += [X[0:n, 0:n] == D,
                        X[0:n, n:2*n] == Z,
                        X[n:2*n, n:2*n] == A]
        # Add the objective.
        D_diag = sum(log(D[i, i]) for i in xrange(n))
        obj, constr = D_diag.canonical_form
        return (obj, constraints + constr)
