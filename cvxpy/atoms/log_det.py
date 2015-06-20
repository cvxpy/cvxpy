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
from cvxpy.atoms.elementwise.log import log
from cvxpy.atoms.affine.index import index
from cvxpy.atoms.affine.transpose import transpose
from cvxpy.constraints.semidefinite import SDP
from cvxpy.expressions.variables.semidef_var import Semidef
import numpy as np
from numpy import linalg as LA

class log_det(Atom):
    """:math:`\log\det A`

    """
    def __init__(self, A):
        super(log_det, self).__init__(A)

    @Atom.numpy_numeric
    def numeric(self, values):
        """Returns the logdet of SDP matrix A.

        For SDP matrix A, this is the sum of logs of eigenvalues of A
        and is equivalent to the nuclear norm of the matrix logarithm of A.
        """
        sign, logdet = LA.slogdet(values[0])
        if sign == 1:
            return logdet
        else:
            return -np.inf

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

    @staticmethod
    def graph_implementation(arg_objs, size, data=None):
        """Reduces the atom to an affine expression and list of constraints.

        Creates the equivalent problem::

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
        A = arg_objs[0] # n by n matrix.
        n, _ = A.size
        X = lu.create_var((2*n, 2*n))
        X, constraints = Semidef(2*n).canonical_form
        Z = lu.create_var((n, n))
        D = lu.create_var((n, 1))
        # Require that X and A are PSD.
        constraints += [SDP(A)]
        # Fix Z as upper triangular, D as diagonal,
        # and diag(D) as diag(Z).
        Z_lower_tri = lu.upper_tri(lu.transpose(Z))
        constraints.append(lu.create_eq(Z_lower_tri))
        # D[i, i] = Z[i, i]
        constraints.append(lu.create_eq(D, lu.diag_mat(Z)))
        # Fix X using the fact that A must be affine by the DCP rules.
        # X[0:n, 0:n] == D
        index.block_eq(X, lu.diag_vec(D), constraints, 0, n, 0, n)
        # X[0:n, n:2*n] == Z,
        index.block_eq(X, Z, constraints, 0, n, n, 2*n)
        # X[n:2*n, n:2*n] == A
        index.block_eq(X, A, constraints, n, 2*n, n, 2*n)
        # Add the objective sum(log(D[i, i])
        obj, constr = log.graph_implementation([D], (n, 1))
        return (lu.sum_entries(obj), constraints + constr)
