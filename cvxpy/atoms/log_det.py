"""
Copyright 2017 Steven Diamond

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

import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.atom import Atom
from cvxpy.atoms.elementwise.log import log
from cvxpy.atoms.affine.index import index
from cvxpy.constraints.semidefinite import SDP
from cvxpy.expressions.variables.semidef_var import Semidef
import numpy as np
from numpy import linalg as LA
import scipy.sparse as sp


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

    # Any argument size is valid.
    def validate_arguments(self):
        n, m = self.args[0].size
        if n != m:
            raise TypeError("The argument to log_det must be a square matrix.")

    def size_from_args(self):
        """Returns the (row, col) size of the expression.
        """
        return (1, 1)

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
        A = arg_objs[0]  # n by n matrix.
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
