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
from cvxpy.atoms.affine.index import index
from cvxpy.constraints.semidefinite import SDP
from numpy import linalg as LA
import numpy as np
import scipy.sparse as sp


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
        X = np.matrix(values[0])
        P = np.matrix(values[1])
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

            DP = P_inv.dot(X)
            DP = DP.dot(X.T)
            DP = DP.dot(P_inv)
            DP = -DP.T
            DP = sp.csc_matrix(DP.T.ravel(order='F')).T
            return [DX, DP]

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
        X = arg_objs[0]  # n by m matrix.
        P = arg_objs[1]  # n by n matrix.
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
