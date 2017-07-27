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
import scipy.sparse as sp
from numpy import linalg as LA
import numpy as np


class sigma_max(Atom):
    """ Maximum singular value. """

    def __init__(self, A):
        super(sigma_max, self).__init__(A)

    @Atom.numpy_numeric
    def numeric(self, values):
        """Returns the largest singular value of A.
        """
        return LA.norm(values[0], 2)

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        # Grad: U diag(e_1) V.T
        U, s, V = LA.svd(values[0])
        ds = np.zeros(len(s))
        ds[0] = 1
        D = U.dot(np.diag(ds)).dot(V)
        return [sp.csc_matrix(D.ravel(order='F')).T]

    def size_from_args(self):
        """Returns the (row, col) size of the expression.
        """
        return (1, 1)

    def sign_from_args(self):
        """Returns sign (is positive, is negative) of the expression.
        """
        # Always positive.
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
        A = arg_objs[0]  # n by m matrix.
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
