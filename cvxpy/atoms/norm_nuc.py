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
import scipy.linalg
import numpy as np
import scipy.sparse as sp


class normNuc(Atom):
    """ Sum of the singular values. """

    def __init__(self, A):
        super(normNuc, self).__init__(A)

    @Atom.numpy_numeric
    def numeric(self, values):
        """Returns the nuclear norm (i.e. the sum of the singular values) of A.
        """
        return scipy.linalg.svdvals(values[0]).sum()

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        # Grad UV^T
        U, _, V = np.linalg.svd(values[0])
        D = U.dot(V)
        return [sp.csc_matrix(D.A.ravel(order='F')).T]

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
        A = arg_objs[0]
        rows, cols = A.size
        # Create the equivalent problem:
        #   minimize (trace(U) + trace(V))/2
        #   subject to:
        #            [U A; A.T V] is positive semidefinite
        X = lu.create_var((rows+cols, rows+cols))
        constraints = []
        # Fix X using the fact that A must be affine by the DCP rules.
        # X[0:rows,rows:rows+cols] == A
        index.block_eq(X, A, constraints,
                       0, rows, rows, rows+cols)
        half = lu.create_const(0.5, (1, 1))
        trace = lu.mul_expr(half, lu.trace(X), (1, 1))
        # Add SDP constraint.
        return (trace, [SDP(X)] + constraints)
