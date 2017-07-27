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
from cvxpy.constraints.semidefinite import SDP
from scipy import linalg as LA
import numpy as np
import scipy.sparse as sp


class lambda_max(Atom):
    """ Maximum eigenvalue; :math:`\lambda_{\max}(A)`.

    """

    def __init__(self, A):
        super(lambda_max, self).__init__(A)

    @Atom.numpy_numeric
    def numeric(self, values):
        """Returns the largest eigenvalue of A.

        Requires that A be symmetric.
        """
        if not (values[0].T == values[0]).all():
            raise ValueError("lambda_max called on a non-symmetric matrix.")
        lo = hi = self.args[0].size[0]-1
        return LA.eigvalsh(values[0], eigvals=(lo, hi))

    def _domain(self):
        """Returns constraints describing the domain of the node.
        """
        return [self.args[0].T == self.args[0]]

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        w, v = LA.eigh(values[0])
        d = np.zeros(w.size)
        d[-1] = 1
        d = np.diag(d)
        D = v.dot(d).dot(v.T)
        return [sp.csc_matrix(D.ravel(order='F')).T]

    def validate_arguments(self):
        """Verify that the argument A is square.
        """
        if not self.args[0].size[0] == self.args[0].size[1]:
            raise ValueError("The argument '%s' to lambda_max must resolve to a square matrix."
                             % self.args[0].name())

    def size_from_args(self):
        """Returns the (row, col) size of the expression.
        """
        return (1, 1)

    def sign_from_args(self):
        """Returns sign (is positive, is negative) of the expression.
        """
        return (False, False)

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
        n, _ = A.size
        # SDP constraint.
        t = lu.create_var((1, 1))
        prom_t = lu.promote(t, (n, 1))
        # I*t - A
        expr = lu.sub_expr(lu.diag_vec(prom_t), A)
        return (t, [SDP(expr)])
