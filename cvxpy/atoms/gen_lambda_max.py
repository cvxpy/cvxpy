"""
Copyright 2013 Steven Diamond

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

from cvxpy.atoms.atom import Atom
from scipy import linalg as LA


class gen_lambda_max(Atom):
    """ Maximum generalized eigenvalue; :math:`\\lambda_{\\max}(A, B)`.
    """

    def __init__(self, A, B):
        super(gen_lambda_max, self).__init__(A, B)

    def numeric(self, values):
        """Returns the largest generalized eigenvalue corresponding to A and B.

        Requires that A is symmetric, B is positive semidefinite.
        """
        lo = hi = self.args[0].shape[0]-1
        return LA.eigvalsh(a=values[0], b=values[1], eigvals=(lo, hi))[0]

    def _domain(self):
        """Returns constraints describing the domain of the node.
        """
        return [self.args[0].H == self.args[0], self.args[1].H == self.args[1],
                self.args[1] >> 0]

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        return NotImplemented

    def validate_arguments(self):
        """Verify that the argument A, B are square and of the same dimension.
        """
        if (not self.args[0].ndim == 2 or
                self.args[0].shape[0] != self.args[0].shape[1] or
                self.args[1].shape[0] != self.args[1].shape[1] or
                self.args[0].shape != self.args[1].shape):
            raise ValueError(
                "The arguments '%s' and '%s' to gen_lambda_max must "
                "be square and have the same dimensions." % (
                 self.args[0].name(), self.args[1].name()))

    def shape_from_args(self):
        """Returns the (row, col) shape of the expression.
        """
        return tuple()

    def sign_from_args(self):
        """Returns sign (is positive, is negative) of the expression.
        """
        return (False, False)

    def is_atom_convex(self):
        """Is the atom convex?
        """
        return False

    def is_atom_concave(self):
        """Is the atom concave?
        """
        return False

    def is_atom_quasiconvex(self):
        """Is the atom quasiconvex?
        """
        return True

    def is_atom_quasiconcave(self):
        """Is the atom quasiconcave?
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
