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
from typing import Tuple

import numpy as np
import scipy.sparse as sp


class lambda_max(Atom):
    """ Maximum eigenvalue; :math:`\\lambda_{\\max}(A)`.
    """

    def __init__(self, A) -> None:
        super(lambda_max, self).__init__(A)

    def numeric(self, values):
        """Returns the largest eigenvalue of A.

        Requires that A be symmetric.
        """
        lo = hi = self.args[0].shape[0]-1
        return LA.eigvalsh(values[0], eigvals=(lo, hi))[0]

    def _domain(self):
        """Returns constraints describing the domain of the node.
        """
        return [self.args[0].H == self.args[0]]

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        w, v = LA.eigh(values[0])
        d = np.zeros(w.shape)
        d[-1] = 1
        d = np.diag(d)
        D = v.dot(d).dot(v.T)
        return [sp.csc_matrix(D.ravel(order='F')).T]

    def validate_arguments(self) -> None:
        """Verify that the argument A is square.
        """
        if not self.args[0].ndim == 2 or self.args[0].shape[0] != self.args[0].shape[1]:
            raise ValueError("The argument '%s' to lambda_max must resolve to a square matrix."
                             % self.args[0].name())

    def shape_from_args(self):
        """Returns the (row, col) shape of the expression.
        """
        return tuple()

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        return (False, False)

    def is_atom_convex(self) -> bool:
        """Is the atom convex?
        """
        return True

    def is_atom_concave(self) -> bool:
        """Is the atom concave?
        """
        return False

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        return False

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return False

    @property
    def value(self):
        if not np.allclose(self.args[0].value, self.args[0].value.T.conj()):
            raise ValueError("Input matrix was not Hermitian/symmetric.")
        if any([p.value is None for p in self.parameters()]):
            return None
        return self._value_impl()
