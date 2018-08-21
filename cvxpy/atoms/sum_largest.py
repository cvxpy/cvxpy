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
import cvxpy.interface as intf
import numpy as np
import scipy.sparse as sp


class sum_largest(Atom):
    """Sum of the largest k values in the matrix X.
    """

    def __init__(self, x, k):
        self.k = k
        super(sum_largest, self).__init__(x)

    def validate_arguments(self):
        """Verify that k is a positive integer.
        """
        if int(self.k) != self.k or self.k <= 0:
            raise ValueError("Second argument must be a positive integer.")
        super(sum_largest, self).validate_arguments()

    def numeric(self, values):
        """Returns the sum of the k largest entries of the matrix.
        """
        value = values[0].flatten()
        indices = np.argsort(-value)[:int(self.k)]
        return value[indices].sum()

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        # Grad: 1 for each of k largest indices.
        value = intf.from_2D_to_1D(values[0].flatten().T)
        indices = np.argsort(-value)[:int(self.k)]
        D = np.zeros((self.args[0].shape[0]*self.args[0].shape[1], 1))
        D[indices] = 1
        return [sp.csc_matrix(D)]

    def shape_from_args(self):
        """Returns the (row, col) shape of the expression.
        """
        return tuple()

    def sign_from_args(self):
        """Returns sign (is positive, is negative) of the expression.
        """
        # Same as argument.
        return (self.args[0].is_nonneg(), self.args[0].is_nonpos())

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
        return True

    def is_decr(self, idx):
        """Is the composition non-increasing in argument idx?
        """
        return False

    def get_data(self):
        """Returns the parameter k.
        """
        return [self.k]
