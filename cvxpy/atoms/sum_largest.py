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

from typing import Tuple

import numpy as np
import scipy.sparse as sp

import cvxpy.interface as intf
from cvxpy.atoms.atom import Atom


class sum_largest(Atom):
    """
    Sum of the largest k values in the expression X
    """

    def __init__(self, x, k) -> None:
        self.k = k
        super(sum_largest, self).__init__(x)

    def validate_arguments(self) -> None:
        """Verify that k is a positive number.
        """
        if self.k <= 0:
            raise ValueError("Second argument must be a positive number.")
        super(sum_largest, self).validate_arguments()

    def numeric(self, values):
        """
        Returns the sum of the k largest entries of the matrix.
        For non-integer k, uses linear interpolation.
        """
        value = values[0].flatten()
        n = len(value)
        k_floor = int(np.floor(self.k))
        k_frac = self.k - k_floor

        if k_floor > 0:
            # Get k_floor largest values
            indices = np.argpartition(-value, kth=min(k_floor, n) - 1)[:k_floor]
            result = value[indices].sum()
        else:
            result = 0.0

        # Add fractional part if needed
        if k_frac > 0 and k_floor < n:
            # Get the (k_floor + 1)-th largest value
            indices_next = np.argpartition(-value, kth=k_floor)[:k_floor + 1]
            # min of largest k_floor+1 is the (k_floor+1)-th largest
            next_value = value[indices_next].min()
            result += k_frac * next_value

        return result

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        # Grad: 1 for each of k_floor largest indices, k_frac for the next one
        value = intf.from_2D_to_1D(values[0].flatten().T)
        n = len(value)
        k_floor = int(np.floor(self.k))
        k_frac = self.k - k_floor

        D = np.zeros((self.args[0].shape[0]*self.args[0].shape[1], 1))

        if k_floor > 0:
            indices = np.argpartition(-value, kth=min(k_floor, n) - 1)[:k_floor]
            D[indices] = 1

        if k_frac > 0 and k_floor < n:
            # Find the (k_floor + 1)-th largest element
            indices_next = np.argpartition(-value, kth=k_floor)[:k_floor + 1]
            # The minimum of these is the (k_floor + 1)-th largest
            next_idx = indices_next[np.argmin(value[indices_next])]
            D[next_idx] = k_frac

        return [sp.csc_array(D)]

    def shape_from_args(self) -> Tuple[int, ...]:
        """Returns the (row, col) shape of the expression.
        """
        return tuple()

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        # Same as argument.
        return (self.args[0].is_nonneg(), self.args[0].is_nonpos())

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
        return True

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return False

    def is_pwl(self) -> bool:
        """Is the atom piecewise linear?
        """
        return all(arg.is_pwl() for arg in self.args)

    def get_data(self):
        """Returns the parameter k.
        """
        return [self.k]
