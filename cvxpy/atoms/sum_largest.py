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

from cvxpy.atoms.axis_atom import AxisAtom


class sum_largest(AxisAtom):
    """
    Sum of the largest k values in the expression X, optionally along an axis.
    """

    def __init__(self, x, k, axis=None, keepdims: bool = False) -> None:
        self.k = k
        super(sum_largest, self).__init__(x, axis=axis, keepdims=keepdims)

    def validate_arguments(self) -> None:
        """Verify that k is a positive number.
        """
        if self.k <= 0:
            raise ValueError("Second argument must be a positive number.")
        super(sum_largest, self).validate_arguments()

    def numeric(self, values):
        """
        Returns the sum of the k largest entries, optionally along an axis.
        For non-integer k, uses linear interpolation.
        """
        x = values[0]
        k_floor = int(np.floor(self.k))
        k_frac = self.k - k_floor

        if self.axis is None:
            value = x.flatten()
            n = len(value)
            result = 0.0

            if k_floor > 0:
                indices = np.argpartition(-value, kth=min(k_floor, n) - 1)[:k_floor]
                result = value[indices].sum()

            if k_frac > 0 and k_floor < n:
                indices_next = np.argpartition(-value, kth=k_floor)[:k_floor + 1]
                next_value = value[indices_next].min()
                result += k_frac * next_value

            return result

        # For axis reduction, normalize axis and move reduction axes to end
        if isinstance(self.axis, int):
            axes = (self.axis,)
        else:
            axes = tuple(self.axis)

        keep = [i for i in range(x.ndim) if i not in axes]
        perm = keep + list(axes)
        x_t = np.transpose(x, perm)
        out_shape = x_t.shape[:len(keep)]
        x_flat = x_t.reshape(out_shape + (-1,))

        n = x_flat.shape[-1]
        result = np.zeros(out_shape)

        if k_floor > 0:
            kth = min(k_floor, n) - 1
            neg_part = np.partition(-x_flat, kth=kth, axis=-1)
            result = -neg_part[..., :k_floor].sum(axis=-1)

        if k_frac > 0 and k_floor < n:
            neg_part2 = np.partition(-x_flat, kth=k_floor, axis=-1)
            next_val = -neg_part2[..., k_floor]
            result = result + k_frac * next_val

        if self.keepdims:
            for a in sorted(axes):
                result = np.expand_dims(result, a)

        return result

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        return self._axis_grad(values)

    def _column_grad(self, value):
        """Gives the (sub/super)gradient of the atom w.r.t. a column argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            value: A numeric value for a column.

        Returns:
            A NumPy ndarray or None.
        """
        value = np.array(value).ravel(order='F')
        n = len(value)
        k_floor = int(np.floor(self.k))
        k_frac = self.k - k_floor

        D = np.zeros((n, 1))

        if k_floor > 0:
            indices = np.argpartition(-value, kth=min(k_floor, n) - 1)[:k_floor]
            D[indices] = 1

        if k_frac > 0 and k_floor < n:
            indices_next = np.argpartition(-value, kth=k_floor)[:k_floor + 1]
            next_idx = indices_next[np.argmin(value[indices_next])]
            D[next_idx] = k_frac

        return D

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
        """Returns the parameter k, axis, and keepdims.
        """
        return [self.k, self.axis, self.keepdims]
