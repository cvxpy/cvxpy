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
from numpy.lib.array_utils import normalize_axis_index, normalize_axis_tuple

from cvxpy.atoms.atom import Atom


class AxisAtom(Atom):
    """
    An abstract base class for atoms that can be applied along an axis.
    """

    # Whether reducing over all axes is equivalent to axis=None.
    # True for reduction atoms (sum, max, min, etc.).
    # False for cumulative atoms (cumsum, cummax, cumprod) that preserve shape.
    _reduce_all_axes_to_none = True

    def __init__(
        self, expr, axis: None | int | tuple[int, ...] = None, keepdims: bool = False
    ) -> None:
        self.axis = axis
        self.keepdims = keepdims
        super(AxisAtom, self).__init__(expr)
        # Normalize axis after init so self.args is available.
        if self.axis is not None:
            ndim = len(self.args[0].shape)
            if ndim == 0:
                return  # 0D arrays: leave axis as-is for subclass to handle
            axes = normalize_axis_tuple(self.axis, ndim)
            if self._reduce_all_axes_to_none and len(axes) == ndim:
                self.axis = None
            elif len(axes) == 1:
                self.axis = axes[0]
            else:
                self.axis = axes

    def shape_from_args(self) -> Tuple[int, ...]:
        """
        Returns the shape of the atom after applying a function along an axis.
        Handles negative axis inputs by normalizing them to positive indices.
        """
        shape = list(self.args[0].shape)
        ndim = len(shape)
        if self.axis is None:
            return (1,) * len(shape) if self.keepdims else ()
        elif isinstance(self.axis, int):
            # Normalize negative axis
            axis = normalize_axis_index(self.axis, ndim)
            if self.keepdims:
                shape[axis] = 1
            else:
                shape = shape[:axis] + shape[axis+1:]
        else:
            # Normalize each axis in the list
            axes =  normalize_axis_tuple(self.axis, ndim)
            if self.keepdims:
                for axis in axes:
                    shape[axis] = 1
            else:
                shape[:] = [shape[i] for i in range(len(shape)) if i not in axes]
        return tuple(shape)

    def get_data(self):
        """
        Returns the axes and the keepdims parameter.
        """
        return [self.axis, self.keepdims]

    def validate_arguments(self) -> None:
        """
        Checks that each axis is within the valid range.
        """
        if self.axis is not None:
            axes = [self.axis] if isinstance(self.axis, int) else self.axis
            dim = self.args[0].ndim
            _ = normalize_axis_tuple(axes, dim)
        super(AxisAtom, self).validate_arguments()

    def _axis_grad(self, values) -> list[sp.csc_array] | None:
        """
        Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.
        Takes axis into account. Works for any number of dimensions.

        CVXPY convention: grad[i, j] = d(output_flat_F[j]) / d(input_flat_F[i])

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        if self.axis is None or self.args[0].ndim < 2:
            value = np.reshape(values[0].T, (self.args[0].size, 1))
            D = self._column_grad(value)
            if D is not None:
                D = sp.csc_array(D)
            return [D]

        input_shape = self.args[0].shape
        ndim = len(input_shape)

        # Normalize axis to tuple
        axis = self.axis
        axes = (axis,) if isinstance(axis, int) else tuple(axis)
        keep = [i for i in range(ndim) if i not in axes]

        reduce_dims = [input_shape[a] for a in axes]
        reduce_size = int(np.prod(reduce_dims))
        output_shape = tuple(input_shape[i] for i in keep)
        input_size = int(np.prod(input_shape))
        output_size = max(1, int(np.prod(output_shape)))

        # F-order strides: stride[k] = prod(input_shape[:k])
        f_strides = np.ones(ndim, dtype=int)
        for k in range(1, ndim):
            f_strides[k] = f_strides[k-1] * input_shape[k-1]

        # Flat input in F-order
        flat_input = values[0].ravel(order='F')

        # All output multi-indices in F-order
        if len(output_shape) == 0:
            out_multis = np.zeros((0, 1), dtype=int)
        else:
            out_multis = np.array(
                np.unravel_index(np.arange(output_size), output_shape, order='F')
            )  # shape: (len(keep), output_size)

        # All reduce-axis multi-indices
        reduce_multis = np.array(
            list(np.ndindex(*reduce_dims))
        )  # shape: (reduce_size, len(axes))

        all_rows = []
        all_cols = []
        all_data = []

        for j in range(output_size):
            om = out_multis[:, j]

            # Build input multi-indices: fix keep axes, vary reduce axes
            in_multis = np.zeros((reduce_size, ndim), dtype=int)
            for idx, k in enumerate(keep):
                in_multis[:, k] = om[idx]
            for idx, a in enumerate(axes):
                in_multis[:, a] = reduce_multis[:, idx]

            # Compute flat F-order indices for this fiber
            fiber_indices = in_multis @ f_strides
            fiber_values = flat_input[fiber_indices]

            d = self._column_grad(fiber_values.reshape(-1, 1))
            if d is None:
                return [None]
            d = np.asarray(d).flatten()

            all_rows.append(fiber_indices)
            all_cols.append(np.full(reduce_size, j, dtype=int))
            all_data.append(d)

        rows = np.concatenate(all_rows)
        cols = np.concatenate(all_cols)
        data = np.concatenate(all_data)
        D = sp.csc_array((data, (rows, cols)), shape=(input_size, output_size))
        return [D]

    def _column_grad(self, value):
        """
        Gives the (sub/super)gradient of the atom w.r.t. a column argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            value: A numeric value for a column.

        Returns:
            A SciPy sparse matrix or None.
        """
        raise NotImplementedError()
