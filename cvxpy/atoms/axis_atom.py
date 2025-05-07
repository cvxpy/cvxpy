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

from typing import List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

from cvxpy.atoms.atom import Atom


class AxisAtom(Atom):
    """
    An abstract base class for atoms that can be applied along an axis.
    """

    def __init__(self, expr, axis: Optional[int] = None, keepdims: bool = False) -> None:
        self.axis = axis
        self.keepdims = keepdims
        super(AxisAtom, self).__init__(expr)

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
            axis = self.axis if self.axis >= 0 else self.axis + ndim
            if axis < 0:
                ValueError(f"axis {self.axis} is out of bounds for array of dimension {ndim}")
            if self.keepdims:
                shape[axis] = 1
            else:
                shape = shape[:axis] + shape[axis+1:]
        else:
            # Normalize each axis in the list
            axes = [axis if axis >= 0 else axis + ndim for axis in self.axis]
            if any(axis < 0 for axis in axes):
                ValueError(f"axis {[axis for axis in self.axis if axis < -ndim][0]}"
                           f" is out of bounds for array of dimension {ndim}")
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
            for axis in axes:
                if axis < 0:
                    axis += dim
                if axis >= dim or axis < 0:
                    raise ValueError(f"axis {axis} is out of bounds for array of dimension {dim}")
        super(AxisAtom, self).validate_arguments()

    def _axis_grad(self, values) -> Optional[List[sp.csc_array]]:
        """
        Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.
        Takes axis into account.

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
        else:
            m, n = self.args[0].shape
            if self.axis == 0:  # function apply to each column
                D = sp.csc_array((m*n, n), dtype=float)
                for i in range(n):
                    value = values[0][:, i]
                    d = self._column_grad(value).T
                    if d is None:
                        return [None]
                    else:
                        d = np.array(d).flatten()
                    row = np.linspace(i*m, i*m+m-1, m)  # [i*m, i*m+1, ..., i*m+m-1]
                    col = np.ones((m))*i
                    D = D + sp.csc_array((d, (row, col)),
                                          shape=(m*n, n))  # d must be 1-D
            else:  # function apply to each row
                values = np.transpose(values[0])
                D = sp.csc_array((m*n, m), dtype=float)
                for i in range(m):
                    value = values[:, i]
                    d = self._column_grad(value).T
                    if d is None:
                        return [None]
                    row = np.linspace(i, i+(n-1)*m, n)  # [0+i, m+i, ..., m(n-1)+i]
                    col = np.ones((n))*i
                    D = D + sp.csc_array((np.array(d)[0], (row, col)),
                                          shape=(m*n, m))  # d must be 1-D
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
