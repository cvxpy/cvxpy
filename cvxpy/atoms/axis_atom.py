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

import abc
from cvxpy.atoms.atom import Atom
import numpy as np
import scipy.sparse as sp


class AxisAtom(Atom):
    """
    An abstract base class for atoms that can be applied along an axis.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, expr, axis=None, keepdims: bool = False) -> None:
        self.axis = axis
        self.keepdims = keepdims
        super(AxisAtom, self).__init__(expr)

    def shape_from_args(self):
        """Depends on axis.
        """
        shape = list(self.args[0].shape)
        if self.keepdims and self.axis is None:
            shape = [1]*len(shape)
        elif self.keepdims and self.axis is not None:
            shape[self.axis] = 1
        elif not self.keepdims and self.axis is None:
            shape = []
        else:
            shape = shape[:self.axis] + shape[self.axis+1:]
        return tuple(shape)

    def get_data(self):
        """Returns the axis being summed.
        """
        return [self.axis, self.keepdims]

    def validate_arguments(self) -> None:
        """Checks that the new shape has the same number of entries as the old.
        """
        if self.axis is not None and self.axis > self.args[0].ndim:
            raise ValueError("Invalid argument for axis.")
        super(AxisAtom, self).validate_arguments()

    def _axis_grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

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
                D = sp.csc_matrix(D)
        else:
            m, n = self.args[0].shape
            if self.axis == 0:  # function apply to each column
                D = sp.csc_matrix((m*n, n), dtype=np.float)
                for i in range(n):
                    value = values[0][:, i]
                    d = self._column_grad(value).T
                    if d is None:
                        return [None]
                    else:
                        d = d.flatten()
                    row = np.linspace(i*n, i*n+m-1, m)  # [i*n, i*n+1, ..., i*n+m-1]
                    col = np.ones((m))*i
                    D = D + sp.csc_matrix((d, (row, col)),
                                          shape=(m*n, n))  # d must be 1-D
            else:  # function apply to each row
                values = np.transpose(values[0])
                D = sp.csc_matrix((m*n, m), dtype=np.float)
                for i in range(m):
                    value = values[:, i]
                    d = self._column_grad(value).T
                    if d is None:
                        return [None]
                    row = np.linspace(i, i+(n-1)*m, n)  # [0+i, m+i, ..., m(n-1)+i]
                    col = np.ones((n))*i
                    D = D + sp.csc_matrix((np.array(d)[0], (row, col)),
                                          shape=(m*n, m))  # d must be 1-D
        return [D]

    def _column_grad(self, value):
        """Gives the (sub/super)gradient of the atom w.r.t. a column argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            value: A numeric value for a column.

        Returns:
            A SciPy sparse matrix or None.
        """
        raise NotImplementedError()
