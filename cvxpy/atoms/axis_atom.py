"""
Copyright 2013 Steven Diamond

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
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

    def __init__(self, expr, axis=None):
        self.axis = axis
        super(AxisAtom, self).__init__(expr)

    def size_from_args(self):
        """Depends on axis.
        """
        if self.axis is None:
            return (1, 1)
        elif self.axis == 0:
            return (1, self.args[0].size[1])
        else:  # axis == 1.
            return (self.args[0].size[0], 1)

    def get_data(self):
        """Returns the axis being summed.
        """
        return [self.axis]

    def validate_arguments(self):
        """Checks that the new shape has the same number of entries as the old.
        """
        if self.axis is not None and self.axis not in [0, 1]:
            raise ValueError("Invalid argument for axis.")

    def _axis_grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.
        Takes axis into account.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        m = self.args[0].size[0]
        n = self.args[0].size[1]
        if self.axis is None:
            value = np.reshape(values[0].T, (m*n, 1))
            D = self._column_grad(value)
            if D is not None:
                D = sp.csc_matrix(D)
        else:
            if self.axis == 0:  # function apply to each column
                D = sp.csc_matrix((m*n, n), dtype=np.float)
                for i in range(n):
                    value = values[0][:, i]
                    d = self._column_grad(value).T
                    if d is None:
                        return [None]
                    row = np.linspace(i*n, i*n+m-1, m)  # [i*n, i*n+1, ..., i*n+m-1]
                    col = np.ones((m))*i
                    D = D + sp.csc_matrix((np.array(d)[0], (row, col)),
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
        return NotImplemented
