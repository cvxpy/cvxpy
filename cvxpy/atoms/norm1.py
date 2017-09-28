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

import numpy as np
import scipy.sparse as sp

from cvxpy.atoms.axis_atom import AxisAtom


class norm1(AxisAtom):
    _allow_complex = True

    def numeric(self, values):
        """Returns the one norm of x.
        """
        if self.axis is None:
            values = np.array(values[0]).flatten()
        else:
            values = np.array(values[0])
        return np.linalg.norm(values, 1, axis=self.axis, keepdims=self.keepdims)

    def sign_from_args(self):
        """Returns sign (is positive, is negative) of the expression.
        """
        # Always positive.
        return (True, False)

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
        return self.args[0].is_nonneg()

    def is_decr(self, idx):
        """Is the composition non-increasing in argument idx?
        """
        return self.args[0].is_nonpos()

    def is_pwl(self):
        """Is the atom piecewise linear?
        """
        return self.args[0].is_pwl()

    def get_data(self):
        return [self.axis]

    def name(self):
        return "%s(%s)" % (self.__class__.__name__,
                           self.args[0].name())

    def _domain(self):
        """Returns constraints describing the domain of the node.
        """
        return []

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
            A NumPy ndarray matrix or None.
        """
        rows = self.args[0].size
        D_null = sp.csc_matrix((rows, 1), dtype='float64')
        D_null += (value > 0)
        D_null -= (value < 0)
        return sp.csc_matrix(D_null.A.ravel(order='F')).T
