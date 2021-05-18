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

import numpy as np
import scipy.sparse as sp

from cvxpy.atoms.axis_atom import AxisAtom
from typing import Tuple


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

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        # Always positive.
        return (True, False)

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
        return self.args[0].is_nonneg()

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return self.args[0].is_nonpos()

    def is_pwl(self) -> bool:
        """Is the atom piecewise linear?
        """
        return self.args[0].is_pwl() and \
            (self.args[0].is_real() or self.args[0].is_imag())

    def get_data(self):
        return [self.axis]

    def name(self) -> str:
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
