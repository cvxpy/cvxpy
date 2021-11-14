"""
Copyright 2018 Akshay Agrawal

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

from cvxpy.atoms.affine.binary_operators import multiply
from cvxpy.atoms.atom import Atom


def diff_pos(x, y):
    r"""The difference :math:`x - y` with domain `\{x, y : x > y > 0\}`.

    This atom is log-log concave.

    Parameters
    ----------
    x : :class:`~cvxpy.expressions.expression.Expression`
        An Expression.
    y : :class:`~cvxpy.expressions.expression.Expression`
        An Expression.
    """
    return multiply(x, one_minus_pos(y/x))


class one_minus_pos(Atom):
    r"""The difference :math:`1 - x` with domain `\{x : 0 < x < 1\}`.

    This atom is log-log concave.

    Parameters
    ----------
    x : :class:`~cvxpy.expressions.expression.Expression`
        An Expression.
    """
    def __init__(self, x) -> None:
        super(one_minus_pos, self).__init__(x)
        self.args[0] = x
        self._ones = np.ones(self.args[0].shape)

    def numeric(self, values):
        return self._ones - values[0]

    def _grad(self, values):
        del values
        return sp.csc_matrix(-1.0 * self._ones)

    def name(self) -> str:
        return "%s(%s)" % (self.__class__.__name__, self.args[0])

    def shape_from_args(self) -> Tuple[int, ...]:
        """Returns the (row, col) shape of the expression.
        """
        return self.args[0].shape

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        return (True, False)

    def is_atom_convex(self) -> bool:
        """Is the atom convex?
        """
        return False

    def is_atom_concave(self) -> bool:
        """Is the atom concave?
        """
        return False

    def is_atom_log_log_convex(self) -> bool:
        """Is the atom log-log convex?
        """
        return False

    def is_atom_log_log_concave(self) -> bool:
        """Is the atom log-log concave?
        """
        return True

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        return False

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return True
