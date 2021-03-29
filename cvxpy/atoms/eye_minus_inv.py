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

from cvxpy.atoms.atom import Atom
from typing import Tuple

import numpy as np


def resolvent(X, s: float):
    r"""The resolvent of a positive matrix, :math:`(sI - X)^{-1}`.

    For an elementwise positive matrix :math:`X` and a positive scalar
    :math:`s`, this atom computes

    .. math::

        (sI - X)^{-1},

    and it enforces the constraint that the spectral radius of :math:`X/s`
    is at most :math:`1`.

    This atom is log-log convex.

    Parameters
    ----------
    X : cvxpy.Expression
        A positive square matrix.
    s : cvxpy.Expression or numeric
        A positive scalar.
    """
    return 1.0 / s * eye_minus_inv(X / s)


class eye_minus_inv(Atom):
    r"""The unity resolvent of a positive matrix, :math:`(I - X)^{-1}`.

    For an elementwise positive matrix :math:`X`, this atom represents

    .. math::

        (I - X)^{-1},

    and it enforces the constraint that the spectral radius of :math:`X`
    is at most :math:`1`.

    This atom is log-log convex.

    Parameters
    ----------
    X : cvxpy.Expression
        A positive square matrix.
    """
    def __init__(self, X) -> None:
        super(eye_minus_inv, self).__init__(X)
        if len(X.shape) != 2 or X.shape[0] != X.shape[1]:
            raise ValueError("The argument to `eye_minus_inv` must be a "
                             "square matrix, received ", X)
        self.args[0] = X

    def numeric(self, values):
        return np.linalg.inv(np.eye(self.args[0].shape[0]) - values[0])

    def name(self) -> str:
        return "%s(%s)" % (self.__class__.__name__, self.args[0])

    def shape_from_args(self):
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
        return True

    def is_atom_log_log_concave(self) -> bool:
        """Is the atom log-log concave?
        """
        return False

    # TODO(akshayka): Figure out monotonicity.
    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        return False

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return False

    def _grad(self, values) -> None:
        return None
