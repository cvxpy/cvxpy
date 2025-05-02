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

from cvxpy.atoms.atom import Atom


class pf_eigenvalue(Atom):
    """The Perron-Frobenius eigenvalue of a positive matrix.

    For an elementwise positive matrix :math:`X`, this atom represents its
    spectral radius, i.e., the magnitude of its largest eigenvalue. Because
    :math:`X` is positive, the spectral radius equals its largest eigenvalue,
    which is guaranteed to be positive.

    This atom is log-log convex.

    Parameters
    ----------
    X : cvxpy.Expression
        A positive square matrix.
    """
    def __init__(self, X) -> None:
        super(pf_eigenvalue, self).__init__(X)
        self.args[0] = X

    def numeric(self, values):
        return np.max(np.abs(np.linalg.eig(values[0])[0]))

    def validate_arguments(self):
        """Verify that the argument is a square matrix."""
        if not self.args[0].ndim == 2 or self.args[0].shape[0] != self.args[0].shape[1]:
            raise ValueError(
                f"The argument {self.args[0].name()} to pf_eigenvalue must be a 2-d square array."
            )
    
    def name(self) -> str:
        return "%s(%s)" % (self.__class__.__name__, self.args[0])

    def shape_from_args(self) -> Tuple[int, ...]:
        """Returns the (row, col) shape of the expression.
        """
        return tuple()

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

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        return True

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return False

    def _grad(self, values) -> None:
        return None
