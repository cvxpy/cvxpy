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
import numpy as np


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
    def __init__(self, X):
        super(pf_eigenvalue, self).__init__(X)
        if len(X.shape) != 2 or X.shape[0] != X.shape[1]:
            raise ValueError("Argument to `spectral radius` must be a "
                             "square matrix, received ", X)
        self.args[0] = X

    def numeric(self, values):
        return np.max(np.abs(np.linalg.eig(values[0])[0]))

    def name(self):
        return "%s(%s)" % (self.__class__.__name__, self.args[0])

    def shape_from_args(self):
        """Returns the (row, col) shape of the expression.
        """
        return tuple()

    def sign_from_args(self):
        """Returns sign (is positive, is negative) of the expression.
        """
        return (True, False)

    def is_atom_convex(self):
        """Is the atom convex?
        """
        return False

    def is_atom_concave(self):
        """Is the atom concave?
        """
        return False

    def is_atom_log_log_convex(self):
        """Is the atom log-log convex?
        """
        return True

    def is_atom_log_log_concave(self):
        """Is the atom log-log concave?
        """
        return False

    # TODO(akshayka): Figure out monotonicity.
    def is_incr(self, idx):
        """Is the composition non-decreasing in argument idx?
        """
        return False

    def is_decr(self, idx):
        """Is the composition non-increasing in argument idx?
        """
        return False

    def _grad(self, values):
        return None
