"""
Copyright, the CVXPY authors

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


class dist_ratio(Atom):
    """norm(x - a)_2 / norm(x - b)_2, with norm(x - a)_2 <= norm(x - b).

    `a` and `b` must be constants.
    """
    def __init__(self, x, a, b):
        super(dist_ratio, self).__init__(x, a, b)
        if not self.args[1].is_constant():
            raise ValueError("`a` must be a constant.")
        if not self.args[2].is_constant():
            raise ValueError("`b` must be a constant.")
        self.a = self.args[1].value
        self.b = self.args[2].value

    def numeric(self, values):
        """Returns the distance ratio.
        """
        return np.linalg.norm(
            values[0] - self.a) / np.linalg.norm(values[0] - self.b)

    def shape_from_args(self):
        """Returns the (row, col) shape of the expression.
        """
        return tuple()

    def sign_from_args(self):
        """Returns sign (is positive, is negative) of the expression.
        """
        # Always nonnegative.
        return (True, False)

    def is_atom_convex(self):
        """Is the atom convex?
        """
        return False

    def is_atom_concave(self):
        """Is the atom concave?
        """
        return False

    def is_atom_quasiconvex(self):
        """Is the atom quasiconvex?
        """
        return True

    def is_atom_quasiconcave(self):
        """Is the atom quasiconvex?
        """
        return False

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
