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


class sign(Atom):
    """Sign of an expression (-1 for x <= 0, +1 for x > 0).
    """
    def __init__(self, x):
        super(sign, self).__init__(x)

    @Atom.numpy_numeric
    def numeric(self, values):
        """Returns the sign of x.
        """
        x = values[0]
        x[x > 0] = 1.0
        x[x <= 0] = -1.0
        return x

    def shape_from_args(self):
        """Returns the (row, col) shape of the expression.
        """
        return tuple()

    def sign_from_args(self):
        """Returns sign (is positive, is negative) of the expression.
        """
        return (self.args[0].is_nonneg(), self.args[0].is_nonpos())

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
        return True

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
