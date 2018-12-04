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
import scipy.sparse as sp


class one_minus(Atom):
    def __init__(self, x):
        super(one_minus, self).__init__(x)
        if x.shape != tuple():
            raise ValueError("Argument to `one_minus` must be a scalar, "
                             "received ", x)
        self.args[0] = x

    def numeric(self, values):
        return 1.0 - values[0]

    def _grad(self, values):
        del values
        return sp.csc_matrix(-1.0)

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
        return False

    def is_atom_log_log_concave(self):
        """Is the atom log-log concave?
        """
        return True

    def is_incr(self, idx):
        """Is the composition non-decreasing in argument idx?
        """
        return False

    def is_decr(self, idx):
        """Is the composition non-increasing in argument idx?
        """
        return True
