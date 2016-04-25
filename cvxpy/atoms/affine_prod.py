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

from cvxpy.atoms.atom import Atom
import cvxpy.utilities as u
import operator as op
import numpy as np

class affine_prod(Atom):
    """Product of two affine expressions.
    """
    OP_NAME = "*"
    OP_FUNC = op.mul

    def __init__(self, x, y):
        super(affine_prod, self).__init__(x, y)

    @Atom.numpy_numeric
    def numeric(self, values):
        """Returns the product of two affine expressions.
        """
        return np.dot(values[0], values[1])

    def size_from_args(self):
        """Returns the (row, col) size of the expression.
        """
        return u.shape.mul_shapes(self.args[0].size, self.args[1].size)

    def sign_from_args(self):
        """Returns sign (is positive, is negative) of the expression.
        """
        # Unknown.
        return (False, False)

    def is_atom_convex(self):
        """Affine times affine is not convex
        """
        return False

    def is_atom_concave(self):
        """Affine times affine is not concave
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

    def validate_arguments(self):
        """Check dimensions of arguments and linearity.
        """
        if not self.args[0].is_affine() or not self.args[1].is_affine():
            raise ValueError("The arguments to affine_prod must be affine.")
        u.shape.mul_shapes(self.args[0].size, self.args[1].size)

    def is_quadratic(self):
        """Is the expression quadratic?
        """
        return True

    @staticmethod
    def graph_implementation(arg_objs, size, data=None):
        return NotImplemented
