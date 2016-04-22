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

import sys
import abc
import cvxpy.utilities as u
from cvxpy.atoms.atom import Atom
import operator as op
if sys.version_info >= (3, 0):
    from functools import reduce

class AffAtom(Atom):
    """ Abstract base class for affine atoms. """
    __metaclass__ = abc.ABCMeta

    def sign_from_args(self):
        """By default, the sign is the most general of all the argument signs.
        """
        return u.sign.sum_signs([arg for arg in self.args])

    def is_atom_convex(self):
        """Is the atom convex?
        """
        return True

    def is_atom_concave(self):
        """Is the atom concave?
        """
        return True

    def is_incr(self, idx):
        """Is the composition non-decreasing in argument idx?
        """
        # Defaults to increasing.
        return True

    def is_decr(self, idx):
        """Is the composition non-increasing in argument idx?
        """
        # Defaults to increasing.
        return False

    def is_quadratic(self):
        return all([arg.is_quadratic() for arg in self.args])
    