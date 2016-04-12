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
    # The curvature of the atom if all arguments conformed to DCP.
    def func_curvature(self):
        return u.Curvature.AFFINE

    def sign_from_args(self):
        """By default, the sign is the most general of all the argument signs.
        """
        arg_signs = [arg._dcp_attr.sign for arg in self.args]
        return reduce(op.add, arg_signs)

    # Doesn't matter for affine atoms.
    def monotonicity(self):
        return len(self.args)*[u.monotonicity.INCREASING]
